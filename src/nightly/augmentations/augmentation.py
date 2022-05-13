import os
import glob
import json
import random
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from copy import deepcopy


class KelvinWB(A.ImageOnlyTransform):
    def __init__(
        self,
        k_min=5000,
        k_max=15000,
        json_file=None,
        always_apply=False,
        p=1
    ):
        super(KelvinWB, self).__init__(always_apply, p)
        self.k_min = k_min
        self.k_max = k_max
        self.json_file = json_file if json_file else os.path.join(os.path.dirname(__file__), "kelvin_table.json")

    def apply(self, image, **kwargs):
        infile = open(self.json_file, 'r')
        temperature_dict = json.load(infile)
        self.k_min = -self.k_min // 100 * -100
        self.k_max = self.k_max // 100 * 100
        num = str(random.randrange(max(1000, self.k_min), min(40001, self.k_max+1), 100))
        temp = temperature_dict[num]
        image = Image.fromarray(image)
        r, g, b = temp[0], temp[1], temp[2]
        matrix = ( b / 255.0, 0.0, 0.0, 0.0,
                   0.0, g / 255.0, 0.0, 0.0,
                   0.0, 0.0, r / 255.0, 0.0 )
        image = image.convert('RGB', matrix)
        image = np.array(image)
        return image


class RandomLocateInFrame(A.BasicTransform):
    def __init__(
        self,
        h,
        w,
        interpolation=cv2.INTER_AREA,
        border_mode=cv2.BORDER_REPLICATE,
        always_apply=False,
        p=1,
    ):
        super(RandomLocateInFrame, self).__init__(always_apply, p)
        self.h = h
        self.w = w
        self.interpolation = interpolation
        self.border_mode = border_mode

    @property
    def target_dependence(self):
        return {'image': ['bboxes', 'labels'],
                'bboxes': ['image'],
                'labels': ['image', 'bboxes']}

    def crop_side_with_safe_bbox(self, frame_side, img_side, safe_set):
        lst = list(set(range(img_side - frame_side)) & safe_set)
        if len(lst) == 0:
            lst.append(0)
        side_min = random.choice(lst)
        side_max = side_min + frame_side + 1
        while side_max < img_side:
            if side_max in safe_set:
                break
            side_max += 1
        return side_min, side_max
            
    def apply(self, image, **params):
        img_h, img_w, _ = image.shape

        h_set = set(range(img_h))
        w_set = set(range(img_w))

        for bbox in params['bboxes']:
            w_set -= set(range(int(img_w * bbox[0] + 0.5), int(img_w * bbox[2] + 0.5) + 1))
            h_set -= set(range(int(img_h * bbox[1] + 0.5), int(img_h * bbox[3] + 0.5) + 1))

        # crop
        if self.h < img_h:
            self.img_h_min, self.img_h_max = self.crop_side_with_safe_bbox(self.h, img_h, h_set)
        else:
            self.img_h_min, self.img_h_max = 0, img_h

        if self.w < img_w:
            self.img_w_min, self.img_w_max = self.crop_side_with_safe_bbox(self.w, img_w, w_set)
        else:
            self.img_w_min, self.img_w_max = 0, img_w
        
        img = image[self.img_h_min:self.img_h_max, self.img_w_min:self.img_w_max, ...]

        # resize and locate
        img_h, img_w, _ = img.shape
        if self.h / img_h < self.w / img_w:
            self.r = self.h / img_h
        else:
            self.r = self.w / img_w

        self.h_min = random.randrange(self.h - int(img_h * self.r + 0.5) + 1)
        self.w_min = random.randrange(self.w - int(img_w * self.r + 0.5) + 1)

        mtrx = np.array(
            [[self.r, 0, self.w_min],
             [0, self.r, self.h_min]]
        , dtype=np.float32)

        img = cv2.warpAffine(img, mtrx, (self.w, self.h), flags=self.interpolation, borderMode=self.border_mode)

        return img

    def apply_to_bboxes(self, bboxes, **params):
        new_bboxes = []
        img_h, img_w, _ = params['image'].shape
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = int(bbox[0] * img_w + 0.5), int(bbox[1] * img_h + 0.5), int(bbox[2] * img_w + 0.5), int(bbox[3] * img_h + 0.5)
            if x_min >= self.img_w_min and y_min >= self.img_h_min and x_max <= self.img_w_max and y_max <= self.img_h_max:
                new_bbox = (
                    ((x_min - self.img_w_min) * self.r + self.w_min) / self.w,
                    ((y_min - self.img_h_min) * self.r + self.h_min) / self.h,
                    ((x_max - self.img_w_min) * self.r + self.w_min) / self.w,
                    ((y_max - self.img_h_min) * self.r + self.h_min) / self.h,
                    bbox[-1]
                    )
                new_bboxes.append(new_bbox)
        return new_bboxes

    def apply_to_labels(self, labels, **params):
        new_labels = []
        img_h, img_w, _ = params['image'].shape
        for bbox, label in zip(params['bboxes'], labels):
            x_min, y_min, x_max, y_max = bbox[0] * img_w, bbox[1] * img_h, bbox[2] * img_w, bbox[3] * img_h
            if x_min >= self.img_w_min and y_min >= self.img_h_min and x_max < self.img_w_max and y_max < self.img_h_max:
                new_labels.append(label)
        
        return new_labels

    @property
    def targets(self):
        return {'image': self.apply, 'bboxes': self.apply_to_bboxes, 'labels': self.apply_to_labels}


class Mosaic(A.BasicTransform):
    def __init__(
        self,
        dataset,
        height_split_range=(0.25, 0.75),
        width_split_range=(0.25, 0.75),
        transforms=[],
        bbox_params=None,
        always_apply=False,
        p=0.5,
    ):
        super(Mosaic, self).__init__(always_apply, p)
        self.dataset = dataset
        self.height_split_range = height_split_range
        self.width_split_range = width_split_range
        if bbox_params is None:
            bbox_params = A.BboxParams(format=dataset.bbox_format, min_area=0.3, min_visibility=0.3, label_fields=['labels'])
        self.transforms = transforms
        self.bbox_params = bbox_params

    @property
    def target_dependence(self):
        return {'image': ['bboxes', 'labels']}

    def get_piece(self, data, h, w):
        transforms = self.transforms + [RandomLocateInFrame(h, w, always_apply=True)]
        augmentation = A.Compose(transforms, bbox_params=self.bbox_params)
        data = augmentation(**data)
        return data

    def locate_bboxes(self, piece, h_r, w_r, h_p, w_p):
        def locate_bbox(bbox):
            return [w_r * w_p + bbox[0][0] * abs(w_p - w_r),
                    h_r * h_p + bbox[0][1] * abs(h_p - h_r),
                    w_r * w_p + bbox[0][2] * abs(w_p - w_r),
                    h_r * h_p + bbox[0][3] * abs(h_p - h_r),
                    bbox[1]]
        return list(map(locate_bbox, zip(piece['bboxes'], piece['labels'])))

    def apply(self, image, **params):
        height, width = params['rows'], params['cols']
        bboxes, labels = [], []
        for bbox in params['bboxes']:
            bboxes.append(list(bbox[:-1]))
            labels.append(bbox[-1])
        params['bboxes'] = bboxes
        params['labels'] = labels
        
        h_r = random.uniform(*self.height_split_range)
        w_r = random.uniform(*self.width_split_range)
        h = int(height * h_r + 0.5)
        w = int(width * w_r + 0.5)
        tl = self.get_piece({'image': image, **params}, h, w)
        tr = self.get_piece(self.dataset[random.randrange(self.dataset.__len__())], h, width - w)
        bl = self.get_piece(self.dataset[random.randrange(self.dataset.__len__())], height - h, w)
        br = self.get_piece(self.dataset[random.randrange(self.dataset.__len__())], height - h, width - w)
        t = np.concatenate((tl['image'], tr['image']), axis=1)
        b = np.concatenate((bl['image'], br['image']), axis=1)
        image = np.concatenate((t, b), axis=0)

        bboxes = self.locate_bboxes(tl, h_r, w_r, 0, 0)
        bboxes += self.locate_bboxes(tr, h_r, w_r, 0, 1)
        bboxes += self.locate_bboxes(bl, h_r, w_r, 1, 0)
        bboxes += self.locate_bboxes(br, h_r, w_r, 1, 1)
        self.bboxes = bboxes

        return image

    def apply_to_bboxes(self, bboxes, **params):
        return deepcopy(self.bboxes)

    @property
    def targets(self):
        return {'image': self.apply, 'bboxes': self.apply_to_bboxes}


class Sticker(A.BasicTransform):
    # crop the boxes from target image and paste it on another image.
    # scale_range: ratio of sticker size to background (float) (0., 1.)
    # rotate_range: sticker rotation angle (degree) (float) (-360., 360.)
    # variation_range: ratio of randomly moving sticker vertices (float) (-0.4, 0.4)
    def __init__(
        self,
        dataset,
        scale_range=(0.2, 0.4),
        rotate_range=(-5, 5),
        variation_range=(-0.1, 0.1),
        backgrounds=None,
        transforms=[],
        bg_transforms=[],
        bbox_params=None,
        annotation=True,
        always_apply=False,
        p=0.5,
    ):
        super(Sticker, self).__init__(always_apply, p)
        self.dataset = dataset
        self.scale_range = scale_range
        self.rotate_range = rotate_range
        self.variation_range = variation_range
        self.backgrounds = backgrounds
        self.transforms = transforms
        self.bg_transforms = bg_transforms
        if bbox_params is None:
            self.bbox_params = A.BboxParams(format=dataset.bbox_format, min_area=0.3, min_visibility=0.3, label_fields=['labels'])
        self.annotation = annotation

    @property
    def target_dependence(self):
        return {'image': ['bboxes', 'labels']}

    def detach(self, image, bbox, label):
        h, w, _ = image.shape
        transforms = [
            A.Crop(int(bbox[0] * w + 0.5), int(bbox[1] * h + 0.5), int(bbox[2] * w + 0.5), int(bbox[3] * h + 0.5), always_apply=True),
            ] + self.transforms
        transformed = A.Compose(transforms, bbox_params=self.bbox_params)
        return transformed(image=image, bboxes=[bbox], labels=[label])

    def attach(self, sticker, bg, mask_board):
        h, w, _ = sticker.shape
        bg_h, bg_w, _ = bg.shape
        
        # make homography matrix
        s = random.uniform(*self.scale_range)
        r = np.pi * random.uniform(*self.rotate_range) / 180
        x_min, y_min, x_max, y_max = 0, 0, w, h
        p_before = [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]]
        # variation
        p_after = [[px + w * random.uniform(*self.variation_range), py + h * random.uniform(*self.variation_range)] for (px, py) in p_before]
        # rotate
        p_after = [[px * np.cos(r) - py * np.sin(r), px * np.sin(r) + py * np.cos(r)] for (px, py) in p_after]
        # calib zero
        px_min, py_min = min(p_after, key=lambda x: x[0])[0], min(p_after, key=lambda x: x[1])[1]
        p_after = [[px - px_min, py - py_min] for (px, py) in p_after]
        px_max, py_max = max(p_after, key=lambda x: x[0])[0], max(p_after, key=lambda x: x[1])[1]
        # fit in bg and scaling
        f = min(bg_w / px_max, bg_h / py_max)
        p_after = [[px * f * s, py * f * s] for (px, py) in p_after]
        # randomly move
        px_max, py_max = max(p_after, key=lambda x: x[0])[0], max(p_after, key=lambda x: x[1])[1]
        px_move, py_move = random.randrange(max(bg_w - int(px_max + 0.5), 0) + 1), random.randrange(max(bg_h - int(py_max + 0.5), 0) + 1)
        p_after = [[px + px_move, py + py_move] for (px, py) in p_after]
        # cast
        p_before, p_after = np.array(p_before, dtype=np.float32), np.array(p_after, dtype=np.float32)
        mtrx = cv2.getPerspectiveTransform(p_before, p_after)

        # make mask
        mask = cv2.warpPerspective(np.ones(sticker.shape, dtype=np.uint8), mtrx, (bg_w, bg_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        if np.sum(mask_board * mask) == 0:
            bg = cv2.warpPerspective(sticker, mtrx, (bg_w, bg_h), bg, borderMode=cv2.BORDER_TRANSPARENT)
            # bg = cv2.bitwise_and(bg, cv2.bitwise_not(mask))
            # bg = cv2.bitwise_xor(bg, cv2.bitwise_or(mask, sticker))
            mask_board += mask
            bbox = (min(p_after, key=lambda x: x[0])[0] / bg_w,
                    min(p_after, key=lambda x: x[1])[1] / bg_h,
                    max(p_after, key=lambda x: x[0])[0] / bg_w,
                    max(p_after, key=lambda x: x[1])[1] / bg_h)
            return bg, bbox, mask_board
        else:
            return None

    def rotate_bbox(self, bbox, M):
        # unit: pixel
        x_min, y_min, x_max, y_max = bbox
        loc00 = M @ np.array([[x_min, y_min, 1]]).T
        loc01 = M @ np.array([[x_min, y_max, 1]]).T
        loc10 = M @ np.array([[x_max, y_min, 1]]).T
        loc11 = M @ np.array([[x_max, y_max, 1]]).T
        con = np.concatenate((loc00, loc01, loc10, loc11), axis=1)
        x_min, y_min = np.min(con, axis=1)
        x_max, y_max = np.max(con, axis=1)
        return x_min, y_min, x_max, y_max

    def trim_bbox(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        return max(x_min, 0.), max(y_min, 0.), min(x_max, 1.), min(y_max, 1.)

    def apply(self, image, **params):
        bboxes, labels = [], []
        for bbox in params['bboxes']:
            bboxes.append(list(bbox[:-1]))
            labels.append(bbox[-1])

        self.attached_bboxes = []
        self.attached_labels = []

        if self.backgrounds is not None:
            bg = cv2.imread(random.choice(glob.glob(os.path.join(self.backgrounds, '*'))))
            bg = A.Compose(self.bg_transforms)(image=bg)['image']
        else:
            bg_sample = self.dataset[random.randrange(self.dataset.__len__())]
            bg_sample = A.Compose(self.bg_transforms, bbox_params=self.bbox_params)(**bg_sample)
            bg = bg_sample['image']
            self.attached_bboxes += bg_sample['bboxes']
            self.attached_labels += bg_sample['labels']

        mask_board = np.zeros(bg.shape, dtype=np.uint8)
        for bbox in self.attached_bboxes:
            bg_h, bg_w, _ = bg.shape
            x_min, y_min, x_max, y_max = int(bbox[0] * bg_w + 0.5), int(bbox[1] * bg_h + 0.5), int(bbox[2] * bg_w + 0.5), int(bbox[3] * bg_h + 0.5)
            mask_board[y_min:y_max+1, x_min:x_max+1, ...] = 1
        
        indexes = list(range(len(bboxes)))
        random.shuffle(indexes)

        for i in indexes:
            x_min, y_min, x_max, y_max = bboxes[i]
            if x_min >= 0 and y_min >= 0 and x_max <= 1 and y_max <= 1:
                data = self.detach(image, bboxes[i], labels[i])
                rst = self.attach(data['image'], bg, mask_board)
                if rst:
                    bg, bbox, mask_board = rst
                    if self.annotation:
                        self.attached_bboxes.append(bbox)
                        self.attached_labels.append(data['labels'][0])

        return bg

    def apply_to_bboxes(self, bboxes, **params):
        bboxes = []
        for bbox, label in zip(self.attached_bboxes, self.attached_labels):
            bbox = self.trim_bbox(bbox)
            bboxes.append((*bbox, label))
        return bboxes

    @property
    def targets(self):
        return {'image': self.apply, 'bboxes': self.apply_to_bboxes}


class RandomLocationCrop(A.BasicTransform):
    def __init__(
        self,
        height,
        width,
        always_apply=False,
        p=1
    ):
        super(RandomLocationCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.scale_rate = scale_rate
        self.scale_limit = scale_limit

    def image_apply(self, image, **kwargs):
        random.randrange()
        h, w, c = image.shape
        y_max = h - self.height
        x_max = w - self.width
        y = random.randint(0, y_max - 1)
        x = random.randint(0, x_max - 1)
        return image[y:y+self.height, x:x+self.width, :]

    @property
    def targets(self):
        return {"image": self.image_apply}


class ShortBaseScale(A.BasicTransform):
    def __init__(
        self,
        scale,
        interpolation=None,
        always_apply=False,
        p=1
    ):
        super(ShortBaseScale, self).__init__(always_apply, p)
        self.scale = scale
        self.interpolation = interpolation if interpolation is not None else cv2.INTER_LINEAR
        self.r = 1

    def image_apply(self, image, **kwargs):
        h, w, c = image.shape
        if h < w:
            size = (-(-w * self.scale // h), self.scale)
        else:
            size = (self.scale, -(-h * self.scale // w))
        return cv2.resize(image, size, interpolation=self.interpolation)

    def bboxes_apply(self, bboxes, **kwargs):
        return [tuple(map(lambda x: -int(-x*self.r), bbox)) for bbox in bboxes]

    @property
    def targets(self):
        return {"image": self.image_apply, 'bboxes': self.bboxes_apply}