import os
import glob
import json
import random
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from copy import deepcopy


class CutMix(A.BasicTransform):
    def __init__(
        self,
        dataset,
        bbox_width_range=(0.3, 0.5),
        bbox_height_range=(0.3, 0.5),
        pad_size=(0.1, 0.1),
        mix_label=True,
        always_apply=False,
        p=1
    ):
        super(CutMix, self).__init__(always_apply, p)
        self.dataset = dataset
        self.sub_sample = None
        self.bbox_width_range = bbox_width_range
        self.bbox_height_range = bbox_height_range
        self.pad_size = pad_size
        self.r = 0.
        self.mix_label = mix_label

    def image_apply(self, image, **kwargs):
        index = random.randrange(self.dataset.__len__())
        self.sub_sample = self.dataset[index]
        bx, by, bw, bh, self.r = rand_bbox(image.shape, self.bbox_width_range, self.bbox_height_range, self.pad_size)
        image[by:by+bh, bx:bx+bw, :] = self.sub_sample["image"][by:by+bh, bx:bx+bw, :]
        return image

    def label_apply(self, label, **kwargs):
        if self.mix_label:
            label = label * (1 - self.r) + self.sub_sample["label"] * self.r
        return label

    @property
    def targets(self):
        return {"image": self.image_apply, "label": self.label_apply}

def rand_bbox(image_shape, bbox_width_range, bbox_height_range, pad_size):
    # shape: (h, w, c)
    W = image_shape[1]
    H = image_shape[0]
    if str(type(bbox_width_range[0])) == "<class 'float'>":
        bw = int(W * random.uniform(*bbox_width_range) + 0.5)
    else:
        bw = random.randrange(*bbox_width_range)
    if str(type(bbox_height_range[0])) == "<class 'float'>":
        bh = int(H * random.uniform(*bbox_height_range) + 0.5)
    else:
        bh = random.randrange(*bbox_height_range)
    if str(type(pad_size[0])) == "<class 'float'>":
        pw = int(W * pad_size[0] + 0.5)
        ph = int(H * pad_size[1] + 0.5)
    else:
        pw, ph = pad_size
    bx = random.randrange(pw, W - pw - bw)
    by = random.randrange(ph, H - ph - bh)
    r = bw * bh / (W * H)
    return bx, by, bw, bh, r


class MixUp(A.BasicTransform):
    def __init__(
        self,
        dataset,
        rate_range=(0.3, 0.5),
        mix_label=True,
        always_apply=False,
        p=1
    ):
        super(MixUp, self).__init__(always_apply, p)
        self.dataset = dataset
        self.sub_sample = None
        self.rate_range = rate_range
        self.r = 0.
        self.mix_label = mix_label

    def image_apply(self, image, **kwargs):
        index = random.randrange(self.dataset.__len__())
        self.sub_sample = self.dataset[index]
        resize = A.Compose([
            A.RandomSizedBBoxSafeCrop(*image.shape[:2])
            ], bbox_params=A.BboxParams(format='albumentations', min_area=0.3, min_visibility=0.3, label_fields=['labels'])
        )
        self.sub_sample = resize(**self.sub_sample)
        self.r = random.uniform(*self.rate_range)
        image = image * (1 - self.r) + self.sub_sample["image"] * self.r
        image = np.uint8(image)
        return image

    def label_apply(self, label, **kwargs):
        if self.mix_label:
            label = label * (1 - self.r) + self.sub_sample["label"] * self.r
        return label

    @property
    def targets(self):
        return {"image": self.image_apply, "label": self.label_apply}


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
    def __init__(
        self,
        dataset,
        scale_range=(0.95, 1.05),
        degree_range=(-5, 5),
        wrap_range=(-0.05, 0.05),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        backgrounds=None,
        transforms=[],
        bg_transforms=[],
        bbox_params=None,
        always_apply=False,
        p=0.5,
    ):
        super(Sticker, self).__init__(always_apply, p)
        self.dataset = dataset
        self.scale_range = scale_range
        self.degree_range = degree_range
        self.wrap_range = wrap_range
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.backgrounds = backgrounds
        self.transforms = transforms
        self.bg_transforms = bg_transforms
        if bbox_params is None:
            self.bbox_params = A.BboxParams(format=dataset.bbox_format, min_area=0.3, min_visibility=0.3, label_fields=['labels'])

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

    def attach(self, sticker, bg, bbox, mask_board):
        h, w, _ = sticker.shape
        bg_h, bg_w, _ = bg.shape
        s = random.uniform(*self.scale_range)
        r = np.pi * random.uniform(*self.degree_range) / 180
        wps = [random.uniform(*self.wrap_range) for _ in range(4)]
        mtrx = np.array([[(s + wps[0]) * np.cos(r), (s + wps[1]) * -np.sin(r), 0],
                         [(s + wps[2]) * np.sin(r), (s + wps[3]) * np.cos(r), 0]])
        x_min, y_min, x_max, y_max = bbox
        _x_min, _y_min, _x_max, _y_max = self.rotate_bbox((x_min * w, y_min * h, x_max * w, y_max * h), mtrx)
        x_min, y_min, x_max, y_max = 0, 0, _x_max - _x_min, _y_max - _y_min

        x_move = random.randrange(max(bg_w - int(x_max + 0.5), 0) + 1)
        y_move = random.randrange(max(bg_h - int(y_max + 0.5), 0) + 1)
        mtrx[0, 2] = x_move - _x_min
        mtrx[1, 2] = y_move - _y_min
        mask = cv2.warpAffine(np.ones(sticker.shape, dtype=np.uint8) * 255, mtrx, (bg_w, bg_h), flags=self.interpolation, borderMode=self.border_mode)

        if np.sum(np.max(mask_board, axis=2) * np.max(mask, axis=2)) == 0:
            sticker = cv2.warpAffine(sticker, mtrx, (bg_w, bg_h), flags=self.interpolation, borderMode=self.border_mode)
            bg = cv2.bitwise_and(bg, cv2.bitwise_not(mask)) + sticker
            mask_board += mask
            bbox = ((x_min + x_move) / bg_w,
                    (y_min + y_move) / bg_h,
                    (x_max + x_move) / bg_w,
                    (y_max + y_move) / bg_h)
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
            mask_board[y_min:y_max+1, x_min:x_max+1, ...] = 255
        
        indexes = list(range(len(bboxes)))
        random.shuffle(indexes)

        for i in indexes:
            x_min, y_min, x_max, y_max = bboxes[i]
            if x_min >= 0 and y_min >= 0 and x_max <= 1 and y_max <= 1:
                data = self.detach(image, bboxes[i], labels[i])
                rst = self.attach(data['image'], bg, data['bboxes'][0], mask_board)
                if rst:
                    bg, bbox, mask_board = rst
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
