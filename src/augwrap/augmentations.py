import os
import random
import numpy as np
import json
from PIL import Image
from albumentations import BasicTransform, ImageOnlyTransform


class CutMix(BasicTransform):
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


class MixUp(BasicTransform):
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
        self.r = random.uniform(*self.rate_range)
        image = image * (1 - self.r) + 0.5 + self.sub_sample["image"] * self.r
        image = np.int16(image)
        return image

    def label_apply(self, label, **kwargs):
        if self.mix_label:
            label = label * (1 - self.r) + self.sub_sample["label"] * self.r
        return label

    @property
    def targets(self):
        return {"image": self.image_apply, "label": self.label_apply}


class KelvinWB(ImageOnlyTransform):
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