import random
import albumentations as A


class CutMix(A.BasicTransform):
    def __init__(
        self,
        sub_dataset,
        height_cut_range=(0.3, 0.5),
        width_cut_range=(0.3, 0.5),
        top_bottom_margin=(0.1, 0.1),  # (min, max)
        left_right_margin=(0.1, 0.1),  # (min, max)
        sub_transforms=None,
        apply_label=True,
        apply_mask=True,
        always_apply=False,
        p=1
    ):
        super(CutMix, self).__init__(always_apply, p)
        self.height_cut_range = height_cut_range
        self.width_cut_range = width_cut_range
        self.top_bottom_margin = top_bottom_margin
        self.left_right_margin = left_right_margin
        self.sub_dataset = sub_dataset
        self.sub_transforms = sub_transforms if sub_transforms is not None else []
        self.apply_label = apply_label
        self.apply_mask = apply_mask

        self.sub_sample = None
        self.cut_h = 0
        self.cut_w = 0
        self.main_cut_y = 0
        self.main_cut_x = 0
        self.sub_cut_y = 0
        self.sub_cut_x = 0

    @property
    def target_dependence(self):
        return {'label': ['image']}

    def image_apply(self, image, **kwargs):
        # get sub_sample
        index = random.randrange(self.sub_dataset.__len__())
        transformed = A.Compose(self.sub_transforms)
        self.sub_sample = transformed(**self.sub_dataset[index])

        sub_image = self.sub_sample['image']
        main_h, main_w = image.shape[:2]
        sub_h, sub_w = sub_image.shape[:2]

        # cut values
        cut_h_min, cut_h_max = self.height_cut_range
        cut_w_min, cut_w_max = self.width_cut_range

        if isinstance(cut_h_min, float):
            cut_h_min = min(int(main_h * cut_h_min), sub_h)
        if isinstance(cut_h_max, float):
            cut_h_max = min(int(main_h * cut_h_max), sub_h)
        if isinstance(cut_w_min, float):
            cut_w_min = min(int(main_w * cut_w_min), sub_w)
        if isinstance(cut_w_max, float):
            cut_w_max = min(int(main_w * cut_w_max), sub_w)

        # margin values
        margin_t, margin_b = self.top_bottom_margin
        margin_l, margin_r = self.left_right_margin

        if isinstance(margin_t, float):
            margin_t = int(main_h * margin_t)
        if isinstance(margin_b, float):
            margin_b = int(main_h * margin_b)
        if isinstance(margin_l, float):
            margin_l = int(main_w * margin_l)
        if isinstance(margin_r, float):
            margin_r = int(main_w * margin_r)

        # cut randomly
        self.cut_h = random.randrange(cut_h_min, cut_h_max + 1)
        self.cut_w = random.randrange(cut_w_min, cut_w_max + 1)
        self.main_cut_y = random.randrange(margin_t, main_h - self.cut_h - margin_b + 1)
        self.main_cut_x = random.randrange(margin_l, main_w - self.cut_w - margin_r + 1)
        self.sub_cut_y = random.randrange(0, sub_h - self.cut_h + 1)
        self.sub_cut_x = random.randrange(0, sub_w - self.cut_w + 1)

        # cut mix
        image[self.main_cut_y:self.main_cut_y+self.cut_h, self.main_cut_x:self.main_cut_x+self.cut_w]\
         = sub_image[self.sub_cut_y:self.sub_cut_y+self.cut_h, self.sub_cut_x:self.sub_cut_x+self.cut_w]

        return image

    def label_apply(self, label, **kwargs):
        if self.apply_label:
            h, w = kwargs['image'].shape[:2]
            r = (self.cut_h * self.cut_w) / (h * w)
            label = label * (1 - r) + self.sub_sample["label"] * r
        return label

    def mask_apply(self, mask, **kwargs):
        if self.apply_mask:
            sub_mask = self.sub_sample['mask']

            mask[self.main_cut_y:self.main_cut_y+self.cut_h, self.main_cut_x:self.main_cut_x+self.cut_w]\
             = sub_mask[self.sub_cut_y:self.sub_cut_y+self.cut_h, self.sub_cut_x:self.sub_cut_x+self.cut_w]

        return mask

    @property
    def targets(self):
        return {'image': self.image_apply, 'label': self.label_apply, 'mask': self.mask_apply}
