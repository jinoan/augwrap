import cv2
import random
import albumentations as A


class MixUp(A.BasicTransform):
    def __init__(
        self,
        sub_dataset,
        mix_range=(0.3, 0.5),
        sub_transforms=None,
        apply_label=True,
        apply_mask=True,
        always_apply=False,
        p=1
    ):
        super(MixUp, self).__init__(always_apply, p)
        self.mix_range = mix_range
        self.sub_dataset = sub_dataset
        self.sub_transforms = sub_transforms if sub_transforms is not None else []
        self.apply_label = apply_label
        self.apply_mask = apply_mask

        self.sub_sample = None
        self.size = None  # (w, h)
        self.mix_rate = 0.

    def image_apply(self, image, **kwargs):
        # get dtype
        dtype = image.dtype

        # get sub_sample
        index = random.randrange(self.sub_dataset.__len__())
        transformed = A.Compose(self.sub_transforms)
        self.sub_sample = transformed(**self.sub_dataset[index])

        # get image size
        sub_image = self.sub_sample['image']
        self.size = image.shape[1::-1]  # (w, h)
        
        # Sizing method might be changed.
        sub_image = cv2.resize(sub_image, self.size, interpolation=cv2.INTER_LINEAR)

        self.mix_rate = random.uniform(*self.mix_range)
        image = image * (1 - self.mix_rate) + sub_image * self.mix_rate
        image = image.astype(dtype)
        return image

    def label_apply(self, label, **kwargs):
        if self.apply_label:
            label = label * (1 - self.mix_rate) + self.sub_sample["label"] * self.mix_rate
        return label

    def mask_apply(self, mask, **kwargs):
        if self.apply_mask:
            sub_mask = cv2.resize(self.sub_sample['mask'], self.size, interpolation=cv2.INTER_LINEAR)
            mask = mask * (1 - self.mix_rate) + sub_mask * self.mix_rate
        return mask

    @property
    def targets(self):
        return {'image': self.image_apply, 'label': self.label_apply, 'mask': self.mask_apply}