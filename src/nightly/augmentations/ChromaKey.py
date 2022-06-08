import random
import albumentations as A


class ChromaKey(A.BasicTransform):
    def __init__(
        self,
        bg_dataset,
        height_cut_range=(0.5, 1.),
        width_cut_range=(0.5, 1.),
        top_bottom_margin=(0., 0.),  # (min, max)
        left_right_margin=(0., 0.),  # (min, max)
        fg_transforms=None,  # do not use spatial transforms
        bg_transforms=None,
        always_apply=False,
        p=1
    ):
        super(ChromaKey, self).__init__(always_apply, p)
        self.height_cut_range = height_cut_range
        self.width_cut_range = width_cut_range
        self.top_bottom_margin = top_bottom_margin
        self.left_right_margin = left_right_margin
        self.bg_dataset = bg_dataset
        self.fg_transforms = fg_transforms if fg_transforms is not None else []
        self.bg_transforms = bg_transforms if bg_transforms is not None else []

        self.sub_sample = None
    
    def image_apply(self, image, **kwargs):
        # get sub_sample
        index = random.randrange(self.bg_dataset.__len__())
        bg_transform = A.Compose(self.bg_transforms)
        self.bg_sample = bg_transform(**self.bg_dataset[index])

        frame = np.ones_like(self.bg_sample['image']) * [0, 225, 0]
        frame_h, frame_w = frame.shape[:2]
        image_h, image_w = image.shape[:2]
        
        # cut values
        cut_h_min, cut_h_max = self.height_cut_range
        cut_w_min, cut_w_max = self.width_cut_range

        if isinstance(cut_h_min, float):
            cut_h_min = min(int(frame_h * cut_h_min), image_h)
        if isinstance(cut_h_max, float):
            cut_h_max = min(int(frame_h * cut_h_max), image_h)
        if isinstance(cut_w_min, float):
            cut_w_min = min(int(frame_w * cut_w_min), image_w)
        if isinstance(cut_w_max, float):
            cut_w_max = min(int(frame_w * cut_w_max), image_w)

        # margin values
        margin_t, margin_b = self.top_bottom_margin
        margin_l, margin_r = self.left_right_margin

        if isinstance(margin_t, float):
            margin_t = int(frame_h * margin_t)
        if isinstance(margin_b, float):
            margin_b = int(frame_h * margin_b)
        if isinstance(margin_l, float):
            margin_l = int(frame_w * margin_l)
        if isinstance(margin_r, float):
            margin_r = int(frame_w * margin_r)

        # cut randomly
        self.cut_h = random.randrange(cut_h_min, cut_h_max + 1)
        self.cut_w = random.randrange(cut_w_min, cut_w_max + 1)
        self.frame_cut_y = random.randrange(margin_t, frame_h - self.cut_h - margin_b + 1)
        self.frame_cut_x = random.randrange(margin_l, frame_w - self.cut_w - margin_r + 1)
        self.image_cut_y = random.randrange(0, image_h - self.cut_h + 1)
        self.image_cut_x = random.randrange(0, image_w - self.cut_w + 1)

        # cut mix
        frame[self.frame_cut_y:self.frame_cut_y+self.cut_h, self.frame_cut_x:self.frame_cut_x+self.cut_w]\
         = image[self.image_cut_y:self.image_cut_y+self.cut_h, self.image_cut_x:self.image_cut_x+self.cut_w]

        frame = frame.astype(np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (50, 150, 0), (100, 255, 255))

        fg_transform = A.Compose(self.fg_transforms)
        frame = fg_transform(image=frame)['image']

        result = cv2.copyTo(self.bg_sample['image'], mask, frame)
        return result

    @property
    def targets(self):
        return {'image': self.image_apply}