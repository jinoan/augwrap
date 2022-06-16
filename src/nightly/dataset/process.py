import cv2
import numpy as np
from copy import copy
from collections import defaultdict
from sklearn.model_selection import KFold as KF
from .base import get_dict_value_index, base_inheritance
# import xml.etree.ElementTree as ET


@base_inheritance
class LoadImage:
    def __init__(self, dataset, targets, color_mode=1):
        # targets: List of inputs target names which contain image paths.
        # color_mode: 1: "color", 0: "grey", -1: "unchanged"
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.targets = targets
        self.color_mode = {1: "color", 0: "grey", -1: "unchanged"}.get(color_mode, color_mode)

    def __getitem__(self, index):
        sample = self.dataset[index]
        for target in self.targets:
            sample[target] = self.load_img(sample[target])
        return sample

    def load_img(self, img_path):
        img = cv2.imread(
            img_path,
            flags={
                "color": cv2.IMREAD_COLOR,
                "grey": cv2.IMREAD_GRAYSCALE,
                "unchanged": cv2.IMREAD_UNCHANGED
            }[self.color_mode]
        )
        # if img.ndim == 2:
        #     img = img[..., np.newaxis]
        return img


@base_inheritance
class ResizeImage:
    def __init__(self, dataset, targets, image_size=(200, 200), interpolation=cv2.INTER_LINEAR):
        # targets: List of inputs target names which contain image.
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.targets = targets
        self.image_size = image_size
        self.interpolation = interpolation

    def __getitem__(self, index):
        sample = self.dataset[index]
        for target in self.targets:
            sample[target] = cv2.resize(sample[target], dsize=self.image_size, interpolation=self.interpolation)
        return sample


@base_inheritance
class NormalizeImage:
    def __init__(self, dataset, targets, feature_range=(-1, 1), dtype=np.float32):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.targets = targets
        self.feature_range = feature_range
        self.dtype = dtype

    def __getitem__(self, index):
        sample = self.dataset[index]
        f_min, f_max = self.feature_range
        for target in self.targets:
            sample[target] = (sample[target].astype(self.dtype) / 255) * (f_max - f_min) + f_min
        return sample


@base_inheritance
class OneHot:
    def __init__(self, dataset, targets, classes, dtype=np.float32):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.targets = targets
        self.classes = classes
        self.dtype = dtype

    def __getitem__(self, index):
        sample = self.dataset[index]
        for target in self.targets:
            sample[target] = self.apply(sample[target])
        return sample

    def apply(self, label):
        if label:
            stack = [np.array(label == cls, dtype=self.dtype) for cls in self.classes]
            output = np.stack(stack, axis=-1)
        else:
            output = np.ones_like(self.classes, dtype=self.dtype) / len(self.classes)
        return output


@base_inheritance
class Sparse:
    def __init__(self, dataset, targets, classes, dtype=np.float32):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.targets = targets
        self.classes = classes
        self.dtype = dtype

    def __getitem__(self, index):
        sample = self.dataset[index]
        for target in self.targets:
            sample[target] = self.classes.index(sample[target])
        return sample


@base_inheritance
class Augmentation:
    def __init__(self, dataset, augmentations):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.augmentations = augmentations

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample = self.augmentations(**sample)
        return sample


@base_inheritance
class Transform:
    def __init__(self, dataset, transforms):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample = self.transforms(sample)
        return sample


@base_inheritance
class TFDataGenerator:
    def __init__(self, dataset, batch_size=1, shuffle=False, tuple_map=None):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tuple_map = tuple_map
        self.indices = np.arange(self.dataset.__len__())
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return -(-self.dataset.__len__() // self.batch_size)

    def __getitem__(self, batch_index):
        indices = self.indices[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        result = self.data_generation(indices)
        if self.tuple_map:
            result = self.to_tuple(result, self.tuple_map)
        # return batch size items
        return result
        
    def data_generation(self, indices):
        result = defaultdict(list)
        for index in indices:
            sample = self.dataset[index]
            for k, v in sample.items():
                result[k].append(v)
        for k, v in result.items():
            result[k] = np.stack(v, axis=0)
        return result

    def to_tuple(self, sample, tuple_map):
        return tuple([sample[k] for k in tuple_map])


class KFold:
    def __init__(self, dataset):
        self.__dict__ = dataset.__dict__.copy()
        self.dataset = dataset

    def split(self, n_splits=5):
        kf = KF(n_splits=n_splits, shuffle=True).split(range(self.dataset.__len__()))
        folds = []
        for train_indices, valid_indices in kf:
            # make train dataset
            train_dataset = copy(self.dataset)
            train_inputs = defaultdict(list)
            for i in train_indices:
                for k, v in get_dict_value_index(self.inputs, i).items():
                    train_inputs[k].append(v)
            train_dataset.__dict__["inputs"] = train_inputs

            # make validation dataset
            valid_dataset = copy(self.dataset)
            valid_inputs = defaultdict(list)
            for i in valid_indices:
                for k, v in get_dict_value_index(self.inputs, i).items():
                    valid_inputs[k].append(v)
            valid_dataset.__dict__["inputs"] = valid_inputs

            folds.append((train_dataset, valid_dataset))
        return folds


@base_inheritance
class MergeDataset:
    def __init__(self, datasets):
        dct = {}
        for dataset in datasets:
            dct.update(dataset.__dict__)
        dct['inputs'] = {}
        for dataset in datasets:
            for k, v in dataset.__dict__['inputs'].items():
                dct['inputs'].setdefault(k, type(v)())
                dct['inputs'][k] += v
        self.__dict__ = dct
        self.datasets = datasets
        
    def __getitem__(self, index):
        for dset in self.datasets:
            if index >= dset.__len__():
                index -= dset.__len__()
                continue
            return dset[index]
            

# @base_inheritance
# class LoadPascalVOCLabels:
#     def __init__(self, dataset):
#         self.__dict__ = dataset.__dict__.copy()
#         self.dataset = dataset
#         self.bbox_format = "albumentations"
#
#     def __getitem__(self, index):
#         sample = self.dataset[index]
#         sample['labels'], sample['bboxes'] = self.__decode_xml(sample.pop('label'))
#         return sample
#
#     def __decode_xml(self, label):
#         tree = ET.parse(label)
#         root = tree.getroot()
#         width = int(root.find('size').find('width').text)
#         height = int(root.find('size').find('height').text)
#         objects = root.findall('object')
#         labels, bboxes = [], []
#         for obj in objects:
#             labels.append(obj.find('name').text)
#             box = obj.find('bndbox')
#             xmin = float(box.find('xmin').text) / width
#             ymin = float(box.find('ymin').text) / height
#             xmax = float(box.find('xmax').text) / width
#             ymax = float(box.find('ymax').text) / height
#             bboxes.append([xmin, ymin, xmax, ymax])
#         return labels, bboxes