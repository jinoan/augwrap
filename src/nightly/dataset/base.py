import logging


def get_dict_value_index(dct, index):
    result = {}
    for k, v in dct.items():
        result[k] = v[index]
    return result


def base_inheritance(cls):
    def inherit_base(dataset, *args, **kwargs):
        cls_name = cls.__name__
        if isinstance(dataset, list):
            sup = dataset[0]
        else:
            sup = dataset
        return type(cls_name, (cls, *sup.__class__.mro()[-3:],), {})(dataset, *args, **kwargs)
    return inherit_base


class BaseDataset:
    def __init__(self, inputs, **kwargs):
        # inputs: Input dataset of dictionary type. example)
        # (i.e. {'image': ['a.jpg', 'b.jpg'], 'label': ['a', 'b']})
        self.__dict__ = kwargs
        self.inputs = inputs

    def __len__(self):
        return len(list(self.inputs.values())[0])

    def __getitem__(self, index):
        return get_dict_value_index(self.inputs, index)


try:
    from torch.utils.data import Dataset


    class TorchBaseDataset(Dataset):
        def __init__(self, inputs, **kwargs):
            # inputs: Input dataset of dictionary type.
            # (i.e. {'image': ['a.jpg', 'b.jpg'], 'label': ['a', 'b']})
            super(TorchBaseDataset, self).__init__()
            self.__dict__ = kwargs
            self.inputs = inputs

        def __len__(self):
            return len(list(self.inputs.values())[0])

        def __getitem__(self, index):
            return get_dict_value_index(self.inputs, index)

except ImportError:
    logging.warning("TorchBaseDataset is unable because Pytorch couldn't be imported.")

try:
    from tensorflow.keras.utils import Sequence


    class TFBaseDataset(Sequence):
        def __init__(self, inputs, **kwargs):
            # inputs: Input dataset of dictionary type. example)
            # (i.e. {'image': ['a.jpg', 'b.jpg'], 'label': ['a', 'b']})
            super(TFBaseDataset, self).__init__()
            self.__dict__ = kwargs
            self.inputs = inputs

        def __len__(self):
            return len(list(self.inputs.values())[0])

        def __getitem__(self, index):
            return get_dict_value_index(self.inputs, index)

except ImportError:
    logging.warning("TFBaseDataset is unable because Tensorflow 2 couldn't be imported.")