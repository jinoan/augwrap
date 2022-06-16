import logging
from copy import deepcopy

is_torch = True
is_tf = True


class BaseDataset:
    def __init__(self, inputs, **kwargs):
        # inputs: Input dataset of dictionary type. example)
        # (i.e. {'image': ['a.jpg', 'b.jpg'], 'label': ['a', 'b']})
        self.__dict__ = kwargs
        self.inputs = inputs

    # @property
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
    is_torch = False

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
    is_tf = False


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

        if sup.__class__.mro()[-2] == BaseDataset:
            mro = sup.__class__.mro()[-2:-1]
        elif is_tf and sup.__class__.mro()[-3] == TFBaseDataset:
            mro = sup.__class__.mro()[-3:-1]
        elif is_torch and sup.__class__.mro()[-4] == TorchBaseDataset:
            mro = sup.__class__.mro()[-4:-2]

        return type(cls_name, (cls, *mro,), {})(dataset, *args, **kwargs)
    return inherit_base