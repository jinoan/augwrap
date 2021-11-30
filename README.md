# AugWrap
[![PyPI version](https://badge.fury.io/py/augwrap.svg)](https://badge.fury.io/py/augwrap)



## Installation

```shell
$ pip install augwrap
```



## Basic Usage

1. Base dataset을 선택합니다. Pytorch 기반 데이터셋을 만들려면 `TorchBaseDataset`을 사용합니다. Tensorflow 기반 데이터셋을 만들려면 `TFBaseDataset`을 사용합니다.

   ```python
   import augwrap.data as D
   
   dataset = D.TorchBaseDataset(images, labels, classes)
   ```

2. `augwrap.data` 안의 모듈을 이용하여 학습 전 데이터를 처리합니다.

   ```python
   dataset = D.LoadImages(dataset)
   dataset = D.ResizeImages(dataset, (256, 256))
   dataset = D.OneHotLabels(dataset)
   ```

3. 데이터 로더를 생성하여 학습에 사용합니다.

   ```python
   from torch.utils.data import DataLoader
   
   data_loader = DataLoader(
       dataset,
       batch_size = 16,
       shuffle = False,
       num_workers = 4
   )
   ```



### Augmentations

Augmentation은 [Albumentations](https://github.com/albumentations-team/albumentations) 라이브러리를 이용하도록 설계했습니다.

`augwrap.data.Augmentations`의 인자로 base dataset에서 파생된 인스턴스와 Albumentations 인스턴스를 받습니다.

`augwrap.augmentations`에 Albumentations와 함께 사용할 수 있는 모듈을 추가하고 있습니다.

```python
import albumentations as A
import augwrap.data as D
import augwrap.augmentations as DA

augmentations = A.Compose([
        A.RandomRotate90(p=1),
        A.GridDistortion(p=0.8),
        A.GaussNoise(p=0.75),
        DA.CutMix(dataset, p=0.8),
])
dataset = D.Augmentations(dataset, augmentations)
```



### Transforms

`augwrap.data.Transforms`를 이용하여 Pytorch의 Transforms를 적용할 수 있습니다.

Augmentation과 동일하게 base dataset에서 파생된 인스턴스와 Transforms 인스턴스를 인자로 받습니다.

`augwrap.transforms`에 Transforms와 함께 사용할 수 있는 모듈을 추가하고 있습니다.

```python
from torchvision.transforms as T
import augwrap.data as D
import augwrap.transforms as DT

transforms = T.Compose([
    DT.ToTorchTensor(),
    DT.TorchNormalize(from_image=True)
])
dataset = D.Transforms(dataset, transforms)
```

