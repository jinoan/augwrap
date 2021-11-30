# AugWrap
[![PyPI version](https://badge.fury.io/py/augwrap.svg)](https://badge.fury.io/py/augwrap)



## Installation

```shell
$ pip install augwrap
```



## Basic Usage

1. Base dataset을 선택합니다. Pytorch 기반 데이터셋을 만들려면 `TorchBaseDataset`을 사용합니다. Tensorflow 기반 데이터셋을 만들려면 `TFBaseDataset`을 사용합니다.

   ```python
   import augwrap as aw
   
   dataset = aw.data.TorchBaseDataset(images, labels, classes)
   ```

2. `augwrap.data` 안의 모듈을 이용하여 학습 전 데이터를 처리합니다.

   ```python
   dataset = aw.data.LoadImages(dataset)
   dataset = aw.data.ResizeImages(dataset, (256, 256))
   dataset = aw.data.OneHotLabels(dataset)
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



## Augmentations

잘 알려진 어그멘테이션 도구인 [Albumentations](https://github.com/albumentations-team/albumentations)을 활용할 수 있도록 만들었습니다.

`augwrap.data.Augmentations`의 생성자 인자로 base dataset에서 파생된 객체와 Albumentations 객체를 받습니다.

Albumentations와 함께 사용할 수 있는 어그멘테이션 모듈을 `augwrap.augmentations`에 추가했습니다.

```python
import albumentations as A
import augwrap as aw

augmentations = A.Compose([
        A.RandomRotate90(p=1),
        A.GridDistortion(p=0.8),
        A.GaussNoise(p=0.75),
        aw.augmentations.CutMix(dataset, p=0.8),
])
dataset = aw.data.Augmentations(dataset, augmentations)
```


