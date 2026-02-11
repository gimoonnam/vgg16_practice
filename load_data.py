import sys 
import os 
import struct
from typing import Tuple, Optional, Union, List, Callable, Any
from pathlib import Path 

import torch 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from numpy.typing import NDArray

import albumentations as A 
from albumentations import Compose
from albumentations.pytorch import ToTensorV2   # Coverting image to Tensor


class CatDogDataLoadandSave:
    """
    This a class for selecting the data for the model
    Attributes
    ----------
    images  : List[Tuple[str, int]]
        List of tuples containing (image_path, label) where label is 0 for cat, 1 for dog
    transforms : albumentations.core.composition.Compose
        transforms that must be applied on the image
    """

    def __init__(self, data_dir: Union[str, Path], transforms: Optional[Callable] = None) -> None:
        cat_path: str = str(data_dir) + '/cats'
        dog_path: str = str(data_dir) + '/dogs'
        cats: List[str] = os.listdir(cat_path)
        dogs: List[str] = os.listdir(dog_path)

        self.images = [(cat_path + '/' + cats[i], 0) for i in range(len(cats)) if cats[i].endswith('.jpg')]
        dogs_list: List[Tuple[str, int]] = [(dog_path + '/' + dogs[i], 1) for i in range(len(dogs)) if dogs[i].endswith('.jpg')]

        self.images.extend(dogs_list)

        self.transforms = self.standard_transforms() if transforms is None else transforms

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img: NDArray[np.uint8] = np.array(Image.open(self.images[index][0]))
        y: int = self.images[index][1]
        if self.transforms is not None:
            augmentations = self.transforms(image=img)
            img = augmentations["image"]
        return img, y

    def standard_transforms(self) -> Compose:
        return Compose([
            A.Resize(224, 224),  # Resizing the image to 224x224 as VGG16 model input size is 224x224
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    

    
    


if __name__ == "__main__":
    
    data_path: str = r'/Users/gimoon/Documents/GitHub/computer_vision/DataSources'

    train_data_path: str = os.path.join(data_path, "cat-and-dog/training_set/")
    test_data_path: str = os.path.join(data_path, "cat-and-dog/test_set/")

    print(os.path.exists(train_data_path))
    print(os.path.exists(test_data_path))
