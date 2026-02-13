
import os 
from typing import Tuple, Optional, Union, List, Callable, Any
from pathlib import Path 

import torch 
from PIL import Image
import numpy as np
from numpy.typing import NDArray

from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.transforms import Compose
from PIL import Image


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
        img: Image.Image = Image.open(self.images[index][0]).convert("RGB")
        y: int = self.images[index][1]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, y


    def standard_transforms(self) -> Compose:
        return Compose([
            transforms.Resize((224, 224)),  # Resizing the image to 224x224 as VGG16 model input size is 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    
def split_dataset(dataset, train_ratio=0.8, random_seed=42):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")

    return random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed) 
    )


if __name__ == "__main__":
    
    data_path: str = r'/Users/gimoon/Documents/GitHub/computer_vision/DataSources'

    train_data_path: str = os.path.join(data_path, "cat-and-dog/training_set/")
    test_data_path: str = os.path.join(data_path, "cat-and-dog/test_set/")

    print(os.path.exists(train_data_path))
    print(os.path.exists(test_data_path))
