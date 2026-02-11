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
    
    # def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
    #     img: NDArray[np.uint8] = np.array(Image.open(self.images[index][0]))
    #     y: int = self.images[index][1]
    #     if self.transforms is not None:
    #         augmentations = self.transforms(image=img)
    #         img = augmentations["image"]
    #     return img, y

    def standard_transforms(self) -> Compose:
        return Compose([
            A.Resize(224, 224),  # Resizing the image to 224x224 as VGG16 model input size is 224x224
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    

    def save_as_ubyte(self, 
                      output_dir: Union[str, Path], 
                      img_size: Tuple[int, int] = (224, 224), 
                      prefix: str = "catdog") -> None:
        """
        Save dataset in MNIST-like ubyte format (IDX file format).
        
        Creates two files:
        - {prefix}-images-idx3-ubyte: Contains all images
        - {prefix}-labels-idx1-ubyte: Contains all labels
        
        Parameters
        ----------
        output_dir : Union[str, Path]
            Directory to save the ubyte files
        img_size : Tuple[int, int], default=(224, 224)
            Size to resize images to (height, width)
        prefix : str, default="catdog"
            Prefix for output filenames
        
        Format specification (MNIST IDX format):
        Images file:
            [magic number (4 bytes)] [num images (4 bytes)] [rows (4 bytes)] [cols (4 bytes)] [pixels...]
        Labels file:
            [magic number (4 bytes)] [num labels (4 bytes)] [labels...]
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images_file = output_path / f"{prefix}-images-idx3-ubyte"
        labels_file = output_path / f"{prefix}-labels-idx1-ubyte"
        
        print(f"Processing {len(self)} images...")
        
        # Prepare transforms for saving (no normalization, just resize)
        save_transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),
        ])
        
        # Collect all images and labels
        all_images: List[NDArray[np.uint8]] = []
        all_labels: List[int] = []
        
        for idx in range(len(self)):
            img_path, label = self.images[idx]
            img = np.array(Image.open(img_path))
            
            # Convert grayscale to RGB if needed
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            
            # Apply resize transform
            augmented = save_transform(image=img)
            img = augmented["image"]
            
            all_images.append(img)
            all_labels.append(label)
        
        # Convert to numpy arrays
        images_array = np.array(all_images, dtype=np.uint8)  # Shape: (N, H, W, C)
        labels_array = np.array(all_labels, dtype=np.uint8)
        
        num_images = len(images_array)
        rows, cols, channels = images_array.shape[1], images_array.shape[2], images_array.shape[3]
        
        # Write images file
        # Magic numbers: 2051 for 3D images (grayscale), 2052 for 4D images (RGB)
        magic_number_images = 2052 if channels == 3 else 2051
        
        with open(images_file, 'wb') as f:
            # Write header
            f.write(struct.pack('>I', magic_number_images))  # Magic number
            f.write(struct.pack('>I', num_images))           # Number of images
            f.write(struct.pack('>I', rows))                 # Number of rows
            f.write(struct.pack('>I', cols))                 # Number of columns
            if channels == 3:
                f.write(struct.pack('>I', channels))         # Number of channels (for RGB)
            
            # Write image data
            f.write(images_array.tobytes())
        
        # Write labels file
        magic_number_labels = 2049  # Magic number for label files
        
        with open(labels_file, 'wb') as f:
            # Write header
            f.write(struct.pack('>I', magic_number_labels))  # Magic number
            f.write(struct.pack('>I', num_images))           # Number of labels
            
            # Write label data
            f.write(labels_array.tobytes())
        
        print(f"✓ Saved {num_images} images to: {images_file}")
        print(f"✓ Saved {num_images} labels to: {labels_file}")
        print(f"  Image dimensions: {rows}x{cols}x{channels}")
        print(f"  Total images file size: {images_file.stat().st_size / (1024**2):.2f} MB")
        print(f"  Total labels file size: {labels_file.stat().st_size / 1024:.2f} KB")


class CatandDogDataLoader(Dataset):
    def __init__(self, raw_folder: str, train: bool = True, transforms: Optional[Callable] = None):
        self.raw_folder = raw_folder
        self.train = train
        self.data, self.targets = self.load_data()
        self.transforms = self.standard_transforms() if transforms is None else transforms


    def load_data(self):
        image_file = f"{'catdog_train' if self.train else 'catdog_test'}-images-idx3-ubyte"
        data = self._read_ubyte_images(os.path.join(self.raw_folder, image_file))

        label_file = f"{'catdog_train' if self.train else 'catdog_test'}-labels-idx1-ubyte"
        targets = self._read_ubyte_labels(os.path.join(self.raw_folder, label_file))

        return data, targets


    def _read_ubyte_images(self, filepath):
        """Read images from ubyte format file."""
        with open(filepath, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            num_images = struct.unpack('>I', f.read(4))[0]
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]
            
            if magic == 2052:  # RGB images
                channels = struct.unpack('>I', f.read(4))[0]
                images = np.frombuffer(f.read(), dtype=np.uint8)
                images = images.reshape(num_images, rows, cols, channels)
            else:  # Grayscale
                channels = 1
                images = np.frombuffer(f.read(), dtype=np.uint8)
                images = images.reshape(num_images, rows, cols)
        
        print(f"Loaded {num_images} images of shape {rows}x{cols}x{channels if magic == 2052 else ''}")
        return images


    def standard_transforms(self) -> Compose:
        return Compose([
            A.Resize(224, 224),  # Resizing the image to 224x224 as VGG16 model input size is 224x224
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


    def _read_ubyte_labels(self, filepath):
        """Read labels from ubyte format file."""
        with open(filepath, 'rb') as f:
            num_labels = struct.unpack('>I', f.read(4))[0]
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        print(f"Loaded {num_labels} labels")
        return labels


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.data[index], torch.int64(self.targets[index])

        if self.transforms is not None:
            transformed = self.transforms(image=img)
            img = transformed["image"]

        return img, target
    


if __name__ == "__main__":
    
    data_path: str = r'/Users/gimoon/Documents/GitHub/computer_vision/DataSources'

    train_data_path: str = os.path.join(data_path, "cat-and-dog/training_set/")
    test_data_path: str = os.path.join(data_path, "cat-and-dog/test_set/")

    print(os.path.exists(train_data_path))
    print(os.path.exists(test_data_path))
