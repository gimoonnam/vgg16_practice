import sys
import os
import time
from typing import Tuple, Optional, Union, List, Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.transforms import Compose
from PIL import Image
from tqdm import tqdm


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

    def __init__(
        self, data_dir: Union[str, Path], transforms: Optional[Callable] = None
    ) -> None:
        cat_path: str = str(data_dir) + "/cats"
        dog_path: str = str(data_dir) + "/dogs"
        cats: List[str] = os.listdir(cat_path)
        dogs: List[str] = os.listdir(dog_path)

        self.images = [
            (cat_path + "/" + cats[i], 0)
            for i in range(len(cats))
            if cats[i].endswith(".jpg")
        ]
        dogs_list: List[Tuple[str, int]] = [
            (dog_path + "/" + dogs[i], 1)
            for i in range(len(dogs))
            if dogs[i].endswith(".jpg")
        ]

        self.images.extend(dogs_list)

        self.transforms = (
            self.standard_transforms() if transforms is None else transforms
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img: Image.Image = Image.open(self.images[index][0]).convert("RGB")
        y: int = self.images[index][1]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, y

    def standard_transforms(self) -> Compose:
        return Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),  # Resizing the image to 224x224 as VGG16 model input size is 224x224
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def split_dataset(dataset, train_ratio=0.8, random_seed=42):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")

    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed),
    )


@torch.no_grad()  # Disable gradient tracking for efficiency
def calc_accuracy_loader(data_loader, model, device, num_batches=None, desc="testing"):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(tqdm(data_loader, desc)):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)
            predicted_labels = torch.argmax(logits, dim=1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def compute_epoch_loss(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    num_examples = 0
    for features, labels in data_loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        epoch_loss += loss.item()
        num_examples += features.size(0)
    return epoch_loss / num_examples


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


_, term_width = os.popen("stty size", "r").read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()
