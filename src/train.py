import os
import sys
from datetime import datetime
from typing import Optional
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


sys.path.insert(0, str(Path(os.getcwd()).resolve().parent))

from models.vgg16 import VGG16
from .data_classes import TrainingConfig


class Trainer:
    def __init__(
        self,
        model: VGG16,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
        scheduler: Optional[StepLR] = None,
    ) -> None:

        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate
            )
        if self.scheduler is None:
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)

    def train(self) -> None:
        self.model.train()
        for epoch in range(self.config.num_epochs):
            ProgressBar = tqdm(self.train_loader, total=len(self.train_loader))

            for inputs, labels in ProgressBar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                avg_val_acc = 0
                if self.val_loader is not None:
                    # validate
                    self.model.eval()
                    num_correct = 0
                    num_samples = 0

                    with torch.no_grad():
                        for inputs_val, labels_val in self.val_loader:
                            inputs_val, labels_val = inputs_val.to(
                                self.device
                            ), labels_val.to(self.device)
                            outputs_val = self.model(inputs_val)
                            _, predictions = outputs_val.max(1)
                            num_correct += (predictions == labels_val).sum()
                            num_samples += predictions.size(0)

                        avg_val_acc = (
                            num_correct / num_samples if num_samples > 0 else 0
                        )

                # Update Progress bar
                ProgressBar.set_description(f"Epoch [{epoch + 1}]")
                ProgressBar.set_postfix(TrainLoss=loss.item(), ValAcc=avg_val_acc)
                self.save_checkpoint(save_path=self.config.save_path, epoch=epoch + 1)

    def save_checkpoint(self, save_path: str, epoch: int):
        checkpoint = {
            "batch_size": self.config.batch_size,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "saved_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        save_pth = os.path.join(
            save_path, f"checkpoint_{epoch}_{checkpoint['saved_datetime']}.pth.tar"
        )
        torch.save(checkpoint, save_pth)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch: int = checkpoint["epoch"]
        loss: float = checkpoint["loss"]

        return self.model, self.optimizer, epoch, loss
