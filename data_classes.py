from typing import Optional
import torch
from dataclasses import dataclass
from torch import nn, optim




@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 20
    momentum: float = 0.9  # for Adam optimizer
    weight_decay: float = 1e-4  # for Adam optimizer
    lr_scheduler_epoch: int = 5 # Epoch interval for decaying learning rate
    num_train_samples: int = 0  # Will be set when dataset is loaded
    
    def steps_per_epoch(self) -> int:
        if self.num_train_samples == 0:
            raise ValueError("num_train_samples must be set before calculating steps_per_epoch")
        return self.num_train_samples // self.batch_size
    
    def total_steps(self) -> int:
        return self.steps_per_epoch() * self.num_epochs

    def lr_scheduler_step_size(self) -> int:
        return self.steps_per_epoch() * self.lr_scheduler_epoch




# 1. Define the Dataclass to hold checkpoint data
@dataclass
class TrainingCheckpoint:
    epoch: int
    model_state: dict
    optimizer_state: dict
    loss: float




# # 2. Setup Dummy Model & Optimizer
# model = nn.Linear(10, 1)
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# # 3. Save Function
# def save_checkpoint(path, checkpoint_dataclass):
#     # Convert dataclass to dict if necessary, or just save the object
#     torch.save(checkpoint_dataclass, path)
#     print(f"Checkpoint saved at {path}")

# # 4. Load Function
# def load_checkpoint(path, device='cpu'):
#     checkpoint = torch.load(path, map_location=device)
#     return checkpoint

# # --- Usage ---
# # Save
# checkpoint = TrainingCheckpoint(
#     epoch=10, 
#     model_state=model.state_dict(),
#     optimizer_state=optimizer.state_dict(),
#     loss=0.5
# )
# save_checkpoint("model_cp.pth", checkpoint)

# # Load
# loaded_cp = load_checkpoint("model_cp.pth")
# model.load_state_dict(loaded_cp.model_state)
# optimizer.load_state_dict(loaded_cp.optimizer_state)
# print(f"Resuming from epoch: {loaded_cp.epoch}")
