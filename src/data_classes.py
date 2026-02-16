from dataclasses import dataclass


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 20
    momentum: float = 0.9  # for Adam optimizer
    weight_decay: float = 1e-4  # for Adam optimizer
    lr_scheduler_epoch: int = 5  # Epoch interval for decaying learning rate
    num_train_samples: int = 0  # Will be set when dataset is loaded
    save_path: str = "./checkpoints"  # Default path for saving checkpoints

    def steps_per_epoch(self) -> int:
        if self.num_train_samples == 0:
            raise ValueError(
                "num_train_samples must be set before calculating steps_per_epoch"
            )
        return self.num_train_samples // self.batch_size

    def total_steps(self) -> int:
        return self.steps_per_epoch() * self.num_epochs

    def lr_scheduler_step_size(self) -> int:
        return self.steps_per_epoch() * self.lr_scheduler_epoch
