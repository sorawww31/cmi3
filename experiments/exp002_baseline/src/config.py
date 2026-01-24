import os
from dataclasses import dataclass, field
from typing import Optional

from utils.env import EnvConfig  # noqa: E402


@dataclass
class ExpConfig:
    debug: bool = False
    seed: int = 42
    folds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_folds: int = 5

    # Wandb
    wandb_project_name: Optional[str] = os.getenv("COMPETITION", "cmi3")

    # Data
    max_length: int = 144
    batch_size: int = 32
    num_workers: int = 4
    sensor_type: str = "imu"  # imu, all

    # Model
    model_name: str = "lstm"  # lstm, cnn1d, transformer
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3

    # Training
    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    patience: int = 5

    # Optimizer: adam, adamw, radam_schedule_free
    optimizer_name: str = "radam_schedule_free"
    warmup_steps: int = 100

    # EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.99


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)
