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
    max_length: int = 120
    batch_size: int = 64
    num_workers: int = 4
    sensor_type: str = "imu"  # imu, all

    # Model Architecture
    model_name: str = "cmi"

    # RNN settings
    rnn_type: str = "gru"  # gru, lstm
    rnn_hidden_size: int = 96
    rnn_num_layers: int = 2
    rnn_dropout: float = 0.2
    rnn_bidirectional: bool = True

    # MLP Head settings
    mlp_hidden_channels: list[int] = field(default_factory=lambda: [96])
    mlp_dropout: float = 0.3

    # Branch encoder settings (channel multiplier for Conv1D layers)
    branch_hidden_multiplier: int = 2  # hidden_channels = [in*2, in*4, in*8]

    # Training
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    patience: int = 10

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
