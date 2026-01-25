"""
Model architectures for CMI Behavior Detection competition.
Scalable multi-branch architecture with configurable feature encoders.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class BranchConfig:
    """Configuration for a single feature branch encoder.

    Attributes:
        name: Unique identifier for the branch
        input_channels: Number of input channels (features in this group)
        channel_indices: Tuple of (start, end) indices to slice input features
        hidden_channels: List of hidden channel sizes for each Conv1D layer
        kernel_size: Kernel size for Conv1D layers
        pool_size: Pooling size for AvgPool1d layers between Conv blocks
    """

    name: str
    input_channels: int
    channel_indices: tuple[int, int]  # (start_idx, end_idx) for slicing
    hidden_channels: list[int] = field(default_factory=lambda: [16, 32, 64])
    kernel_size: int = 3
    pool_size: int = 2


class Conv1DBlock(nn.Module):
    """1D Convolutional Block with Conv1D, BatchNorm, ReLU"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            stride=stride,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FeatureBranch(nn.Module):
    """Dynamic feature branch encoder using Conv1D blocks.

    Creates a sequence of Conv1DBlocks with pooling layers based on config.
    """

    def __init__(self, config: BranchConfig):
        super().__init__()
        self.config = config

        layers = []
        in_ch = config.input_channels

        for i, out_ch in enumerate(config.hidden_channels):
            layers.append(Conv1DBlock(in_ch, out_ch, kernel_size=config.kernel_size))
            # Add pooling after each layer except the last
            if i < len(config.hidden_channels) - 1:
                layers.append(nn.AvgPool1d(config.pool_size))
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)

    @property
    def output_channels(self) -> int:
        """Returns the output channel size of this branch."""
        return self.config.hidden_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, Channels, Time)
        Returns:
            Encoded tensor of shape (Batch, output_channels, Time')
        """
        return self.encoder(x)


class MLPHead(nn.Module):
    """MLP classification head with configurable layers."""

    def __init__(
        self,
        in_channel: int,
        hidden_channels: list[int],
        out_channel: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_ch = in_channel

        for hidden_ch in hidden_channels:
            layers.extend(
                [
                    nn.Linear(prev_ch, hidden_ch),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
            prev_ch = hidden_ch

        layers.append(nn.Linear(prev_ch, out_channel))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CMIModel(nn.Module):
    """Scalable Multi-Branch Model for CMI competition.

    Architecture:
        1. Feature branches: Separate Conv1D encoders for each feature group
        2. RNN layer: GRU/LSTM/Transformer for temporal modeling
        3. Global Average Pooling: Aggregate temporal features
        4. MLP Head: Classification head

    Args:
        branch_configs: List of BranchConfig for each feature group
        num_classes: Number of output classes
        rnn_type: Type of RNN layer ("gru", "lstm", "transformer")
        rnn_hidden_size: Hidden size for RNN layer
        rnn_num_layers: Number of RNN layers
        rnn_dropout: Dropout for RNN layer
        rnn_bidirectional: Whether to use bidirectional RNN
        mlp_hidden_channels: Hidden channel sizes for MLP head
        mlp_dropout: Dropout for MLP head
    """

    def __init__(
        self,
        branch_configs: list[BranchConfig],
        num_classes: int = 18,
        rnn_type: Literal["gru", "lstm"] = "gru",
        rnn_hidden_size: int = 96,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.2,
        rnn_bidirectional: bool = True,
        mlp_hidden_channels: list[int] | None = None,
        mlp_dropout: float = 0.3,
    ):
        super().__init__()

        self.branch_configs = branch_configs
        self.rnn_bidirectional = rnn_bidirectional

        # --- 1. Feature Branch Encoders ---
        self.branches = nn.ModuleDict(
            {cfg.name: FeatureBranch(cfg) for cfg in branch_configs}
        )

        # Calculate total output channels from all branches
        total_encoder_channels = sum(
            branch.output_channels for branch in self.branches.values()
        )

        # --- 2. RNN Layer ---
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=total_encoder_channels,
            hidden_size=rnn_hidden_size,
            batch_first=True,
            bidirectional=rnn_bidirectional,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
        )

        # --- 3. Global Average Pooling (applied in forward) ---

        # --- 4. MLP Head ---
        # RNN output size: hidden_size * 2 if bidirectional
        rnn_output_size = rnn_hidden_size * (2 if rnn_bidirectional else 1)
        if mlp_hidden_channels is None:
            mlp_hidden_channels = [rnn_output_size // 2]

        self.head = MLPHead(
            in_channel=rnn_output_size,
            hidden_channels=mlp_hidden_channels,
            out_channel=num_classes,
            dropout=mlp_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, Time, Channel)
        Returns:
            Logits of shape (Batch, num_classes)
        """
        # x: (Batch, Time, Channel) -> (Batch, Channel, Time)
        x = x.transpose(1, 2)

        # --- 1. Feature Branch Encoding ---
        encoded_features = []
        for cfg in self.branch_configs:
            branch = self.branches[cfg.name]
            start, end = cfg.channel_indices
            branch_input = x[:, start:end, :]  # (Batch, branch_channels, Time)
            encoded = branch(branch_input)  # (Batch, output_channels, Time')
            encoded_features.append(encoded)

        # Concatenate all branch outputs along channel dimension
        x = torch.cat(encoded_features, dim=1)  # (Batch, total_channels, Time')

        # --- 2. RNN Layer ---
        # RNN expects (Batch, Time, Features)
        x = x.transpose(1, 2)  # (Batch, Time', total_channels)
        self.rnn.flatten_parameters()  # Suppress contiguous memory warning
        output, _ = self.rnn(x)  # (Batch, Time', hidden*directions)

        # --- 3. Global Average Pooling ---
        # Average over time dimension
        feature = output.mean(dim=1)  # (Batch, hidden*directions)

        # --- 4. Classification ---
        logits = self.head(feature)
        return logits


# ============================================================================
# Default branch configurations
# ============================================================================

DEFAULT_BRANCH_CONFIGS = [
    BranchConfig(
        name="imu",
        input_channels=3,
        channel_indices=(0, 3),
        hidden_channels=[8, 16, 32],
        kernel_size=3,
        pool_size=2,
    ),
    BranchConfig(
        name="euler",
        input_channels=3,
        channel_indices=(3, 6),
        hidden_channels=[8, 16, 32],
        kernel_size=3,
        pool_size=2,
    ),
]


def get_model(
    model_name: str = "cmi",
    num_classes: int = 18,
    branch_configs: list[BranchConfig] | None = None,
    rnn_type: str = "gru",
    rnn_hidden_size: int = 96,
    rnn_num_layers: int = 2,
    rnn_dropout: float = 0.2,
    rnn_bidirectional: bool = True,
    mlp_hidden_channels: list[int] | None = None,
    mlp_dropout: float = 0.3,
    **kwargs,
) -> nn.Module:
    """Factory function to create model instance.

    Args:
        model_name: Model identifier (currently only "cmi" supported)
        num_classes: Number of output classes
        branch_configs: List of BranchConfig for feature branches.
                       If None, uses DEFAULT_BRANCH_CONFIGS.
        rnn_type: Type of RNN layer ("gru" or "lstm")
        rnn_hidden_size: Hidden size for RNN
        rnn_num_layers: Number of RNN layers
        rnn_dropout: Dropout for RNN
        rnn_bidirectional: Whether to use bidirectional RNN
        mlp_hidden_channels: Hidden channels for MLP head
        mlp_dropout: Dropout for MLP head

    Returns:
        Configured CMIModel instance
    """
    if branch_configs is None:
        branch_configs = DEFAULT_BRANCH_CONFIGS

    return CMIModel(
        branch_configs=branch_configs,
        num_classes=num_classes,
        rnn_type=rnn_type,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
        rnn_dropout=rnn_dropout,
        rnn_bidirectional=rnn_bidirectional,
        mlp_hidden_channels=mlp_hidden_channels,
        mlp_dropout=mlp_dropout,
    )
