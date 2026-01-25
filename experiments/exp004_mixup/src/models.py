"""
Model architectures for CMI Behavior Detection competition.
Scalable multi-branch architecture with configurable feature encoders.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BranchConfig:
    """Configuration for a single feature branch encoder.

    Attributes:
        name: Unique identifier for the branch
        columns: List of column names for this branch (e.g., ['acc_x', 'acc_y', 'acc_z'])
        hidden_channels: List of hidden channel sizes for each Conv1D layer
        kernel_size: Kernel size for Conv1D layers
        pool_size: Pooling size for AvgPool1d layers between Conv blocks
        channel_indices: Tuple of (start, end) indices (auto-computed from columns)
    """

    name: str
    columns: list[str]  # Column names for this branch
    hidden_channels: list[int] = field(default_factory=lambda: [16, 32, 64])
    kernel_size: int = 3
    pool_size: int = 2
    channel_indices: tuple[int, int] | None = None  # Set by resolve_branch_indices()

    @property
    def input_channels(self) -> int:
        """Returns the number of input channels (derived from columns)."""
        return len(self.columns)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 各時刻の特徴量から「重要度スコア」を出す層
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (Batch, Time_Steps, Features)

        # 1. 重要度スコアを計算
        # (Batch, Time, Features) -> (Batch, Time, 1)
        scores = self.attention_weights(x)

        # 2. Softmaxで確率（重み）に変換
        # dim=1 (時間方向) で合計1になるようにする
        weights = F.softmax(scores, dim=1)

        # 3. 加重和 (Weighted Sum)
        # 特徴量(x) に 重み(weights) を掛けて足し合わせる
        # (Batch, Time, Features) * (Batch, Time, 1) -> sum -> (Batch, Features)
        output = torch.sum(x * weights, dim=1)

        return output


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


class AttentionPooling1d(nn.Module):
    def __init__(self, input_dim, kernel_size):
        super().__init__()
        # 重要度を計算するための層
        self.attn_layer = nn.Linear(input_dim, 1)
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        x: (Batch, Features, Time)
        """
        batch_size, features, time_steps = x.size()

        # パディングが必要か確認
        pad_size = 0
        if time_steps % self.kernel_size != 0:
            pad_size = self.kernel_size - (time_steps % self.kernel_size)
            # F.padは (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right, ...)
            # Time次元（最後）の右側にpaddingを追加
            x = F.pad(x, (0, pad_size))
            time_steps += pad_size

        # 1. 時間軸を分割するためにReshape
        # (B, F, T) -> (B, F, T_new, K)
        # ※ Tは最後の次元なので、そのまま分割可能です
        x_reshaped = x.view(
            batch_size, features, time_steps // self.kernel_size, self.kernel_size
        )

        # 2. 次元入れ替え (Permute)
        # nn.Linearは「最後の次元」に対して作用するため、Features(F)を最後に持ってくる必要があります
        # (B, F, T_new, K) -> (B, T_new, K, F)
        x_permuted = x_reshaped.permute(0, 2, 3, 1)

        # 3. スコア計算
        # (B, T_new, K, F) -> (B, T_new, K, 1)
        scores = self.attn_layer(x_permuted)

        # 4. Kernel内(dim=2)でSoftmaxをとる -> 重み
        # これで「K個の近傍点の中で、どれを重視するか」が決まります
        weights = F.softmax(scores, dim=2)

        # 5. 加重和 (Weighted Sum)
        # Data:    (B, T_new, K, F)
        # Weights: (B, T_new, K, 1)
        # 掛け合わせて dim=2 (K) で合計 -> (B, T_new, F)
        weighted_sum = torch.sum(x_permuted * weights, dim=2)

        # 6. 元の形式 (B, F, T) に戻す
        # (B, T_new, F) -> (B, F, T_new)
        output = weighted_sum.permute(0, 2, 1)

        return output


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
                # layers.append(AttentionPooling1d(out_ch, config.kernel_size))
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

        self.global_pool = AttentionPooling(rnn_output_size)

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
            if cfg.channel_indices is None:
                raise RuntimeError(
                    f"channel_indices not set for branch '{cfg.name}'. "
                    "Call resolve_branch_indices() before creating the model."
                )
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
        feature = self.global_pool(output)  # (Batch, hidden*directions)

        # --- 4. Classification ---
        logits = self.head(feature)
        return logits


# ============================================================================
# Default branch configurations (column-name based)
# ============================================================================

DEFAULT_BRANCH_CONFIGS = [
    BranchConfig(
        name="imu",
        columns=["acc_x", "acc_y", "acc_z"],
        hidden_channels=[8, 16, 32],
    ),
    BranchConfig(
        name="euler",
        columns=["roll", "pitch", "yaw"],
        hidden_channels=[8, 16, 32],
    ),
]


def resolve_branch_indices(
    branch_configs: list[BranchConfig],
    sensor_cols: list[str],
) -> list[BranchConfig]:
    """Resolve channel_indices for each BranchConfig based on sensor column order.

    Args:
        branch_configs: List of BranchConfig with column names defined
        sensor_cols: Ordered list of all sensor columns used in the dataset

    Returns:
        Updated list of BranchConfig with channel_indices set

    Raises:
        ValueError: If a column in branch_configs is not found in sensor_cols
    """
    col_to_idx = {col: idx for idx, col in enumerate(sensor_cols)}

    for cfg in branch_configs:
        # Find start and end indices for this branch's columns
        indices = []
        for col in cfg.columns:
            if col not in col_to_idx:
                raise ValueError(
                    f"Column '{col}' in branch '{cfg.name}' not found in sensor_cols. "
                    f"Available: {sensor_cols}"
                )
            indices.append(col_to_idx[col])

        # Verify columns are contiguous
        if indices != list(range(min(indices), max(indices) + 1)):
            raise ValueError(
                f"Columns in branch '{cfg.name}' must be contiguous in sensor_cols. "
                f"Got indices: {indices}"
            )

        cfg.channel_indices = (min(indices), max(indices) + 1)

    return branch_configs


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
