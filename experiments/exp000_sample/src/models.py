"""
Model architectures for CMI Behavior Detection competition.
Includes LSTM and 1D CNN based classifiers.
"""

import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    """1D Convolutional Block with Conv1D, BatchNorm, ReLU, and MaxPool1D"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int
    ):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, drop_out=0.5):
        # 1. 親クラスの初期化（必須！）
        super().__init__()

        self.linear1 = nn.Linear(in_features=in_channel, out_features=mid_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_out)
        self.linear2 = nn.Linear(in_features=mid_channel, out_features=out_channel)

    def forward(self, x):
        x = self.linear1(x)  # B, in_channel -> B, mid_channel
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear2(x)  # B, mid_channel -> B, out_channel


class CMIModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()

        # --- 1. Encoder (Conv1dだけで軽く作る) ---
        self.encoder = nn.Sequential(
            # 入力: (Batch, input_channels, time)
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 長さを半分に
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # さらに半分に
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # --- 2. Global Pooling ---
        # 時系列の長さがいくつであっても、最後は必ず長さ1にする便利な層
        self.pool = nn.AdaptiveAvgPool1d(1)

        # --- 3. Head ---
        # Encoderの出力チャネル数が64なので、入力も64
        self.head = MLPHead(
            in_channel=64, mid_channel=32, out_channel=num_classes, drop_out=0.2
        )

    def forward(self, x):
        # x: (Batch, Time, Channel) from DataLoader
        # Conv1d expects (Batch, Channel, Time), so we transpose
        x = x.transpose(1, 2)  # -> (Batch, Channel, Time)

        # 特徴抽出
        x = self.encoder(x)  # -> (Batch, 64, New_Time)

        # Pooling
        x = self.pool(x)  # -> (Batch, 64, 1)

        # 余計な次元を削除 (Batch, 64) にする
        x = x.squeeze(-1)

        # 分類
        y = self.head(x)
        return y


def get_model(
    model_name: str,
    input_size: int,
    num_classes: int = 18,
    **kwargs,
) -> nn.Module:
    """モデル名からモデルインスタンスを取得"""

    return CMIModel(input_channels=input_size, num_classes=num_classes)
