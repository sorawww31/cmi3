"""
Dataset module for CMI Behavior Detection competition.
Handles loading and preprocessing of sensor data sequences.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocess import Preprocessor

# センサーカラム定義（前処理後: オイラー角を使用）
IMU_COLS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "roll",
    "pitch",
    "yaw",
]
THM_COLS = [f"thm_{i}" for i in range(1, 6)]
TOF_COLS = [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]

ALL_SENSOR_COLS = IMU_COLS + THM_COLS + TOF_COLS


# ジェスチャーラベル定義
TARGET_GESTURES = [
    "Above ear - pull hair",
    "Cheek - pinch skin",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Neck - pinch skin",
    "Neck - scratch",
]

NON_TARGET_GESTURES = [
    "Write name on leg",
    "Wave hello",
    "Glasses on/off",
    "Text on phone",
    "Write name in air",
    "Feel around in tray and pull out an object",
    "Scratch knee/leg skin",
    "Pull air toward your face",
    "Drink from bottle/cup",
    "Pinch knee/leg skin",
]

ALL_GESTURES = TARGET_GESTURES + NON_TARGET_GESTURES
GESTURE_TO_IDX = {gesture: idx for idx, gesture in enumerate(ALL_GESTURES)}
IDX_TO_GESTURE = {idx: gesture for gesture, idx in GESTURE_TO_IDX.items()}


def load_train_data(input_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """訓練データとデモグラフィックデータを読み込む"""
    input_dir = Path(input_dir)
    train_df = pd.read_csv(input_dir / "train.csv")
    demo_df = pd.read_csv(input_dir / "train_demographics.csv")
    return train_df, demo_df


def get_sequence_data(
    df: pd.DataFrame,
    sequence_id: int,
    sensor_cols: list[str] = ALL_SENSOR_COLS,
) -> np.ndarray:
    """指定されたシーケンスIDのセンサーデータを取得"""
    seq_df = df[df["sequence_id"] == sequence_id]
    return seq_df[sensor_cols].values.astype(np.float32)


def pad_sequence(
    data: np.ndarray,
    max_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """シーケンスをmax_lengthにパディング（または切り詰め）"""
    seq_len, n_features = data.shape

    if seq_len >= max_length:
        return data[:max_length]

    padded = np.full((max_length, n_features), pad_value, dtype=np.float32)
    padded[:seq_len] = data
    return padded


class CMIDataset(Dataset):
    """CMI Behavior Detection用のPyTorch Dataset"""

    def __init__(
        self,
        df: pd.DataFrame,
        sequence_ids: list[int],
        max_length: int = 500,
        sensor_cols: Optional[list[str]] = None,
        is_train: bool = True,
    ):
        """
        Args:
            df: センサーデータを含むDataFrame
            sequence_ids: 使用するシーケンスIDのリスト
            max_length: パディング後のシーケンス長
            sensor_cols: 使用するセンサーカラム（Noneの場合はIMUのみ）
            is_train: 訓練モードかどうか
        """
        self.df = df
        self.sequence_ids = sequence_ids
        self.max_length = max_length
        self.sensor_cols = sensor_cols if sensor_cols else IMU_COLS
        self.is_train = is_train

        # シーケンスごとのメタデータを事前に取得
        self.metadata = self._prepare_metadata()

    def _prepare_metadata(self) -> dict:
        """各シーケンスのメタデータを準備"""
        metadata = {}
        for seq_id in self.sequence_ids:
            seq_df = self.df[self.df["sequence_id"] == seq_id]
            if len(seq_df) > 0:
                first_row = seq_df.iloc[0]
                metadata[seq_id] = {
                    "gesture": first_row.get("gesture", None),
                    "subject": first_row.get("subject", None),
                    "sequence_type": first_row.get("sequence_type", None),
                }
        return metadata

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __getitem__(self, idx: int) -> dict:
        seq_id = self.sequence_ids[idx]

        # センサーデータ取得
        data = get_sequence_data(self.df, seq_id, self.sensor_cols)

        # 欠損値を0で埋める
        data = np.nan_to_num(data, nan=0.0)

        # パディング
        data = pad_sequence(data, self.max_length)

        # (seq_len, n_features) -> (n_features, seq_len) for Conv1D
        # または (seq_len, n_features) のままLSTMへ

        result = {
            "sequence_id": seq_id,
            "data": torch.from_numpy(data),  # (max_length, n_features)
        }

        if self.is_train:
            gesture = self.metadata[seq_id]["gesture"]
            label = GESTURE_TO_IDX[gesture]
            result["label"] = torch.tensor(label, dtype=torch.long)

            # Binary label (target vs non-target)
            is_target = gesture in TARGET_GESTURES
            result["is_target"] = torch.tensor(is_target, dtype=torch.float32)

        return result


def create_dataloaders(
    df: pd.DataFrame,
    train_ids: list[int],
    val_ids: list[int],
    batch_size: int = 32,
    max_length: int = 500,
    sensor_cols: Optional[list[str]] = None,
    num_workers: int = 4,
) -> tuple:
    """訓練・検証用のDataLoaderを作成"""
    from torch.utils.data import DataLoader

    preprocessor = Preprocessor()
    train_df = df[df["sequence_id"].isin(train_ids)].copy()
    val_df = df[df["sequence_id"].isin(val_ids)].copy()
    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.transform(val_df)

    train_dataset = CMIDataset(
        train_df, train_ids, max_length, sensor_cols, is_train=True
    )
    val_dataset = CMIDataset(val_df, val_ids, max_length, sensor_cols, is_train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
