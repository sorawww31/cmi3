"""
Dataset module for CMI Behavior Detection competition.
Handles loading and preprocessing of sensor data sequences.

Memory-optimized version:
- Pre-converts all sequences to NumPy arrays (discards DataFrame)
- Uses sequence index mapping for O(1) access
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocess import Preprocessor
from src.time_augmentation import (
    ButterFilter,
    Compose,
    GaussianNoise,
    OneOf,
    PinkNoiseSNR,
    SignalTransform,
    TimeShift,
    TimeStretch,
)

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


def pad_sequence(
    data: np.ndarray,
    max_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """シーケンスをmax_lengthにパディング（前方パディング）または切り詰め"""
    seq_len, n_features = data.shape

    if seq_len >= max_length:
        # 長い場合は末尾を使用（最新のデータを保持）
        return data[-max_length:]

    # 前方パディング: [PAD, PAD, ..., data]
    padded = np.full((max_length, n_features), pad_value, dtype=np.float32)
    padded[-seq_len:] = data  # データを末尾に配置
    return padded


def preconvert_sequences(
    df: pd.DataFrame,
    sequence_ids: list[int],
    sensor_cols: list[str],
    max_length: int,
) -> tuple[np.ndarray, dict[int, dict]]:
    """
    DataFrameから全シーケンスをNumPy配列に事前変換（メモリ効率化）

    Args:
        df: センサーデータを含むDataFrame
        sequence_ids: 使用するシーケンスIDのリスト
        sensor_cols: 使用するセンサーカラム
        max_length: パディング後のシーケンス長

    Returns:
        data_array: (n_sequences, max_length, n_features) のNumPy配列
        metadata: シーケンスIDをキーとするメタデータ辞書
    """
    n_sequences = len(sequence_ids)
    n_features = len(sensor_cols)

    # 事前にメモリを確保
    data_array = np.zeros((n_sequences, max_length, n_features), dtype=np.float32)
    metadata = {}

    # sequence_idでグループ化して効率的にアクセス
    grouped = df.groupby("sequence_id")

    for idx, seq_id in enumerate(sequence_ids):
        if seq_id in grouped.groups:
            seq_df = grouped.get_group(seq_id)
            seq_data = seq_df[sensor_cols].values.astype(np.float32)

            # 欠損値を0で埋める
            seq_data = np.nan_to_num(seq_data, nan=0.0)

            # パディング
            seq_data = pad_sequence(seq_data, max_length)
            data_array[idx] = seq_data

            # メタデータを保存
            first_row = seq_df.iloc[0]
            metadata[seq_id] = {
                "gesture": first_row.get("gesture", None),
                "subject": first_row.get("subject", None),
                "sequence_type": first_row.get("sequence_type", None),
                "idx": idx,  # 配列内のインデックス
            }

    return data_array, metadata


class CMIDataset(Dataset):
    """
    CMI Behavior Detection用のPyTorch Dataset（メモリ最適化版）

    DataFrameを保持せず、事前変換されたNumPy配列を使用
    """

    def __init__(
        self,
        data_array: np.ndarray,
        sequence_ids: list[int],
        metadata: dict[int, dict],
        is_train: bool = True,
        alpha: float = 0.3,
        transforms: SignalTransform | None = None,
    ):
        """
        Args:
            data_array: (n_sequences, max_length, n_features) のNumPy配列
            sequence_ids: 使用するシーケンスIDのリスト
            metadata: シーケンスIDをキーとするメタデータ辞書
            is_train: 訓練モードかどうか
            alpha: Mixup parameter for Beta distribution
            transforms: Data augmentation transforms
        """
        self.data_array = data_array
        self.sequence_ids = sequence_ids
        self.metadata = metadata
        self.is_train = is_train
        self.alpha = alpha
        self.transforms = transforms
        self.num_classes = len(TARGET_GESTURES) + len(NON_TARGET_GESTURES)

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def _get_data(self, idx: int):
        seq_id = self.sequence_ids[idx]
        seq_idx = self.metadata[seq_id]["idx"]
        data = self.data_array[seq_idx]

        label_idx = None
        if self.is_train:
            gesture = self.metadata[seq_id]["gesture"]
            label_idx = GESTURE_TO_IDX[gesture]

        return data, label_idx, seq_id

    def __getitem__(self, idx: int) -> dict:
        # Default behavior (no mixup or val)
        X, y_idx, seq_id = self._get_data(idx)

        # Apply augmentation if available (only on is_train usually, but handled by is_train flag in creation or passed transforms)
        if self.is_train and self.transforms:
            X = self.transforms(X)

        # Determine if we should apply mixup
        # ユーザー参照コード: p <= 0.0 -> Original, Else -> Mixup
        # np.random.rand() returns [0, 1), so p <= 0.0 only if p=0. basically always Mixup.
        # We will follow this logic for 'train' mode.

        do_mixup = False
        if self.is_train and self.alpha > 0:
            p = np.random.rand()
            if (
                p > 0.0
            ):  # Reference logic inverted: if p <= 0.0 -> straight. Else -> mix.
                do_mixup = True

        if do_mixup:
            # Mixup
            lam = np.random.beta(self.alpha, self.alpha)
            j = np.random.randint(0, len(self.sequence_ids))

            X2, y2_idx, _ = self._get_data(j)

            if self.is_train and self.transforms:
                X2 = self.transforms(X2)

            # Mix Data
            # X, X2 are pre-padded numpy arrays of same shape (max_len, features)
            X_mixed = lam * X + (1 - lam) * X2

            # Mix Labels (One-hot)
            y_onehot = np.zeros(self.num_classes, dtype=np.float32)
            y_onehot[y_idx] = 1.0

            y2_onehot = np.zeros(self.num_classes, dtype=np.float32)
            y2_onehot[y2_idx] = 1.0

            y_mixed = lam * y_onehot + (1 - lam) * y2_onehot

            result = {
                "sequence_id": seq_id,
                "data": torch.from_numpy(X_mixed.astype(np.float32)),
                "label": torch.from_numpy(y_mixed.astype(np.float32)),  # Soft label
                "label_origin": torch.tensor(
                    y_idx, dtype=torch.long
                ),  # Hard label for metrics
            }

            # is_target (Optional: can mix or just take original? usually just keep original for tracking)
            is_target = IDX_TO_GESTURE[y_idx] in TARGET_GESTURES
            result["is_target"] = torch.tensor(is_target, dtype=torch.float32)

        else:
            # Standard return
            result = {
                "sequence_id": seq_id,
                "data": torch.from_numpy(X),
            }

            if self.is_train:
                result["label"] = torch.tensor(y_idx, dtype=torch.long)
                result["label_origin"] = torch.tensor(y_idx, dtype=torch.long)

                is_target = IDX_TO_GESTURE[y_idx] in TARGET_GESTURES
                result["is_target"] = torch.tensor(is_target, dtype=torch.float32)

        return result


def create_dataloaders(
    df: pd.DataFrame,
    train_ids: list[int],
    val_ids: list[int],
    batch_size: int = 32,
    max_length: int = 500,
    sensor_cols: list[str] = [],
    num_workers: int = 4,
) -> tuple:
    """
    訓練・検証用のDataLoaderを作成（メモリ最適化版）

    改善点:
    - 全シーケンスをNumPy配列に事前変換（O(1)アクセス）
    - DataFrameへの参照を解放

    注意: fit_transformはtrainのみ、transformはvalに適用（データリーク防止）
    """
    from torch.utils.data import DataLoader

    if not sensor_cols:
        raise ValueError("sensor_cols must be provided")

    # 1. train/valデータを分離
    train_df = df[df["sequence_id"].isin(train_ids)]
    val_df = df[df["sequence_id"].isin(val_ids)]

    # 2. 前処理: trainでfit、valはtransformのみ（データリーク防止）
    preprocessor = Preprocessor()
    train_df_processed = preprocessor.fit_transform(train_df)
    val_df_processed = preprocessor.transform(val_df)

    # 3. 前処理済みデータをNumPy配列に変換
    train_data, train_metadata = preconvert_sequences(
        train_df_processed, train_ids, sensor_cols, max_length
    )
    val_data, val_metadata = preconvert_sequences(
        val_df_processed, val_ids, sensor_cols, max_length
    )

    # 4. DataFrameへの参照を解放（メモリ節約）
    del train_df, val_df, train_df_processed, val_df_processed

    # 5. Dataset作成
    # Augmentation definition
    train_transforms = Compose(
        [
            OneOf(
                [
                    GaussianNoise(p=0, max_noise_amplitude=0.05),
                    PinkNoiseSNR(p=0.0, min_snr=4.0, max_snr=20.0),
                    ButterFilter(p=0.0),
                ]
            ),
            TimeShift(p=0.0, padding_mode="zero", max_shift_pct=0.25),
            TimeStretch(p=0.0, max_rate=1.5, min_rate=0.5),
        ]
    )

    train_dataset = CMIDataset(
        train_data,
        train_ids,
        train_metadata,
        is_train=True,
        alpha=0.4,
        transforms=train_transforms,
    )
    val_dataset = CMIDataset(val_data, val_ids, val_metadata, is_train=True, alpha=0.0)

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
