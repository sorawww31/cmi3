"""
Utility functions for CMI Behavior Detection competition.
"""

import random

import numpy as np
import torch
from sklearn.metrics import f1_score

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

ALL_GESTURES = TARGET_GESTURES + [
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

IDX_TO_GESTURE = {idx: gesture for idx, gesture in enumerate(ALL_GESTURES)}


def set_seed(seed: int) -> None:
    """再現性のためのシード設定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_hierarchical_f1(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> float:
    """
    Hierarchical macro F1 metric for the CMI 2025 challenge.

    This metric averages:
    1. Binary F1 on whether the gesture is one of the target or non-target types.
    2. Macro F1 on gesture, where all non-target sequences are collapsed into a single non_target class.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Convert indices to gestures
    y_true_gestures = [IDX_TO_GESTURE[i] for i in y_true]
    y_pred_gestures = [IDX_TO_GESTURE[i] for i in y_pred]

    # Binary F1 (Target vs Non-Target)
    y_true_bin = [g in TARGET_GESTURES for g in y_true_gestures]
    y_pred_bin = [g in TARGET_GESTURES for g in y_pred_gestures]
    f1_binary = f1_score(y_true_bin, y_pred_bin, pos_label=True, zero_division=0)

    # Multi-class F1 (target gestures + collapsed non-target)
    y_true_mc = [g if g in TARGET_GESTURES else "non_target" for g in y_true_gestures]
    y_pred_mc = [g if g in TARGET_GESTURES else "non_target" for g in y_pred_gestures]
    f1_macro = f1_score(y_true_mc, y_pred_mc, average="macro", zero_division=0)

    return 0.5 * f1_binary + 0.5 * f1_macro


def count_parameters(model: torch.nn.Module) -> int:
    """モデルの学習可能なパラメータ数をカウント"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_predictions(
    predictions: dict[int, str],
    output_path: str,
) -> None:
    """予測結果をCSVとして保存"""
    import pandas as pd

    df = pd.DataFrame(
        [
            {"sequence_id": seq_id, "gesture": gesture}
            for seq_id, gesture in predictions.items()
        ]
    )
    df.to_csv(output_path, index=False)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
