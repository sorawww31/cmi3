"""
Inference script for CMI Behavior Detection competition.
Uses kaggle_evaluation.cmi_inference_server for Code Competition submission.

Usage:
    - Kaggle notebook上で実行する
    - /kaggle/input/XXX にソースコード一式 (experiments/exp007_scale/) が格納
    - /kaggle/input/YYY/{MODEL_VERSION} にモデル + config.yaml が格納
"""

import os
import sys
from pathlib import Path

import kaggle_evaluation.cmi_inference_server
import numpy as np
import pandas as pd
import polars as pl
import torch
import yaml

# ==============================================================================
# ユーザー設定（ここだけ変更する）
# ==============================================================================
SRC_DATASET = "XXX"  # ソースコード用データセット名
MODEL_DATASET = "YYY"  # モデル用データセット名
MODEL_VERSION = "013"  # モデルバージョン

SRC_DIR = Path(f"/kaggle/input/{SRC_DATASET}")
MODEL_DIR = Path(f"/kaggle/input/{MODEL_DATASET}/{MODEL_VERSION}")
DATA_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")

# ==============================================================================
# sys.path設定 & import
# ==============================================================================
sys.path.insert(0, str(SRC_DIR))

from src.columns import build_branch_configs, get_sensor_cols  # noqa: E402
from src.dataset import GESTURE_TO_IDX, IDX_TO_GESTURE, pad_sequence  # noqa: E402
from src.models import get_model  # noqa: E402
from src.preprocess import Preprocessor  # noqa: E402

# ==============================================================================
# Config読み込み（Hydraなし）
# ==============================================================================
with open(MODEL_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)["exp"]

features = cfg["features"]
sensor_cols = get_sensor_cols(features)
num_classes = len(GESTURE_TO_IDX)
max_length = cfg["max_length"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Preprocessor を訓練データでfit（スケーリング内蔵）
# ==============================================================================
print("Fitting preprocessor on training data...")
train_df = pd.read_csv(DATA_DIR / "train.csv")

preprocessor = Preprocessor(apply_scaling=True)
preprocessor.fit_transform(train_df)

del train_df
print("Preprocessor fitted.")

# ==============================================================================
# 5-Foldモデルのロード
# ==============================================================================
print("Loading models...")
models = []
for fold in cfg.get("folds", [0, 1, 2, 3, 4]):
    model = get_model(
        model_name=cfg["model_name"],
        num_classes=num_classes,
        branch_configs=build_branch_configs(features, cfg["branch_hidden_multiplier"]),
        sensor_cols=sensor_cols,
        rnn_type=cfg["rnn_type"],
        rnn_hidden_size=cfg["rnn_hidden_size"],
        rnn_num_layers=cfg["rnn_num_layers"],
        rnn_dropout=cfg["rnn_dropout"],
        rnn_bidirectional=cfg["rnn_bidirectional"],
        mlp_hidden_channels=list(cfg["mlp_hidden_channels"]),
        mlp_dropout=cfg["mlp_dropout"],
    )
    model.load_state_dict(
        torch.load(
            MODEL_DIR / f"model_fold{fold}_best.pt",
            map_location=DEVICE,
            weights_only=True,
        )
    )
    model.to(DEVICE)
    model.eval()
    models.append(model)

print(f"Loaded {len(models)} models.")


# ==============================================================================
# 推論関数
# ==============================================================================
@torch.no_grad()
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """1シーケンスに対する推論を行う。"""
    # Polars → Pandas変換
    seq_pd = sequence.to_pandas()

    # 前処理 + スケーリング（trainでfitしたpreprocessorでtransform）
    seq_processed = preprocessor.transform(seq_pd)

    # センサーカラム抽出 → パディング → Tensor化
    data = seq_processed[sensor_cols].values.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)
    data = pad_sequence(data, max_length)
    input_tensor = torch.from_numpy(data).unsqueeze(0).to(DEVICE)

    # 5-Fold Ensemble（softmax確率の平均）
    probs = torch.stack(
        [torch.softmax(model(input_tensor), dim=1) for model in models]
    ).mean(dim=0)

    pred_idx = torch.argmax(probs, dim=1).item()
    return IDX_TO_GESTURE[pred_idx]


# ==============================================================================
# サーバー起動
# ==============================================================================
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            str(DATA_DIR / "test.csv"),
            str(DATA_DIR / "test_demographics.csv"),
        )
    )
