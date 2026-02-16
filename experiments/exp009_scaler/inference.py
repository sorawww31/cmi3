"""
Inference script for CMI Behavior Detection competition.
Uses kaggle_evaluation.cmi_inference_server for Code Competition submission.

Usage:
    - Kaggle notebook上で実行する
    - /kaggle/input/XXX にソースコード一式 (experiments/exp009_scaler/) が格納
    - /kaggle/input/YYY/{MODEL_VERSION} にモデル + config.yaml + scaler_fold*.npz が格納
"""

import os
import sys
from pathlib import Path

import kaggle_evaluation.cmi_inference_server
import pandas as pd
import polars as pl
import torch
import yaml

# ==============================================================================
# ユーザー設定（ここだけ変更する）
# ==============================================================================
DATASETS = "/kaggle/input/datasets/sorawww31/exp009-scaler"
SRC_DATASET = f"{DATASETS}/experiments"  # ソースコード用データセット名
MODEL_DATASET = f"{DATASETS}/output"  # モデル用データセット名
MODEL_VERSION = "013"  # モデルバージョン

SRC_DIR = Path(f"{SRC_DATASET}")
MODEL_DIR = Path(f"{MODEL_DATASET}/{MODEL_VERSION}")
DATA_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")

# ==============================================================================
# sys.path設定 & import
# ==============================================================================
sys.path.insert(0, str(SRC_DIR))

from src.columns import build_branch_configs, get_sensor_cols  # noqa: E402
from src.dataset import (  # noqa: E402
    GESTURE_TO_IDX,
    IDX_TO_GESTURE,
    SequenceScaler,
    preconvert_sequences,
)
from src.models import get_model  # noqa: E402
from src.preprocess import Preprocessor  # noqa: E402

# ==============================================================================
# Config読み込み（Hydraなし）
# ==============================================================================
with open(MODEL_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

features = cfg["features"]
sensor_cols = get_sensor_cols(features)
num_classes = len(GESTURE_TO_IDX)
max_length = cfg["max_length"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Preprocessor を訓練データから構築（前処理パイプラインの再現）
# ==============================================================================
print("Building preprocessor from training data...")
train_df = pd.read_csv(DATA_DIR / "train.csv")

preprocessor = Preprocessor(apply_scaling=False)
preprocessor.fit_transform(train_df)
# fit_transform で内部パラメータを学習、以降は transform のみ使用

del train_df
print("Preprocessor ready.")

# ==============================================================================
# 5-Foldモデル + fold毎のScalerをロード
# ==============================================================================
print("Loading models and per-fold scalers...")
models = []
scalers = []

for fold in cfg.get("folds", [0, 1, 2, 3, 4]):
    # Model
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

    # Scaler: fold毎に保存されたパラメータをロード
    scaler_path = MODEL_DIR / f"scaler_fold{fold}.npz"
    scaler = SequenceScaler.load(scaler_path)
    scalers.append(scaler)
    print(f"  Fold {fold}: model + scaler loaded")

print(f"Loaded {len(models)} models with per-fold scalers.")


# ==============================================================================
# 推論関数（訓練時と同一のデータフロー）
# ==============================================================================
@torch.no_grad()
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """1シーケンスに対する推論を行う。

    データフロー（create_dataloaders と同一）:
        1. preprocessor.transform() で前処理
        2. preconvert_sequences() で nan_to_num + pad_sequence
        3. fold毎のscaler.transform() でスケーリング（訓練時と完全一致）
    """
    # Polars → Pandas変換
    seq_pd = sequence.to_pandas()

    # ダミーの sequence_id を付与（preconvert_sequences が groupby で使用）
    dummy_seq_id = 0
    seq_pd["sequence_id"] = dummy_seq_id

    # 1. 前処理（fit済みpreprocessorでtransformのみ）
    seq_processed = preprocessor.transform(seq_pd)

    # 2. preconvert_sequences（nan_to_num + pad_sequence）
    data_array, lengths, _ = preconvert_sequences(
        seq_processed, [dummy_seq_id], sensor_cols, max_length
    )

    # 3. 各fold: fold固有のscalerでスケーリング → fold固有のモデルで推論
    all_probs = []
    for model, scaler in zip(models, scalers):
        scaled_data = scaler.transform(data_array.copy(), lengths)
        input_tensor = torch.from_numpy(scaled_data).to(DEVICE)
        probs = torch.softmax(model(input_tensor), dim=1)
        all_probs.append(probs)

    # softmax確率の平均でアンサンブル
    avg_probs = torch.stack(all_probs).mean(dim=0)
    pred_idx = torch.argmax(avg_probs, dim=1).item()
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

print("Complete Prediction")
