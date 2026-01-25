"""
Main training script for CMI Behavior Detection competition.
Uses Hydra for configuration, Wandb for experiment tracking,
and GroupKFold for cross-validation by subject.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from sklearn.model_selection import GroupKFold

import wandb

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))  # exp/exp000_sample
project_root = os.path.join(current_dir, "../../")  # rootへ移動
sys.path.append(os.path.normpath(project_root))

# ruffの警告を無視

from utils.logger import get_logger  # noqa: E402
from utils.timing import trace  # noqa: E402

# ローカルモジュール（相対インポートでなく直接追加）
sys.path.insert(0, current_dir)
from src.config import Config, ExpConfig  # noqa: E402
from src.dataset import (  # noqa: E402
    ALL_SENSOR_COLS,
    GESTURE_TO_IDX,
    IMU_COLS,
    create_dataloaders,
    load_train_data,
)
from src.models import get_model  # noqa: E402
from src.train import train_fold  # noqa: E402
from src.utils import count_parameters, get_device, set_seed  # noqa: E402

from utils.env import EnvConfig  # noqa: E402

load_dotenv()
LOGGER = None


####################
# Config 設定
####################
# hydra用にdefaultを設定
cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)


####################
# 実験用コード
####################
def log_config(cfg: Config, logger) -> None:
    logger.info(
        "\nConfig: %s",
        json.dumps(OmegaConf.to_container(cfg, resolve=True), default=str, indent=4),
    )


def init_output_dir(cfg: Config):
    this_file_path = Path(__file__).resolve()
    cfg.env.output_dir = this_file_path.parent / "outputs"
    cfg.env.exp_output_dir = (
        cfg.env.output_dir / HydraConfig.get().runtime.choices["exp"]
    )
    output_dir = cfg.env.exp_output_dir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_config(cfg: Config, logger) -> None:
    """設定をexp_output_dirにconfig.yamlとして保存する"""
    config_path = Path(cfg.env.exp_output_dir) / "config.yaml"
    OmegaConf.save(cfg, config_path)
    logger.info(f"Config saved to: {config_path}")


def get_sensor_cols(sensor_type: str) -> list[str]:
    """センサータイプに応じたカラムを取得"""
    if sensor_type == "imu":
        return IMU_COLS
    elif sensor_type == "all":
        return ALL_SENSOR_COLS
    else:
        raise ValueError(f"Unknown sensor_type: {sensor_type}")


def prepare_fold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFoldでsubject毎にfold分割"""
    # シーケンス毎の情報を取得
    seq_info = df.groupby("sequence_id").first().reset_index()
    sequence_ids = seq_info["sequence_id"].values
    subjects = seq_info["subject"].values

    gkf = GroupKFold(n_splits=n_folds)

    # ダミーのyを使用
    splits = []
    for train_idx, val_idx in gkf.split(sequence_ids, groups=subjects):
        train_seq_ids = sequence_ids[train_idx].tolist()
        val_seq_ids = sequence_ids[val_idx].tolist()
        splits.append((train_seq_ids, val_seq_ids))

    return splits


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:
    global LOGGER
    output_dir = init_output_dir(cfg)
    LOGGER = get_logger(__name__, output_dir)
    LOGGER.info("output_dir: %s", output_dir)
    LOGGER.info("Start CMI Behavior Detection Training")

    exp_name = (
        f"{Path(sys.argv[0]).parent.name}/{HydraConfig.get().runtime.choices['exp']}"
    )

    log_config(cfg, LOGGER)
    save_config(cfg, LOGGER)

    # Set seed
    set_seed(cfg.exp.seed)

    # Device
    device = get_device()
    LOGGER.info(f"Using device: {device}")

    # Debug mode adjustments
    if cfg.exp.debug:
        LOGGER.info("Running in DEBUG mode")
        cfg.exp.epochs = 2
        cfg.exp.folds = [0]

    # Initialize wandb
    wandb.init(
        project=cfg.exp.wandb_project_name,
        name=exp_name,
        notes=", ".join(HydraConfig.get().overrides.task),
        config=cast(dict[str, Any], OmegaConf.to_container(cfg.exp, resolve=True)),
        mode="disabled" if cfg.exp.debug else "online",
    )

    # Load data
    LOGGER.info("Loading data...")

    with trace("load_train_data"):
        train_df, demo_df = load_train_data(
            Path(cfg.env.input_dir) / "cmi-detect-behavior-with-sensor-data"
        )

    LOGGER.info(f"Train data shape: {train_df.shape}")
    LOGGER.info(f"Number of sequences: {train_df['sequence_id'].nunique()}")
    LOGGER.info(f"Number of subjects: {train_df['subject'].nunique()}")

    # Debug: use subset of data
    if cfg.exp.debug:
        unique_seqs = train_df["sequence_id"].unique()[:100]
        train_df = train_df[train_df["sequence_id"].isin(unique_seqs)]
        LOGGER.info(f"Debug mode: using {len(unique_seqs)} sequences")

    # Prepare fold splits
    LOGGER.info("Preparing fold splits...")
    splits = prepare_fold_splits(train_df, cfg.exp.n_folds)

    # Get sensor columns
    sensor_cols = get_sensor_cols(cfg.exp.sensor_type)
    input_size = len(sensor_cols)
    num_classes = len(GESTURE_TO_IDX)
    LOGGER.info(f"Input size: {input_size}, Num classes: {num_classes}")

    # Training loop
    all_fold_scores = []

    for fold in cfg.exp.folds:
        LOGGER.info(f"{'=' * 50}")
        LOGGER.info(f"Training Fold {fold}")
        LOGGER.info(f"{'=' * 50}")

        train_ids, val_ids = splits[fold]
        LOGGER.info(f"Train sequences: {len(train_ids)}, Val sequences: {len(val_ids)}")

        # Create dataloaders
        LOGGER.info("Creation Dataloders")
        with trace("create_dataloaders"):
            train_loader, val_loader = create_dataloaders(
                train_df,
                train_ids,
                val_ids,
                batch_size=cfg.exp.batch_size,
                max_length=cfg.exp.max_length,
                sensor_cols=sensor_cols,
                num_workers=cfg.exp.num_workers,
            )

        # Create model
        LOGGER.info("Creation CMIModel")

        # Build branch configs based on sensor columns
        from src.models import BranchConfig

        branch_configs = []
        mult = cfg.exp.branch_hidden_multiplier

        if cfg.exp.sensor_type == "imu":
            # IMU: acc (3ch) + euler (3ch)
            imu_ch = 3
            euler_ch = 3
            branch_configs = [
                BranchConfig(
                    name="imu",
                    input_channels=imu_ch,
                    channel_indices=(0, imu_ch),
                    hidden_channels=[
                        imu_ch * mult,
                        imu_ch * mult * 2,
                        imu_ch * mult * 4,
                    ],
                ),
                BranchConfig(
                    name="euler",
                    input_channels=euler_ch,
                    channel_indices=(imu_ch, imu_ch + euler_ch),
                    hidden_channels=[
                        euler_ch * mult,
                        euler_ch * mult * 2,
                        euler_ch * mult * 4,
                    ],
                ),
            ]
        elif cfg.exp.sensor_type == "all":
            # All sensors: acc (3ch) + euler (3ch) + thm (5ch) + tof (5ch)
            branch_configs = [
                BranchConfig(
                    name="imu",
                    input_channels=3,
                    channel_indices=(0, 3),
                    hidden_channels=[3 * mult, 3 * mult * 2, 3 * mult * 4],
                ),
                BranchConfig(
                    name="euler",
                    input_channels=3,
                    channel_indices=(3, 6),
                    hidden_channels=[3 * mult, 3 * mult * 2, 3 * mult * 4],
                ),
                BranchConfig(
                    name="thm",
                    input_channels=5,
                    channel_indices=(6, 11),
                    hidden_channels=[5 * mult, 5 * mult * 2, 5 * mult * 4],
                ),
                BranchConfig(
                    name="tof",
                    input_channels=5,
                    channel_indices=(11, 16),
                    hidden_channels=[5 * mult, 5 * mult * 2, 5 * mult * 4],
                ),
            ]

        model = get_model(
            model_name=cfg.exp.model_name,
            num_classes=num_classes,
            branch_configs=branch_configs,
            rnn_type=cfg.exp.rnn_type,
            rnn_hidden_size=cfg.exp.rnn_hidden_size,
            rnn_num_layers=cfg.exp.rnn_num_layers,
            rnn_dropout=cfg.exp.rnn_dropout,
            rnn_bidirectional=cfg.exp.rnn_bidirectional,
            mlp_hidden_channels=list(cfg.exp.mlp_hidden_channels),
            mlp_dropout=cfg.exp.mlp_dropout,
        )
        model = model.to(device)
        LOGGER.info(
            f"Model: {cfg.exp.model_name}, Parameters: {count_parameters(model):,}"
        )

        # Train
        with trace(f"train_fold_{fold}"):
            results = train_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg,
                fold=fold,
                output_dir=Path(output_dir),
                logger=LOGGER,
                device=device,
            )

        LOGGER.info(
            f"Fold {fold} Best Score: {results['best_score']:.4f} at epoch {results['best_epoch']}"
        )
        all_fold_scores.append(results["best_score"])

    # Summary
    mean_score = np.mean(all_fold_scores)
    std_score = np.std(all_fold_scores)
    LOGGER.info(f"\n{'=' * 50}")
    LOGGER.info(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")
    LOGGER.info(f"{'=' * 50}")

    wandb.log(
        {
            "cv_mean": mean_score,
            "cv_std": std_score,
        }
    )

    wandb.finish()
    LOGGER.info("Training completed!")


if __name__ == "__main__":
    main()
