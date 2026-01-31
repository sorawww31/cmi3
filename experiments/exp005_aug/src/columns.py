"""
Centralized column configuration for CMI Behavior Detection.

This module provides:
1. FeatureGroup definitions for each sensor group
2. Helper functions to generate column lists and BranchConfigs
3. Single source of truth for column ordering
"""

from dataclasses import dataclass


@dataclass
class FeatureGroup:
    """特徴量グループの定義"""

    name: str
    columns: list[str]


# ============================================================================
# Feature Group Definitions
# 新しい特徴量を追加する場合: ここに FeatureGroup を追加するだけ！
# ============================================================================

FEATURE_GROUPS: dict[str, FeatureGroup] = {
    "rot2": FeatureGroup(
        "rot2",
        [],
    ),
    "euler": FeatureGroup(
        "euler",
        ["roll", "pitch", "yaw"],
    ),
    "imu": FeatureGroup(
        "imu",
        [
            "acc_x",
            "acc_y",
            "acc_z",
            "linear_acc_x",
            "linear_acc_y",
            "linear_acc_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "rot_w",
            "rot_angle",
            "rot_angle_vel",
            "angular_vel_x",
            "angular_vel_y",
            "angular_vel_z",
            "angular_dist",
            "jerk_x",
            "jerk_y",
            "jerk_z",
        ],
    ),
    "acc2": FeatureGroup(
        "acc2",
        [
            "acc_x2",
            "acc_y2",
            "acc_z2",
        ],
    ),
    "thm": FeatureGroup("thm", [f"thm_{i}" for i in range(1, 6)]),
    "tof": FeatureGroup(
        "tof", [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]
    ),
}


def get_sensor_cols(group_names: list[str]) -> list[str]:
    """グループ名リストから正しい順序のカラムリストを生成

    返り値のリスト順 = モデルのスライス順 が保証される

    Args:
        group_names: 使用する特徴量グループ名のリスト (例: ["imu", "euler"])

    Returns:
        正しい順序で連結されたカラム名リスト

    Raises:
        ValueError: 未知のグループ名が指定された場合
    """
    cols = []
    for name in group_names:
        if name not in FEATURE_GROUPS:
            available = list(FEATURE_GROUPS.keys())
            raise ValueError(f"Unknown feature group: '{name}'. Available: {available}")
        cols.extend(FEATURE_GROUPS[name].columns)
    return cols


def build_branch_configs(
    group_names: list[str],
    hidden_multiplier: int = 2,
) -> list:
    """グループ名からBranchConfigリストを生成

    Args:
        group_names: 使用する特徴量グループ名のリスト
        hidden_multiplier: hidden_channels計算用の倍率

    Returns:
        BranchConfigのリスト (channel_indicesは未設定)
    """
    from src.models import BranchConfig

    configs = []
    for name in group_names:
        if name not in FEATURE_GROUPS:
            available = list(FEATURE_GROUPS.keys())
            raise ValueError(f"Unknown feature group: '{name}'. Available: {available}")

        group = FEATURE_GROUPS[name]
        num_ch = len(group.columns)
        configs.append(
            BranchConfig(
                name=group.name,
                columns=group.columns,
                hidden_channels=[
                    num_ch * hidden_multiplier,
                    num_ch * hidden_multiplier * 2,
                    num_ch * hidden_multiplier * 4,
                    num_ch * hidden_multiplier * 8,
                ],
            )
        )
    return configs
