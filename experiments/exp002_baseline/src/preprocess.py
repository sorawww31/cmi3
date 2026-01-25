"""
Preprocessing module for CMI Behavior Detection competition.

Implements:
1. Missing value imputation (IMU/THM: NaN->0, TOF: NaN/-1->0)
2. Quaternion (rot_x, rot_y, rot_z, rot_w) to Euler angles (roll, pitch, yaw) conversion
3. Handedness correction (flip acc_x, pitch, yaw for handness=1)
4. Sensor swap for handness=1 (THM5<->THM3, TOF5<->TOF3)
5. StandardScaler normalization with fit_transform/transform pattern

Memory-optimized version:
- Minimizes DataFrame copies by using inplace operations where safe
- Single copy at entry point, all subsequent operations are inplace
"""

from typing import Optional

import numpy as np
import pandas as pd

# センサーカラムの定義
IMU_COLS = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
THM_COLS = [f"thm_{i}" for i in range(1, 6)]
TOF_COLS = [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]


def fill_missing_values_inplace(df: pd.DataFrame) -> None:
    """
    欠損値を補完する（インプレース操作）
    - IMU, THM: NaN -> 0
    - TOF: NaN, -1 -> 0

    Args:
        df: センサーデータを含むDataFrame（直接変更される）
    """
    # IMUカラムの欠損値を0に
    imu_cols_exist = [col for col in IMU_COLS if col in df.columns]
    if imu_cols_exist:
        df[imu_cols_exist] = df[imu_cols_exist].fillna(0)

    # THMカラムの欠損値を0に
    thm_cols_exist = [col for col in THM_COLS if col in df.columns]
    if thm_cols_exist:
        df[thm_cols_exist] = df[thm_cols_exist].fillna(0)

    # TOFカラムの欠損値と-1を0に
    tof_cols_exist = [col for col in TOF_COLS if col in df.columns]
    if tof_cols_exist:
        df[tof_cols_exist] = df[tof_cols_exist].fillna(0)
        df[tof_cols_exist] = df[tof_cols_exist].replace(-1, 0)


def quaternion_to_euler(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    クォータニオン (w, x, y, z) からオイラー角 (roll, pitch, yaw) に変換

    Args:
        w, x, y, z: クォータニオン成分の配列

    Returns:
        roll, pitch, yaw: オイラー角（ラジアン）
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # ジンバルロック対策: sinpを[-1, 1]にクリップ
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def add_euler_angles(df: pd.DataFrame) -> pd.DataFrame:
    """
    rot_x, rot_y, rot_z, rot_wからオイラー角（roll, pitch, yaw）を計算して追加
    元のrot_*列は削除

    Args:
        df: rot_x, rot_y, rot_z, rot_w列を含むDataFrame

    Returns:
        オイラー角が追加されたDataFrame（デフラグメント済み）
    """
    roll, pitch, yaw = quaternion_to_euler(
        df["rot_w"].values,
        df["rot_x"].values,
        df["rot_y"].values,
        df["rot_z"].values,
    )

    # クォータニオン列を除いたDataFrameを作成
    cols_to_keep = [
        c for c in df.columns if c not in ["rot_w", "rot_x", "rot_y", "rot_z"]
    ]

    # オイラー角のDataFrameを作成
    euler_df = pd.DataFrame({"roll": roll, "pitch": pitch, "yaw": yaw}, index=df.index)

    # pd.concatで一括結合（断片化を回避）
    return pd.concat([df[cols_to_keep], euler_df], axis=1)


def correct_handedness_inplace(
    df: pd.DataFrame, handness_col: str = "handness"
) -> None:
    """
    handness=1の被験者に対して（インプレース操作）:
    1. acc_x, pitch, yawの符号を反転
    2. THM5 <-> THM3 を入れ替え
    3. TOF5 <-> TOF3 を入れ替え

    Args:
        df: センサーデータを含むDataFrame（直接変更される）
        handness_col: 利き手を示す列名
    """
    # handness=1のマスク
    mask = df[handness_col] == 1

    if not mask.any():
        return

    # 1. acc_x, pitch, yawの符号反転
    for col in ["acc_x", "pitch", "yaw"]:
        if col in df.columns:
            df.loc[mask, col] = -df.loc[mask, col]

    # 2. THM5 <-> THM3 の入れ替え
    if "thm_5" in df.columns and "thm_3" in df.columns:
        thm5_values = df.loc[mask, "thm_5"].values.copy()
        df.loc[mask, "thm_5"] = df.loc[mask, "thm_3"].values
        df.loc[mask, "thm_3"] = thm5_values

    # 3. TOF5 <-> TOF3 の入れ替え (各チャンネル0-63)
    for v in range(64):
        tof5_col = f"tof_5_v{v}"
        tof3_col = f"tof_3_v{v}"
        if tof5_col in df.columns and tof3_col in df.columns:
            tof5_values = df.loc[mask, tof5_col].values.copy()
            df.loc[mask, tof5_col] = df.loc[mask, tof3_col].values
            df.loc[mask, tof3_col] = tof5_values


class Preprocessor:
    """
    前処理クラス。sklearn風のfit_transform/transformインターフェースを提供

    処理内容:
    1. 欠損値補完 (IMU/THM: NaN->0, TOF: NaN/-1->0)
    2. クォータニオン → オイラー角変換
    3. Handedness補正
    4. StandardScaler正規化

    Memory-optimized:
    - コピーは各メソッドのエントリポイントで1回のみ
    - 内部処理は全てインプレース
    """

    def __init__(
        self,
        apply_fill_missing: bool = True,
        apply_euler: bool = True,
        apply_handedness_correction: bool = True,
        apply_scaling: bool = True,
        handness_col: str = "handness",
        feature_cols: Optional[list[str]] = None,
    ):
        """
        Args:
            apply_fill_missing: 欠損値補完を適用するか
            apply_euler: クォータニオン→オイラー角変換を適用するか
            apply_handedness_correction: handness補正を適用するか
            apply_scaling: StandardScaler正規化を適用するか
            handness_col: 利き手を示す列名
            feature_cols: 正規化対象のカラムリスト（Noneの場合は自動検出）
        """
        self.apply_fill_missing = apply_fill_missing
        self.apply_euler = apply_euler
        self.apply_handedness_correction = apply_handedness_correction
        self.apply_scaling = apply_scaling
        self.handness_col = handness_col
        self.feature_cols = feature_cols

        # StandardScalerのパラメータ
        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None
        self.fitted_ = False

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """正規化対象のカラムを取得"""
        if self.feature_cols is not None:
            return [col for col in self.feature_cols if col in df.columns]

        # 数値カラムを自動検出（メタデータカラムを除外）
        exclude_cols = {
            "sequence_id",
            "subject",
            "gesture",
            "sequence_type",
            "handness",
            self.handness_col,
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in exclude_cols]

    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値補完、オイラー角変換、handness補正を適用"""

        # 1. 欠損値補完
        if self.apply_fill_missing:
            fill_missing_values_inplace(df)

        # 2. クォータニオン → オイラー角変換
        if self.apply_euler:
            required_cols = ["rot_w", "rot_x", "rot_y", "rot_z"]
            if all(col in df.columns for col in required_cols):
                df = add_euler_angles(df)

        # 3. Handedness補正
        if self.apply_handedness_correction:
            if self.handness_col in df.columns:
                correct_handedness_inplace(df, self.handness_col)

        return df

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """
        訓練データから正規化パラメータを学習

        Args:
            df: 訓練データ

        Returns:
            self
        """
        # コピーを作成（元データを変更しない）
        df_processed = df.copy()

        # 前処理を適用
        df_processed = self._apply_preprocessing(df_processed)

        # 正規化対象カラムを取得
        feature_cols = self._get_feature_cols(df_processed)

        if self.apply_scaling and feature_cols:
            # 平均と標準偏差を計算
            self.mean_ = df_processed[feature_cols].mean()
            self.std_ = df_processed[feature_cols].std()
            # 標準偏差が0の場合は1に置き換え（ゼロ除算防止）
            self.std_ = self.std_.replace(0, 1)

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        学習済みパラメータで正規化を適用

        Args:
            df: 変換対象データ

        Returns:
            変換後のDataFrame
        """
        if not self.fitted_:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")

        # コピーを作成（元データを変更しない）
        df_processed = df.copy()

        # 前処理を適用
        df_processed = self._apply_preprocessing(df_processed)

        # 正規化を適用
        if self.apply_scaling and self.mean_ is not None and self.std_ is not None:
            feature_cols = self.mean_.index.tolist()
            existing_cols = [col for col in feature_cols if col in df_processed.columns]
            df_processed[existing_cols] = (
                df_processed[existing_cols] - self.mean_[existing_cols]
            ) / self.std_[existing_cols]

        return df_processed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        訓練データにfitしてtransformを適用（メモリ最適化版）

        コピーは1回のみ作成し、fit + transform を効率的に実行

        Args:
            df: 訓練データ

        Returns:
            変換後のDataFrame
        """
        # コピーを1回だけ作成
        df_processed = df.copy()

        # 前処理を適用
        df_processed = self._apply_preprocessing(df_processed)

        # 正規化対象カラムを取得
        feature_cols = self._get_feature_cols(df_processed)

        if self.apply_scaling and feature_cols:
            # 平均と標準偏差を計算（fit）
            self.mean_ = df_processed[feature_cols].mean()
            self.std_ = df_processed[feature_cols].std()
            self.std_ = self.std_.replace(0, 1)

            # 正規化を適用（transform）
            df_processed[feature_cols] = (
                df_processed[feature_cols] - self.mean_
            ) / self.std_

        self.fitted_ = True
        return df_processed
