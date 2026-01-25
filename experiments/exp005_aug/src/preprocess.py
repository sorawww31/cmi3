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
from scipy.spatial.transform import Rotation as R

# センサーカラムの定義
IMU_COLS = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
THM_COLS = [f"thm_{i}" for i in range(1, 6)]
TOF_COLS = [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]


# ============================================================================
# Accelerometer Feature Engineering Functions
# ============================================================================


def remove_gravity_from_acc(
    acc_data: np.ndarray | pd.DataFrame,
    rot_data: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """加速度データから重力成分を除去し、線形加速度を計算

    Args:
        acc_data: 加速度データ (N, 3) または DataFrame with acc_x, acc_y, acc_z
        rot_data: クォータニオンデータ (N, 4) または DataFrame with rot_x, rot_y, rot_z, rot_w

    Returns:
        linear_accel: 重力除去後の線形加速度 (N, 3)
    """
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[["acc_x", "acc_y", "acc_z"]].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
            linear_accel[i, :] = acc_values[i, :]

    return linear_accel


def calculate_angular_velocity_from_quat(
    rot_data: np.ndarray | pd.DataFrame,
    time_delta: float = 1 / 200,  # 200Hz sampling rate
) -> np.ndarray:
    """クォータニオンから角速度を計算

    Args:
        rot_data: クォータニオンデータ (N, 4) または DataFrame
        time_delta: サンプリング間隔（秒）

    Returns:
        angular_vel: 角速度 (N, 3) [rad/s]
    """
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i + 1]

        if (
            np.all(np.isnan(q_t))
            or np.all(np.isclose(q_t, 0))
            or np.all(np.isnan(q_t_plus_dt))
            or np.all(np.isclose(q_t_plus_dt, 0))
        ):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass

    return angular_vel


def calculate_angular_distance(
    rot_data: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """連続するクォータニオン間の角距離を計算

    Args:
        rot_data: クォータニオンデータ (N, 4) または DataFrame

    Returns:
        angular_dist: 角距離 (N,) [rad]
    """
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i + 1]

        if (
            np.all(np.isnan(q1))
            or np.all(np.isclose(q1, 0))
            or np.all(np.isnan(q2))
            or np.all(np.isclose(q2, 0))
        ):
            angular_dist[i] = 0
            continue

        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0

    return angular_dist


def calculate_jerk(
    acc_data: np.ndarray | pd.DataFrame,
    time_delta: float = 1,
) -> np.ndarray:
    """加速度の微分（躍度: Jerk）を計算

    Args:
        acc_data: 加速度データ (N, 3) または DataFrame
        time_delta: サンプリング間隔

    Returns:
        jerk: 躍度 (N, 3)
    """
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[["acc_x", "acc_y", "acc_z"]].values
    else:
        acc_values = acc_data

    # np.gradient または diff で微分
    # ここでは単純な差分を使用 (境界は0埋めなど)
    diff = np.diff(acc_values, axis=0, prepend=acc_values[:1])
    jerk = diff / time_delta

    # ノイズ対策で移動平均などをかける場合もあるが、まずは生Jerk
    return jerk


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
        df[tof_cols_exist] = df[tof_cols_exist].replace(-1, 255)


def quat_to_euler(df: pd.DataFrame) -> pd.DataFrame:
    """
    クォータニオン (rot_x, rot_y, rot_z, rot_w) からオイラー角 (roll, pitch, yaw) に変換
    Scipyを使用 (degrees=True)

    Args:
        df: rot_x, rot_y, rot_z, rot_w列を含むDataFrame

    Returns:
        roll, pitch, yaw を含むDataFrame
    """
    # 1. 回転オブジェクトを作成
    # Scipyのクオータニオン順序は (x, y, z, w)
    quats = df[["rot_x", "rot_y", "rot_z", "rot_w"]].values

    # ゼロノルムのクォータニオンをチェックして単位クォータニオン (0,0,0,1) に置換
    norms = np.linalg.norm(quats, axis=1)
    zero_norm_mask = norms < 1e-9

    if zero_norm_mask.any():
        # コピーを作成して書き換え
        quats = quats.copy()
        quats[zero_norm_mask] = [0, 0, 0, 1]

    r = R.from_quat(quats)

    # 2. オイラー角に変換 ('xyz'は回転の適用順序)
    euler_angles = r.as_euler("xyz", degrees=True)

    # 必要ならゼロノルムだった行のオイラー角を0にする（Identityなら自然に0になるが明示的にも可能）
    # Identity (0,0,0,1) -> Euler (0,0,0) なのでそのままでOK

    return pd.DataFrame(euler_angles, columns=["roll", "pitch", "yaw"], index=df.index)


def euler_to_quat(df: pd.DataFrame) -> pd.DataFrame:
    """
    オイラー角 (roll, pitch, yaw) からクォータニオン (rot_x, rot_y, rot_z, rot_w) に変換
    Scipyを使用 (degrees=True)

    Args:
        df: roll, pitch, yaw列を含むDataFrame

    Returns:
        rot_x, rot_y, rot_z, rot_w を含むDataFrame
    """
    # 1. 回転オブジェクトを作成
    r = R.from_euler("xyz", df[["roll", "pitch", "yaw"]].values, degrees=True)

    # 2. クォータニオンに戻す
    quats = r.as_quat()

    return pd.DataFrame(
        quats, columns=["rot_x", "rot_y", "rot_z", "rot_w"], index=df.index
    )


def add_euler_angles(df: pd.DataFrame) -> pd.DataFrame:
    """
    rot_x, rot_y, rot_z, rot_wからオイラー角を計算して追加
    ※ rot_*列は保持する（後で補正後に更新するため）

    Args:
        df: rot_x, rot_y, rot_z, rot_w列を含むDataFrame

    Returns:
        オイラー角が追加されたDataFrame
    """
    euler_df = quat_to_euler(df)

    # 既存のcolumnsと重複しないか確認しつつ結合
    # もし既にroll/pitch/yawがある場合は上書きしたいが、pd.concatは重複名でエラーにならないため
    # 明示的に除外してから結合するか、単にconcatする
    cols_to_keep = [c for c in df.columns if c not in ["roll", "pitch", "yaw"]]
    return pd.concat([df[cols_to_keep], euler_df], axis=1)


def add_rotation_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    rot_angle と rot_angle_vel を追加する関数
    """
    df_eng = df.copy()

    # --- 1. rot_angle (基準からの回転角度) ---
    # クォータニオンの実部(w)から計算
    # np.clipは数値誤差で1.0を超えてNaNになるのを防ぐため必須
    # abs()をとるのは、qと-qが同じ回転を表すため
    df_eng["rot_angle"] = 2 * np.arccos(np.clip(df_eng["rot_w"].abs(), -1.0, 1.0))

    # --- 2. rot_angle_vel (角速度の推定) ---
    # ステップ間のクォータニオンの変化量を計算

    # グループごとに計算するためのヘルパー関数
    def calculate_angular_velocity(group):
        # 現在のq
        q_curr = group[["rot_w", "rot_x", "rot_y", "rot_z"]].values

        # 1ステップ前のq (shift)
        q_prev = np.roll(q_curr, 1, axis=0)
        q_prev[0] = q_curr[0]  # 先頭は差分なしとする(0埋め)

        # クォータニオンの内積: w1*w2 + x1*x2 + ...
        # 行ごとの和 (axis=1) をとる
        dot_product = np.sum(q_curr * q_prev, axis=1)

        # 最短経路をとるため絶対値 (qと-qの等価性)
        dot_product = np.abs(dot_product)

        # 数値安定性のためのクリッピング (-1 ~ 1)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # 角度差を計算 (これが1ステップあたりの回転量)
        delta_theta = 2 * np.arccos(dot_product)

        return pd.Series(delta_theta, index=group.index)

    # sequence_idごとに適用 (これ重要！またがないように)
    # ※ tqdmなどが不要なら単純に apply してください
    if "sequence_id" in df_eng.columns:
        df_eng["rot_angle_vel"] = 0.0
        for _, group in df_eng.groupby("sequence_id"):
            df_eng.loc[group.index, "rot_angle_vel"] = calculate_angular_velocity(group)
    else:
        # sequence_idがない場合は全体を1つのシーケンスとして扱う（非推奨だが念のため）
        df_eng["rot_angle_vel"] = calculate_angular_velocity(df_eng)

    # 先頭行などがNaNになる場合のケア（0で埋める）
    df_eng["rot_angle_vel"] = df_eng["rot_angle_vel"].fillna(0)

    return df_eng


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
        apply_linear_acc: bool = True,
        apply_angular_vel: bool = True,
        apply_angular_dist: bool = True,
        apply_jerk: bool = True,
        apply_acc_squared: bool = True,
        apply_rotation_angle_features: bool = True,
        handness_col: str = "handness",
        feature_cols: Optional[list[str]] = None,
        sampling_rate: float = 200.0,
    ):
        """
        Args:
            apply_fill_missing: 欠損値補完を適用するか
            apply_euler: クォータニオン→オイラー角変換を適用するか
            apply_handedness_correction: handness補正を適用するか
            apply_scaling: StandardScaler正規化を適用するか
            apply_linear_acc: 重力除去後の線形加速度を計算するか
            apply_angular_vel: 角速度を計算するか
            apply_angular_dist: 角距離を計算するか
            handness_col: 利き手を示す列名
            feature_cols: 正規化対象のカラムリスト（Noneの場合は自動検出）
            sampling_rate: サンプリングレート (Hz)
        """
        self.apply_fill_missing = apply_fill_missing
        self.apply_euler = apply_euler
        self.apply_handedness_correction = apply_handedness_correction
        self.apply_scaling = apply_scaling
        self.apply_linear_acc = apply_linear_acc
        self.apply_angular_vel = apply_angular_vel
        self.apply_angular_dist = apply_angular_dist
        self.apply_jerk = apply_jerk
        self.apply_acc_squared = apply_acc_squared
        self.apply_rotation_angle_features = apply_rotation_angle_features
        self.handness_col = handness_col
        self.feature_cols = feature_cols
        self.sampling_rate = sampling_rate

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
        """欠損値補完、オイラー角変換、handness補正、特徴量生成を適用"""

        # 1. 欠損値補完
        if self.apply_fill_missing:
            fill_missing_values_inplace(df)

        # 2. クォータニオン → オイラー角変換 (Roll/Pitch/Yaw 生成)
        required_quat_cols = ["rot_w", "rot_x", "rot_y", "rot_z"]
        has_quat = all(col in df.columns for col in required_quat_cols)

        if self.apply_euler and has_quat:
            df = add_euler_angles(df)

        # 3. Handedness補正 (Acc & Euler Inversion, Sensor Swap)
        if self.apply_handedness_correction:
            if self.handness_col in df.columns:
                correct_handedness_inplace(df, self.handness_col)

        # 4. オイラー角 → クォータニオン (補正されたオイラー角から再生成)
        # Euler変換とHandedness補正の両方が適用された場合のみ再生成
        if self.apply_euler and has_quat and self.apply_handedness_correction:
            if all(c in df.columns for c in ["roll", "pitch", "yaw"]):
                new_quats = euler_to_quat(df)
                # rotカラムを更新
                df[required_quat_cols] = new_quats[required_quat_cols]

        # 5. 特徴量生成 (線形加速度、角速度など)
        # ※ 補正後の acc, rot を使用して計算する
        has_acc = all(col in df.columns for col in ["acc_x", "acc_y", "acc_z"])
        new_features = {}

        if has_quat and has_acc and self.apply_linear_acc:
            # 重力除去後の線形加速度を計算
            linear_acc = remove_gravity_from_acc(df, df)
            new_features["linear_acc_x"] = linear_acc[:, 0]
            new_features["linear_acc_y"] = linear_acc[:, 1]
            new_features["linear_acc_z"] = linear_acc[:, 2]

        if has_quat and self.apply_angular_vel:
            # 角速度を計算
            time_delta = 1.0 / self.sampling_rate
            angular_vel = calculate_angular_velocity_from_quat(df, time_delta)
            new_features["angular_vel_x"] = angular_vel[:, 0]
            new_features["angular_vel_y"] = angular_vel[:, 1]
            new_features["angular_vel_z"] = angular_vel[:, 2]

        if has_quat and self.apply_angular_dist:
            # 角距離を計算
            angular_dist = calculate_angular_distance(df)
            new_features["angular_dist"] = angular_dist

        if has_acc and self.apply_jerk:
            # Jerk (躍度)
            jerk = calculate_jerk(df)
            new_features["jerk_x"] = jerk[:, 0]
            new_features["jerk_y"] = jerk[:, 1]
            new_features["jerk_z"] = jerk[:, 2]

        if has_acc and self.apply_acc_squared:
            # 加速度の二乗
            new_features["acc_x2"] = df["acc_x"] ** 2
            new_features["acc_y2"] = df["acc_y"] ** 2
            new_features["acc_z2"] = df["acc_z"] ** 2

        if new_features:
            new_features_df = pd.DataFrame(new_features, index=df.index)
            df = pd.concat([df, new_features_df], axis=1)

        # 6. その他の特徴量 (rot_angle, rot_angle_vel)
        if self.apply_rotation_angle_features and has_quat:
            # sequence_idが必要
            if "sequence_id" in df.columns:
                df = add_rotation_angle_features(df)

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
