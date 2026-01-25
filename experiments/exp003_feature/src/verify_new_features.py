import numpy as np
import pandas as pd

from src.preprocess import Preprocessor


def verify_features():
    # Accelerometer data to test Jerk (diff) and Squared
    # t=0: 1.0 -> 1.0^2 = 1.0
    # t=1: 2.0 -> 2.0^2 = 4.0, Jerk = (2.0 - 1.0) / 0.005 = 1.0 / 0.005 = 200
    # t=2: 3.0 -> 3.0^2 = 9.0, Jerk = (3.0 - 2.0) / 0.005 = 200
    data = {
        "acc_x": [1.0, 2.0, 3.0],
        "acc_y": [0.0, 0.0, 0.0],
        "acc_z": [0.0, 0.0, 0.0],
        "rot_x": [0.0, 0.0, 0.0],
        "rot_y": [0.0, 0.0, 0.0],
        "rot_z": [0.0, 0.0, 0.0],
        "rot_w": [1.0, 1.0, 1.0],
        "handness": [0, 0, 0],
    }

    df = pd.DataFrame(data)

    processor = Preprocessor(
        apply_fill_missing=True,
        apply_euler=False,
        apply_handedness_correction=False,
        apply_scaling=False,
        apply_linear_acc=False,
        apply_angular_vel=False,
        apply_angular_dist=False,
        apply_jerk=True,
        apply_acc_squared=True,
        sampling_rate=200.0,  # dt = 0.005s
    )

    df_processed = processor.fit_transform(df)

    print("--- Processed DataFrame ---")
    print(df_processed[["acc_x", "jerk_x", "acc_x2"]])

    # Validation
    # Squared Accuracy
    assert np.allclose(df_processed["acc_x2"], [1.0, 4.0, 9.0]), (
        "Squared acceleration mismatch"
    )

    # Jerk
    # First element diff depends on implementation.
    # Current impl: np.diff(prepend=acc[:1]) -> first element diff is 0 (1-1)
    # So Jerk[0] should be 0.
    # Jerk[1] = (2-1)/0.005 = 200
    # Jerk[2] = (3-2)/0.005 = 200
    expected_jerk = [0.0, 200.0, 200.0]
    print(f"Jerk X: {df_processed['jerk_x'].values}")
    assert np.allclose(df_processed["jerk_x"], expected_jerk), (
        "Jerk calculation mismatch"
    )

    print("Verification Passed!")


if __name__ == "__main__":
    verify_features()
