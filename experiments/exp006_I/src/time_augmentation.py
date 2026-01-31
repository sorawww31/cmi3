# import colorednoise as cn
import numpy as np
from scipy.signal import butter, lfilter


class SignalTransform:
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: np.ndarray, **params):
        if self.always_apply:
            return self.apply(x, **params)
        else:
            if np.random.rand() < self.p:
                return self.apply(x, **params)
            else:
                return x

    def apply(self, x: np.ndarray, **params):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        return trns(y)


class GaussianNoise(SignalTransform):
    def __init__(
        self,
        always_apply: bool = False,
        p: float = 0.5,
        max_noise_amplitude: float = 0.20,
        **kwargs,
    ):
        super().__init__(always_apply, p)
        self.noise_amplitude = (0.0, max_noise_amplitude)

    def apply(self, x: np.ndarray, **params):
        noise_amplitude = np.random.uniform(*self.noise_amplitude)
        noise = np.random.randn(*x.shape)  # shape (L, N)
        augmented = (x + noise * noise_amplitude).astype(x.dtype)
        return augmented


class PinkNoiseSNR(SignalTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, x: np.ndarray, **params):
        # Check if x contains any zeros or very small values to avoid division by zero
        # Add epsilon if needed, but here we assume preprocessed standardized data
        snr = np.random.uniform(self.min_snr, self.max_snr)

        # Calculate signal amplitude for each channel
        # x shape: (L, N) -> max along axis 0
        a_signal = np.sqrt((x**2).max(axis=0))  # shape: (N,)

        # Handle cases where signal is practically zero
        a_signal = np.maximum(a_signal, 1e-6)

        a_noise = a_signal / (10 ** (snr / 20))  # shape: (N,)

        # Generate pink noise for each channel
        # cn.powerlaw_psd_gaussian(1, len(x)) generates 1D pink noise
        pink_noise = np.stack(
            [cn.powerlaw_psd_gaussian(1, len(x)) for _ in range(x.shape[1])], axis=1
        )

        a_pink = np.sqrt((pink_noise**2).max(axis=0))  # shape: (N,)
        a_pink = np.maximum(a_pink, 1e-6)

        pink_noise_normalized = pink_noise * (a_noise / a_pink)
        augmented = (x + pink_noise_normalized).astype(x.dtype)
        return augmented


class TimeStretch(SignalTransform):
    def __init__(self, max_rate=1.5, min_rate=0.5, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.max_rate = max_rate
        self.min_rate = min_rate

    def apply(self, x: np.ndarray, **params):
        """
        Stretch a 1D or 2D array in time using linear interpolation.
        - x: np.ndarray of shape (L,) or (L, N)
        """
        rate = np.random.uniform(self.min_rate, self.max_rate)
        L = x.shape[0]
        L_new = int(L / rate)

        # Avoid empty sequence
        L_new = max(1, L_new)

        orig_idx = np.linspace(0, L - 1, num=L)
        new_idx = np.linspace(0, L - 1, num=L_new)

        if x.ndim == 1:
            stretched = np.interp(new_idx, orig_idx, x)
        elif x.ndim == 2:
            stretched = np.stack(
                [np.interp(new_idx, orig_idx, x[:, i]) for i in range(x.shape[1])],
                axis=1,
            )
        else:
            raise ValueError("Only 1D or 2D arrays are supported.")

        augmented = stretched
        if augmented.shape[0] > L:
            augmented = augmented[:L]
        elif augmented.shape[0] < L:
            # Pad with zeros
            pad_width = L - augmented.shape[0]
            if x.ndim == 1:
                augmented = np.pad(
                    augmented, (0, pad_width), mode="constant", constant_values=0
                )
            else:
                augmented = np.pad(
                    augmented,
                    ((0, pad_width), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

        return augmented.astype(x.dtype)


class TimeShift(SignalTransform):
    def __init__(
        self, always_apply=False, p=0.5, max_shift_pct=0.25, padding_mode="replace"
    ):
        super().__init__(always_apply, p)

        assert 0 <= max_shift_pct <= 1.0, "`max_shift_pct` must be between 0 and 1"
        assert padding_mode in ["replace", "zero"], (
            "`padding_mode` must be either 'replace' or 'zero'"
        )

        self.max_shift_pct = max_shift_pct
        self.padding_mode = padding_mode

    def apply(self, x: np.ndarray, **params):
        assert x.ndim == 2, "`x` must be a 2D array with shape (L, N)"

        L = x.shape[0]
        max_shift = int(L * self.max_shift_pct)
        shift = np.random.randint(-max_shift, max_shift + 1)

        # Roll along time axis (axis=0)
        augmented = np.roll(x, shift, axis=0)

        if self.padding_mode == "zero":
            if shift > 0:
                augmented[:shift, :] = 0
            elif shift < 0:
                augmented[shift:, :] = 0

        return augmented


class ButterFilter(SignalTransform):
    def __init__(
        self, always_apply=False, p=0.5, cutoff_freq=20, sampling_rate=200, order=4
    ):
        super().__init__(always_apply, p)

        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order

    def apply(self, x: np.ndarray, **params):
        assert x.ndim == 2, "`x` must be a 2D array with shape (L, N)"
        return self.butter_lowpass_filter(x)

    def butter_lowpass_filter(self, data: np.ndarray):
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff_freq / nyquist
        # Avoid valid range errors
        normal_cutoff = min(normal_cutoff, 0.99)

        b, a = butter(self.order, normal_cutoff, btype="low", analog=False)
        filtered_data = lfilter(b, a, data, axis=0)  # filter each channel independently
        return filtered_data.astype(data.dtype)
