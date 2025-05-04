from typing import Tuple

import numpy as np
import scipy.signal as sg

import utils.constants as c


def welch_psd(rr: np.ndarray, fs: float = 4.0) -> Tuple[float, float, float, float]:
    """Compute frequency-domain HRV metrics from unevenly spaced RR intervals using Welch's method.

    Parameters
    ----------
    rr : np.ndarray
        Array of RR intervals in milliseconds.
    fs : float, optional
        Target sampling frequency for resampling (Hz). Default is 4.0 Hz.

    Returns
    -------
    Tuple[float, float, float, float]
        Tuple of (lf, hf, tp, lf_hf)
    """
    t = np.cumsum(rr) / 1000.0  # Convert RR intervals to cumulative time in seconds
    even_t = np.arange(0, t[-1], 1 / fs)  # Evenly spaced time grid
    even_rr = np.interp(even_t, t, rr)
    even_rr = sg.detrend(even_rr)  # Remove linear trend (center the signal)
    f, pxx = sg.welch(even_rr, fs=fs, nperseg=256)

    lf_band = np.logical_and(f >= c.LF_LOW, f < c.LF_HIGH)
    hf_band = np.logical_and(f >= c.HF_LOW, f < c.HF_HIGH)
    freq_mask = (f >= c.LOW_THRESHOLD) & (f < c.HIGH_THRESHOLD)

    # Integrate power in each band
    lf = np.trapezoid(pxx[lf_band], f[lf_band])
    hf = np.trapezoid(pxx[hf_band], f[hf_band])
    tp = np.trapezoid(pxx[freq_mask], f[freq_mask])

    # Compute LF/HF ratio, handle division by zero
    lf_hf = lf / hf if hf else np.nan

    return lf, hf, tp, lf_hf
