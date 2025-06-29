import numpy as np
import scipy.signal as sg

import utils.constants as c


def welch_psd(rr: np.ndarray, fs: float = 4.0) -> tuple[float, float, float, float]:
    """Compute frequency-domain HRV metrics from unevenly spaced RR intervals using Welch's method.

    Parameters
    ----------
    rr : np.ndarray
        Array of RR intervals in milliseconds.
    fs : float, optional
        Target sampling frequency for resampling (Hz). Default is 4.0 Hz.

    Returns
    -------
    tuple[float, float, float, float]
        Tuple of (lf, hf, tp, lf_hf)
    """
    try:
        if len(rr) < c.MIN_RR_INTERVALS:
            return np.nan, np.nan, np.nan, np.nan

        if np.any(np.isnan(rr)) or np.any(np.isinf(rr)):
            return np.nan, np.nan, np.nan, np.nan

        # Evenly space the time grid with the target sampling frequency.
        t = np.cumsum(rr) / 1000.0
        even_t = np.arange(0, t[-1], 1 / fs)

        # Require at least 256 points for reliable spectral analysis (power of 2 for FFT efficiency)
        if len(even_t) < 256:
            return np.nan, np.nan, np.nan, np.nan

        even_rr = np.interp(even_t, t, rr)
        even_rr_detrended = even_rr - np.mean(even_rr)
        # Use 256-point segments for optimal FFT efficiency
        segment_length = min(256, len(even_rr_detrended) // 2)
        if segment_length < 64:
            return np.nan, np.nan, np.nan, np.nan

        f, pxx = sg.welch(even_rr_detrended, fs=fs, nperseg=segment_length)

        lf_band = np.logical_and(f >= c.LF_LOW, f < c.LF_HIGH)
        hf_band = np.logical_and(f >= c.HF_LOW, f < c.HF_HIGH)
        freq_mask = (f >= c.LOW_THRESHOLD) & (f < c.HIGH_THRESHOLD)

        # Integrate power in each band
        lf = np.trapezoid(pxx[lf_band], f[lf_band])
        hf = np.trapezoid(pxx[hf_band], f[hf_band])
        tp = np.trapezoid(pxx[freq_mask], f[freq_mask])

        lf_hf = lf / hf if hf > 0 else np.nan

        return lf, hf, tp, lf_hf

    except Exception:
        return np.nan, np.nan, np.nan, np.nan
