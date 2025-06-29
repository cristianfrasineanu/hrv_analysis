import numpy as np
import pandas as pd

from utils.constants import (
    COL_DFA_ALPHA1,
    COL_LF_HF,
    COL_MEAN_HR_BPM,
    COL_RMSSD_MS,
    COL_SAMPEN,
    COL_SDNN_MS,
)


def time_domain(rr: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Calculate time domain HRV metrics from RR intervals.

    Parameters
    ----------
    rr : np.ndarray
        Array of RR intervals in milliseconds

    Returns
    -------
    tuple[float, float, float, float, float, float]
        Tuple of (mean_rr, mean_hr, sdnn, rmssd, pnn50, lnrmssd)
    """
    diff_rr = np.diff(rr)
    sdnn = np.std(rr, ddof=1)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    pnn50 = 100 * np.mean(np.abs(diff_rr) > 50)
    mean_rr = np.mean(rr)
    mean_hr = 60000 / mean_rr
    lnrmssd = np.log(rmssd)
    return (float(mean_rr), float(mean_hr), float(sdnn), float(rmssd), float(pnn50), float(lnrmssd))


def compute_rolling_stats(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling means and z-scores for key HRV metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with HRV metrics
    window : int
        Window size for rolling mean in days

    Returns
    -------
    pd.DataFrame
        DataFrame containing only rolling means and z-scores
    """
    key_metrics = [
        col
        for col in [
            COL_RMSSD_MS,
            COL_SDNN_MS,
            COL_MEAN_HR_BPM,
            COL_LF_HF,
            COL_SAMPEN,
            COL_DFA_ALPHA1,
        ]
        if col in df.columns
    ]

    # Shift by 1 to avoid look-ahead bias - use only past data for current day's calculations
    rolling_mean = pd.DataFrame(df[key_metrics].rolling(window=window, min_periods=7).mean()).shift(
        1
    )
    rolling_std = (
        df[key_metrics]
        .rolling(window=window, min_periods=7)
        .std()
        .shift(1)
        .replace(0, np.finfo(float).eps)
    )

    z_scores = (df[key_metrics] - rolling_mean) / rolling_std

    # Add the suffixes after the computation to avoid NaNs and mismatched columns.
    new_metrics = pd.concat(
        [
            rolling_mean.add_suffix("_rolling_mean"),
            rolling_std.add_suffix("_rolling_std"),
            z_scores.add_suffix("_zscore"),
        ],
        axis=1,
    )

    # Drop rows where we don't have enough data points for rolling calculations
    return new_metrics.dropna()
