#!/usr/bin/env python3
"""
batch_rr.py  Bulk-process RR interval data to calculate
heart rate variability (HRV) metrics.

This script processes RR interval data files in EliteHRV .txt format to
calculate various HRV metrics:
- Time domain: SDNN, RMSSD, pNN50, mean HR/RR
- Frequency domain: LF, HF, Total Power, LF/HF ratio
- Non-linear: Sample Entropy, Detrended Fluctuation Analysis

You can get these files from the EliteHRV app via the "Export Data" feature in Settings.

Usage
-----
$ python batch_rr.py /path/to/folder/with/txt [--excel] [--window WINDOW]

Arguments:
    folder                  Path to folder containing EliteHRV .txt files
    --excel                 Also write .xlsx with rolling stats and visualizations
    --window WINDOW         Window size in days for rolling statistics (default: 14)
"""

import argparse
import glob
import os
import sys
import warnings
from io import BytesIO
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import scipy.signal as sg
from antropy import sample_entropy
from nolds import dfa

import constants as c
from hrv_types import (
    FreqDomainMetrics,
    HRVMetrics,
    NonlinearMetrics,
    OptionalHRVMetrics,
    RRIntervals,
    TimeDomainMetrics,
)


# ---------- helper functions ----------


def read_rr_intervals(fp: str) -> RRIntervals:
    """Read and validate RR intervals from an EliteHRV .txt file.

    Parameters
    ----------
    fp : str
        Path to the EliteHRV .txt file containing RR intervals

    Returns
    -------
    RRIntervals
        Array of valid RR intervals in milliseconds

    Raises
    ------
    ValueError
        If more than 3% of intervals are artifacts

    Notes
    -----
    Validation criteria:
    - RR intervals must be between 300ms and 2000ms
    - Consecutive intervals cannot differ by more than 300ms
    - Warnings are logged for each artifact found
    """
    with open(fp, "r") as f:
        data = f.read().strip().replace("\n", ",")
    rr = np.array([float(x) for x in data.split(",") if x])

    # Validate RR intervals using range and jump filters
    range_mask = (rr >= c.MIN_RR_MS) & (rr <= c.MAX_RR_MS)
    # Prepending the first RR interval to the array to avoid off by one errors.
    jump_mask = np.abs(np.diff(rr, prepend=rr[0])) <= c.MAX_JUMP_MS
    valid_mask = range_mask & jump_mask

    artifacts = ~valid_mask
    artifact_count = np.sum(artifacts)
    for i, is_artifact in enumerate(artifacts):
        if is_artifact:
            warnings.warn(f"Artifact found in {fp}: RR={rr[i]:.1f}ms at index {i}")

    artifact_percent = (artifact_count / len(rr)) * 100
    if artifact_percent > c.MAX_ARTIFACT_PERCENT:
        warnings.warn(
            f"Too many artifacts ({artifact_percent:.1f}%) in {fp}. Will skip the reading."
        )
        return []

    return rr[valid_mask]


def time_domain(rr: RRIntervals) -> TimeDomainMetrics:
    """Calculate time domain HRV metrics from RR intervals.

    Parameters
    ----------
    rr : RRIntervals
        Array of RR intervals in milliseconds

    Returns
    -------
    TimeDomainMetrics
        Tuple of (mean_rr, mean_hr, sdnn, rmssd, pnn50, lnrmssd)
    """
    diff_rr = np.diff(rr)
    sdnn = np.std(rr, ddof=1)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    pnn50 = 100 * np.mean(np.abs(diff_rr) > 50)
    mean_rr = np.mean(rr)
    mean_hr = 60000 / mean_rr
    return mean_rr, mean_hr, sdnn, rmssd, pnn50, np.log(rmssd)


def welch_psd(rr: RRIntervals, fs: float = 4.0) -> FreqDomainMetrics:
    """
    Compute frequency-domain HRV metrics from unevenly spaced RR intervals using Welch's method.

    This function first resamples the input RR interval series (in milliseconds) to an evenly spaced
    time grid at the specified sampling frequency (fs, in Hz). It then detrends the resampled signal
    and estimates its power spectral density (PSD) using Welch's method. The function calculates the
    absolute power in the standard HRV frequency bands:
        - LF (Low Frequency, 0.04–0.15 Hz)
        - HF (High Frequency, 0.15–0.40 Hz)
        - Total Power (0.0033–0.40 Hz)
    and the LF/HF ratio.

    Parameters
    ----------
    rr : RRIntervals
        Array of RR intervals in milliseconds.
    fs : float, optional
        Target sampling frequency for resampling (Hz). Default is 4.0 Hz.

    Returns
    -------
    FreqDomainMetrics
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


def nonlinear(rr: RRIntervals) -> NonlinearMetrics:
    """Calculate non-linear HRV metrics.

    This function computes two non-linear HRV metrics:
    1. Sample Entropy (SampEn): A measure of time series complexity that quantifies
       the unpredictability of RR interval fluctuations. Higher values indicate
       more complex and less predictable patterns.
    2. Detrended Fluctuation Analysis (DFA) alpha1: A measure of long-range
       correlations in the RR intervals. Values around 1.0 indicate healthy
       fractal-like behavior, while values significantly different from 1.0
       may indicate pathological conditions.

    Parameters
    ----------
    rr : RRIntervals
        Array of RR intervals in milliseconds.

    Returns
    -------
    NonlinearMetrics
        Tuple of (sampen, alpha1)
    """
    try:
        sampen = sample_entropy(rr, order=2, tolerance=0.2 * np.std(rr))
    except Exception:
        sampen = np.nan
    try:
        alpha1 = dfa(rr, nvals=np.arange(4, 17))
    except Exception:
        alpha1 = np.nan
    return sampen, alpha1


def compute_rolling_stats(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute rolling means and z-scores for key HRV metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with HRV metrics from main()
    window : int, default=14
        Window size for rolling mean in days

    Returns
    -------
    pd.DataFrame
        DataFrame containing only rolling means and z-scores
    """
    key_metrics = [
        col
        for col in [
            c.COL_RMSSD_MS,
            c.COL_SDNN_MS,
            c.COL_MEAN_HR_BPM,
            c.COL_LF_HF,
            c.COL_SAMPEN,
            c.COL_DFA_ALPHA1,
        ]
        if col in df.columns
    ]

    rolling_mean = df[key_metrics].rolling(window=window, min_periods=7).mean().shift(1)
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


def create_visualizations(df: pd.DataFrame, rolling_df: pd.DataFrame) -> BytesIO:
    """Create focused time series plots for key HRV metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw metrics
    rolling_df : pd.DataFrame
        DataFrame with rolling statistics

    Returns
    -------
    BytesIO
        In-memory buffer containing the plots
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("HRV Readiness Metrics", fontsize=16)

    # Plot 1: Dual-axis RMSSD-z & HR-z
    ax1_twin = ax1.twinx()

    # RMSSD z-score
    ax1.plot(
        rolling_df.index,
        rolling_df[f"{c.COL_RMSSD_MS}_zscore"],
        "b-",
        label="RMSSD z-score",
    )
    # Add RMSSD thresholds
    ax1.axhline(y=-1.5, color="b", linestyle="--", alpha=0.5, label="RMSSD warning")
    ax1.axhline(y=-2, color="b", linestyle=":", alpha=0.5, label="RMSSD alert")
    ax1.set_ylabel("RMSSD z-score", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # HR z-score
    ax1_twin.plot(
        rolling_df.index,
        rolling_df[f"{c.COL_MEAN_HR_BPM}_zscore"],
        "r-",
        label="HR z-score",
    )
    # Add HR thresholds
    ax1_twin.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="HR warning")
    ax1_twin.axhline(y=1.5, color="r", linestyle=":", alpha=0.5, label="HR alert")
    ax1_twin.set_ylabel("HR z-score", color="r")
    ax1_twin.tick_params(axis="y", labelcolor="r")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("RMSSD & HR Z-scores")
    ax1.grid(True, alpha=0.3)

    # Plot 2: DFA α1
    ax2.plot(df.index, df[c.COL_DFA_ALPHA1], "b-", label="DFA α1")
    ax2.axhline(
        y=0.75, color="r", linestyle="--", alpha=0.5, label="Warning threshold (0.75)"
    )
    ax2.set_title("DFA α1")
    ax2.set_ylabel("DFA α1")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Rotate x-axis labels
    for ax in [ax1, ax2]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def process_rr_file(path: str) -> OptionalHRVMetrics:
    """Process a single RR interval file and return its metrics.

    Parameters
    ----------
    path : str
        Path to the RR interval file

    Returns
    -------
    OptionalHRVMetrics
        Dictionary containing the date and calculated metrics, or None if processing fails
    """
    date_match = c.FILENAME_RE.search(os.path.basename(path))
    if not date_match:
        warnings.warn(f"Date not found in filename: {path}")
        return None

    # Parse only the date part and normalize to midnight
    date = pd.to_datetime(date_match.group(1)).normalize()

    rr = read_rr_intervals(path)
    if len(rr) < c.MIN_RR_INTERVALS:
        warnings.warn(f"{path}: too few RR intervals ({len(rr)}), skipped.")
        return None

    mean_rr, mean_hr, sdnn, rmssd, pnn50, lnrmssd = time_domain(rr)
    lf, hf, tp, lf_hf = welch_psd(rr)
    sampen, alpha1 = nonlinear(rr)

    return {
        "Date": date,
        c.COL_MEAN_RR_MS: mean_rr,
        c.COL_MEAN_HR_BPM: mean_hr,
        c.COL_SDNN_MS: sdnn,
        c.COL_RMSSD_MS: rmssd,
        c.COL_LN_RMSSD: lnrmssd,
        c.COL_PNN50_PCT: pnn50,
        c.COL_LF_POWER: lf,
        c.COL_HF_POWER: hf,
        c.COL_TOTAL_POWER: tp,
        c.COL_LF_HF: lf_hf,
        c.COL_SAMPEN: sampen,
        c.COL_DFA_ALPHA1: alpha1,
    }


def save_metrics(
    df: pd.DataFrame, directory: str, to_excel: bool = False, window: int = 14
) -> None:
    """Save the metrics DataFrame to CSV and optionally to Excel.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metrics
    directory : str
        Directory to save the output files
    to_excel : bool, default=False
        Whether to also write Excel output
    window : int, default=14
        Window size in days for rolling statistics
    """
    out_csv = os.path.join(directory, c.OUTPUT_CSV)
    df.to_csv(out_csv, float_format="%.4f")
    print(f"Wrote {out_csv} ({len(df)} days).")

    if to_excel:
        out_xlsx = out_csv.replace(".csv", ".xlsx")
        print("\nDataFrame Head:")
        print(df.head())

        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            # Write base metrics to first sheet
            df.to_excel(writer, sheet_name="Raw Metrics", float_format="%.4f")

            # Add rolling statistics to second sheet
            rolling_df = compute_rolling_stats(df, window=window)
            rolling_df.to_excel(writer, sheet_name="Rolling Stats", float_format="%.4f")

            # Add visualizations to third sheet
            img_data = create_visualizations(df, rolling_df)
            worksheet = writer.book.create_sheet("z-score trend")
            img = openpyxl.drawing.image.Image(img_data)
            worksheet.add_image(img, "A1")

        print(f"…and {out_xlsx} with three sheets")


def main(directory: str, to_excel: bool = False, window: int = 14) -> None:
    """Process all RR interval files in directory and save metrics.

    Parameters
    ----------
    directory : str
        Path to directory containing RR interval files
    to_excel : bool, default=False
        Whether to also write Excel output
    window : int, default=14
        Window size in days for rolling statistics
    """
    files = sorted(glob.glob(os.path.join(directory, "*.txt")))
    if not files:
        sys.exit("No HRV input files found in %s" % directory)

    rows: List[HRVMetrics] = []
    last_date = None

    for path in files:
        date_match = c.FILENAME_RE.search(os.path.basename(path))
        if not date_match:
            warnings.warn(f"Date not found in filename: {path}")
            continue

        current_date = pd.to_datetime(date_match.group(1)).normalize()

        # Check for gaps between readings
        if last_date is not None:
            days_gap = (current_date - last_date).days
            if days_gap > c.MAX_DAYS_GAP:
                warnings.warn(
                    f"Large gap detected: {days_gap} days between {last_date.date()} "
                    f"and {current_date.date()}. Skipping previous reading from {last_date.date()}"
                )
                # Remove the last metrics since there was a gap
                if rows:
                    rows.pop()
                last_date = None

        metrics = process_rr_file(path)
        if metrics:
            rows.append(metrics)
            last_date = current_date

    if not rows:
        sys.exit("No valid HRV files found after processing")

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    save_metrics(df, directory, to_excel, window)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder with EliteHRV .txt files")
    parser.add_argument(
        "--excel",
        action="store_true",
        help="also write .xlsx, including rolling stats and visualizations",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=14,
        help="window size in days for rolling statistics (default: 14)",
    )
    args = parser.parse_args()
    main(args.folder, to_excel=args.excel, window=args.window)
