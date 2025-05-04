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
from typing import List

import pandas as pd

import utils.constants as c

from . import (
    nonlinear,
    read_rr_intervals,
    save_metrics,
    time_domain,
    welch_psd,
)
from .utils.hrv_types import HRVMetrics, OptionalHRVMetrics


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


def main(
    directory: str, to_excel: bool = False, window: int = c.DEFAULT_WINDOW_DAYS
) -> None:
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
    save_metrics(df, directory, window, to_excel)


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
        default=c.DEFAULT_WINDOW_DAYS,
        help=f"window size in days for rolling statistics (default: {c.DEFAULT_WINDOW_DAYS})",
    )

    args = parser.parse_args()
    kwargs = {"to_excel": args.excel}
    if args.window != c.DEFAULT_WINDOW_DAYS:
        kwargs["window"] = args.window
    main(args.folder, **kwargs)
