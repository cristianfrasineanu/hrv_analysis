#!/usr/bin/env python
"""
batch_rr.py  Bulk-process RR interval data to calculate
heart rate variability (HRV) metrics.

This script processes RR interval data files to calculate various HRV metrics that
help you assess your autonomic nervous system balance:
- Time domain: SDNN, RMSSD, pNN50, mean HR/RR
- Frequency domain: LF, HF, Total Power, LF/HF ratio
- Non-linear: Sample Entropy, Detrended Fluctuation Analysis

Supported file formats:
- Plain text files (.txt): Each line or comma-separated value is an RR interval in milliseconds
- CSV files (.csv): Must contain a column named 'RR' (case-insensitive) with RR intervals in milliseconds

You can get .txt files from the EliteHRV app via the "Export Data" feature in Settings.
CSV files are supported from apps like ECGApp and other HRV monitoring tools.

Usage
-----
$ python batch_rr.py /path/to/folder/with/files [--excel] [--window WINDOW]

Arguments:
    folder                  Path to folder containing RR interval files (.txt or .csv)
    --excel                 Also write .xlsx with rolling stats and visualizations
    --window WINDOW         Window size in days for rolling statistics
"""

import argparse
import glob
import os
import sys
import warnings

import pandas as pd

import utils.constants as c
from analysis import (
    nonlinear,
    time_domain,
    welch_psd,
)
from data_io import (
    read_rr_intervals,
    save_metrics,
)
from utils.hrv_types import HRVMetrics, OptionalHRVMetrics


def _find_hrv_files(directory: str) -> list[str]:
    """Find all supported HRV files in the given directory.

    Parameters
    ----------
    directory : str
        Directory to search for files

    Returns
    -------
    list[str]
        List of file paths sorted by timestamp (earliest first)
    """
    files = []
    for pattern in c.SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, pattern)))

    # Sort by extracted timestamp instead of filename
    return sorted(files, key=_extract_date)


def _extract_date(path: str) -> pd.Timestamp:
    """Extract date from file path, using filename pattern or file modification time as fallback.

    Parameters
    ----------
    path : str
        Path to the file

    Returns
    -------
    pd.Timestamp
        Normalized date (midnight) for the file
    """
    date_match = c.FILENAME_RE.search(os.path.basename(path))
    if date_match:
        return pd.to_datetime(date_match.group(1)).normalize()
    else:
        mtime = os.path.getmtime(path)
        date = pd.to_datetime(mtime, unit="s").normalize()
        warnings.warn(c.DATE_NOT_FOUND_MSG.format(path, date.date()))
        return date


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
    rr = read_rr_intervals(path)
    if len(rr) < c.MIN_RR_INTERVALS:
        warnings.warn(c.TOO_FEW_RR_INTERVALS_MSG.format(path, len(rr)))
        return None

    mean_rr, mean_hr, sdnn, rmssd, pnn50, lnrmssd = time_domain(rr)
    lf, hf, tp, lf_hf = welch_psd(rr)
    sampen, alpha1 = nonlinear(rr)

    return {
        "Date": _extract_date(path),
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


def main(directory: str, to_excel: bool = False, window: int = c.DEFAULT_WINDOW_DAYS) -> None:
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
    files = _find_hrv_files(directory)
    if not files:
        sys.exit("No HRV input files found in %s" % directory)

    rows: list[HRVMetrics] = []
    last_date = None

    for path in files:
        current_date = _extract_date(path)

        # Check for gaps between readings
        if last_date is not None:
            days_gap = (current_date - last_date).days
            if days_gap > c.MAX_DAYS_GAP:
                warnings.warn(
                    c.LARGE_GAP_DETECTED_MSG.format(
                        days_gap, last_date.date(), current_date.date(), last_date.date()
                    )
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
    parser.add_argument("folder", help="folder with RR interval files (.txt or .csv)")
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
