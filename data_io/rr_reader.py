import warnings
# TODO: add logging module
import csv
import os

import numpy as np

import utils.constants as c


def read_rr_intervals(fp: str) -> np.ndarray:
    """Read and validate RR intervals from a file.

    Supported formats:
    - Plain text files (.txt): Each line or comma-separated value is an RR interval in milliseconds
    - CSV files (.csv): Must contain a column named 'RR' (case-insensitive) with RR intervals in milliseconds

    Parameters
    ----------
    fp : str
        Path to the file containing RR intervals

    Returns
    -------
    np.ndarray
        Array of valid RR intervals in milliseconds after artifact removal

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If path is not a file, if CSV file doesn't contain an 'RR' column,
        if no valid numeric data is found, if more than MAX_ARTIFACT_PERCENT of intervals are artifacts,
        or if the file extension is not supported

    Notes
    -----
    Validation criteria:
    - RR intervals must be between MIN_RR_MS and MAX_RR_MS
    - Consecutive intervals cannot differ by more than MAX_JUMP_MS
    - Warnings are logged for each artifact found
    - If more than MAX_ARTIFACT_PERCENT of intervals are artifacts, raises ValueError
    """
    if not os.path.exists(fp):
        raise FileNotFoundError(c.FILE_NOT_FOUND_MSG.format(fp))

    if not os.path.isfile(fp):
        raise ValueError(c.NOT_A_FILE_MSG.format(fp))

    ext = os.path.splitext(fp)[1].lower()
    if ext == ".csv":
        rr = _read_rr_csv(fp)
    elif ext == ".txt":
        rr = _read_rr_plain(fp)
    else:
        raise ValueError(f"Unsupported file extension: {ext} in {fp}")

    if rr.size == 0:
        raise ValueError(c.NO_NUMERIC_DATA_MSG.format(fp))

    return _validate_rr(rr, fp)


def _read_rr_plain(fp: str) -> np.ndarray:
    """Read RR intervals from plain text file.

    Supports newline or comma-separated values.
    This is the default format for EliteHRV exports.

    Parameters
    ----------
    fp : str
        Path to the plain text file

    Returns
    -------
    np.ndarray
        Array of RR intervals in milliseconds

    Raises
    ------
    ValueError
        If no valid numeric data is found in the file
    """
    with open(fp) as f:
        data = f.read().replace("\n", ",")

    rr_values = []
    for raw_rr in data.split(","):
        raw_rr = raw_rr.strip()
        if not raw_rr:
            continue
        try:
            rr_values.append(float(raw_rr))
        except ValueError:
            continue

    if not rr_values:
        raise ValueError(c.NO_NUMERIC_DATA_MSG.format(fp))

    return np.asarray(rr_values, dtype=float)


def _read_rr_csv(fp: str) -> np.ndarray:
    """Read RR intervals from CSV file.

    Extracts values from the 'RR' column (case-insensitive).
    Some apps like ECGApp use 'RR' as the column name to export the HR data.
    May contain other PQRS ECG data but we only need the RR intervals.

    Parameters
    ----------
    fp : str
        Path to the CSV file

    Returns
    -------
    np.ndarray
        Array of RR intervals in milliseconds

    Raises
    ------
    ValueError
        If 'RR' column is not found in the CSV file or if no valid numeric data is found
    """
    rr_values: list[float] = []
    with open(fp, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader, [])]
        try:
            rr_idx = header.index(c.RR_COLUMN_NAME)
        except ValueError as exc:
            raise ValueError(c.RR_COLUMN_NOT_FOUND_MSG.format(fp)) from exc

        for row in reader:
            if len(row) <= rr_idx:
                continue
            val = row[rr_idx].strip()
            if not val:
                continue

            try:
                rr_values.append(float(val))
            except ValueError:
                continue

    if not rr_values:
        raise ValueError(c.NO_NUMERIC_DATA_MSG.format(fp))

    return np.asarray(rr_values, dtype=float)


# TODO: Maybe let the caller decide on the failure mode.
def _validate_rr(rr: np.ndarray, fp: str) -> np.ndarray:
    """Validate RR intervals and remove artifacts/ectopic beats.

    Parameters
    ----------
    rr : np.ndarray
        Array of RR intervals in milliseconds
    fp : str
        File path for error reporting

    Returns
    -------
    np.ndarray
        Array of valid RR intervals after artifact removal

    Notes
    -----
    Validation criteria:
    - RR intervals must be between MIN_RR_MS and MAX_RR_MS
    - Consecutive intervals cannot differ by more than MAX_JUMP_MS
    - If more than MAX_ARTIFACT_PERCENT are artifacts, returns empty array
    """
    if rr.size == 0:
        return rr

    range_mask = (rr >= c.MIN_RR_MS) & (rr <= c.MAX_RR_MS)
    # Prepending the first RR interval to the array to avoid off by one errors.
    jump_mask = np.abs(np.diff(rr, prepend=rr[0])) <= c.MAX_JUMP_MS
    valid_mask = range_mask & jump_mask

    artifacts = ~valid_mask
    for i in np.where(artifacts)[0]:
        warnings.warn(c.ARTIFACT_FOUND_MSG.format(fp, rr[i], i))

    artifact_percent = artifacts.mean() * 100
    if artifact_percent > c.MAX_ARTIFACT_PERCENT:
        raise ValueError(c.TOO_MANY_ARTIFACTS_MSG.format(artifact_percent, fp))

    return rr[valid_mask]
