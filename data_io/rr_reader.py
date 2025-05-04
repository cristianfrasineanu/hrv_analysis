import warnings
from typing import List

import numpy as np

import utils.constants as c


def read_rr_intervals(fp: str) -> List[float]:
    """Read and validate RR intervals from an EliteHRV .txt file.

    Parameters
    ----------
    fp : str
        Path to the EliteHRV .txt file containing RR intervals

    Returns
    -------
    List[float]
        Array of valid RR intervals in milliseconds

    Raises
    ------
    ValueError
        If more than MAX_ARTIFACT_PERCENT of intervals are artifacts

    Notes
    -----
    Validation criteria:
    - RR intervals must be between MIN_RR_MS and MAX_RR_MS
    - Consecutive intervals cannot differ by more than MAX_JUMP_MS
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

    return rr[valid_mask].tolist()
