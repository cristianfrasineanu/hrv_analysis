"""Constants used for HRV analysis.

This module contains all the constants used throughout the HRV analysis process,
including validation thresholds, frequency bands, and output configurations.
"""

import re

# Regular expression for parsing EliteHRV filenames
FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s+\d{2}-\d{2}-\d{2}")

# Minimum number of RR intervals required for analysis
MIN_RR_INTERVALS = 100

# Maximum allowed gap between readings (in days)
MAX_DAYS_GAP = 4  # Skip readings that are more than 4 days apart

# Frequency bands for spectral analysis (in Hz)
LF_LOW = 0.04
LF_HIGH = 0.15
HF_LOW = 0.15
HF_HIGH = 0.40
LOW_THRESHOLD = 0.0033
HIGH_THRESHOLD = 0.40

# RR interval validation thresholds
MIN_RR_MS = 300
MAX_RR_MS = 2000
MAX_JUMP_MS = 500  # Maximum allowed difference between consecutive RR intervals
MAX_ARTIFACT_PERCENT = 5  # Maximum allowed percentage of artifacts. This is the Kubios default.

# Output file names
OUTPUT_CSV = "all_readiness_metrics.csv"

# Column name constants
COL_RMSSD_MS = "RMSSD_ms"
COL_SDNN_MS = "SDNN_ms"
COL_MEAN_HR_BPM = "MeanHR_bpm"
COL_LF_HF = "LF_HF"
COL_SAMPEN = "SampEn"
COL_DFA_ALPHA1 = "DFA_alpha1"
COL_MEAN_RR_MS = "MeanRR_ms"
COL_LN_RMSSD = "lnRMSSD"
COL_PNN50_PCT = "pNN50_pct"
COL_LF_POWER = "LF_power"
COL_HF_POWER = "HF_power"
COL_TOTAL_POWER = "TotalPower"

# Rolling statistics
DEFAULT_WINDOW_DAYS = 14

# Supported file extensions
SUPPORTED_EXTENSIONS = ["*.txt", "*.csv"]

# String constants for RR reader error messages and column names
RR_COLUMN_NAME = "rr"
RR_COLUMN_NOT_FOUND_MSG = "'RR' column not found in {}"
TOO_MANY_ARTIFACTS_MSG = "Too many artifacts ({:.1f}%) in {}. Skipped."
ARTIFACT_FOUND_MSG = "Artifact found in {}: RR={:.1f}ms at index {}"
FILE_NOT_FOUND_MSG = "File not found: {}"
NOT_A_FILE_MSG = "Path is not a file: {}"
NO_NUMERIC_DATA_MSG = "No valid numeric RR data found in {}"

# String constants for error messages and column names used in batch_rr.py
DATE_NOT_FOUND_MSG = "Date not found in filename: {}. Using file modification date {} instead."
TOO_FEW_RR_INTERVALS_MSG = "{}: too few RR intervals ({}), skipped."
LARGE_GAP_DETECTED_MSG = "Large gap detected: {} days between {} and {}. Skipping previous reading from {}"

# TODO: Add Poincare and constants for the graph thresholds.
