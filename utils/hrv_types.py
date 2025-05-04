"""Type aliases for HRV analysis.

This module contains type aliases to make the code more readable and maintainable.
"""

from typing import Dict, TypeAlias, Union

import numpy as np
import pandas as pd

# Basic types
RRIntervals: TypeAlias = np.ndarray

# Metric types
# mean_rr, mean_hr, sdnn, rmssd, pnn50, lnrmssd
TimeDomainMetrics: TypeAlias = tuple[float, float, float, float, float, float]
FreqDomainMetrics: TypeAlias = tuple[float, float, float, float]  # lf, hf, tp, lf_hf
NonlinearMetrics: TypeAlias = tuple[float, float]  # sampen, alpha1

# Dictionary types
HRVMetrics: TypeAlias = Dict[str, Union[float, pd.Timestamp]]
OptionalHRVMetrics: TypeAlias = Union[HRVMetrics, None]
