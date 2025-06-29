import numpy as np
from antropy import sample_entropy
from nolds import dfa


def nonlinear(rr: np.ndarray) -> tuple[float, float]:
    """Calculate non-linear HRV metrics.

    Parameters
    ----------
    rr : np.ndarray
        Array of RR intervals in milliseconds.

    Returns
    -------
    tuple[float, float]
        Tuple of (sampen, alpha1)
    """
    try:
        sampen = sample_entropy(rr, order=2, tolerance=0.2 * np.std(rr))
    except Exception:
        sampen = np.nan
    try:
        result = dfa(rr, nvals=np.arange(4, 17), debug_data=False)
        alpha1 = float(result[0] if isinstance(result, tuple) else result)
    except Exception:
        alpha1 = np.nan
    return sampen, alpha1
