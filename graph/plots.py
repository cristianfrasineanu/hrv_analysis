from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd

import utils.constants as c


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
