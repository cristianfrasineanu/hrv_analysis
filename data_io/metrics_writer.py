import os

import openpyxl
import pandas as pd

import utils.constants as c
from analysis.time_domain import compute_rolling_stats
from graph.plots import create_visualizations


def save_metrics(
    df: pd.DataFrame,
    directory: str,
    window: int,
    to_excel: bool = False,
) -> None:
    """Save the metrics DataFrame to CSV and optionally to Excel.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metrics
    directory : str
        Directory to save the output files
    window : int,
        Window size in days for rolling statistics
    to_excel : bool, default=False
        Whether to also write Excel output
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
