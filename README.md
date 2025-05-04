# HRV Readiness Analysis

This project processes RR interval data (the difference between each R peak on an ECG given in ms) from EliteHRV exports (or any other app that allows you to export the chest strap data) to calculate heart rate variability (HRV) metrics and give you a more in-depth perspective of your training readiness. It's like having Kubios Premium, but without the premium price tag! 😉 The lite version doesn't allow you to batch process multiple files concurrently.

## Features

### Time Domain Metrics

- SDNN: Standard deviation of NN intervals (overall HRV)
- RMSSD: Root mean square of successive differences (parasympathetic activity). This is the one given by most apps
- pNN50: Percentage of NN intervals differing by >50ms
- Mean HR: Average heart rate in BPM that was captured during the recording

### Frequency Domain Metrics

- LF Power: Low frequency power (0.04-0.15 Hz standard band)
- HF Power: High frequency power (0.15-0.40 Hz standard band)
- LF/HF Ratio: Sympathovagal balance
- Total Power: Overall HRV power

### Non-linear Metrics

- Sample Entropy: Complexity of HRV
- DFA α1: Detrended Fluctuation Analysis (fractal scaling)
  > **Note:** For accurate DFA α1 calculations, recordings should be at least 5 minutes long. Shorter recordings may yield unreliable results.

### Readiness Indicators

The script calculates z-scores for key metrics to assess readiness:

#### RMSSD Z-scores

- Warning: Below -1.5 SD (potential stress/overreaching)
- Alert: Below -2.0 SD (high stress/overtraining risk)

#### Mean HR Z-scores

- Warning: Above +1.0 SD (elevated resting HR)
- Alert: Above +1.5 SD (significantly elevated resting HR)

#### DFA α1

- Warning threshold: Below 0.75 (reduced complexity)

## Usage

```bash
python batch_rr.py /path/to/elitehrv/export [--excel] [--window WINDOW]
```

> **Note:** The folder must contain all the EliteHRV .txt files that you want to process. Each file should be named in the format "YYYY-MM-DD HH-MM-SS.txt" (e.g., "2024-03-20 07-30-00.txt").

Options:

- `--excel`: Flag to generate Excel report with visualizations for the z-scores, annotated with warning thresholds.
- `--window`: Rolling window size in days (default: 14). Ideally the window should be as large as possible to even out false flags.

## Output

1. CSV file with all metrics
2. Excel report (optional) with:
   - Raw metrics
   - Rolling statistics and z-scores
   - Trend visualization

## Disclaimer

While this tool aims to provide similar insights to Kubios Premium, it is not a medical device or professional analysis tool. This is a tool for personal use and should not replace professional medical advice.

## Data Source

This script processes RR interval data exported from the EliteHRV app. You can export your data from EliteHRV's settings menu.
