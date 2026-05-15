"""
Create a balanced synthetic version of pilot_files/all_task_ready.csv.

Input:
  pilot_files/all_task_ready.csv

Output:
  pilot_files/all_task_ready_synthetic.csv
  pilot_files/all_task_ready_synthetic_summary.txt

Balancing rule:
  - group A: Sleep_Hours < 7
  - group B: Sleep_Hours >= 7
  - generate synthetic rows only for group A until row counts match group B

Synthetic generation:
  - numeric EEG/PM features sampled uniformly within observed min/max of the <7h group
  - Task sampled from observed <7h rows
  - Gender sampled from observed <7h rows
  - Sleep_Hours sampled uniformly from [5.0, 6.9]

Important:
  This file is for exploratory balancing experiments only.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).parent.parent.parent
INPUT_FILE = BASE_DIR / "pilot_files" / "all_task_ready.csv"
OUTPUT_FILE = BASE_DIR / "pilot_files" / "all_task_ready_synthetic.csv"
SUMMARY_FILE = BASE_DIR / "pilot_files" / "all_task_ready_synthetic_summary.txt"

RANDOM_STATE = 42
LOW_MIN = 5.0
LOW_MAX = 6.9


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    df = pd.read_csv(INPUT_FILE)
    low_df = df[df["Sleep_Hours"] < 7].copy()
    high_df = df[df["Sleep_Hours"] >= 7].copy()

    if low_df.empty or high_df.empty:
        raise ValueError("Both <7h and >=7h groups must be present.")

    n_to_generate = len(high_df) - len(low_df)
    if n_to_generate <= 0:
        raise ValueError("The <7h group is already at least as large as the >=7h group.")

    pow_cols = [c for c in df.columns if c.startswith("POW.")]
    pm_cols = [c for c in df.columns if c.startswith("PM.")]
    numeric_feature_cols = pow_cols + pm_cols

    low_min = low_df[numeric_feature_cols].min()
    low_max = low_df[numeric_feature_cols].max()

    task_values = low_df["Task"].dropna().astype(str).to_numpy()
    gender_values = low_df["Gender"].dropna().astype(str).to_numpy()
    source_subjects = low_df["Subject_ID"].dropna().astype(str).unique().tolist()

    synthetic_rows = []
    for i in range(n_to_generate):
        row = {}
        for col in numeric_feature_cols:
            mn = float(low_min[col])
            mx = float(low_max[col])
            if np.isfinite(mn) and np.isfinite(mx):
                row[col] = mn if mn == mx else float(rng.uniform(mn, mx))
            else:
                row[col] = np.nan

        row["Subject_ID"] = f"SYNTH_LOW_{i+1:04d}_FROM_{rng.choice(source_subjects)}"
        row["Task"] = str(rng.choice(task_values))
        row["Sleep_Hours"] = float(rng.uniform(LOW_MIN, LOW_MAX))
        row["Gender"] = str(rng.choice(gender_values))
        synthetic_rows.append(row)

    synth_df = pd.DataFrame(synthetic_rows, columns=df.columns)
    final_df = pd.concat([df, synth_df], ignore_index=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    summary = (
        f"Input file: {INPUT_FILE}\n"
        f"Original rows: {len(df)}\n"
        f"Original <7h rows: {len(low_df)}\n"
        f"Original >=7h rows: {len(high_df)}\n"
        f"Synthetic <7h rows generated: {len(synth_df)}\n"
        f"Final rows: {len(final_df)}\n\n"
        "Method:\n"
        "- Numeric EEG/PM features sampled within observed <7h min/max ranges\n"
        "- Task sampled from observed <7h rows\n"
        "- Gender sampled from observed <7h rows\n"
        "- Sleep_Hours sampled uniformly from 5.0 to 6.9\n\n"
        "Final class counts:\n"
        f"<7h rows: {(final_df['Sleep_Hours'] < 7).sum()}\n"
        f">=7h rows: {(final_df['Sleep_Hours'] >= 7).sum()}\n"
    )
    SUMMARY_FILE.write_text(summary)

    print(summary)
    print(f"Saved synthetic-balanced dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
