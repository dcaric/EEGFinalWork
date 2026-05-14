import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_FILE = os.path.join(PROJECT_ROOT, 'pilot_files/all_task_ready.csv')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'task_vs_channels_median.csv')
OUTPUT_PNG = os.path.join(PROJECT_ROOT, 'visuals/task_vs_channels_median.png')

# =========================
# Load data
# =========================
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=['Task']).copy()
df['Task'] = df['Task'].astype(str).str.strip()

# =========================
# Define channel groups (bands aggregated)
# =========================
channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

channel_map = {}
for ch in channels:
    channel_map[ch] = [c for c in df.columns if c.startswith(f'POW.{ch}.')]

subjects = sorted(df['Subject_ID'].dropna().unique())
all_tasks = sorted(df['Task'].unique())

# =========================
# Build per-subject task x channel matrices
# Value = mean power across all bands for that channel during task
# =========================
all_matrices = []

for subj in subjects:
    sub_df = df[df['Subject_ID'] == subj].copy()

    subj_matrix = pd.DataFrame(
        data=0.0,
        index=all_tasks,
        columns=channels
    )

    for task in all_tasks:
        task_df = sub_df[sub_df['Task'] == task].copy()

        if len(task_df) > 0:
            for ch, cols in channel_map.items():
                valid_cols = [c for c in cols if c in task_df.columns]
                if valid_cols:
                    subj_matrix.loc[task, ch] = task_df[valid_cols].apply(pd.to_numeric, errors='coerce').mean().mean()
                else:
                    subj_matrix.loc[task, ch] = 0.0
        else:
            subj_matrix.loc[task, :] = 0.0

    subj_matrix = subj_matrix.fillna(0.0)
    all_matrices.append(subj_matrix)

# =========================
# Stack and compute median across subjects
# =========================
stacked = np.stack([m.values.astype(float) for m in all_matrices], axis=0)
median_matrix = np.median(stacked, axis=0)

median_df = pd.DataFrame(
    median_matrix,
    index=all_tasks,
    columns=channels
)

# Save raw median
median_df.to_csv(OUTPUT_CSV, index=True)

# =========================
# Z-score normalize within each channel
# =========================
zscore_df = median_df.copy()
for ch in channels:
    col = zscore_df[ch]
    zscore_df[ch] = (col - col.mean()) / col.std()

# =========================
# Plot z-score normalized heatmap
# =========================
plt.figure(figsize=(10, max(8, len(all_tasks) * 0.4)))
sns.heatmap(
    zscore_df,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    linecolor='white',
    annot_kws={'size': 6}
)
plt.title('Median Channel Activation per Task (Z-Score Normalized Within Each Channel)')
plt.xlabel('Brain Channels')
plt.ylabel('Tasks')
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved raw median matrix to: {OUTPUT_CSV}")
print(f"Saved z-score heatmap to: {OUTPUT_PNG}")