"""
Per-Task FFT-Sleep Correlation Analysis.

For each of the 36 cognitive tasks, computes Pearson r between
subject-level mean FFT band power and Sleep_Hours.

Outputs
-------
1. visuals_per_task/per_task_fft_heatmap.png
   → 36 tasks × 5 bands heatmap (r averaged across 14 channels)
2. visuals_per_task/top_task_band_pairs.png
   → Top 10 strongest individual task-band-channel pairs
3. Console: ranked summary table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr
import os

# =========================
# Config
# =========================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_FILE = os.path.join(PROJECT_ROOT, 'pilot_files/all_task_ready.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'visuals_per_task')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BANDS    = ['Theta', 'Alpha', 'BetaL', 'BetaH', 'Gamma']
CHANNELS = ['AF3','AF4','F3','F4','F7','F8','FC5','FC6','O1','O2','P7','P8','T7','T8']

# =========================
# Load data
# =========================
df = pd.read_csv(INPUT_FILE)
pow_cols = [c for c in df.columns if c.startswith('POW.')]
tasks = sorted(df['Task'].unique())
print(f"Loaded: {len(df)} rows, {len(tasks)} tasks, {df['Subject_ID'].nunique()} subjects")

# =========================
# Step 1: Compute per-task, per-feature Pearson r with Sleep_Hours
# =========================
results = []  # (task, feature, r, p, n)

for task in tasks:
    task_df = df[df['Task'] == task]
    # Average FFT per subject for this task
    subj_avg = task_df.groupby('Subject_ID').agg(
        {**{f: 'mean' for f in pow_cols}, 'Sleep_Hours': 'first'}
    ).dropna()
    n = len(subj_avg)
    if n < 10:
        continue
    for feat in pow_cols:
        r, p = pearsonr(subj_avg[feat], subj_avg['Sleep_Hours'])
        results.append({
            'Task': task,
            'Feature': feat,
            'Channel': feat.split('.')[1],
            'Band': feat.split('.')[2],
            'r': r,
            'p': p,
            'n': n,
        })

results_df = pd.DataFrame(results)
print(f"Computed {len(results_df)} correlations ({len(tasks)} tasks × {len(pow_cols)} features)")

# =========================
# Step 2: Build 36×5 heatmap matrix (average r across channels per band)
# =========================
heatmap_data = results_df.groupby(['Task', 'Band'])['r'].mean().unstack('Band')
heatmap_data = heatmap_data[BANDS]  # consistent column order

# Sort tasks by max |r| across bands (most predictive at top)
heatmap_data['max_abs_r'] = heatmap_data.abs().max(axis=1)
heatmap_data = heatmap_data.sort_values('max_abs_r', ascending=True)  # ascending for imshow (top = bottom of array)
heatmap_data = heatmap_data.drop(columns='max_abs_r')

# --- Plot heatmap ---
fig, ax = plt.subplots(figsize=(8, 14))

vmax = max(abs(heatmap_data.values.min()), abs(heatmap_data.values.max()))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(heatmap_data.values, cmap='RdBu_r', norm=norm, aspect='auto')

# Labels
ax.set_xticks(range(len(BANDS)))
ax.set_xticklabels(BANDS, fontsize=10, fontweight='bold')
ax.set_yticks(range(len(heatmap_data)))
ax.set_yticklabels(heatmap_data.index, fontsize=8)

# Annotate cells with r values
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data.values[i, j]
        color = 'white' if abs(val) > vmax * 0.6 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, color=color)

ax.set_title('Per-Task FFT–Sleep Correlation (Pearson r, avg across channels)',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('FFT Band', fontsize=11)
ax.set_ylabel('Cognitive Task', fontsize=11)

cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('Mean Pearson r', fontsize=10)

plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, 'per_task_fft_heatmap.png')
plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved heatmap: {heatmap_path}")

# =========================
# Step 3: Top 10 strongest task-band-channel combinations
# =========================
results_df['abs_r'] = results_df['r'].abs()
top10 = results_df.nlargest(10, 'abs_r').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 7))

labels = [f"{row['Task']}  •  {row['Feature'].replace('POW.','')}" for _, row in top10.iterrows()]
colors = ['#E74C3C' if r > 0 else '#4C72B0' for r in top10['r']]

bars = ax.barh(range(len(top10)), top10['r'], color=colors, edgecolor='white',
               linewidth=1.2, alpha=0.85, height=0.7)

# Add p-value annotations — always on the right end (positive side) for readability
for i, (_, row) in enumerate(top10.iterrows()):
    p_str = f"p={row['p']:.3f}" if row['p'] >= 0.001 else f"p={row['p']:.1e}"
    sign = '+' if row['r'] > 0 else ''
    annotation = f"r={sign}{row['r']:.3f}, {p_str}, n={row['n']:.0f}"
    # Place annotation outside the bar end
    if row['r'] > 0:
        ax.text(row['r'] + 0.008, i, annotation,
                va='center', ha='left', fontsize=8, color='#333')
    else:
        ax.text(row['r'] - 0.008, i, annotation,
                va='center', ha='right', fontsize=8, color='#333')

ax.set_yticks(range(len(top10)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Pearson r', fontsize=11)
ax.set_title('Top 10 Task–Band–Channel Pairs by |r| with Sleep Hours',
             fontsize=13, fontweight='bold')
ax.axvline(0, color='#333', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)
ax.set_axisbelow(True)
ax.margins(x=0.25)  # extra horizontal margin for annotations

plt.tight_layout()
top10_path = os.path.join(OUTPUT_DIR, 'top_task_band_pairs.png')
plt.savefig(top10_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved top-10 chart: {top10_path}")

# =========================
# Step 4: Console summary
# =========================
print(f"\n{'='*80}")
print(f"  TOP 15 TASK-BAND-CHANNEL PAIRS (by |r|)")
print(f"{'='*80}")
top15 = results_df.nlargest(15, 'abs_r')
for _, row in top15.iterrows():
    sig = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01 else ('*' if row['p'] < 0.05 else 'n.s.'))
    print(f"  {row['Task']:40s} {row['Feature']:20s}  r={row['r']:+.4f}  p={row['p']:.4f}  n={row['n']:.0f}  {sig}")

# Heatmap summary: which tasks have the strongest average signal?
print(f"\n{'='*80}")
print(f"  TOP 10 TASKS by max |mean r| across bands")
print(f"{'='*80}")
task_max = heatmap_data.abs().max(axis=1).sort_values(ascending=False).head(10)
for task, maxr in task_max.items():
    best_band = heatmap_data.loc[task].abs().idxmax()
    best_r = heatmap_data.loc[task, best_band]
    print(f"  {task:40s}  best band={best_band:6s}  mean r={best_r:+.4f}")

print(f"\nDone. Output directory: {OUTPUT_DIR}")
