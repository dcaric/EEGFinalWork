"""
Sleep-Group Bar Charts: FFT features (Gamma, BetaH) across sleep groups.
Full dataset (all_task_ready.csv, 83 subjects).

Generates two figures:
  1. 3-group split: <6h / 6–8h / ≥8h   (n=6 / 68 / 9  — limited power)
  2. 2-group split: ≤7h / >7h           (n=63 / 20 — better power)

Each subplot: grouped bars + SEM error bars + Mann-Whitney U significance stars.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from itertools import combinations
import os

# =========================
# Config
# =========================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_FILE = os.path.join(PROJECT_ROOT, 'pilot_files/all_task_ready.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'visuals_sleep_groups')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features to plot — positive AND negative correlates from bar chart analysis
FEATURES = [
    # Frontal Gamma (top positive correlates)
    'POW.F7.Gamma',  'POW.F8.Gamma',  'POW.AF3.Gamma', 'POW.AF4.Gamma',
    # Frontal BetaH (top positive correlates)
    'POW.F7.BetaH',  'POW.F8.BetaH',  'POW.AF3.BetaH', 'POW.AF4.BetaH',
    # Posterior Gamma
    'POW.P7.Gamma',  'POW.P8.Gamma',  'POW.O1.Gamma',  'POW.O2.Gamma',
    # Frontal Theta & Alpha (top negative correlates)
    'POW.AF3.Theta', 'POW.AF4.Theta', 'POW.AF3.Alpha', 'POW.AF4.Alpha',
]

COLORS_3 = ['#E74C3C', '#4C72B0', '#2ECC71']   # red / blue / green
COLORS_2 = ['#E74C3C', '#4C72B0']               # red / blue

# =========================
# Load & aggregate per subject
# =========================
df = pd.read_csv(INPUT_FILE)
subject_avg = df.groupby('Subject_ID').agg(
    {**{f: 'mean' for f in FEATURES}, 'Sleep_Hours': 'first'}
)
print(f"Subjects loaded: {len(subject_avg)}")

# =========================
# Define sleep groups
# =========================
def assign_3group(h):
    if h < 6:
        return 'Short (<6h)'
    elif h < 8:
        return 'Normal (6–8h)'
    else:
        return 'Long (≥8h)'

def assign_2group(h):
    if h <= 7:
        return '≤7h'
    else:
        return '>7h'

subject_avg['Group_3'] = subject_avg['Sleep_Hours'].apply(assign_3group)
subject_avg['Group_2'] = subject_avg['Sleep_Hours'].apply(assign_2group)

GROUP_3_ORDER = ['Short (<6h)', 'Normal (6–8h)', 'Long (≥8h)']
GROUP_2_ORDER = ['≤7h', '>7h']

# Print group sizes
for g in GROUP_3_ORDER:
    n = (subject_avg['Group_3'] == g).sum()
    print(f"  3-group  {g}: n={n}")
for g in GROUP_2_ORDER:
    n = (subject_avg['Group_2'] == g).sum()
    print(f"  2-group  {g}: n={n}")


# =========================
# Plotting helper
# =========================
def sig_label(p, alpha_corrected=0.05):
    """Uncorrected significance label."""
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'

def sig_label_corrected(p, alpha_corrected):
    """After Bonferroni correction."""
    if p < alpha_corrected: return f'* (Bonf.)'
    return 'n.s. (Bonf.)'


def plot_sleep_group_bars(subject_df, group_col, group_order, colors, title, out_path):
    """Grid of bar charts — one per feature, with Bonferroni correction."""
    n_pairs = len(list(combinations(range(len(group_order)), 2)))
    n_comparisons = len(FEATURES) * n_pairs  # total family of tests
    alpha_corrected = 0.05 / n_comparisons
    print(f"\nBonferroni correction: {n_comparisons} comparisons → α_corrected = {alpha_corrected:.6f}")
    n_feats = len(FEATURES)
    n_cols = 4
    n_rows = int(np.ceil(n_feats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    summary_rows = []

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        group_data = []
        means, sems = [], []

        for g in group_order:
            vals = subject_df.loc[subject_df[group_col] == g, feat].dropna()
            group_data.append(vals)
            means.append(vals.mean())
            sems.append(vals.sem())

        # --- Bars ---
        x = np.arange(len(group_order))
        bar_width = 0.55
        bars = ax.bar(x, means, bar_width, yerr=sems, capsize=5,
                      color=colors, edgecolor='white', linewidth=1.2, alpha=0.85,
                      error_kw={'linewidth': 1.5})

        # --- Significance brackets ---
        pair_results = []
        for (i1, g1), (i2, g2) in combinations(enumerate(group_order), 2):
            if len(group_data[i1]) < 3 or len(group_data[i2]) < 3:
                pair_results.append((i1, i2, 1.0))  # skip tiny groups
                continue
            stat, p = mannwhitneyu(group_data[i1], group_data[i2], alternative='two-sided')
            pair_results.append((i1, i2, p))

        # Draw brackets for the most relevant comparisons
        y_max = max(m + s for m, s in zip(means, sems)) if means else 0
        bracket_height = y_max * 0.06
        y_offset = y_max * 0.04

        for idx, (i1, i2, p) in enumerate(pair_results):
            uncorr = sig_label(p)
            bonf   = sig_label_corrected(p, alpha_corrected)
            # Show Bonferroni-corrected label on chart
            display_label = uncorr if p >= 0.05 else f'{uncorr} (n.s. Bonf.)'
            if p < alpha_corrected:
                display_label = f'{uncorr} (sig. Bonf.)'
            y_bar = y_max + y_offset + idx * (bracket_height + y_max * 0.05)
            ax.plot([i1, i1, i2, i2], [y_bar, y_bar + bracket_height, y_bar + bracket_height, y_bar],
                    color='#333', linewidth=1.2)
            ax.text((i1 + i2) / 2, y_bar + bracket_height, display_label,
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color='#E74C3C' if p < alpha_corrected else '#888')

            # Record for summary
            summary_rows.append({
                'Feature': feat,
                'Comparison': f'{group_order[i1]} vs {group_order[i2]}',
                'U_p': p,
                'Uncorrected': uncorr,
                'Bonferroni': bonf
            })

        # Labels
        short_name = feat.replace('POW.', '')
        ax.set_title(short_name, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{g}\n(n={len(d)})' for g, d in zip(group_order, group_data)],
                           fontsize=9)
        ax.set_ylabel('Mean Power', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

    # Hide empty subplots
    for j in range(len(FEATURES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Print summary table
    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print()


# =========================
# Generate both versions
# =========================
plot_sleep_group_bars(
    subject_avg, 'Group_3', GROUP_3_ORDER, COLORS_3,
    'FFT Power by Sleep Group (3-way: <6h / 6–8h / ≥8h)',
    os.path.join(OUTPUT_DIR, 'sleep_group_3way_fft.png')
)

plot_sleep_group_bars(
    subject_avg, 'Group_2', GROUP_2_ORDER, COLORS_2,
    'FFT Power by Sleep Group (2-way: ≤7h / >7h)',
    os.path.join(OUTPUT_DIR, 'sleep_group_2way_fft.png')
)

print(f"\nSaved to: {OUTPUT_DIR}")
