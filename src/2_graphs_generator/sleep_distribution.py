import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# Setup paths
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

METADATA_PATH = os.path.join(PROJECT_ROOT, 'DatasetSub_shorten.csv')
OUTPUT_PNG = os.path.join(PROJECT_ROOT, 'visuals', 'sleep_hours_distribution.png')

# =========================
# Load metadata — all subjects, not just quality-filtered ones
# =========================
metadata_df = pd.read_csv(METADATA_PATH, sep=';', header=1, decimal=',')
metadata_df = metadata_df.dropna(subset=['ID'])
sleep_per_subject = metadata_df['Average Sleep Hours per Night'].astype(float)

print(f"Total subjects: {len(sleep_per_subject)}")
print(f"Sleep range: {sleep_per_subject.min():.1f}h – {sleep_per_subject.max():.1f}h")
print(f"Mean: {sleep_per_subject.mean():.2f}h, Median: {sleep_per_subject.median():.2f}h, Std: {sleep_per_subject.std():.2f}h")

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

# Fixed bin edges — aligned with sleep group thresholds (6h and 8h)
# Each bin is centred on a half-hour value (5.0, 5.5, 6.0, …)
bin_edges = [4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75]
counts, bins, patches = ax.hist(
    sleep_per_subject, 
    bins=bin_edges, 
    edgecolor='white', 
    linewidth=1.2,
    color='#4C72B0',
    alpha=0.85
)

# Color bins by sleep group — use bin centre to avoid edge ambiguity
for patch, left_edge, right_edge in zip(patches, bins[:-1], bins[1:]):
    centre = (left_edge + right_edge) / 2
    if centre < 6.0:
        patch.set_facecolor('#E74C3C')   # Short sleep — red
    elif centre < 8.0:
        patch.set_facecolor('#4C72B0')   # Normal sleep — blue
    else:
        patch.set_facecolor('#2ECC71')   # Long sleep — green

# Reference lines
ax.axvline(sleep_per_subject.mean(), color='#E67E22', linestyle='--', linewidth=2, label=f'Mean ({sleep_per_subject.mean():.1f}h)')
ax.axvline(sleep_per_subject.median(), color='#8E44AD', linestyle='-.', linewidth=2, label=f'Median ({sleep_per_subject.median():.1f}h)')

# Labels
ax.set_title('Distribution of Average Sleep Hours per Night', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Sleep Hours', fontsize=12)
ax.set_ylabel('Number of Subjects', fontsize=12)

# Legend for sleep groups + stats
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E74C3C', label='Short (< 6h)'),
    Patch(facecolor='#4C72B0', label='Normal (6–8h)'),
    Patch(facecolor='#2ECC71', label='Long (≥ 8h)'),
    plt.Line2D([0], [0], color='#E67E22', linestyle='--', linewidth=2, label=f'Mean ({sleep_per_subject.mean():.1f}h)'),
    plt.Line2D([0], [0], color='#8E44AD', linestyle='-.', linewidth=2, label=f'Median ({sleep_per_subject.median():.1f}h)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.set_yticks(range(0, int(counts.max()) + 2))
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSaved to: {OUTPUT_PNG}")
