import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create a folder for these new graphs
if not os.path.exists('visuals_fft'):
    os.makedirs('visuals_fft')

# Load your clean pilot data
df = pd.read_csv('pilot_files/pilot_task_ready.csv')

# Identify the columns
pow_cols = [col for col in df.columns if col.startswith('POW.')]
pm_cols = [col for col in df.columns if col.startswith('PM.')]

# --- TASK 1: FFT vs. Sleep Hours (Bar Chart) ---
# Calculate correlations
fft_sleep_corr = df[pow_cols + ['Sleep_Hours']].corr()['Sleep_Hours'].drop('Sleep_Hours').sort_values()

plt.figure(figsize=(10, 18))
# Color bars: Red for negative correlation, Blue for positive
colors = ['red' if x < 0 else 'blue' for x in fft_sleep_corr]
fft_sleep_corr.plot(kind='barh', color=colors)

plt.title('Correlation: All FFT Features vs. Sleep Hours', fontsize=14, pad=20)
plt.xlabel('Correlation Coefficient (r)', fontsize=12)
plt.ylabel('FFT Sensor-Band Pairs', fontsize=10)
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(axis='x', linestyle='--', alpha=0.5)

plt.savefig('visuals_fft/fft_sleep_correlation_bars.png', bbox_inches='tight')
print("✅ Bar chart saved to visuals_fft/fft_sleep_correlation_bars.png")


# --- TASK 2: FFT vs. Performance Metrics (Heatmaps) ---
# Calculate the correlation matrix specifically for FFT vs PM
corr_matrix = df[pow_cols + pm_cols].corr()
pow_vs_pm = corr_matrix.loc[pow_cols, pm_cols]

# 2a. The "Big Picture" Heatmap (No numbers, just colors to see patterns)
plt.figure(figsize=(12, 15))
sns.heatmap(pow_vs_pm, annot=False, cmap='RdBu_r', center=0)
plt.title('Overview: All FFT vs. Performance Metrics', fontsize=15)
plt.savefig('visuals_fft/fft_vs_metrics_overview.png', bbox_inches='tight')

# 2b. Regional Heatmaps (The ones you'll actually use for the text)
regions = {
    'Frontal': ['AF3', 'F7', 'F3', 'FC5', 'FC6', 'F4', 'F8', 'AF4'],
    'Temporal': ['T7', 'T8'],
    'Parietal_Occipital': ['P7', 'P8', 'O1', 'O2']
}

for region_name, sensors in regions.items():
    # Filter columns for this specific region
    region_cols = [col for col in pow_cols if any(sensor in col for sensor in sensors)]
    
    if region_cols:
        region_df = pow_vs_pm.loc[region_cols]
        plt.figure(figsize=(10, len(region_cols)*0.4 + 2))
        sns.heatmap(region_df, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
        plt.title(f'Heatmap: {region_name} FFT vs. Performance Metrics', fontsize=14)
        plt.savefig(f'visuals_fft/fft_vs_metrics_{region_name.lower()}.png', bbox_inches='tight')
        print(f"✅ Region heatmap saved: {region_name}")

plt.show()