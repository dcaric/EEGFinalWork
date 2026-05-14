##graph that answers: "In which task does the brain wave signal best predict sleep?"

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

df = pd.read_csv(os.path.join(PROJECT_ROOT, 'pilot_files/pilot_categorized_tasks.csv'))

pow_cols = [col for col in df.columns if col.startswith('POW.')]

#correlations between Sleep and FFT for each category
categories = df['Task_Category'].unique()
task_explainer_data = []

for cat in categories:
    subset = df[df['Task_Category'] == cat]
    if len(subset) > 5:  # Minimum rows needed for a meaningful correlation
        corrs = subset[pow_cols].corrwith(subset['Sleep_Hours'])
        corrs.name = cat
        task_explainer_data.append(corrs)

explainer_df = pd.concat(task_explainer_data, axis=1)


plt.figure(figsize=(12, 18))
sns.heatmap(explainer_df, annot=False, cmap='RdBu_r', center=0)
plt.title('Task Explainer: Which Categories track Sleep best?\n(Correlation between FFT and Sleep Hours)', fontsize=16)
plt.xlabel('Task Category (Brain Domain)', fontsize=12)
plt.ylabel('FFT Sensors and Bands', fontsize=12)


if not os.path.exists('visuals'):
    os.makedirs('visuals')

plt.savefig('visuals/task_explainer_heatmap.png', bbox_inches='tight')
print("✅ Task Explainer Heatmap saved in 'visuals/task_explainer_heatmap.png'")