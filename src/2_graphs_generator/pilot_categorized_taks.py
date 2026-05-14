import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

output_folder = os.path.join(PROJECT_ROOT, 'visuals_tasks_per_metric')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

df = pd.read_csv(os.path.join(PROJECT_ROOT, 'pilot_files/pilot_categorized_tasks.csv'))

df['Sleep_Group'] = df['Sleep_Hours'].apply(lambda x: '6h (Short)' if x <= 6 else '7h+ (Normal)')

# List of Performance Metrics to analyze
metrics = [
    'PM.Engagement.Scaled', 'PM.Excitement.Scaled', 'PM.Stress.Scaled', 
    'PM.Relaxation.Scaled', 'PM.Interest.Scaled', 'PM.Focus.Scaled', 'PM.Attention.Scaled'
]

#  7 Graphs
for metric in metrics:
    plt.figure(figsize=(14, 8))
    
    # Barplot to show the average score per category/sleep group
    # ci=68 represents the Standard Error (the 'uncertainty' in the data)
    sns.barplot(data=df, x='Task_Category', y=metric, hue='Sleep_Group', 
                palette='coolwarm', ci=68, capsize=.1)
    
    display_name = metric.replace('PM.', '').replace('.Scaled', '')
    
    plt.title(f'Impact of Sleep Duration on {display_name} by Task Domain', fontsize=18, pad=20)
    plt.ylabel(f'Mean {display_name} Score (0.0 - 1.0)', fontsize=14)
    plt.xlabel('Brain Functional Domains', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1) 
    plt.legend(title='Sleep Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    file_path = os.path.join(output_folder, f"{display_name}_Comparison.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

print(f"✅ Success! 7 graphs generated in the '{output_folder}' folder.")