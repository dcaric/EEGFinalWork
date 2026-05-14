import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

output_dir = os.path.join(PROJECT_ROOT, 'individual_graphs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


df = pd.read_csv(os.path.join(PROJECT_ROOT, 'pilot_files/pilot_categorized_tasks.csv'))

# Metrics for the clusters
metrics = [c for c in df.columns if c.startswith('PM.')]
subjects = df['Subject_ID'].unique()

print(f"--- Generating Clustered Heatmaps for {len(subjects)} subjects ---")

for sid in subjects:
    person_df = df[df['Subject_ID'] == sid]
    person_sleep = person_df['Sleep_Hours'].iloc[0]
    
    
    pivot = person_df.groupby('Task')[metrics].mean()

    #Drop metrics that are entirely empty for this person
    pivot = pivot.dropna(axis=1, how='all')
    
    # Drop tasks that have an missing metric values
    # Ward method cannot handle NaN values
    pivot = pivot.dropna(axis=0, how='any')
    
    # Check if we still have a valid table (need at least 2 tasks and 2 metrics to cluster)
    if pivot.shape[0] > 1 and pivot.shape[1] > 1:
        try:
            # method='ward' handles the clustering
            # standard_scale=1 normalizes the metrics
            cg = sns.clustermap(pivot, 
                                method='ward', 
                                cmap='RdYlGn', 
                                standard_scale=1, 
                                figsize=(12, 12))
            
            cg.fig.suptitle(f'Brain Signature (Ward Clustering): Subject {sid}\n({person_sleep}h Sleep)', y=1.05, fontsize=15)
            
            save_path = os.path.join(output_dir, f'clustered_map_{sid}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"✅ Generated map for {sid}")
            
        except Exception as e:
            print(f"❌ Could not cluster {sid} due to: {e}")
    else:
        print(f"⚠️ Skipping {sid}: Not enough complete data (Rows: {pivot.shape[0]}, Cols: {pivot.shape[1]})")

print(f"\n🚀 Success! Check the '{output_dir}' folder.")