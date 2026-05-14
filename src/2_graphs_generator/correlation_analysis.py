#FIFTH STEP- correlation matrix will try to find 
# brain metrics that change the most  when sleep changes

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

if not os.path.exists('visuals'):
    os.makedirs('visuals')

df = pd.read_csv('pilot_files/pilot_task_ready.csv')

rename_map = {col: col.replace('PM.', '').replace('.Scaled', '') for col in df.columns}
df_renamed = df.rename(columns=rename_map)

all_metrics = ['Sleep_Hours', 'Attention', 'Engagement', 'Excitement', 'Stress', 'Relaxation', 'Interest', 'Focus']
corr_matrix = df_renamed[all_metrics].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, 
            annot=True,          
            cmap='RdBu_r',       
            fmt=".2f",           
            linewidths=.5,       
            square=True,         
            cbar_kws={"shrink": .8})

plt.title('Complete Correlation Matrix: Sleep vs. All Performance Metrics', fontsize=15, pad=20)
plt.savefig('visuals/complete_performance_heatmap.png', bbox_inches='tight')
plt.show()