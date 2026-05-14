import os
import pandas as pd

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Load your prepared data
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'pilot_files/pilot_task_ready.csv'))
stimuli = pd.read_csv(os.path.join(PROJECT_ROOT, 'all_data', 'stimuli.csv'))

# Normalize task names for matching
df['Task_clean'] = df['Task'].astype(str).str.strip().str.lower()
stimuli['Task_clean'] = stimuli['original_stimuli_name'].astype(str).str.strip().str.lower()

# Exclude eyesopen and eyesclose from the allowed stimuli list
stimuli = stimuli[~stimuli['Task_clean'].isin(['eyesopen', 'eyesclose'])].copy()

# Keep only tasks that are present in stimuli.csv
df = df.merge(
    stimuli[['Task_clean', 'description', 'cognitive_domain']],
    on='Task_clean',
    how='inner'
)

def categorize_task(task_name, description='', cognitive_domain=''):
    t = str(task_name).lower()
    d = str(description).lower()
    c = str(cognitive_domain).lower()

    # 1. MEMORY
    if any(x in t for x in ['memor', 'recall', 'sequence', 'next_in_seq']) \
       or 'memory' in c:
        return 'Memory'

    # 2. EMOTIONAL
    if any(x in t for x in ['image', 'happy', 'sad', 'angry']) \
       or 'emotion' in c:
        return 'Emotional'

    # 3. COGNITIVE / LANGUAGE
    if any(x in t for x in ['prefrontal', 'reading', 'sentence', 'words', 'saying']) \
       or any(x in c for x in ['language', 'speech', 'verbal fluency', 'semantic']) \
       or any(x in d for x in ['reading', 'speech', 'language', 'verbal']):
        return 'Cognitive_Language'

    # 4. VISUAL / SPATIAL
    if any(x in t for x in ['valdo', 'occipital', 'parietal', 'imagine', '2d_3d', 'text_element']) \
       or any(x in c for x in ['visual', 'spatial', 'pattern recognition']) \
       or any(x in d for x in ['visual', 'spatial', 'navigation', 'imagery']):
        return 'Visual_Spatial'

    # 5. MOTOR / ACTION
    if any(x in t for x in ['gyrus_left_open', 'gyrus_right_open', 'gyrus_coord', 'gyrus_pause']) \
       or any(x in c for x in ['motor', 'coordination', 'proprioception', 'bimanual']) \
       or any(x in d for x in ['motor', 'movement', 'coordination', 'proprioception']):
        return 'Motor_Action'

    # 6. AUDITORY
    if 'instruments' in t \
       or 'auditory' in c \
       or any(x in d for x in ['auditory', 'listening', 'sound', 'music']):
        return 'Auditory'

    return 'Other_Stimuli'

df['Task_Category'] = df.apply(
    lambda row: categorize_task(
        row['Task'],
        row.get('description', ''),
        row.get('cognitive_domain', '')
    ),
    axis=1
)

# Keep only relevant columns (drop helper columns)
df_final = df.drop(columns=['Task_clean', 'description', 'cognitive_domain'], errors='ignore').copy()
df_final.to_csv('pilot_categorized_tasks.csv', index=False)

print("Data categorized using stimuli.csv and cleaned!")

counts = df_final['Task_Category'].value_counts()
for category, count in counts.items():
    unique_tasks = df_final[df_final['Task_Category'] == category]['Task'].unique()
    print(f"\n{category} ({count} rows)")
    print(f"Tasks: {', '.join(unique_tasks)}")

other_count = (df_final['Task_Category'] == 'Other_Stimuli').sum()
if other_count == 0:
    print("✨ Great! 0 tasks left in 'Other_Stimuli'. Everything is categorized.")
else:
    print(f"⚠️ Warning: {other_count} tasks are still in 'Other_Stimuli'.")

excluded_tasks = set(pd.read_csv(os.path.join(PROJECT_ROOT, 'pilot_task_ready.csv'))['Task'].astype(str).str.strip().str.lower()) - set(df['Task_clean'])
print(f"\nExcluded tasks ({len(excluded_tasks)}):")
print(sorted(excluded_tasks))