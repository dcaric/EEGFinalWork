import pandas as pd
import os
import sys

# =========================
# 1. SETUP PATHS
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR = os.path.join(PROJECT_ROOT, "all_data")

METADATA_PATH = os.path.join(PROJECT_ROOT, 'DatasetSub_shorten.csv')
STIMULI_PATH = os.path.join(BASE_DIR, 'stimuli.csv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'all_task_ready.csv')

# =========================
# 2. LOAD METADATA & STIMULI
# =========================
print("--- Step 1: Loading Metadata and Stimuli ---")
# Delimiter is ';' and header is at index 1
metadata_df = pd.read_csv(METADATA_PATH, sep=';', header=1, decimal=',')
metadata_df = metadata_df.dropna(subset=['ID'])

# Identify subjects with existing folders
metadata_df['Folder_Exists'] = metadata_df['ID'].apply(lambda x: os.path.isdir(os.path.join(BASE_DIR, str(x))))
available_df = metadata_df[metadata_df['Folder_Exists'] == True].copy()
available_df['Sleep'] = available_df['Average Sleep Hours per Night'].astype(float)

# Load stimuli for filtering
stimuli = pd.read_csv(STIMULI_PATH)
stimuli['Task_clean'] = stimuli['original_stimuli_name'].astype(str).str.strip().str.lower()
allowed_tasks = set(stimuli['Task_clean']) - {'eyesopen', 'eyesclose'}

print(f"Total available subjects: {len(available_df)}")

# =========================
# 3. QUALITY SETTINGS
# =========================
MIN_OVERALL_CQ = 0.7
MIN_CHANNEL_CQ = 2

def quality_mask(data: pd.DataFrame) -> pd.Series:
    mask = data['EEG.Interpolated'] == 0

    overall_cq = pd.to_numeric(data['CQ.Overall'], errors='coerce')
    if overall_cq.dropna().max() is not None and overall_cq.dropna().max() <= 1.0:
        mask &= overall_cq >= MIN_OVERALL_CQ
    else:
        mask &= overall_cq >= (MIN_OVERALL_CQ * 100)

    channel_cq_cols = [c for c in data.columns if c.startswith('CQ.') and c != 'CQ.Overall']
    channel_cq = data[channel_cq_cols].apply(pd.to_numeric, errors='coerce')
    mask &= (channel_cq >= MIN_CHANNEL_CQ).all(axis=1)

    return mask

# =========================
# 4. PROCESSING LOOP
# =========================
all_task_features = []

print("\n--- Step 2: Processing All Subjects ---")

for index, row in available_df.iterrows():
    subject_id = str(row['ID'])
    data_path = os.path.join(BASE_DIR, subject_id, 'data.csv')
    marker_path = os.path.join(BASE_DIR, subject_id, 'marker.csv')

    if not os.path.exists(data_path) or not os.path.exists(marker_path):
        continue

    print(f"Processing {subject_id}...")

    # Load Data and Markers
    # Data has header at row 1 (0-indexed)
    try:
        data = pd.read_csv(data_path, header=1, low_memory=False)
        markers = pd.read_csv(marker_path)
    except Exception as e:
        print(f"  Error loading {subject_id}: {e}")
        continue

    # Cleaning (Quality Check) 
    clean_data = data[quality_mask(data)].copy()
    
    # identify key columns to analyze (FFT/Power and Performance Metrics)
    pow_cols = [c for c in clean_data.columns if 'POW.' in c]
    pm_cols = [c for c in clean_data.columns if 'PM.' in c and '.Scaled' in c]
    metrics_to_avg = pow_cols + pm_cols

    # Parse MarkerIndex for inline event sync (positive = start, negative = end)
    clean_data = clean_data.copy()
    clean_data['MarkerIndex'] = pd.to_numeric(clean_data['MarkerIndex'], errors='coerce')

    # Build lookup: marker_id -> task name, using marker.csv only for names
    id_to_task = dict(zip(markers['marker_id'], markers['type']))

    subjects_with_data = False

    for marker_id, task_type in id_to_task.items():
        task_clean = str(task_type).strip().lower()

        # Filter to only allowed tasks from stimuli.csv
        if task_clean not in allowed_tasks:
            continue

        # Synchronize using inline MarkerIndex events in data.csv
        # Positive index = task start, negative = task end
        start_rows = clean_data[clean_data['MarkerIndex'] == marker_id]
        end_rows   = clean_data[clean_data['MarkerIndex'] == -marker_id]

        if start_rows.empty or end_rows.empty:
            continue

        t_start = start_rows['Timestamp'].iloc[0]
        t_end   = end_rows['Timestamp'].iloc[0]

        task_segment = clean_data[(clean_data['Timestamp'] >= t_start) &
                                  (clean_data['Timestamp'] <= t_end)]

        if not task_segment.empty:
            # check for missing FFT data reference
            if 'POW.AF3.Theta' in task_segment.columns and task_segment['POW.AF3.Theta'].isna().all():
                continue

            # calculate mean for this segment
            task_averages = task_segment[metrics_to_avg].mean().to_dict()

            # Attach metadata
            task_averages.update({
                'Subject_ID': subject_id,
                'Task': task_type,
                'Sleep_Hours': row['Sleep'],
                'Gender': row['Gender']
            })
            all_task_features.append(task_averages)
            subjects_with_data = True

    if not subjects_with_data:
        print(f"  WARNING: No task segments found for {subject_id}")

# =========================
# 5. SAVE FINAL DATASET
# =========================
if all_task_features:
    final_df = pd.DataFrame(all_task_features)
    # Final check: drop rows where reference FFT is NaN (just in case)
    if 'POW.AF3.Theta' in final_df.columns:
        final_df = final_df.dropna(subset=['POW.AF3.Theta'])
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n🚀 SUCCESS! '{OUTPUT_FILE}' created.")
    print(f"Total task segments analyzed: {len(final_df)}")
    print(f"Unique subjects in output:   {final_df['Subject_ID'].nunique()}")
    print(f"Total subjects available:    {len(available_df)}")
else:
    print("\n⚠️ No task data was processed. Check your paths and filters.")
