#FORTH STEP- looks at the raw headset data and calculates exactly how "clean" 
# the signal was for each person


import pandas as pd
import os

mapping_df = pd.read_csv('pilot_data_map.csv')

quality_summary = []

MIN_OVERALL_CQ = 0.7
MIN_CHANNEL_CQ = 2

print("--- Starting Signal Quality Check ---")

#   14 EEG channels used by the Emotiv headset
eeg_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
cq_columns = [f'CQ.{ch}' for ch in eeg_channels]

for index, row in mapping_df.iterrows():
    subject_id = row['Subject_ID']
    print(f"Analyzing {subject_id}...")

    data = pd.read_csv(row['Data_File'], header=1, low_memory=False)

    # Calculate Interpolation Rate
    # 0 = Good signal, 1 = Interpolated (noisy/lost)
    total_samples = len(data)
    eeg_interpolated = pd.to_numeric(data['EEG.Interpolated'], errors='coerce')
    interpolated_samples = eeg_interpolated.sum()
    usable_percent = 100 * (1 - (interpolated_samples / total_samples))

    # Calculate pass rates for the same quality rules used in cleaning
    overall_cq = pd.to_numeric(data['CQ.Overall'], errors='coerce')
    if overall_cq.dropna().max() is not None and overall_cq.dropna().max() <= 1.0:
        overall_threshold = MIN_OVERALL_CQ
    else:
        overall_threshold = MIN_OVERALL_CQ * 100

    channel_cq = data[cq_columns].apply(pd.to_numeric, errors='coerce')

    interpolated_mask = eeg_interpolated == 0
    overall_mask = overall_cq >= overall_threshold
    channel_mask = (channel_cq >= MIN_CHANNEL_CQ).all(axis=1)
    final_mask = interpolated_mask & overall_mask & channel_mask

    # Calculate Average Channel Quality (CQ)
    #  0 is 'No Contact' and 4 is 'Perfect'
    avg_cq = channel_cq.mean().to_dict()
    
    subject_report = {
        'Subject_ID': subject_id,
        'Usable_Data_%': round(usable_percent, 2),
        'Overall_CQ': round(overall_cq.mean(), 2),
        'Interpolated_Pass_%': round(100 * interpolated_mask.mean(), 2),
        'Overall_CQ_Pass_%': round(100 * overall_mask.mean(), 2),
        'All_Channels_CQ_Pass_%': round(100 * channel_mask.mean(), 2),
        'Final_Retained_%': round(100 * final_mask.mean(), 2)
    }
    # Add individual channel averages to the report
    subject_report.update({k: round(v, 2) for k, v in avg_cq.items()})
    
    quality_summary.append(subject_report)

quality_df = pd.DataFrame(quality_summary)
quality_df.to_csv('pilot_signal_quality.csv', index=False)

print("\n--- Pilot Signal Quality Report ---")
print(quality_df[['Subject_ID', 'Usable_Data_%', 'Overall_CQ', 'Final_Retained_%']])

print("\n🚀 Success! Detailed report saved as 'pilot_signal_quality.csv'")
