import pandas as pd
import os

# ── Load source data ──────────────────────────────────────────────────────────
df = pd.read_csv('pilot_files/all_task_ready.csv')
pow_cols = [c for c in df.columns if c.startswith('POW.')]

# ── Held-out subjects (reserved for thesis defense demo) ─────────────────────
# ba16 (Long sleeper) and ba45 (Normal sleeper) are excluded from all experiments.
# Use train_final_model.py + predict.py on these two at the defense.
HELD_OUT = ['BA16', 'BA45']
df = df[~df['Subject_ID'].isin(HELD_OUT)]
print(f'Held out: {HELD_OUT} — training on {df["Subject_ID"].nunique()} subjects')

OUTPUT_DIR = 'modelling_tables'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Experiment 1: FFT only ────────────────────────────────────────────────────
exp1 = df.groupby('Subject_ID').agg(
    {**{c: 'mean' for c in pow_cols},
     'Sleep_Hours': 'first'}
).reset_index()

exp1.to_csv(f'{OUTPUT_DIR}/exp1_fft_only.csv', index=False)
print(f'Exp1: {exp1.shape}')  # (83, 72) — Subject_ID + 70 POW + Sleep_Hours

# ── Experiment 2: FFT + Gender ────────────────────────────────────────────────
exp2 = df.groupby('Subject_ID').agg(
    {**{c: 'mean' for c in pow_cols},
     'Sleep_Hours': 'first',
     'Gender': 'first'}
).reset_index()

exp2['Gender'] = exp2['Gender'].map({'M': 0, 'F': 1})  # values in dataset are 'M'/'F'

exp2.to_csv(f'{OUTPUT_DIR}/exp2_fft_gender.csv', index=False)
print(f'Exp2: {exp2.shape}')  # (83, 73) — Subject_ID + 70 POW + Sleep_Hours + Gender

# ── Experiment 3: FFT from top-5 predictive tasks only ───────────────────────
# Instead of averaging POW across all 36 tasks, only use rows from the 5 tasks
# that showed the strongest correlation with Sleep_Hours in the EDA.
# Same 70 features, same structure as exp1 — but averaged over these 5 tasks only.
# Hypothesis: task-specific EEG from the most predictive tasks beats the global average.
TOP_TASKS = [
    'gyrus_left_closed_eyes',
    'gyrus_right_closed_eyes',
    'gyrus_coord',
    'prefrontal1',
    'prefrontal2',
]

df_top = df[df['Task'].isin(TOP_TASKS)]

# Check how many subjects have data for these tasks
subjects_covered = df_top['Subject_ID'].nunique()
print(f'Exp3: {subjects_covered} subjects have at least one of the top-5 tasks')

exp3 = df_top.groupby('Subject_ID').agg(
    {**{c: 'mean' for c in pow_cols},
     'Sleep_Hours': 'first'}
).reset_index()

exp3.to_csv(f'{OUTPUT_DIR}/exp3_task_specific.csv', index=False)
print(f'Exp3: {exp3.shape}')  # expected (~83, 72) — same structure as exp1
print(f'Exp3 NaN check: {exp3.isnull().sum().sum()} NaN values')

