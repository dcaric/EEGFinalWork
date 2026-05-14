# Enhanced Preparation Audit

This folder is a sandbox for checking whether the preprocessing logic that led to
`pilot_files/all_task_ready.csv` is methodologically sound.

It is intentionally separate from the main pipeline so the original dataset and
scripts remain untouched.

## Purpose

The goal is not to rebuild the whole thesis dataset yet.
The goal is to verify, on a very small subset of subjects, whether:

- task segmentation behaves as expected
- marker filtering matters
- quality filtering removes too much or too little data
- extracted task rows resemble the structure of the original `all_task_ready.csv`

## Current test subjects

- `BA9`
- `BA39`

These subjects were chosen because local folders exist in:

- `data/ba9`
- `data/ba39`

and both subjects also exist in the metadata file.

## Files in this folder

- `build_two_subjects_filtered.py`
  - creates a small test dataset using only `BA9` and `BA39`
  - applies the same `stimuli.csv` task filtering logic used by the original full pipeline
  - excludes non-task markers such as calibration and instruction phases
  - writes a diagnostics file for auditing segmentation outcomes

- `all_task_ready_2subjects_filtered.csv`
  - output test dataset

- `all_task_ready_2subjects_filtered_diagnostics.csv`
  - row-level diagnostics showing which markers were kept, skipped, or failed

## Why this matters

An earlier unfiltered 2-subject test showed about 96 task-like segments per subject,
which was far above the expected ~36 real tasks. That revealed that the marker stream
contains many non-task events such as:

- phase markers
- instruction markers
- calibration markers
- active-period markers

This means the original `stimuli.csv` filtering step is important and should not be removed.

## Intended interpretation

If the filtered 2-subject dataset produces roughly the expected number of cognitive tasks,
that supports the view that the original full preprocessing pipeline is structurally reasonable.

If the filtered output still looks strange, then the segmentation logic itself should be reviewed more deeply.

## Current conclusion from the 2-subject audit

The filtered audit produced results much closer to the original pipeline expectations:

- `BA9` kept `36` tasks
- `BA39` kept `34` tasks
- non-task markers were heavily filtered out
- only two expected task segments were missing start/end markers in `BA39`

So this supports the idea that the original procedure was not fundamentally wrong.
The preprocessing may still be lossy, but the marker/task filtering logic appears to be doing the right kind of work.

### Main takeaway

The sandbox suggests:

- the original pipeline is probably methodologically limited, but not broken
- the poor model results are still more likely due to:
  - weak signal
  - coarse target
  - small sample
  - heavy averaging
