# New Sampling Strategy

## Why This Change Was Proposed

The previous modeling pipeline aggregated the EEG features too early and reduced the data to a subject-level matrix such as:

- `81 subjects x 70 FFT features`

This likely removed useful within-subject variation across:

- windows
- stimuli/tasks

The new idea is to keep more granular observations for modeling, then collapse predictions back to the subject level only at the very end.

## Professor's Suggested Logic

Instead of one row per person, use one row per:

- `subject x stimulus x window`

Example row IDs:

- `valdo1-BA1-window1`
- `valdo1-BA1-window2`
- `gyrus1-BA7-window5`

If that becomes too large, aggregate windows only within each stimulus using the median, so rows become:

- `valdo1-BA1`
- `valdo2-BA1`
- `gyrus1-BA7`

Then:

1. predict `Short` / `Normal` / `Long` per stimulus row
2. collect all stimulus predictions for one subject
3. use majority vote to get the final subject-level prediction

This also makes it possible to inspect which stimuli tend to push the model toward `Short` or `Long`.

## New Prototype Files

To implement this idea, a new preparation script was created:

- [build_windowed_stimulus_datasets.py](/Users/dcaric/Working/pythonWorking/PaolaFinalWork/enhanced_preparation/build_windowed_stimulus_datasets.py)

It generates two datasets:

- window-level dataset:
  - [all_task_window_ready.csv](/Users/dcaric/Working/pythonWorking/PaolaFinalWork/enhanced_preparation/all_task_window_ready.csv)
- stimulus-level median dataset:
  - [all_task_stimulus_median_ready.csv](/Users/dcaric/Working/pythonWorking/PaolaFinalWork/enhanced_preparation/all_task_stimulus_median_ready.csv)

It also saves diagnostics:

- [all_task_window_ready_diagnostics.csv](/Users/dcaric/Working/pythonWorking/PaolaFinalWork/enhanced_preparation/all_task_window_ready_diagnostics.csv)

## What Counts As a Window Here

Each cleaned raw row from `data.csv` already contains:

- the `70` FFT features (`POW.*`)
- the `PM.*` scaled metrics

So in the new pipeline, one cleaned raw row inside a valid task segment is treated as one FFT window.

## Two-Subject Prototype Result

The new script was tested on the only currently available local raw folders:

- `BA9`
- `BA39`

Result:

- allowed tasks from reference dataset: `36`
- `BA9`: `36` kept tasks
- `BA39`: `34` kept tasks
- total valid stimulus-subject pairs: `70`
- total valid window rows: `8585`

Saved outputs:

- window rows: `8585`
- stimulus-median rows: `70`

Per-subject counts:

- `BA9`: `4643` valid window rows, `36` stimulus rows
- `BA39`: `3942` valid window rows, `34` stimulus rows

Diagnostics summary:

- `70` task segments processed successfully
- `128` markers excluded as non-task events
- `2` missing start/end cases:
  - `image_sad` for `BA39`
  - `image_sad2` for `BA39`

This supports the idea that the new sampling strategy is technically feasible on the raw recordings.

## Important Interpretation

This new representation gives many more rows, but it does **not** create more independent people.

So future modeling must still:

- split train/test by `Subject_ID`
- never mix rows from the same person across train and test

Otherwise there will be leakage.

## New Modeling Script

A new classifier script was created:

- [run_stimulus_majority_vote_cv.py](/Users/dcaric/Working/pythonWorking/PaolaFinalWork/src/3_modelling/run_stimulus_majority_vote_cv.py)

It does the following:

- trains on `subject x stimulus` rows
- uses subject-safe splitting with leave-one-subject-out CV
- predicts one class per stimulus row
- aggregates back to one final class per subject using majority vote
- saves:
  - per-stimulus predictions
  - per-subject final predictions
  - task-level prediction summaries

This script is the direct implementation of the professor's idea.

## Prototype Modeling Smoke Test

The script was run on the 2-subject prototype stimulus-median dataset to verify that the full logic works:

- input rows: `70`
- features: `77`
  - `70` FFT features
  - `7` PM scaled features
- subjects: `2`
- tasks: `36`
- available classes in this tiny prototype:
  - `Normal`: `34`
  - `Long`: `36`

Observed leave-one-subject-out result:

- when `BA39` was held out, all `34` stimulus rows were predicted as `Long`
- when `BA9` was held out, all `36` stimulus rows were predicted as `Normal`
- subject-level majority-vote accuracy: `0.0000`

This is not a meaningful scientific result, because with only two subjects each fold trains on only one person. However, it is still useful as a technical verification because it proves that:

- the stimulus-level classifier runs successfully
- the majority-vote aggregation works
- per-stimulus and per-subject outputs are generated correctly
- the workflow is ready to be rerun on the full dataset once all subject folders are available

## Why This Could Be Better Than the Old Pipeline

Potential advantages:

- preserves stimulus-specific EEG information
- avoids collapsing all subject data into one single average too early
- allows the model to detect that some tasks may be more sleep-sensitive than others
- supports analysis of which stimuli tend to produce `Short` or `Long` predictions

## What Still Needs To Happen

The current prototype used only `2` subjects because only two raw folders are currently available locally.

The next real step is:

1. run the new preparation script on the full set of subject folders
2. produce the full `subject x stimulus` median dataset
3. run the majority-vote modeling pipeline on that full dataset
4. compare the new stimulus-level approach against the old fully aggregated subject-level approach

## Current Takeaway

The old pipeline was probably too aggressively aggregated.

The new pipeline keeps:

- more rows
- more task structure
- more within-subject information

This does not guarantee a strong final model, but it is a much better test of whether sleep-related signal is present in specific stimuli rather than only in a single subject-average representation.
