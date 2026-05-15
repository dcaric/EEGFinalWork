# Thesis Summary

## Modeling Summary

Several modeling approaches were tested on the current EEG dataset.

- `RandomForestRegressor` on subject-averaged FFT features:
  - gave the best regression result among the main thesis models
  - only slightly outperformed the dummy baseline
  - indicated only weak predictive signal for exact `Sleep_Hours`

- `RandomForestClassifier` for 3 classes (`Short`, `Normal`, `Long`):
  - failed to detect minority classes
  - behaved almost like a majority-class predictor
  - showed that class imbalance was a major problem

- Binary classification (`Short` vs `NotShort`):
  - `RandomForest` still failed to detect short sleepers
  - simplifying the target alone did not solve the problem

- Small-sample model comparison:
  - `SVR`, `Ridge`, `ElasticNet`, `LogisticRegression`, `SVC`
  - `LogRegBalanced` was the only classifier that detected any short sleepers
  - this suggested that some weak structure exists, but not enough for a reliable classifier

- Tuned logistic regression with threshold search:
  - did not improve the result
  - increased false positives without creating a useful classifier
  - confirmed that tuning alone could not rescue the dataset

- Grouped regression using all rows from `all_task_ready.csv`:
  - used all task rows with `GroupKFold` by `Subject_ID`
  - did not improve performance
  - confirmed that more rows from the same people do not add more independent signal

- Subject-level clustering:
  - `KMeans` and `AgglomerativeClustering`
  - found some structure in EEG feature space
  - but clusters did not separate sleep hours meaningfully
  - showed that dominant structure in the data is not organized around sleep duration

- Balanced and normalized analyses from `subject_normalized_analysis`:
  - balancing the number of subjects above and below 6 hours still did not improve model quality
  - this strongly suggested that the issue is not only class imbalance

- Synthetic balancing experiment on `all_task_ready.csv`:
  - generated additional synthetic `<7h` rows to match the number of `>=7h` rows
  - improved regression and short-sleeper classification metrics numerically
  - suggested that sample sparsity in the low-sleep region was an important bottleneck
  - however, these gains cannot be treated as equivalent to real-world generalization because the added samples were artificial

## What These Models Showed

Together, the models showed:

- the data contain at most weak sleep-related signal
- no tested model produced robust prediction of sleep duration
- failure was consistent across regression, classification, balancing, tuning, and clustering
- this consistency strengthens the conclusion, because it does not depend on one weak modeling choice

An additional synthetic-balancing experiment showed that performance can improve when the low-sleep region is artificially densified. This does not overturn the main conclusion from the real data, but it does suggest that low-sleep underrepresentation was one of the key practical constraints.

## Data Used

The main data source was:

- `pilot_files/all_task_ready.csv`

This dataset contains:

- `83` subjects
- approximately `36` tasks per subject in most cases
- `70` FFT power features (`POW.*`)
- `7` performance and mood features (`PM.*`)
- subject-level `Sleep_Hours`
- `Gender`

Several derived modeling tables were created from this source:

- FFT-only subject averages
- FFT plus gender
- top-task subject averages
- grouped all-task supervised versions
- normalized and balanced subject-level analyses

An additional exploratory dataset was also created:

- `pilot_files/all_task_ready_synthetic.csv`

This file was generated from `all_task_ready.csv` by:

- splitting rows into `<7h` and `>=7h`
- keeping all original rows
- generating synthetic rows only for the `<7h` group until row counts matched
- sampling EEG/PM numeric features within the observed `<7h` min/max ranges
- sampling `Task` and `Gender` from observed `<7h` rows
- sampling `Sleep_Hours` uniformly between `5.0` and `6.9`

The synthetic balancing summary was:

- original rows: `2830`
- original `<7h` rows: `1197`
- original `>=7h` rows: `1633`
- synthetic `<7h` rows added: `436`
- final rows: `3266`
- final class counts:
  - `<7h`: `1633`
  - `>=7h`: `1633`

## What Was Good About the Data

- the preprocessing pipeline was structurally coherent
- task segmentation appears to be largely correct after filtering
- the dataset captures many tasks per subject
- FFT features were complete and usable
- the work established a reproducible modeling and evaluation pipeline
- the preprocessing audit showed that the original preparation logic was methodologically limited, but not broken

## What Was Not Good for Prediction

The main limitations were:

- small number of independent subjects: only `83`
- severe class imbalance:
  - very few short sleepers
  - very few long sleepers
- narrow target range:
  - most values are concentrated around `6–7` hours
- coarse target:
  - self-reported habitual sleep duration, not objective sleep measurement
- repeated rows per subject do not add true sample independence
- heavy feature compression:
  - full task segments were reduced to mean summaries
- mismatch between task EEG and habitual sleep label may be present
- averaged band-power features may be too simple to capture subtle sleep-related effects

These limitations prevented the models from becoming more accurate.

In addition, synthetic balancing showed that the model can improve when more low-sleep-like samples are present, but because these samples were artificially generated, that improvement cannot be interpreted as proof that the original real dataset contained strong recoverable biological signal.

## Synthetic Balancing Experiment

To test whether underrepresentation of lower-sleep rows was a major bottleneck, a supplementary synthetic-balancing experiment was conducted using `all_task_ready.csv`.

### Procedure

- `all_task_ready.csv` was split into two groups:
  - `<7h`
  - `>=7h`
- New synthetic rows were generated only for the `<7h` group
- Each synthetic row used:
  - EEG and PM values sampled within the observed `<7h` min/max range for each numeric feature
  - `Task` sampled from real `<7h` rows
  - `Gender` sampled from real `<7h` rows
  - `Sleep_Hours` sampled uniformly from `5.0` to `6.9`
- The resulting file was saved as:
  - `pilot_files/all_task_ready_synthetic.csv`

### Synthetic Data Generation Script

The synthetic balancing file was generated with:

- `src/3_modelling/generate_synthetic_balanced_all_task.py`

The script used the following logic:

- input file: `pilot_files/all_task_ready.csv`
- split rows into:
  - `<7h`
  - `>=7h`
- keep all original rows
- generate only as many new `<7h` rows as needed to match the `>=7h` count
- for every synthetic row:
  - sample each `POW.*` and `PM.*` feature uniformly within the observed `<7h` min/max range for that feature
  - sample `Task` from real `<7h` rows
  - sample `Gender` from real `<7h` rows
  - sample `Sleep_Hours` uniformly from `5.0` to `6.9`
  - assign a synthetic subject identifier of the form `SYNTH_LOW_...`

The exact summary printed by the generator was:

```text
Input file: /Users/dcaric/Working/pythonWorking/PaolaFinalWork/pilot_files/all_task_ready.csv
Original rows: 2830
Original <7h rows: 1197
Original >=7h rows: 1633
Synthetic <7h rows generated: 436
Final rows: 3266

Method:
- Numeric EEG/PM features sampled within observed <7h min/max ranges
- Task sampled from observed <7h rows
- Gender sampled from observed <7h rows
- Sleep_Hours sampled uniformly from 5.0 to 6.9

Final class counts:
<7h rows: 1633
>=7h rows: 1633
```

### What Happened After Rebuilding the Modeling Tables

For the synthetic experiment, `src/3_modelling_scripts/modelling_tables.py` was temporarily switched from:

```python
df = pd.read_csv('pilot_files/all_task_ready.csv')
```

to:

```python
df = pd.read_csv('pilot_files/all_task_ready_synthetic.csv')
```

After rebuilding the modeling tables, the script printed:

```text
Held out: ['BA16', 'BA45'] — training on 517 subjects
Exp1: (517, 72)
Exp2: (517, 73)
Exp3: 145 subjects have at least one of the top-5 tasks
Exp3: (145, 72)
Exp3 NaN check: 0 NaN values
```

These counts are larger because synthetic rows were assigned synthetic subject IDs.
Therefore, these are not real independent human subjects.

### Comparison Before and After Synthetic Balancing

For the original real-data `exp3_task_specific` experiment:

- subjects: `81`
- regression:
  - null RMSE: `0.7672`
  - RF RMSE: `0.7474`
  - RF MAE: `0.6119`
  - RF R²: `0.0508`
- classification:
  - accuracy: `0.8272`
  - macro F1: `0.3018`
  - `Short` precision: `0.00`
  - `Short` recall: `0.00`
  - `Short` F1: `0.00`

The original real-data console output was:

```text
Loaded exp3_task_specific: (81, 72)
Features: 70  |  Subjects: 81
Sleep hours — mean=6.73, std=0.77, range=[5.0, 8.5]
Categories  — {'Long': np.int64(8), 'Normal': np.int64(67), 'Short': np.int64(6)}

Selecting top 15 features...
Top 15 features: ['POW.F4.Theta', 'POW.AF3.BetaH', 'POW.F4.Alpha', 'POW.T8.Theta', 'POW.P7.Gamma', 'POW.AF3.Theta', 'POW.T7.BetaH', 'POW.AF3.BetaL', 'POW.FC5.BetaH', 'POW.T7.Gamma', 'POW.AF4.Theta', 'POW.AF3.Alpha', 'POW.F3.Theta', 'POW.AF4.BetaH', 'POW.T7.BetaL']

=======================================================
  REGRESSION  (5-fold CV)
=======================================================
  Fold 1/5: n_test=17, RMSE=0.699h
  Fold 2/5: n_test=16, RMSE=0.650h
  Fold 3/5: n_test=16, RMSE=0.760h
  Fold 4/5: n_test=16, RMSE=0.999h
  Fold 5/5: n_test=16, RMSE=0.557h

  Null baseline RMSE : 0.7672 h
  RF model RMSE      : 0.7474 h
  RF model MAE       : 0.6119 h
  RF model R²        : 0.0508

=======================================================
  CLASSIFICATION  (5-fold stratified CV)
=======================================================
  Fold 1/5: n_test=17, accuracy=0.824
  Fold 2/5: n_test=16, accuracy=0.875
  Fold 3/5: n_test=16, accuracy=0.812
  Fold 4/5: n_test=16, accuracy=0.812
  Fold 5/5: n_test=16, accuracy=0.812

  Accuracy        : 0.8272
  F1 (macro)      : 0.3018
  F1 (weighted)   : 0.7489

              precision    recall  f1-score   support

       Short       0.00      0.00      0.00         6
      Normal       0.83      1.00      0.91        67
        Long       0.00      0.00      0.00         8
```

For the synthetic-balanced `exp3_task_specific` experiment:

- subjects reported by the table: `145`
- regression:
  - null RMSE: `0.7738`
  - RF RMSE: `0.7069`
  - RF MAE: `0.5685`
  - RF R²: `0.1654`
- classification:
  - accuracy: `0.6897`
  - macro F1: `0.3685`
  - `Short` precision: `0.44`
  - `Short` recall: `0.23`
  - `Short` F1: `0.30`

The synthetic-balanced console output was:

```text
Loaded exp3_task_specific: (145, 72)
Features: 70  |  Subjects: 145
Sleep hours — mean=6.42, std=0.77, range=[5.0, 8.5]
Categories  — {'Long': np.int64(8), 'Normal': np.int64(102), 'Short': np.int64(35)}

Selecting top 15 features...
Top 15 features: ['POW.AF3.BetaH', 'POW.T7.Gamma', 'POW.AF4.BetaH', 'POW.T7.BetaH', 'POW.AF3.BetaL', 'POW.F3.BetaH', 'POW.AF3.Theta', 'POW.T7.BetaL', 'POW.P7.Gamma', 'POW.AF3.Alpha', 'POW.F4.Theta', 'POW.AF4.Theta', 'POW.F7.BetaH', 'POW.F7.Gamma', 'POW.FC5.Gamma']

=======================================================
  REGRESSION  (5-fold CV)
=======================================================
  Fold 1/5: n_test=29, RMSE=0.657h
  Fold 2/5: n_test=29, RMSE=0.644h
  Fold 3/5: n_test=29, RMSE=0.572h
  Fold 4/5: n_test=29, RMSE=0.842h
  Fold 5/5: n_test=29, RMSE=0.784h

  Null baseline RMSE : 0.7738 h
  RF model RMSE      : 0.7069 h
  RF model MAE       : 0.5685 h
  RF model R²        : 0.1654

=======================================================
  CLASSIFICATION  (5-fold stratified CV)
=======================================================
  Fold 1/5: n_test=29, accuracy=0.724
  Fold 2/5: n_test=29, accuracy=0.793
  Fold 3/5: n_test=29, accuracy=0.621
  Fold 4/5: n_test=29, accuracy=0.655
  Fold 5/5: n_test=29, accuracy=0.655

  Accuracy        : 0.6897
  F1 (macro)      : 0.3685
  F1 (weighted)   : 0.6381

              precision    recall  f1-score   support

       Short       0.44      0.23      0.30        35
      Normal       0.72      0.90      0.80       102
        Long       0.00      0.00      0.00         8
```

### Interpretation

The synthetic-balanced experiment improved the numerical results, especially:

- regression RMSE improved from `0.7474` to `0.7069`
- regression R² improved from `0.0508` to `0.1654`
- the `Short` class was no longer completely ignored

This suggests that underrepresentation of low-sleep rows was indeed one of the important bottlenecks.

However, the experiment must be interpreted carefully:

- the added samples were artificial
- the model was exposed to synthetically densified low-sleep feature space
- the larger reported subject count does not represent more real people
- the experiment does not prove that the original dataset contained enough real independent biological signal for this level of performance

Therefore, this experiment should be treated as a sensitivity analysis rather than proof of real-world generalization.

## Why This Work Still Brings Value

Even though the work did not produce a final strong predictor, it still makes a meaningful contribution.

It showed:

- which modeling strategies were worth testing
- which ones failed and why
- that the main issue is likely data informativeness, not only algorithm choice
- that proper grouped validation is necessary to avoid misleading results
- that non-task markers must be filtered carefully
- that unsupervised structure exists, but not in a way that aligns with sleep duration
- that synthetic balancing can improve metrics, which supports the view that low-sleep sample sparsity was a real limitation

This work therefore reduces uncertainty for future development. It narrows the search space and prevents future work from repeating the same ineffective directions.

## Future Steps Toward Better Models

To move toward a better final model, future work should focus on:

- collecting more independent subjects
- increasing the number of short- and long-sleep participants
- using more precise labels:
  - actigraphy
  - polysomnography-derived total sleep time
  - repeated nightly sleep measures
- extracting richer EEG features:
  - variance
  - temporal dynamics
  - connectivity
  - asymmetry
  - within-task variability
- testing task-specific rather than only global averages
- preserving more within-task information instead of using only means
- aligning EEG recording more closely with the sleep-measurement window
- exploring larger external datasets if available
- testing whether improvements seen under synthetic balancing can be reproduced with genuinely new real low-sleep participants

## Overall Thesis Message

This work rigorously evaluated whether task-based EEG summary features can predict habitual sleep duration. Across multiple preprocessing choices and modeling strategies, prediction on the real dataset remained weak and close to baseline. A supplementary synthetic-balancing experiment improved the numerical results, suggesting that sparsity in the low-sleep region was one important bottleneck, but these gains cannot be interpreted as equivalent to generalization on new real subjects. Taken together, the findings indicate that the present dataset and feature representation are insufficient for robust final sleep prediction, while still providing a valuable methodological foundation for future work by identifying the main bottlenecks, validating the preprocessing logic, and clarifying what improvements are required to reach a stronger final model.
