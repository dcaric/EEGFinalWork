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

## What These Models Showed

Together, the models showed:

- the data contain at most weak sleep-related signal
- no tested model produced robust prediction of sleep duration
- failure was consistent across regression, classification, balancing, tuning, and clustering
- this consistency strengthens the conclusion, because it does not depend on one weak modeling choice

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

## Why This Work Still Brings Value

Even though the work did not produce a final strong predictor, it still makes a meaningful contribution.

It showed:

- which modeling strategies were worth testing
- which ones failed and why
- that the main issue is likely data informativeness, not only algorithm choice
- that proper grouped validation is necessary to avoid misleading results
- that non-task markers must be filtered carefully
- that unsupervised structure exists, but not in a way that aligns with sleep duration

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

## Overall Thesis Message

This work rigorously evaluated whether task-based EEG summary features can predict habitual sleep duration. Across multiple preprocessing choices and modeling strategies, prediction remained weak and close to baseline. This indicates that the present dataset and feature representation are insufficient for robust sleep prediction, but the study still provides a valuable methodological foundation for future work by identifying the main bottlenecks, validating the preprocessing logic, and clarifying what improvements are required to reach a stronger final model.
