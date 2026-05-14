# Conclusion

## What should be reported in the main thesis text

The strongest and most defensible results to present in the main thesis are:

- Regression on sleep hours using the `exp3_task_specific` feature table with Random Forest
- Binary classification (`Short` vs `NotShort`) using balanced logistic regression from `compare_small_sample_models.py`

These two results best represent the dataset because:

- the regression task preserves the continuous sleep-hours target and showed the clearest, although still weak, predictive signal
- the balanced logistic regression was the only tested classifier that detected any short sleepers at all

### Recommended main regression result

From `src/3_modelling/run_cv.py` and `src/3_modelling/compare_small_sample_models.py`:

- Model: Random Forest Regressor
- RMSE: `0.7474`
- MAE: `0.6119`
- R²: `0.0508`
- Null baseline RMSE: `0.7672`

Interpretation:

The regression model performed slightly better than a trivial baseline that predicts the mean sleep duration for every subject. This indicates that the EEG-derived features contained a small amount of predictive information, but the overall explanatory power remained very limited, with the model accounting for only about 5% of the variance in sleep hours.

### Recommended main classification result

From `src/3_modelling/compare_small_sample_models.py`:

- Model: Balanced Logistic Regression
- Accuracy: `0.8642`
- Precision for `Short`: `0.2222`
- Recall for `Short`: `0.3333`
- F1-score for `Short`: `0.2667`

Interpretation:

Balanced logistic regression was the only tested classifier that identified any short sleepers. It detected approximately 2 of the 6 short-sleep subjects. However, precision was low, meaning that many predicted short-sleep cases were incorrect. This shows that some weak structure may be present in the data, but not enough for reliable sleep-group classification.

## What should not be emphasized as main results

The following findings are useful as supporting evidence, but should not be the headline results:

- The original 3-class classification (`Short`, `Normal`, `Long`) from `src/3_modelling/run_cv.py`
- The binary Random Forest classifier from `src/3_modelling/run_cv_binary.py`
- The tuned logistic regression threshold search from `src/3_modelling/tune_logreg_short_detection.py`
- The weaker regression baselines such as Ridge and ElasticNet from `src/3_modelling/compare_small_sample_models.py`

Why these should be secondary:

- the 3-class classifier collapsed to majority-class prediction and failed to detect minority classes
- the binary Random Forest also failed to detect any short sleepers
- the tuned logistic regression produced worse performance than the simpler untuned balanced logistic regression
- the weaker regression baselines mainly confirm that algorithm changes alone do not solve the problem

These results are still valuable because they strengthen the conclusion that the main limitation is the dataset rather than a poor choice of model.

## Core findings

The main findings across all experiments are:

- The dataset contains only weak predictive signal for sleep hours.
- Regression performs slightly better than a null baseline, but only marginally.
- Classification is strongly limited by class imbalance.
- Model choice can help a little, but it does not solve the main problem.
- Splitting one subject into smaller chunks would not create more independent subject-level samples and would risk data leakage if not handled very carefully.

### Why the models struggled

The weak model performance is best explained by three factors:

1. Small sample size

Only 81 subjects were available for training and cross-validation after excluding held-out subjects.

2. Severe class imbalance

For the binary task there were:

- `6` short sleepers
- `75` not-short sleepers

For the 3-class task there were:

- `6` short sleepers
- `67` normal sleepers
- `8` long sleepers

This means the minority classes were too small for stable learning.

3. Limited target variation

Sleep hours were concentrated in a narrow range:

- mean: `6.73`
- standard deviation: `0.77`
- range: `5.0` to `8.5`

This reduces the amount of variation available for the model to learn.

## Overall conclusion

The dataset is not useless, but it is not strong enough to support reliable prediction of sleep duration or sleep category with the current subject-level EEG feature set and sample size.

The best regression model showed only a slight improvement over baseline, indicating that a small amount of signal exists. The best classifier, balanced logistic regression, was able to identify some short sleepers, but performance remained weak and unstable. More complex models and threshold tuning did not materially improve the outcome.

Therefore, the most appropriate conclusion is that the main limitation lies in the dataset, especially the low number of independent subjects, severe class imbalance, and limited discriminative signal in the extracted features.

## Thesis-ready conclusion paragraph

Although the dataset enabled exploratory analysis and limited subject-level prediction, the modelling results showed that the available EEG-derived features were not sufficiently discriminative for reliable sleep prediction. Random Forest regression slightly outperformed a null baseline, suggesting only weak predictive signal for continuous sleep hours. For classification, balanced logistic regression was the only tested model that detected any short sleepers, but performance remained modest, with low precision and only partial recall of the minority class. Additional modelling strategies, including binary reformulation, alternative algorithms, and threshold tuning, did not substantially improve results. Taken together, these findings indicate that the main constraints were the small number of independent subjects, the severe imbalance between sleep groups, and the limited strength of the extracted EEG features as predictors of sleep duration.

## Practical recommendation for future work

Future work should focus more on data quality and dataset scale than on trying increasingly complex models. The most important improvements would be:

- collecting substantially more subjects
- increasing the number of short- and long-sleep participants
- testing alternative feature engineering strategies
- validating on larger public datasets if a compatible dataset can be identified

As a practical target, a much stronger study design for classification would require at least dozens of subjects in each class, and ideally well over 150 total balanced subjects.
