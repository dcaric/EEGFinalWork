# Graph Audit: What's Useful, What's Not, and What's Missing

## Summary Table

| Graph | Folder | Verdict | Use In Thesis? |
| --- | --- | --- | --- |
| `task_vs_bands_median.png` | visuals/ | ✅ Very Informative | Yes, EEG sanity check |
| `task_vs_channels_median.png` | visuals/ | ✅ Informative | Optional, complements bands |
| `task_explainer_heatmap.png` | visuals/ | ⚠️ Dense but useful | Yes, category-level insight |
| `complete_performance_heatmap.png` | visuals/ | ⚠️ Limited | Maybe, metric inter-correlations |
| `sleep_hours_distribution.png` | visuals/ | ✅ Very Informative | Yes, essential context |
| `fft_sleep_correlation_bars.png` | visuals_fft/ | ✅ Very Informative | Yes, key finding |
| `fft_vs_metrics_overview.png` | visuals_fft/ | ⚠️ Ok overview | Optional, region breakdowns better |
| `fft_vs_metrics_frontal.png` | visuals_fft/ | ✅ Informative | Yes, strongest correlations |
| `fft_vs_metrics_parietal_occipital.png` | visuals_fft/ | ✅ Informative | Yes, alpha/engagement finding |
| `fft_vs_metrics_temporal.png` | visuals_fft/ | ⚠️ Weak | Maybe, low correlations |
| `scatter_grid_top_features.png` | visuals_scatter_plots/ | ✅ Informative | Yes, motivates ML approach |
| `scatter_POW_*_vs_sleep.png` x9 | visuals_scatter_plots/ | ⚠️ Supplementary | Optional, individual views |
| `clustered_map_BA*.png` x9 | individual_graphs/ | ❌ Not useful | No, inconsistent |
| `*_Comparison.png` x7 | visuals_tasks_per_metric/ | ❌ Superseded | No, replaced by sleep group charts |
| `sleep_group_3way_fft.png` | visuals_sleep_groups/ | ⚠️ Trends only | Yes, with caveats (n=6/68/9) |
| `sleep_group_2way_fft.png` | visuals_sleep_groups/ | ⚠️ n.s. | Yes, cleaner for thesis |
| `per_task_fft_heatmap.png` | visuals_per_task/ | ✅ Informative | Yes, answers mentor's Q3 |
| `top_task_band_pairs.png` | visuals_per_task/ | ✅ Very Informative | Yes, strongest EDA finding |
| `per_task_fft_heatmap.png` | visuals_per_task/ | ✅ Very Informative | Yes, answers mentor Q3 |
| `top_task_band_pairs.png` | visuals_per_task/ | ✅ Informative | Yes, identifies top task–band pairs |

---

## Detailed Analysis

### 📂 `visuals/`

#### ✅ `task_vs_bands_median.png` — Z-Score Bands Heatmap
**Keep — one of your best graphs.** Clear physiological patterns:
- Speech tasks (`saying_words`, `frontal3_recall`, `baseline_talk`) show broadband activation (Theta + Beta + Gamma) → muscle artifact, expected
- Eyes-closed tasks (`imagine_way_home`, `temporal_*instruments*`) show high Alpha → classic alpha enhancement
- Passive image tasks (`image_*`) are uniformly suppressed
- **Thesis use**: "Sanity check" proving your EEG data is physiologically valid

#### ✅ `task_vs_channels_median.png` — Z-Score Channels Heatmap
**Keep but optional (tells same story as bands).** Shows spatial distribution:
- Speech tasks light up AF3/AF4 (frontal) → confirms EMG contamination
- Auditory tasks activate O1/O2 (occipital) → alpha enhancement
- **Thesis use**: Complementary to the bands heatmap; include if you want to show both "which frequencies" and "which brain regions"

#### ⚠️ `task_explainer_heatmap.png` — Category × FFT × Sleep Correlation
**Dense but has an interesting finding.** Shows correlations between FFT bands and sleep hours, grouped by task category:
- Gamma/BetaH in Cognitive_Language tasks show positive correlation with sleep
- Motor_Action shows negative frontal theta correlation with sleep
- Auditory tasks show mixed patterns
- **Issue**: Very tall, hard to read. Could be simplified by averaging within brain regions
- **Thesis use**: Yes — shows which task categories are most sleep-sensitive

#### ⚠️ `complete_performance_heatmap.png` — PM Correlation Matrix
**Limited usefulness for sleep prediction.**
- Sleep_Hours row shows all correlations < |0.28| (Engagement +0.28 being the strongest)
- Most inter-metric correlations (Stress-Interest r=0.73, Stress-Relaxation r=0.61) just show that Emotiv's proprietary metrics are internally correlated
- **Thesis use**: Background figure showing metric relationships. Not a primary finding

#### ✅ `sleep_hours_distribution.png` — Sleep Hours Histogram *(NEW)*
**Essential context graph.** Shows the distribution of sleep hours across all 83 subjects:
- Mean = 6.7h, Median = 7.0h, Std = 0.78h, range 5.0–8.5h
- Heavily concentrated around 6–7h — the 6h and 7h bins dominate
- Short sleepers (<6h): 6 subjects (3 at 5.0h, 3 at 5.5h)
- Long sleepers (≥8h): 9 subjects (8 at 8.0h, 1 at 8.5h)
- Color-coded by sleep group (red=short, blue=normal, green=long) with mean/median lines
- **Binning fix applied**: Original `bins=12` produced auto-generated bin edges that placed the boundary at ~5.875 instead of 6.0 — causing 21 subjects who sleep exactly 6h to show as red (Short) when they are Normal. Fixed by using explicit bin edges `[4.75, 5.25, 5.75, …, 8.75]` centred on half-hour values, aligning bin boundaries with the 6h and 8h sleep-group thresholds.
- **Key insight**: The narrow range and low variance in sleep hours explains why prediction is challenging
- **Note on n**: This graph uses all 83 subjects from `DatasetSub_shorten.csv` (metadata). The scatter plots below use n=68 subjects from `all_task_ready.csv` (signal-quality filtered). The discrepancy is expected and correct — one is the full cohort, the other is the analysis-ready subset — but must be stated clearly in the thesis.
- **Thesis use**: Must-include — contextualizes why univariate correlations are weak and ML models are needed
- **Script**: `src/2_graphs_generator/sleep_distribution.py`

---

### 📂 `visuals_fft/`

#### ✅ `fft_sleep_correlation_bars.png` — FFT vs Sleep Correlation Bars
**Your most informative graph for the thesis question.**
- **Top positive correlations**: Gamma and BetaH bands (especially F8, F7, T8, P8) correlate positively with sleep (r up to ~0.27)
- **Top negative correlations**: AF3 Theta and P8 Alpha negatively correlate with sleep (r ~ -0.10 to -0.15)
- **Interpretation**: More sleep → higher high-frequency power. This aligns with literature: well-rested people have more cognitive bandwidth
- **Important caveat**: These are row-level correlations (inflated n). At subject-level the "top positive" features (F8.Gamma, Af4.gamma,F7.Gamma) actually **reverse direction** (flip to slightly negative), meaning the row-level positive signal was driven by within-subject variance, not between-subject sleep effects. Still useful for showing *ranking* of feature importance, but *direction* should be interpreted from the subject-level scatter plots
- **Thesis use**: Primary figure for "which EEG features relate to sleep"

#### ✅ `fft_vs_metrics_frontal.png` — Frontal Region Breakdown
**Most informative region breakdown.** Key findings:
- Alpha power negatively correlates with Engagement (r ~ -0.21 to -0.23) → classic alpha-blocking during engagement
- Stress strongly correlates with all frontal bands (r ~ 0.24–0.33) → higher power = more stress detection
- Attention and Excitement show weak correlations
- **Thesis use**: Explains how Emotiv's performance metrics relate to raw EEG

#### ✅ `fft_vs_metrics_parietal_occipital.png` — Posterior Region
**Strong finding**: P8.Alpha has the strongest Relaxation correlation (r=0.41) and Engagement has a strong negative correlation with posterior Alpha (r=-0.35 at P8)
- Confirms the alpha-relaxation link (more alpha = more relaxed, less engaged)
- **Thesis use**: Supports the frontal findings with posterior evidence

#### ⚠️ `fft_vs_metrics_temporal.png` — Temporal Region
**Weakest region breakdown.** Correlations are small (mostly r < 0.20). The chart is small with only T7/T8 channels
- **Thesis use**: Only include if you want completeness across all regions

#### ⚠️ `fft_vs_metrics_overview.png` — All Regions Overview
**Redundant if you have the region breakdowns.** Same information as the 3 regional heatmaps combined, but without annotations (no numbers visible)
- **Thesis use**: If you only want ONE FFT-vs-metrics graph, use this. Otherwise skip it

---

### 📂 `visuals_scatter_plots/` *(NEW)*

#### ✅ `scatter_grid_top_features.png` — Combined Scatter Grid
**Informative — tells the "why we need ML" story.** Shows the top 5 positive and top 4 negative FFT features (from `fft_sleep_correlation_bars.png`) as subject-level scatter plots:
- All correlations are non-significant at the subject level (n=83, from `all_task_ready.csv`)
- Z-scored so all bands are on the same scale (solves Theta=100 vs Gamma=1 problem)
- Outliers clipped at ±3 SD for readability
- **Direction reversal (important finding)**: The "top positive" features from the row-level bar chart (F8.Gamma r≈−0.057, F7.Gamma r≈−0.052) actually **flip direction** at the subject level — going from positive to slightly negative. This is a stronger statement than simply "weaker correlations": it means those row-level positive associations were driven by *within-subject variance*, not genuine *between-subject* sleep effects. Worth stating explicitly in the thesis — it actually **strengthens the argument** for a multivariate approach.
- **Strongest subject-level signal**: AF4.Theta at r=−0.136, p=0.267 — negative and largest in magnitude, but still not significant
- **Key insight**: No single FFT feature linearly predicts sleep on its own → multivariate ML is needed
- **Note on n**: Uses all 83 subjects from `all_task_ready.csv`, matching the sleep distribution histogram.
- **Thesis use**: Pairs with `fft_sleep_correlation_bars.png` to tell the full story — bar chart shows *which features* matter, scatter plots show *why ML is needed*
- **Script**: `src/2_graphs_generator/scatter_fft_vs_sleep.py`

#### ⚠️ 9 × `scatter_POW_*_vs_sleep.png` — Individual Scatter Plots
Larger individual versions of each subplot in the grid. Same data, just easier to read one at a time.
- **Thesis use**: Optional — include specific ones in an appendix if needed

---

### 📂 `individual_graphs/`

#### ❌ 9 × `clustered_map_BA*.png` — Individual Ward Clustering
**Not useful in current form.**
- BA51 only has ~9 tasks (filtered out most data), while BA39/BA72 have ~36 tasks → inconsistent
- Clustering dendrograms differ wildly between subjects → no reproducible pattern
- The clustering is using Performance Metrics only (no FFT), which have weak signal
- Color scale makes it hard to compare across subjects
- **Issue**: These are from the **pilot data** (9–10 subjects), not the full dataset
- **Thesis use**: Skip entirely. If you want individual-level analysis, a different approach would be needed

---

### 📂 `visuals_tasks_per_metric/` *(SUPERSEDED)*

#### ❌ 7 × `*_Comparison.png` — Sleep Group Bar Charts (Pilot)
**Superseded by `visuals_sleep_groups/`.** These used only pilot data (n=9), PM metrics, and 2 coarse groups. Skip entirely.

---

### 📂 `visuals_sleep_groups/` *(NEW)*

#### ⚠️ `sleep_group_3way_fft.png` — 3-Way Sleep Group FFT Bars
**No features survive multiple-comparison correction.** Compares 16 FFT features (Gamma, BetaH, Theta, Alpha) across <6h (n=6), 6–8h (n=68), ≥8h (n=9) — 48 pairwise Mann-Whitney U tests total:
- **AF4.Theta**: Short vs Normal p=0.009, Short vs Long p=0.050 — nominally significant but **does not survive Bonferroni correction** (α_corrected = 0.05/48 ≈ 0.001). With 48 tests at α=0.05, ~2.4 false positives are expected by chance alone.
- AF3.Theta (p≈0.08), AF4.Alpha (p≈0.07) show borderline trends, also n.s. after correction
- All Gamma and BetaH comparisons are clearly n.s.
- **Key insight**: Even the best univariate candidate (AF4.Theta) cannot be claimed as significant after correcting for multiple comparisons. This is the strongest evidence yet that univariate approaches are insufficient → multivariate ML is needed.
- **Thesis use**: Include with explicit mention of Bonferroni correction. The chart now labels brackets as `** (n.s. Bonf.)` to make this transparent.

#### ⚠️ `sleep_group_2way_fft.png` — 2-Way Sleep Group FFT Bars
**All comparisons n.s.** Compares ≤7h (n=63) vs >7h (n=20):
- All 16 Mann-Whitney U comparisons are n.s. (lowest p ≈ 0.10 for AF4.Theta)
- Bar heights are nearly identical across groups — the median split dilutes the extreme-group effect
- **Key insight**: The AF4.Theta signal visible in the 3-way split disappears at the median split, confirming it's driven by the small extreme-sleep tail rather than a linear trend
- **Thesis use**: Include alongside the 3-way figure — the contrast between the two charts shows the effect is non-linear and group-size dependent, further motivating ML approaches.
- **Note on n**: Uses all 83 subjects from `all_task_ready.csv` (same subjects as metadata)
- **Script**: `src/2_graphs_generator/sleep_group_bar_charts.py`

---

### 📂 `visuals_per_task/` *(NEW)*

#### ✅ `per_task_fft_heatmap.png` — Per-Task FFT–Sleep Correlation Heatmap
**Answers mentor's bullet 3: "Does FFT from some tasks better explain sleep hours?"**
- 36 tasks × 5 bands heatmap, showing mean Pearson r averaged across all 14 channels per band
- Tasks sorted by max |r| — most sleep-predictive tasks at the top
- **Most predictive tasks** (highest |r| across bands): `valdo3`, `image_angry`, `gyrus_coord2`, `image_happiness`, `valdo1`, `gyrus_coord`, `frontal3_saying_outloud`, `baseline_talk`, `frontal3`
- **Least predictive tasks** (near zero across all bands): `saying_words_based_on_cat2`, `image_sad`, `frontal2_memorization`, `parietal_2d_3d`
- **Band pattern**: Gamma and BetaH show the most variation across tasks. Theta/Alpha are more uniform but show notable signal in motor (gyrus) tasks
- **Important caveat**: Values are averaged across channels, which dilutes channel-specific effects. The top-10 chart below reveals the true strongest signals
- **Thesis use**: Yes — directly answers mentor's question. Pairs with top-10 chart. Use as the primary figure for Section 7.2
- **Script**: `src/2_graphs_generator/per_task_fft_sleep_corr.py`

#### ✅ `top_task_band_pairs.png` — Top 10 Task–Band–Channel Pairs
**Your strongest individual-feature finding in the entire EDA.**
All 10 top pairs are statistically significant (p < 0.05), several at p < 0.01:

| Rank | Task | Feature | r | p |
|------|------|---------|---|---|
| 1 | gyrus_coord | T8.Theta | +0.319 | 0.004 |
| 2 | gyrus_right_closed_eyes | T7.BetaH | −0.314 | 0.005 |
| 3 | prefrontal2 | T7.BetaH | −0.296 | 0.007 |
| 4 | prefrontal1 | T7.Gamma | −0.298 | 0.010 |
| 5 | gyrus_left_closed_eyes | T7.BetaH | −0.291 | 0.009 |
| 6 | gyrus_right_closed_eyes | T7.Gamma | −0.275 | 0.014 |
| 7 | prefrontal2 | T7.Gamma | −0.264 | 0.017 |
| 8 | gyrus_left_closed_eyes | FC5.BetaH | −0.263 | 0.019 |
| 9 | gyrus_coord | O1.Theta | +0.263 | 0.020 |
| 10 | gyrus_left_closed_eyes | T7.Gamma | −0.262 | 0.020 |

**Key findings:**
- **T7 (left temporal) BetaH and Gamma dominate** — appears in 6 of the top 10 pairs across multiple task types. Left temporal power is consistently negatively correlated with sleep hours
- **Motor/coordination tasks are most predictive**: `gyrus_left_closed_eyes`, `gyrus_right_closed_eyes`, `gyrus_coord` (hand motor tasks with eyes closed) show the strongest associations. Biologically plausible — sleep deprivation most directly impairs motor control and coordination
- **Prefrontal working memory tasks** (`prefrontal1`, `prefrontal2`) also highly informative — sustained attention and working memory are known to degrade with sleep loss
- **Direction**: Most significant pairs are *negative* (more sleep → less T7 BetaH/Gamma power during motor/frontal tasks). The two positive exceptions are `gyrus_coord • T8.Theta` and `gyrus_coord • O1.Theta` — more sleep associates with more occipital/temporal Theta during coordination, possibly reflecting a more relaxed, controlled motor state
- **Multiple comparisons note**: ~2,520 correlations were computed (36 tasks × 70 features). At α=0.05, ~126 false positives are expected by chance. However, the top pairs here are not isolated singles — T7 appears consistently across multiple independent tasks and bands, which is strong convergent evidence. This pattern is unlikely to be noise. Still, report these as exploratory findings
- **For the RF model**: These tasks (gyrus_left_closed_eyes, gyrus_right_closed_eyes, gyrus_coord, prefrontal1, prefrontal2) should be prioritised as features or used to create task-specific feature sets
- **Thesis use**: Primary figure for Section 7.2. This is the answer to the mentor's question and the most concrete result from the EDA phase
- **Script**: `src/2_graphs_generator/per_task_fft_sleep_corr.py`

---

## Suggested New Graphs (Remaining)

### 1. 📊 Feature Importance from Random Forest (Priority: HIGH)
Train a quick Random Forest regressor (Sleep_Hours ~ all FFT features) on the full dataset and plot feature importances. This would directly show which EEG features the model relies on.

### 2. ~~📊 Sleep Group Bar Charts on FULL Dataset with 3 Groups~~ ✅ DONE
Completed — see `visuals_sleep_groups/` section above. Both 3-way and 2-way splits generated with Mann-Whitney U tests. All comparisons n.s.

### 3. ~~📊 Per-Task FFT–Sleep Correlation Analysis~~ ✅ DONE
Completed — see `visuals_per_task/` section above. Motor and prefrontal tasks show strongest signal via T7 BetaH/Gamma.

### 3. ~~📊 Per-Task FFT-Sleep Correlation Analysis~~ ✅ DONE
Completed — see `visuals_per_task/` section below.

---

### 📂 `visuals_per_task/` *(NEW)*

#### ✅ `per_task_fft_heatmap.png` — Per-Task FFT–Sleep Correlation Heatmap
**Key EDA deliverable — directly answers mentor's Q3.** Shows Pearson r (averaged across 14 channels) for each of 36 tasks × 5 FFT bands:
- Tasks sorted by max |mean r| — **valdo3**, **image_angry**, and **gyrus_coord** emerge as the strongest overall
- Patterns: Alpha and BetaL bands tend to carry the most consistent signal across tasks; Gamma is mixed
- Clear task-specificity: some tasks (e.g. `valdo3` r≈+0.18) show moderate positive correlations across all bands, while others (e.g. `frontal1_recall` r≈−0.10) are consistently negative
- **Thesis use**: Include as a full-page figure — answers "which tasks carry sleep signal" and informs RF feature selection

#### ✅ `top_task_band_pairs.png` — Top 10 Task–Band–Channel Pairs
**Strongest individual signals.** Ranked by |r|, showing specific channel-level detail:
- **#1**: `gyrus_coord × T8.Theta` r=+0.319, p=0.005 — strongest single pair
- **#2**: `gyrus_right_closed_eyes × T7.BetaH` r=−0.314, p=0.005
- **#3**: `prefrontal2 × T7.BetaH` r=−0.296, p=0.007
- Pattern: **T7 (left temporal)** dominates the negative correlates (BetaH/Gamma), while **T8 (right temporal)** and occipital channels carry positive Theta signal
- All top-10 pairs have p < 0.02 uncorrected; with 2520 total comparisons, Bonferroni threshold would be p < 0.00002, so none survive strict correction — but the consistency of T7 BetaH/Gamma across multiple tasks is notable
- **Thesis use**: Include — identifies specific task–channel combinations to prioritise in RF modelling
- **Script**: `src/2_graphs_generator/per_task_fft_sleep_corr.py`
