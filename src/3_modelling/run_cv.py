"""
K-Fold Cross-Validation for Random Forest sleep prediction.
Runs both regression (predict hours) and classification (predict category).

HOW TO USE:
  1. Set EXPERIMENT below to choose which feature table to use
  2. Set K for number of folds (5 recommended)
  3. Run: python run_cv.py

EXPERIMENTS:
  'exp1_fft_only'    → 70 FFT features (14 channels × 5 bands)
  'exp2_fft_gender'  → 70 FFT features + Gender
  'exp3_task_specific' → FFT from 5 specific tasks only (build table first)

SLEEP CATEGORIES (for classification):
  Short  → Sleep_Hours < 6
  Normal → 6 <= Sleep_Hours <= 8
  Long   → Sleep_Hours > 8

OUTPUTS (saved to results/<experiment>/):
  Regression:
    _regression_predictions.csv  — Subject_ID, Actual, Predicted, Error
    _regression_metrics.txt      — RMSE, MAE, R², null baseline
    _regression_scatter.png      — predicted vs actual scatter plot

  Classification:
    _classification_predictions.csv — Subject_ID, Actual_hours, Actual_cat, Predicted_cat
    _classification_metrics.txt     — accuracy, F1 per class, confusion matrix
    _classification_confusion.png   — confusion matrix heatmap

  Both:
    _feature_importance.png      — top-20 feature importances
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from pathlib import Path

# ── SETTINGS — change these ───────────────────────────────────────────────────
EXPERIMENT = 'exp3_task_specific'   # ← change to switch experiments
K          = 5                  # ← number of folds (3 or 5)
# ─────────────────────────────────────────────────────────────────────────────

# Sleep category bins
BINS   = [0, 6, 8, 99]
LABELS = ['Short', 'Normal', 'Long']

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'modelling_tables'
OUT_DIR  = BASE_DIR / 'results' / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / f'{EXPERIMENT}.csv')
print(f"Loaded {EXPERIMENT}: {df.shape}")

feature_cols = [c for c in df.columns if c not in ['Subject_ID', 'Sleep_Hours']]
X       = df[feature_cols].values
y_hours = df['Sleep_Hours'].values
y_cat   = pd.cut(y_hours, bins=BINS, labels=LABELS, right=False).astype(str)

print(f"Features: {len(feature_cols)}  |  Subjects: {len(y_hours)}")
print(f"Sleep hours — mean={y_hours.mean():.2f}, std={y_hours.std():.2f}, range=[{y_hours.min()}, {y_hours.max()}]")
print(f"Categories  — {dict(zip(*np.unique(y_cat, return_counts=True)))}")

# ── Feature selection (optional) ──────────────────────────────────────────────
# Set TOP_N = None to use all features, or an integer to keep only the top-N
# most important features (ranked by RF trained on the full dataset).
# Note: feature importance is computed on all data before CV — this is a
# slight optimistic bias but acceptable for exploratory/thesis work.
TOP_N = 15   # ← change to e.g. 15 to enable feature selection

if TOP_N is not None:
    print(f"\nSelecting top {TOP_N} features...")
    selector = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                                     random_state=42, n_jobs=-1)
    selector.fit(X, y_hours)
    top_idx    = np.argsort(selector.feature_importances_)[::-1][:TOP_N]
    X          = X[:, top_idx]
    feature_cols = [feature_cols[i] for i in top_idx]
    print(f"Top {TOP_N} features: {feature_cols}")
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — REGRESSION (predict exact sleep hours)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  REGRESSION  ({K}-fold CV)")
print(f"{'='*55}")

kf = KFold(n_splits=K, shuffle=True, random_state=42)
reg_preds = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test   = X[train_idx], X[test_idx]
    y_train, y_test   = y_hours[train_idx], y_hours[test_idx]
    subj_test         = df['Subject_ID'].values[test_idx]

    model = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                                  random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    for sid, actual, pred in zip(subj_test, y_test, y_pred):
        reg_preds.append({'Subject_ID': sid, 'Actual': actual,
                          'Predicted': pred, 'Error': pred - actual})

    fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  Fold {fold+1}/{K}: n_test={len(y_test)}, RMSE={fold_rmse:.3f}h")

reg_df  = pd.DataFrame(reg_preds)
actuals = reg_df['Actual'].values
preds   = reg_df['Predicted'].values

rmse      = np.sqrt(mean_squared_error(actuals, preds))
mae       = mean_absolute_error(actuals, preds)
r2        = r2_score(actuals, preds)
null_rmse = actuals.std()

print(f"\n  Null baseline RMSE : {null_rmse:.4f} h")
print(f"  RF model RMSE      : {rmse:.4f} h")
print(f"  RF model MAE       : {mae:.4f} h")
print(f"  RF model R²        : {r2:.4f}")

# Save regression results
reg_df.to_csv(OUT_DIR / f'{EXPERIMENT}_regression_predictions.csv', index=False)
(OUT_DIR / f'{EXPERIMENT}_regression_metrics.txt').write_text(
    f"Experiment : {EXPERIMENT}\n"
    f"Mode       : Regression ({K}-fold CV)\n"
    f"N subjects : {len(actuals)}\n"
    f"N features : {len(feature_cols)}\n\n"
    f"Null baseline RMSE : {null_rmse:.4f} h\n"
    f"RF model RMSE      : {rmse:.4f} h\n"
    f"RF model MAE       : {mae:.4f} h\n"
    f"RF model R²        : {r2:.4f}\n"
)

# Scatter plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(actuals, preds, alpha=0.7, edgecolors='steelblue',
           facecolors='lightblue', s=70)
mn, mx = min(actuals.min(), preds.min()), max(actuals.max(), preds.max())
ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect prediction')
ax.set_xlabel('Actual Sleep Hours')
ax.set_ylabel('Predicted Sleep Hours')
ax.set_title(f'{EXPERIMENT} — Regression\nRMSE={rmse:.3f}h   R²={r2:.3f}   ({K}-fold CV)')
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / f'{EXPERIMENT}_regression_scatter.png', dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — CLASSIFICATION (predict Short / Normal / Long)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  CLASSIFICATION  ({K}-fold stratified CV)")
print(f"{'='*55}")

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
cls_preds = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_cat)):
    X_train, X_test   = X[train_idx], X[test_idx]
    yc_train, yc_test = y_cat[train_idx], y_cat[test_idx]
    subj_test         = df['Subject_ID'].values[test_idx]
    hours_test        = y_hours[test_idx]

    clf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train, yc_train)
    yc_pred = clf.predict(X_test)

    for sid, hrs, actual, pred in zip(subj_test, hours_test, yc_test, yc_pred):
        cls_preds.append({'Subject_ID': sid, 'Actual_hours': hrs,
                          'Actual_cat': actual, 'Predicted_cat': pred})

    fold_acc = accuracy_score(yc_test, yc_pred)
    print(f"  Fold {fold+1}/{K}: n_test={len(yc_test)}, accuracy={fold_acc:.3f}")

cls_df     = pd.DataFrame(cls_preds)
y_true_cat = cls_df['Actual_cat'].values
y_pred_cat = cls_df['Predicted_cat'].values

acc    = accuracy_score(y_true_cat, y_pred_cat)
f1_mac = f1_score(y_true_cat, y_pred_cat, average='macro', zero_division=0)
f1_wei = f1_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
cm     = confusion_matrix(y_true_cat, y_pred_cat, labels=LABELS)
report = classification_report(y_true_cat, y_pred_cat, labels=LABELS, zero_division=0)

print(f"\n  Accuracy        : {acc:.4f}")
print(f"  F1 (macro)      : {f1_mac:.4f}")
print(f"  F1 (weighted)   : {f1_wei:.4f}")
print(f"\n{report}")
print(f"  Note: class_weight='balanced' used to handle Short/Long class imbalance")

# Save classification results
cls_df.to_csv(OUT_DIR / f'{EXPERIMENT}_classification_predictions.csv', index=False)
(OUT_DIR / f'{EXPERIMENT}_classification_metrics.txt').write_text(
    f"Experiment : {EXPERIMENT}\n"
    f"Mode       : Classification ({K}-fold stratified CV)\n"
    f"Categories : Short (<6h), Normal (6-8h), Long (>8h)\n"
    f"N subjects : {len(y_true_cat)}\n"
    f"N features : {len(feature_cols)}\n\n"
    f"Accuracy   : {acc:.4f}\n"
    f"F1 (macro) : {f1_mac:.4f}\n"
    f"F1 (weighted): {f1_wei:.4f}\n\n"
    f"Classification report:\n{report}\n"
    f"Confusion matrix (rows=actual, cols=predicted):\n"
    f"Labels: {LABELS}\n{cm}\n"
)

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title(f'{EXPERIMENT} — Classification\nAccuracy={acc:.3f}   F1(macro)={f1_mac:.3f}')
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=13)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(OUT_DIR / f'{EXPERIMENT}_classification_confusion.png', dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE (train on all data — same for both modes)
# ══════════════════════════════════════════════════════════════════════════════
model_full = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                                   random_state=42, n_jobs=-1)
model_full.fit(X, y_hours)
importance = pd.Series(model_full.feature_importances_,
                        index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(11, 5))
top20 = importance.head(20)
ax.bar(range(len(top20)), top20.values, color='steelblue')
ax.set_xticks(range(len(top20)))
ax.set_xticklabels(top20.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Feature Importance')
ax.set_title(f'{EXPERIMENT} — Top 20 Feature Importances (trained on all {len(y_hours)} subjects)')
fig.tight_layout()
fig.savefig(OUT_DIR / f'{EXPERIMENT}_feature_importance.png', dpi=150)
plt.close()

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")
