"""
Binary K-Fold Cross-Validation for sleep prediction.

This script is separate from run_cv.py so the original 3-class thesis pipeline
remains unchanged. It focuses on a simpler and more realistic question for this
dataset:

  Short vs Not-Short

Definition:
  Short     -> Sleep_Hours < 6
  NotShort  -> Sleep_Hours >= 6

Why this exists:
  The 3-class setup is heavily imbalanced, so it is useful to test whether the
  model can at least identify the low-sleep group in a binary setting.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold


# ── SETTINGS ──────────────────────────────────────────────────────────────────
EXPERIMENT = "exp3_task_specific"
K = 5
TOP_N = 15
POSITIVE_LABEL = "Short"
NEGATIVE_LABEL = "NotShort"
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "modelling_tables"
OUT_DIR = BASE_DIR / "results" / f"{EXPERIMENT}_binary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_binary_labels(y_hours: np.ndarray) -> np.ndarray:
    return np.where(y_hours < 6, POSITIVE_LABEL, NEGATIVE_LABEL)


def safe_splits(y: np.ndarray, preferred_k: int) -> int:
    counts = pd.Series(y).value_counts()
    return min(preferred_k, int(counts.min()))


df = pd.read_csv(DATA_DIR / f"{EXPERIMENT}.csv")
print(f"Loaded {EXPERIMENT}: {df.shape}")

feature_cols = [c for c in df.columns if c not in ["Subject_ID", "Sleep_Hours"]]
X = df[feature_cols].values
y_hours = df["Sleep_Hours"].values
y_bin = make_binary_labels(y_hours)

print(f"Features: {len(feature_cols)}  |  Subjects: {len(y_hours)}")
print(
    f"Sleep hours — mean={y_hours.mean():.2f}, std={y_hours.std():.2f}, "
    f"range=[{y_hours.min()}, {y_hours.max()}]"
)
print(f"Binary classes — {dict(zip(*np.unique(y_bin, return_counts=True)))}")

if TOP_N is not None:
    print(f"\nSelecting top {TOP_N} features...")
    selector = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    selector.fit(X, y_hours)
    top_idx = np.argsort(selector.feature_importances_)[::-1][:TOP_N]
    X = X[:, top_idx]
    feature_cols = [feature_cols[i] for i in top_idx]
    print(f"Top {TOP_N} features: {feature_cols}")

k_splits = safe_splits(y_bin, K)
if k_splits < 2:
    raise ValueError("Not enough subjects in the minority class for stratified CV.")

print(f"\n{'=' * 55}")
print(f"  BINARY CLASSIFICATION  ({k_splits}-fold stratified CV)")
print(f"{'=' * 55}")

skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=42)
rf_preds = []
dummy_preds = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_bin)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_bin[train_idx], y_bin[test_idx]
    subj_test = df["Subject_ID"].values[test_idx]
    hours_test = y_hours[test_idx]

    clf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)

    for sid, hrs, actual, pred, base in zip(subj_test, hours_test, y_test, y_pred, y_dummy):
        rf_preds.append(
            {
                "Subject_ID": sid,
                "Sleep_Hours": hrs,
                "Actual": actual,
                "Predicted": pred,
            }
        )
        dummy_preds.append(
            {
                "Subject_ID": sid,
                "Sleep_Hours": hrs,
                "Actual": actual,
                "Predicted": base,
            }
        )

    fold_acc = accuracy_score(y_test, y_pred)
    fold_f1 = f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    print(f"  Fold {fold + 1}/{k_splits}: n_test={len(y_test)}, accuracy={fold_acc:.3f}, F1-Short={fold_f1:.3f}")

rf_df = pd.DataFrame(rf_preds)
dummy_df = pd.DataFrame(dummy_preds)

y_true = rf_df["Actual"].values
y_pred = rf_df["Predicted"].values
y_dummy = dummy_df["Predicted"].values
labels = [POSITIVE_LABEL, NEGATIVE_LABEL]

rf_acc = accuracy_score(y_true, y_pred)
rf_f1_short = f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
rf_precision_short = precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
rf_recall_short = recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)

dummy_acc = accuracy_score(y_true, y_dummy)
dummy_f1_short = f1_score(y_true, y_dummy, pos_label=POSITIVE_LABEL, zero_division=0)

cm = confusion_matrix(y_true, y_pred, labels=labels)
report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

print(f"\n  Dummy accuracy    : {dummy_acc:.4f}")
print(f"  RF accuracy       : {rf_acc:.4f}")
print(f"  Dummy F1 (Short)  : {dummy_f1_short:.4f}")
print(f"  RF F1 (Short)     : {rf_f1_short:.4f}")
print(f"  RF precision      : {rf_precision_short:.4f}")
print(f"  RF recall         : {rf_recall_short:.4f}")
print(f"\n{report}")

rf_df.to_csv(OUT_DIR / f"{EXPERIMENT}_binary_predictions.csv", index=False)
(OUT_DIR / f"{EXPERIMENT}_binary_metrics.txt").write_text(
    f"Experiment : {EXPERIMENT}\n"
    f"Mode       : Binary classification ({k_splits}-fold stratified CV)\n"
    f"Positive   : {POSITIVE_LABEL} (Sleep_Hours < 6)\n"
    f"Negative   : {NEGATIVE_LABEL} (Sleep_Hours >= 6)\n"
    f"N subjects : {len(y_true)}\n"
    f"N features : {len(feature_cols)}\n\n"
    f"Dummy accuracy   : {dummy_acc:.4f}\n"
    f"RF accuracy      : {rf_acc:.4f}\n"
    f"Dummy F1(Short)  : {dummy_f1_short:.4f}\n"
    f"RF F1(Short)     : {rf_f1_short:.4f}\n"
    f"RF precision     : {rf_precision_short:.4f}\n"
    f"RF recall        : {rf_recall_short:.4f}\n\n"
    f"Classification report:\n{report}\n"
    f"Confusion matrix (rows=actual, cols=predicted):\n"
    f"Labels: {labels}\n{cm}\n"
)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"{EXPERIMENT} — Binary Classification\nAccuracy={rf_acc:.3f}   F1(Short)={rf_f1_short:.3f}")
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            fontsize=13,
        )
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{EXPERIMENT}_binary_confusion.png", dpi=150)
plt.close()

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")
