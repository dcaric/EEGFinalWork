"""
Run subject-safe stimulus-level classification with subject-level majority vote.

Idea:
  - One row = one subject x stimulus aggregate (typically median across windows)
  - Train the model on stimulus rows
  - Split by Subject_ID to avoid leakage
  - Predict one class per stimulus row
  - Collapse each held-out subject to a final class using majority vote

This follows the professor's suggestion:
  keep more granular rows than a single 81 x 70 subject matrix, then aggregate
  predictions back to the subject level.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut


BASE_DIR = Path(__file__).parent.parent.parent
INPUT_FILE = BASE_DIR / "enhanced_preparation" / "all_task_stimulus_median_ready.csv"
OUT_DIR = BASE_DIR / "results" / "stimulus_majority_vote_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BINS = [0, 6, 8, 99]
LABELS = ["Short", "Normal", "Long"]
RANDOM_STATE = 42


def majority_vote(preds: list[str], proba: np.ndarray, classes: np.ndarray) -> tuple[str, dict[str, int]]:
    counts = Counter(preds)
    max_count = max(counts.values())
    top_labels = [label for label, count in counts.items() if count == max_count]

    if len(top_labels) == 1:
        return top_labels[0], dict(counts)

    proba_sums = pd.Series(proba.sum(axis=0), index=classes)
    top_label = proba_sums.loc[top_labels].sort_values(ascending=False).index[0]
    return str(top_label), dict(counts)


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    feature_cols = [c for c in df.columns if c.startswith("POW.") or c.startswith("PM.")]
    if not feature_cols:
        raise ValueError(f"No POW./PM. features found in {INPUT_FILE}")

    X = df[feature_cols].to_numpy()
    y_hours = df["Sleep_Hours"].to_numpy()
    y_cat = pd.cut(y_hours, bins=BINS, labels=LABELS, right=False).astype(str)
    groups = df["Subject_ID"].astype(str).to_numpy()

    print(f"Loaded stimulus-level table: {df.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Subjects: {df['Subject_ID'].nunique()} | Stimulus rows: {len(df)}")
    print(f"Tasks: {df['Task'].nunique()} | Category counts: {dict(zip(*np.unique(y_cat, return_counts=True)))}")

    unique_subjects = pd.unique(groups)
    if len(unique_subjects) < 2:
        raise ValueError("At least 2 subjects are required for leave-one-subject-out CV.")

    logo = LeaveOneGroupOut()
    stimulus_predictions = []
    subject_predictions = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_cat, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]

        test_df = df.iloc[test_idx].copy()
        test_subject = str(test_df["Subject_ID"].iloc[0])
        true_subject_label = str(test_df["Sleep_Hours"].pipe(lambda s: pd.cut(s, bins=BINS, labels=LABELS, right=False)).astype(str).iloc[0])

        clf = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        row_pred = clf.predict(X_test)
        row_proba = clf.predict_proba(X_test)

        for _, row, pred in zip(test_idx, test_df.itertuples(index=False), row_pred):
            stimulus_predictions.append(
                {
                    "Fold": fold,
                    "Subject_ID": row.Subject_ID,
                    "Task": row.Task,
                    "Sleep_Hours": row.Sleep_Hours,
                    "Actual_Cat": true_subject_label,
                    "Predicted_Cat": pred,
                }
            )

        final_pred, vote_counts = majority_vote(list(row_pred), row_proba, clf.classes_)
        subject_predictions.append(
            {
                "Fold": fold,
                "Subject_ID": test_subject,
                "Sleep_Hours": float(test_df["Sleep_Hours"].iloc[0]),
                "Actual_Cat": true_subject_label,
                "Predicted_Cat": final_pred,
                "Num_Stimuli": len(test_df),
                "Votes_Short": vote_counts.get("Short", 0),
                "Votes_Normal": vote_counts.get("Normal", 0),
                "Votes_Long": vote_counts.get("Long", 0),
            }
        )

        print(
            f"Fold {fold}/{len(unique_subjects)}: subject={test_subject}, "
            f"stimuli={len(test_df)}, actual={true_subject_label}, predicted={final_pred}, votes={vote_counts}"
        )

    stimulus_df = pd.DataFrame(stimulus_predictions)
    subject_df = pd.DataFrame(subject_predictions)

    y_true = subject_df["Actual_Cat"].to_numpy()
    y_pred = subject_df["Predicted_Cat"].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    task_summary = (
        stimulus_df.groupby(["Task", "Actual_Cat", "Predicted_Cat"])
        .size()
        .reset_index(name="Count")
        .sort_values(["Task", "Actual_Cat", "Predicted_Cat"])
    )

    subject_df.to_csv(OUT_DIR / "subject_majority_vote_predictions.csv", index=False)
    stimulus_df.to_csv(OUT_DIR / "stimulus_level_predictions.csv", index=False)
    task_summary.to_csv(OUT_DIR / "task_prediction_summary.csv", index=False)

    metrics_text = (
        f"Input file: {INPUT_FILE}\n"
        f"Subjects: {df['Subject_ID'].nunique()}\n"
        f"Stimulus rows: {len(df)}\n"
        f"Tasks: {df['Task'].nunique()}\n"
        f"Features: {len(feature_cols)}\n"
        f"Evaluation: Leave-One-Subject-Out CV\n\n"
        f"Subject-level accuracy: {acc:.4f}\n"
        f"Subject-level F1 (macro): {f1_macro:.4f}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix (rows=actual, cols=predicted; labels={LABELS}):\n{cm}\n"
    )
    (OUT_DIR / "stimulus_majority_vote_metrics.txt").write_text(metrics_text)

    print("\nSubject-level majority-vote results")
    print(f"Accuracy        : {acc:.4f}")
    print(f"F1 (macro)      : {f1_macro:.4f}")
    print(f"\n{report}")
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
