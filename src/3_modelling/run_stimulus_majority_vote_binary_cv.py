"""
Run subject-safe stimulus-level binary classification with subject-level majority vote.

Binary target:
  LowSleep   -> Sleep_Hours < 7
  HigherSleep -> Sleep_Hours >= 7

Pipeline:
  - one row = one subject x stimulus aggregate (median across windows)
  - train on stimulus rows
  - split by Subject_ID to avoid leakage
  - predict one class per stimulus row
  - collapse each held-out subject to a final class using majority vote
"""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut


BASE_DIR = Path(__file__).parent.parent.parent
INPUT_FILE = BASE_DIR / "enhanced_preparation" / "all_task_stimulus_median_ready.csv"
OUT_DIR = BASE_DIR / "results" / "stimulus_majority_vote_binary_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["LowSleep", "HigherSleep"]
RANDOM_STATE = 42


def make_binary_label(hours: float) -> str:
    return "LowSleep" if float(hours) < 7 else "HigherSleep"


def majority_vote(preds: list[str], proba: np.ndarray, classes: np.ndarray) -> tuple[str, dict[str, int]]:
    counts = Counter(preds)
    max_count = max(counts.values())
    top_labels = [label for label, count in counts.items() if count == max_count]

    if len(top_labels) == 1:
        return top_labels[0], dict(counts)

    proba_sums = pd.Series(proba.sum(axis=0), index=classes)
    top_label = proba_sums.loc[top_labels].sort_values(ascending=False).index[0]
    return str(top_label), dict(counts)


def save_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Binary Subject-Level Majority Vote\nConfusion Matrix")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12,
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "binary_subject_majority_vote_confusion.png", dpi=150)
    plt.close(fig)


def save_subject_vote_chart(subject_df: pd.DataFrame) -> None:
    plot_df = subject_df.copy()
    plot_df["Correct"] = plot_df["Actual_Cat"] == plot_df["Predicted_Cat"]
    plot_df["Label"] = plot_df.apply(
        lambda r: f"{r['Subject_ID']}\nA:{r['Actual_Cat']} P:{r['Predicted_Cat']}",
        axis=1,
    )
    colors = ["seagreen" if ok else "indianred" for ok in plot_df["Correct"]]

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.32), 4))
    ax.bar(plot_df["Label"], plot_df["Num_Stimuli"], color=colors)
    ax.set_ylabel("Number of Stimuli")
    ax.set_title("Binary Subject-Level Majority Vote Outcomes")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "binary_subject_majority_vote_overview.png", dpi=150)
    plt.close(fig)


def save_task_prediction_chart(task_summary: pd.DataFrame) -> None:
    if task_summary.empty:
        return

    pivot = (
        task_summary.pivot_table(
            index="Task",
            columns="Predicted_Cat",
            values="Count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=LABELS, fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.28)))
    left = np.zeros(len(pivot))
    colors = {"LowSleep": "#d95f02", "HigherSleep": "#1b9e77"}

    for label in LABELS:
        values = pivot[label].to_numpy()
        ax.barh(pivot.index, values, left=left, label=label, color=colors[label])
        left += values

    ax.set_xlabel("Number of Stimulus-Level Predictions")
    ax.set_ylabel("Task")
    ax.set_title("Binary Predicted Class Distribution by Task")
    ax.legend(title="Predicted")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "binary_task_prediction_distribution.png", dpi=150)
    plt.close(fig)


def save_task_prediction_heatmap(task_summary: pd.DataFrame) -> None:
    if task_summary.empty:
        return

    pivot = (
        task_summary.pivot_table(
            index="Task",
            columns="Predicted_Cat",
            values="Count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=LABELS, fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(6, max(6, len(pivot) * 0.28)))
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Task")
    ax.set_title("Binary Task x Predicted Class Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "binary_task_prediction_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    feature_cols = [c for c in df.columns if c.startswith("POW.") or c.startswith("PM.")]
    if not feature_cols:
        raise ValueError(f"No POW./PM. features found in {INPUT_FILE}")

    X = df[feature_cols].to_numpy()
    y_hours = df["Sleep_Hours"].to_numpy()
    y_cat = np.array([make_binary_label(v) for v in y_hours], dtype=object)
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
        y_train = y_cat[train_idx]

        test_df = df.iloc[test_idx].copy()
        test_subject = str(test_df["Subject_ID"].iloc[0])
        true_subject_label = make_binary_label(float(test_df["Sleep_Hours"].iloc[0]))

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
                "Votes_LowSleep": vote_counts.get("LowSleep", 0),
                "Votes_HigherSleep": vote_counts.get("HigherSleep", 0),
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

    subject_df.to_csv(OUT_DIR / "binary_subject_majority_vote_predictions.csv", index=False)
    stimulus_df.to_csv(OUT_DIR / "binary_stimulus_level_predictions.csv", index=False)
    task_summary.to_csv(OUT_DIR / "binary_task_prediction_summary.csv", index=False)

    metrics_text = (
        f"Input file: {INPUT_FILE}\n"
        f"Subjects: {df['Subject_ID'].nunique()}\n"
        f"Stimulus rows: {len(df)}\n"
        f"Tasks: {df['Task'].nunique()}\n"
        f"Features: {len(feature_cols)}\n"
        f"Binary target: LowSleep (<7h) vs HigherSleep (>=7h)\n"
        f"Evaluation: Leave-One-Subject-Out CV\n\n"
        f"Subject-level accuracy: {acc:.4f}\n"
        f"Subject-level F1 (macro): {f1_macro:.4f}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix (rows=actual, cols=predicted; labels={LABELS}):\n{cm}\n"
    )
    (OUT_DIR / "binary_stimulus_majority_vote_metrics.txt").write_text(metrics_text)

    save_confusion_matrix(cm)
    save_subject_vote_chart(subject_df)
    save_task_prediction_chart(task_summary)
    save_task_prediction_heatmap(task_summary)

    print("\nSubject-level binary majority-vote results")
    print(f"Accuracy        : {acc:.4f}")
    print(f"F1 (macro)      : {f1_macro:.4f}")
    print(f"\n{report}")
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
