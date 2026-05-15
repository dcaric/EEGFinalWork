"""
Run a lighter subject-safe binary classification on the window-level dataset.

Binary target:
  LowSleep    -> Sleep_Hours < 7
  HigherSleep -> Sleep_Hours >= 7

Pipeline:
  - one row = one subject x stimulus x window
  - cap the number of windows per subject x stimulus
  - split by Subject_ID to avoid leakage
  - predict one class per window
  - aggregate window predictions to one vote per stimulus
  - aggregate stimulus votes to one final subject-level prediction
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
INPUT_FILE = BASE_DIR / "enhanced_preparation" / "all_task_window_ready.csv"
OUT_DIR = BASE_DIR / "results" / "stimulus_majority_vote_binary_all_window_downsampled_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["LowSleep", "HigherSleep"]
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_WINDOWS_PER_STIMULUS = 80


def make_binary_label(hours: float) -> str:
    return "LowSleep" if float(hours) < 7 else "HigherSleep"


def downsample_windows(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby("Stimulus_Subject_ID", sort=False):
        if len(group) <= MAX_WINDOWS_PER_STIMULUS:
            parts.append(group)
            continue

        # Evenly spaced sampling preserves coverage across the whole segment.
        idx = np.linspace(0, len(group) - 1, MAX_WINDOWS_PER_STIMULUS, dtype=int)
        sampled = group.iloc[idx].copy()
        sampled["Window_Index_Downsampled"] = range(1, len(sampled) + 1)
        parts.append(sampled)

    out = pd.concat(parts, ignore_index=True)
    if "Window_Index_Downsampled" not in out.columns:
        out["Window_Index_Downsampled"] = out["Window_Index"]
    else:
        out["Window_Index_Downsampled"] = out["Window_Index_Downsampled"].fillna(out["Window_Index"])
    return out


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
    ax.set_title("Downsampled All-Window Binary Subject Vote\nConfusion Matrix")
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
    fig.savefig(OUT_DIR / "binary_all_window_downsampled_subject_confusion.png", dpi=150)
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
    ax.set_title("Downsampled All-Window Binary Subject Vote Outcomes")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "binary_all_window_downsampled_subject_overview.png", dpi=150)
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

    ax.set_xlabel("Number of Stimulus-Level Votes")
    ax.set_ylabel("Task")
    ax.set_title("Downsampled All-Window Binary Predicted Class Distribution by Task")
    ax.legend(title="Predicted")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "binary_all_window_downsampled_task_distribution.png", dpi=150)
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
    ax.set_title("Downsampled All-Window Binary Task x Predicted Class Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "binary_all_window_downsampled_task_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    raw_df = pd.read_csv(INPUT_FILE)
    df = downsample_windows(raw_df)

    feature_cols = [c for c in df.columns if c.startswith("POW.") or c.startswith("PM.")]
    if not feature_cols:
        raise ValueError(f"No POW./PM. features found in {INPUT_FILE}")

    X = df[feature_cols].to_numpy()
    y_hours = df["Sleep_Hours"].to_numpy()
    y_cat = np.array([make_binary_label(v) for v in y_hours], dtype=object)
    groups = df["Subject_ID"].astype(str).to_numpy()

    print(f"Loaded raw window-level table: {raw_df.shape}")
    print(f"Downsampled window-level table: {df.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Subjects: {df['Subject_ID'].nunique()} | Window rows: {len(df)}")
    print(f"Tasks: {df['Task'].nunique()} | Category counts: {dict(zip(*np.unique(y_cat, return_counts=True)))}")
    print(f"Max windows per stimulus after downsampling: {MAX_WINDOWS_PER_STIMULUS}")

    unique_subjects = pd.unique(groups)
    if len(unique_subjects) < 2:
        raise ValueError("At least 2 subjects are required for leave-one-subject-out CV.")

    logo = LeaveOneGroupOut()
    window_predictions = []
    stimulus_predictions = []
    subject_predictions = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_cat, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y_cat[train_idx]

        test_df = df.iloc[test_idx].copy()
        test_subject = str(test_df["Subject_ID"].iloc[0])
        true_subject_label = make_binary_label(float(test_df["Sleep_Hours"].iloc[0]))

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        row_pred = clf.predict(X_test)
        row_proba = clf.predict_proba(X_test)

        window_fold_df = test_df.copy().reset_index(drop=True)
        window_fold_df["Predicted_Cat"] = row_pred
        window_fold_df["Actual_Cat"] = true_subject_label
        window_predictions.extend(
            window_fold_df[
                [
                    "Subject_ID",
                    "Task",
                    "Window_Index",
                    "Window_Index_Downsampled",
                    "Sleep_Hours",
                    "Actual_Cat",
                    "Predicted_Cat",
                ]
            ].to_dict("records")
        )

        stimulus_fold_rows = []
        for stimulus_id, stim_df in window_fold_df.groupby("Stimulus_Subject_ID", sort=False):
            idx = stim_df.index.to_numpy()
            stim_pred, stim_votes = majority_vote(list(row_pred[idx]), row_proba[idx], clf.classes_)
            stimulus_row = {
                "Fold": fold,
                "Stimulus_Subject_ID": stimulus_id,
                "Subject_ID": test_subject,
                "Task": str(stim_df["Task"].iloc[0]),
                "Sleep_Hours": float(stim_df["Sleep_Hours"].iloc[0]),
                "Actual_Cat": true_subject_label,
                "Predicted_Cat": stim_pred,
                "Num_Windows": len(stim_df),
                "Votes_LowSleep": stim_votes.get("LowSleep", 0),
                "Votes_HigherSleep": stim_votes.get("HigherSleep", 0),
            }
            stimulus_fold_rows.append(stimulus_row)

        stimulus_fold_df = pd.DataFrame(stimulus_fold_rows)
        stimulus_predictions.extend(stimulus_fold_rows)

        stim_pred_list = stimulus_fold_df["Predicted_Cat"].tolist()
        stim_vote_matrix = stimulus_fold_df[["Votes_LowSleep", "Votes_HigherSleep"]].to_numpy(dtype=float)
        final_pred, vote_counts = majority_vote(stim_pred_list, stim_vote_matrix, np.array(LABELS))
        subject_predictions.append(
            {
                "Fold": fold,
                "Subject_ID": test_subject,
                "Sleep_Hours": float(test_df["Sleep_Hours"].iloc[0]),
                "Actual_Cat": true_subject_label,
                "Predicted_Cat": final_pred,
                "Num_Stimuli": len(stimulus_fold_df),
                "Num_Windows": len(test_df),
                "Votes_LowSleep": vote_counts.get("LowSleep", 0),
                "Votes_HigherSleep": vote_counts.get("HigherSleep", 0),
            }
        )

        print(
            f"Fold {fold}/{len(unique_subjects)}: subject={test_subject}, "
            f"windows={len(test_df)}, stimuli={len(stimulus_fold_df)}, actual={true_subject_label}, "
            f"predicted={final_pred}, votes={vote_counts}"
        )

    window_df = pd.DataFrame(window_predictions)
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

    subject_df.to_csv(OUT_DIR / "binary_all_window_downsampled_subject_predictions.csv", index=False)
    stimulus_df.to_csv(OUT_DIR / "binary_all_window_downsampled_stimulus_predictions.csv", index=False)
    window_df.to_csv(OUT_DIR / "binary_all_window_downsampled_window_predictions.csv", index=False)
    task_summary.to_csv(OUT_DIR / "binary_all_window_downsampled_task_prediction_summary.csv", index=False)

    metrics_text = (
        f"Input file: {INPUT_FILE}\n"
        f"Original window rows: {len(raw_df)}\n"
        f"Downsampled window rows: {len(df)}\n"
        f"Subjects: {df['Subject_ID'].nunique()}\n"
        f"Stimulus votes: {len(stimulus_df)}\n"
        f"Tasks: {df['Task'].nunique()}\n"
        f"Features: {len(feature_cols)}\n"
        f"RandomForest n_estimators: {N_ESTIMATORS}\n"
        f"Binary target: LowSleep (<7h) vs HigherSleep (>=7h)\n"
        f"Evaluation: Leave-One-Subject-Out CV\n"
        f"Aggregation: window -> stimulus vote -> subject vote\n"
        f"Max windows per stimulus: {MAX_WINDOWS_PER_STIMULUS}\n\n"
        f"Subject-level accuracy: {acc:.4f}\n"
        f"Subject-level F1 (macro): {f1_macro:.4f}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix (rows=actual, cols=predicted; labels={LABELS}):\n{cm}\n"
    )
    (OUT_DIR / "binary_all_window_downsampled_metrics.txt").write_text(metrics_text)

    save_confusion_matrix(cm)
    save_subject_vote_chart(subject_df)
    save_task_prediction_chart(task_summary)
    save_task_prediction_heatmap(task_summary)

    print("\nSubject-level downsampled all-window binary majority-vote results")
    print(f"Accuracy        : {acc:.4f}")
    print(f"F1 (macro)      : {f1_macro:.4f}")
    print(f"\n{report}")
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
