"""
Tune balanced logistic regression for detecting Short sleepers.

This script performs subject-level nested cross-validation:
  - Outer CV estimates generalization performance
  - Inner CV tunes logistic-regression strength and decision threshold

Target:
  Short     -> Sleep_Hours < 6
  NotShort  -> Sleep_Hours >= 6

Why:
  Overall accuracy is misleading for this imbalanced dataset.
  We tune for F1_Short and Recall_Short instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT = "exp3_task_specific"
OUTER_K = 5
INNER_K = 4
TOP_N = 15
POSITIVE_LABEL = "Short"
NEGATIVE_LABEL = "NotShort"
C_GRID = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "modelling_tables"
OUT_DIR = BASE_DIR / "results" / f"{EXPERIMENT}_logreg_tuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def print_header(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def make_binary_labels(y_hours: np.ndarray) -> np.ndarray:
    return np.where(y_hours < 6, POSITIVE_LABEL, NEGATIVE_LABEL)


def safe_splits(y: np.ndarray, preferred_k: int) -> int:
    counts = pd.Series(y).value_counts()
    return min(preferred_k, int(counts.min()))


def select_top_features(
    X: np.ndarray, y_hours: np.ndarray, feature_cols: list[str], top_n: int | None
) -> tuple[np.ndarray, list[str]]:
    if top_n is None or top_n >= len(feature_cols):
        return X, feature_cols

    selector = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    selector.fit(X, y_hours)
    top_idx = np.argsort(selector.feature_importances_)[::-1][:top_n]
    return X[:, top_idx], [feature_cols[i] for i in top_idx]


def build_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=20000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def threshold_predictions(prob_short: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(prob_short >= threshold, POSITIVE_LABEL, NEGATIVE_LABEL)


def tune_on_training(X_train: np.ndarray, y_train: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    inner_splits = safe_splits(y_train, INNER_K)
    if inner_splits < 2:
        raise ValueError("Not enough Short subjects in the training fold for inner CV tuning.")

    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)
    rows = []

    for c_value in C_GRID:
        model = build_model(c_value)
        prob_matrix = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=1,
        )
        short_idx = list(model.classes_).index(POSITIVE_LABEL) if hasattr(model, "classes_") else 1
        # cross_val_predict returns columns ordered by estimator.classes_; infer safely from fit below
        fitted = build_model(c_value).fit(X_train, y_train)
        short_idx = list(fitted.named_steps["model"].classes_).index(POSITIVE_LABEL)
        prob_short = prob_matrix[:, short_idx]

        for threshold in THRESHOLDS:
            pred = threshold_predictions(prob_short, threshold)
            rows.append(
                {
                    "C": c_value,
                    "Threshold": threshold,
                    "F1_Short": f1_score(y_train, pred, pos_label=POSITIVE_LABEL, zero_division=0),
                    "Precision_Short": precision_score(
                        y_train, pred, pos_label=POSITIVE_LABEL, zero_division=0
                    ),
                    "Recall_Short": recall_score(
                        y_train, pred, pos_label=POSITIVE_LABEL, zero_division=0
                    ),
                    "Accuracy": accuracy_score(y_train, pred),
                }
            )

    tuning = pd.DataFrame(rows).sort_values(
        by=["F1_Short", "Recall_Short", "Accuracy"],
        ascending=[False, False, False],
    )
    best = tuning.iloc[0]
    return float(best["C"]), float(best["Threshold"]), tuning


def main() -> None:
    df = pd.read_csv(DATA_DIR / f"{EXPERIMENT}.csv")
    feature_cols = [c for c in df.columns if c not in ["Subject_ID", "Sleep_Hours"]]
    X = df[feature_cols].to_numpy()
    y_hours = df["Sleep_Hours"].to_numpy()
    y = make_binary_labels(y_hours)

    X, selected_features = select_top_features(X, y_hours, feature_cols, TOP_N)

    print(f"Loaded {EXPERIMENT}: {df.shape}")
    print(f"Subjects: {len(df)} | Features used: {X.shape[1]}")
    print(f"Binary classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    outer_splits = safe_splits(y, OUTER_K)
    if outer_splits < 2:
        raise ValueError("Not enough Short subjects for outer CV.")

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    fold_rows = []
    pred_rows = []
    all_tuning_tables = []

    print_header(f"TUNED LOGISTIC REGRESSION ({outer_splits}-fold outer CV)")

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subjects = df["Subject_ID"].to_numpy()[test_idx]
        test_hours = y_hours[test_idx]

        best_c, best_threshold, tuning_table = tune_on_training(X_train, y_train)
        tuning_table = tuning_table.copy()
        tuning_table["Outer_Fold"] = fold
        all_tuning_tables.append(tuning_table)

        model = build_model(best_c)
        model.fit(X_train, y_train)
        classes = list(model.named_steps["model"].classes_)
        short_idx = classes.index(POSITIVE_LABEL)
        prob_short = model.predict_proba(X_test)[:, short_idx]
        y_pred = threshold_predictions(prob_short, best_threshold)

        acc = accuracy_score(y_test, y_pred)
        f1_short = f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        precision_short = precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        recall_short = recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)

        fold_rows.append(
            {
                "Fold": fold,
                "Best_C": best_c,
                "Best_Threshold": best_threshold,
                "Accuracy": round(float(acc), 4),
                "F1_Short": round(float(f1_short), 4),
                "Precision_Short": round(float(precision_short), 4),
                "Recall_Short": round(float(recall_short), 4),
            }
        )

        for sid, hrs, actual, pred, prob in zip(test_subjects, test_hours, y_test, y_pred, prob_short):
            pred_rows.append(
                {
                    "Fold": fold,
                    "Subject_ID": sid,
                    "Sleep_Hours": hrs,
                    "Actual": actual,
                    "Predicted": pred,
                    "Prob_Short": round(float(prob), 6),
                    "Best_C": best_c,
                    "Best_Threshold": best_threshold,
                }
            )

        print(
            f"Fold {fold}/{outer_splits}: "
            f"C={best_c}, threshold={best_threshold:.2f}, "
            f"acc={acc:.3f}, F1_Short={f1_short:.3f}, recall={recall_short:.3f}"
        )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)

    y_true = pred_df["Actual"].to_numpy()
    y_pred = pred_df["Predicted"].to_numpy()

    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    overall_precision = precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    overall_recall = recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[POSITIVE_LABEL, NEGATIVE_LABEL])
    report = classification_report(
        y_true, y_pred, labels=[POSITIVE_LABEL, NEGATIVE_LABEL], zero_division=0
    )

    print_header("OVERALL RESULTS")
    print(f"Accuracy        : {overall_acc:.4f}")
    print(f"F1_Short        : {overall_f1:.4f}")
    print(f"Precision_Short : {overall_precision:.4f}")
    print(f"Recall_Short    : {overall_recall:.4f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix [Short, NotShort]:")
    print(cm)

    fold_df.to_csv(OUT_DIR / f"{EXPERIMENT}_tuned_logreg_fold_metrics.csv", index=False)
    pred_df.to_csv(OUT_DIR / f"{EXPERIMENT}_tuned_logreg_predictions.csv", index=False)
    pd.concat(all_tuning_tables, ignore_index=True).to_csv(
        OUT_DIR / f"{EXPERIMENT}_tuned_logreg_inner_search.csv", index=False
    )

    summary = (
        f"Experiment: {EXPERIMENT}\n"
        f"Subjects: {len(df)}\n"
        f"Selected features ({len(selected_features)}):\n"
        + "\n".join(selected_features)
        + "\n\nOverall metrics:\n"
        f"Accuracy        : {overall_acc:.4f}\n"
        f"F1_Short        : {overall_f1:.4f}\n"
        f"Precision_Short : {overall_precision:.4f}\n"
        f"Recall_Short    : {overall_recall:.4f}\n\n"
        "Classification report:\n"
        f"{report}\n"
        "Confusion matrix [Short, NotShort]:\n"
        f"{cm}\n\n"
        "Fold metrics:\n"
        f"{fold_df.to_string(index=False)}\n"
    )
    (OUT_DIR / f"{EXPERIMENT}_tuned_logreg_summary.txt").write_text(summary)

    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
