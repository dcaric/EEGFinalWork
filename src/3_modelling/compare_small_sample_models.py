"""
Compare small-sample models for subject-level sleep prediction.

This script keeps the same subject-level setup as the thesis pipeline, but
tries models that are often better behaved than Random Forest on small datasets.

Tasks:
  1. Regression on Sleep_Hours
  2. Binary classification: Short vs NotShort

Definition:
  Short     -> Sleep_Hours < 6
  NotShort  -> Sleep_Hours >= 6
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


# ── SETTINGS ──────────────────────────────────────────────────────────────────
EXPERIMENT = "exp3_task_specific"
K = 5
TOP_N = 15
POSITIVE_LABEL = "Short"
NEGATIVE_LABEL = "NotShort"
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "modelling_tables"
OUT_DIR = BASE_DIR / "results" / f"{EXPERIMENT}_model_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def print_header(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def make_binary_labels(y_hours: np.ndarray) -> np.ndarray:
    return np.where(y_hours < 6, POSITIVE_LABEL, NEGATIVE_LABEL)


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
    X_selected = X[:, top_idx]
    selected_cols = [feature_cols[i] for i in top_idx]

    print_header(f"TOP {top_n} FEATURES")
    print(selected_cols)

    return X_selected, selected_cols


def regression_table(X: np.ndarray, y_hours: np.ndarray) -> pd.DataFrame:
    cv = KFold(n_splits=K, shuffle=True, random_state=42)

    models = {
        "DummyMean": DummyRegressor(strategy="mean"),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=20000)),
            ]
        ),
        "SVR_RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, epsilon=0.2)),
            ]
        ),
    }

    rows = []
    for name, model in models.items():
        pred = cross_val_predict(model, X, y_hours, cv=cv, n_jobs=1)
        rmse = np.sqrt(mean_squared_error(y_hours, pred))
        mae = mean_absolute_error(y_hours, pred)
        r2 = r2_score(y_hours, pred)
        rows.append(
            {
                "Model": name,
                "RMSE": round(float(rmse), 4),
                "MAE": round(float(mae), 4),
                "R2": round(float(r2), 4),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["RMSE", "MAE"], ascending=[True, True])


def classification_table(X: np.ndarray, y_bin: np.ndarray) -> pd.DataFrame:
    class_counts = pd.Series(y_bin).value_counts()
    min_count = int(class_counts.min())
    k_splits = min(K, min_count)
    if k_splits < 2:
        raise ValueError("Not enough minority-class subjects for stratified CV.")

    cv = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=42)

    models = {
        "DummyMajority": DummyClassifier(strategy="most_frequent"),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LogRegBalanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=20000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "SVC_RBF_Balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    rows = []
    for name, model in models.items():
        pred = cross_val_predict(model, X, y_bin, cv=cv, n_jobs=1)
        acc = accuracy_score(y_bin, pred)
        f1_short = f1_score(y_bin, pred, pos_label=POSITIVE_LABEL, zero_division=0)
        precision_short = precision_score(y_bin, pred, pos_label=POSITIVE_LABEL, zero_division=0)
        recall_short = recall_score(y_bin, pred, pos_label=POSITIVE_LABEL, zero_division=0)
        rows.append(
            {
                "Model": name,
                "Accuracy": round(float(acc), 4),
                "F1_Short": round(float(f1_short), 4),
                "Precision_Short": round(float(precision_short), 4),
                "Recall_Short": round(float(recall_short), 4),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(by=["F1_Short", "Recall_Short", "Accuracy"], ascending=[False, False, False])
    )


def main() -> None:
    df = pd.read_csv(DATA_DIR / f"{EXPERIMENT}.csv")
    feature_cols = [c for c in df.columns if c not in ["Subject_ID", "Sleep_Hours"]]
    X = df[feature_cols].to_numpy()
    y_hours = df["Sleep_Hours"].to_numpy()
    y_bin = make_binary_labels(y_hours)

    print(f"Loaded {EXPERIMENT}: {df.shape}")
    print(f"Features: {len(feature_cols)}  |  Subjects: {len(df)}")
    print(
        f"Sleep hours — mean={y_hours.mean():.2f}, std={y_hours.std():.2f}, "
        f"range=[{y_hours.min()}, {y_hours.max()}]"
    )
    print(f"Binary classes — {dict(zip(*np.unique(y_bin, return_counts=True)))}")

    X_selected, selected_features = select_top_features(X, y_hours, feature_cols, TOP_N)

    print_header(f"REGRESSION MODEL COMPARISON ({K}-fold CV)")
    reg_results = regression_table(X_selected, y_hours)
    print(reg_results.to_string(index=False))
    reg_results.to_csv(OUT_DIR / f"{EXPERIMENT}_regression_model_comparison.csv", index=False)

    print_header(f"BINARY MODEL COMPARISON ({min(K, int(pd.Series(y_bin).value_counts().min()))}-fold stratified CV)")
    cls_results = classification_table(X_selected, y_bin)
    print(cls_results.to_string(index=False))
    cls_results.to_csv(OUT_DIR / f"{EXPERIMENT}_binary_model_comparison.csv", index=False)

    print_header("INTERPRETATION")
    print("Lower RMSE is better for regression.")
    print("For binary classification, F1_Short and Recall_Short matter more than accuracy.")
    print("If all models stay near the dummy baseline, the bottleneck is likely the dataset, not the algorithm.")

    summary_text = (
        f"Experiment: {EXPERIMENT}\n"
        f"Subjects: {len(df)}\n"
        f"Features used: {len(X_selected[0]) if len(X_selected) else 0}\n"
        f"Binary class counts: {dict(zip(*np.unique(y_bin, return_counts=True)))}\n\n"
        f"Top {TOP_N} features:\n"
        + "\n".join(selected_features)
        + "\n\nRegression model comparison:\n"
        + reg_results.to_string(index=False)
        + "\n\nBinary model comparison:\n"
        + cls_results.to_string(index=False)
        + "\n\nInterpretation:\n"
        + "Lower RMSE is better for regression.\n"
        + "For binary classification, F1_Short and Recall_Short matter more than accuracy.\n"
        + "If all models stay near the dummy baseline, the bottleneck is likely the dataset, not the algorithm.\n"
    )
    (OUT_DIR / f"{EXPERIMENT}_model_comparison_summary.txt").write_text(summary_text)
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
