"""
Diagnostic script for judging whether a modelling table contains usable signal.

This script is separate from run_cv.py so the main thesis pipeline stays unchanged.
It answers questions like:
  - Is the Random Forest actually beating a trivial baseline?
  - Is the bad classification result mostly caused by class imbalance?
  - Are there any features with at least moderate correlation to Sleep_Hours?

HOW TO USE:
  1. Set EXPERIMENT below
  2. Run: python check_data_signal.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict


# ── SETTINGS ──────────────────────────────────────────────────────────────────
EXPERIMENT = "exp3_task_specific"
K = 5
TOP_N = 15
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "modelling_tables"


def build_labels(y_hours: np.ndarray, mode: str) -> np.ndarray:
    """Create sleep labels using one of two threshold conventions."""
    if mode == "thesis_3way":
        # Matches run_cv.py exactly:
        # Short < 6, Normal 6-8, Long > 8
        # Implemented with right=False, so 8.0 becomes Long.
        bins = [0, 6, 8, 99]
        labels = ["Short", "Normal", "Long"]
        return pd.cut(y_hours, bins=bins, labels=labels, right=False).astype(str)

    if mode == "inclusive_3way":
        # More intuitive boundary:
        # Short < 6, Normal 6-8 inclusive, Long > 8
        labels = np.where(
            y_hours < 6,
            "Short",
            np.where(y_hours > 8, "Long", "Normal"),
        )
        return labels.astype(str)

    raise ValueError(f"Unknown label mode: {mode}")


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def safe_stratified_splits(y_cat: np.ndarray, preferred_k: int) -> int:
    """Avoid invalid stratified CV when a class has fewer members than K."""
    counts = pd.Series(y_cat).value_counts()
    min_count = int(counts.min())
    if min_count < 2:
        return 0
    return min(preferred_k, min_count)


def summarize_target(df: pd.DataFrame) -> None:
    y_hours = df["Sleep_Hours"]
    print_header("TARGET SUMMARY")
    print(f"Subjects: {len(df)}")
    print(
        "Sleep hours: "
        f"mean={y_hours.mean():.3f}, std={y_hours.std():.3f}, "
        f"min={y_hours.min():.1f}, median={y_hours.median():.1f}, max={y_hours.max():.1f}"
    )
    print("\nValue counts:")
    print(y_hours.value_counts().sort_index().to_string())

    for mode in ("thesis_3way", "inclusive_3way"):
        labels = build_labels(y_hours.to_numpy(), mode)
        counts = pd.Series(labels).value_counts().sort_index()
        print(f"\nClass counts [{mode}]:")
        print(counts.to_string())


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


def regression_diagnostics(X: np.ndarray, y_hours: np.ndarray) -> None:
    print_header(f"REGRESSION DIAGNOSTICS ({K}-fold CV)")

    cv = KFold(n_splits=K, shuffle=True, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf_pred = cross_val_predict(rf, X, y_hours, cv=cv, n_jobs=1)

    dummy = DummyRegressor(strategy="mean")
    dummy_pred = cross_val_predict(dummy, X, y_hours, cv=cv, n_jobs=1)

    rf_rmse = np.sqrt(mean_squared_error(y_hours, rf_pred))
    rf_mae = mean_absolute_error(y_hours, rf_pred)
    rf_r2 = r2_score(y_hours, rf_pred)

    dummy_rmse = np.sqrt(mean_squared_error(y_hours, dummy_pred))
    dummy_r2 = r2_score(y_hours, dummy_pred)

    print(f"Dummy RMSE : {dummy_rmse:.4f}")
    print(f"RF RMSE    : {rf_rmse:.4f}")
    print(f"RF MAE     : {rf_mae:.4f}")
    print(f"Dummy R²   : {dummy_r2:.4f}")
    print(f"RF R²      : {rf_r2:.4f}")
    print(f"RMSE gain  : {dummy_rmse - rf_rmse:.4f}")


def classification_diagnostics(X: np.ndarray, y_hours: np.ndarray, mode: str) -> None:
    y_cat = build_labels(y_hours, mode)
    k_splits = safe_stratified_splits(y_cat, K)
    print_header(f"CLASSIFICATION DIAGNOSTICS [{mode}] ({k_splits}-fold stratified CV)")

    counts = pd.Series(y_cat).value_counts().sort_index()
    print("Class counts:")
    print(counts.to_string())
    if k_splits == 0:
        print(
            "\nSkipped CV classification: the smallest class has fewer than 2 subjects, "
            "so stratified cross-validation is not reliable."
        )
        return

    if k_splits != K:
        print(
            f"\nAdjusted folds from {K} to {k_splits} because the smallest class "
            f"has only {counts.min()} subject(s)."
        )

    cv = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_pred = cross_val_predict(rf, X, y_cat, cv=cv, n_jobs=1)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy_pred = cross_val_predict(dummy, X, y_cat, cv=cv, n_jobs=1)

    rf_acc = accuracy_score(y_cat, rf_pred)
    rf_f1_macro = f1_score(y_cat, rf_pred, average="macro", zero_division=0)
    dummy_acc = accuracy_score(y_cat, dummy_pred)
    dummy_f1_macro = f1_score(y_cat, dummy_pred, average="macro", zero_division=0)

    labels = sorted(pd.unique(y_cat))

    print(f"Dummy accuracy   : {dummy_acc:.4f}")
    print(f"RF accuracy      : {rf_acc:.4f}")
    print(f"Dummy macro F1   : {dummy_f1_macro:.4f}")
    print(f"RF macro F1      : {rf_f1_macro:.4f}")
    print("\nRF classification report:")
    print(classification_report(y_cat, rf_pred, labels=labels, zero_division=0))


def correlation_diagnostics(df: pd.DataFrame, feature_cols: list[str]) -> None:
    print_header("FEATURE-SIGNAL CHECK")

    corr = (
        df[feature_cols + ["Sleep_Hours"]]
        .corr(numeric_only=True)["Sleep_Hours"]
        .drop("Sleep_Hours")
    )
    top = corr.abs().sort_values(ascending=False).head(15)

    print("Top absolute correlations with Sleep_Hours:")
    for name, value in top.items():
        print(f"{name:20s} abs={abs(corr[name]):.4f} signed={corr[name]:.4f}")

    print(f"\nMedian absolute correlation : {corr.abs().median():.4f}")
    print(f"Num features |corr| >= 0.20 : {int((corr.abs() >= 0.20).sum())}")
    print(f"Num features |corr| >= 0.30 : {int((corr.abs() >= 0.30).sum())}")
    print(f"Num features |corr| >= 0.40 : {int((corr.abs() >= 0.40).sum())}")


def main() -> None:
    df = pd.read_csv(DATA_DIR / f"{EXPERIMENT}.csv")
    feature_cols = [c for c in df.columns if c not in ["Subject_ID", "Sleep_Hours"]]
    X = df[feature_cols].to_numpy()
    y_hours = df["Sleep_Hours"].to_numpy()

    print(f"Loaded {EXPERIMENT}: {df.shape}")
    summarize_target(df)
    correlation_diagnostics(df, feature_cols)
    X_selected, _ = select_top_features(X, y_hours, feature_cols, TOP_N)
    regression_diagnostics(X_selected, y_hours)
    classification_diagnostics(X_selected, y_hours, mode="thesis_3way")
    classification_diagnostics(X_selected, y_hours, mode="inclusive_3way")

    print_header("INTERPRETATION GUIDE")
    print("If RF is close to dummy baseline, the table likely contains weak predictive signal.")
    print("If inclusive_3way looks much better than thesis_3way, thresholds are part of the problem.")
    print("If regression improves a bit but classification does not, the target may be too imbalanced.")


if __name__ == "__main__":
    main()
