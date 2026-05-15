"""
Run subject-safe stimulus-level regression with subject-level aggregation.

Input:
  enhanced_preparation/all_task_stimulus_median_ready.csv

Target:
  Sleep_Hours (continuous)

Pipeline:
  - one row = one subject x stimulus aggregate (median across windows)
  - train on stimulus rows
  - split by Subject_ID to avoid leakage
  - predict one sleep-hours value per stimulus row
  - aggregate stimulus predictions back to one subject-level prediction

Saved outputs:
  - stimulus-level predictions
  - subject-level predictions using mean aggregation
  - subject-level predictions using median aggregation
  - metrics text file
  - scatter plots for mean and median aggregation
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut


BASE_DIR = Path(__file__).parent.parent.parent
INPUT_FILE = BASE_DIR / "enhanced_preparation" / "all_task_stimulus_median_ready.csv"
OUT_DIR = BASE_DIR / "results" / "stimulus_regression_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 500


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_scatter(df: pd.DataFrame, title: str, filename: str, rmse_value: float, r2_value: float) -> None:
    actuals = df["Actual_Sleep_Hours"].to_numpy()
    preds = df["Predicted_Sleep_Hours"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actuals, preds, alpha=0.75, edgecolors="steelblue", facecolors="lightblue", s=70)
    mn = min(actuals.min(), preds.min())
    mx = max(actuals.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Sleep Hours")
    ax.set_ylabel("Predicted Sleep Hours")
    ax.set_title(f"{title}\nRMSE={rmse_value:.3f}  R²={r2_value:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    feature_cols = [c for c in df.columns if c.startswith("POW.") or c.startswith("PM.")]
    if not feature_cols:
        raise ValueError(f"No POW./PM. features found in {INPUT_FILE}")

    X = df[feature_cols].to_numpy()
    y = df["Sleep_Hours"].to_numpy(dtype=float)
    groups = df["Subject_ID"].astype(str).to_numpy()

    print(f"Loaded stimulus-level table: {df.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Subjects: {df['Subject_ID'].nunique()} | Stimulus rows: {len(df)}")
    print(f"Tasks: {df['Task'].nunique()} | Sleep mean={y.mean():.2f}, std={y.std():.2f}, range=[{y.min()}, {y.max()}]")

    logo = LeaveOneGroupOut()
    unique_subjects = pd.unique(groups)
    if len(unique_subjects) < 2:
        raise ValueError("At least 2 subjects are required for leave-one-subject-out CV.")

    stimulus_predictions = []
    subject_predictions = []
    dummy_subject_predictions = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        test_df = df.iloc[test_idx].copy()
        test_subject = str(test_df["Subject_ID"].iloc[0])
        actual_subject_hours = float(test_df["Sleep_Hours"].iloc[0])

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        dummy_pred = dummy.predict(X_test)

        fold_stimulus = test_df[["Subject_ID", "Task", "Sleep_Hours", "Num_Windows"]].copy()
        fold_stimulus["Fold"] = fold
        fold_stimulus["Predicted_Sleep_Hours"] = y_pred
        fold_stimulus["Dummy_Predicted_Sleep_Hours"] = dummy_pred
        stimulus_predictions.extend(fold_stimulus.to_dict("records"))

        pred_mean = float(np.mean(y_pred))
        pred_median = float(np.median(y_pred))
        dummy_mean = float(np.mean(dummy_pred))

        subject_predictions.append(
            {
                "Fold": fold,
                "Subject_ID": test_subject,
                "Actual_Sleep_Hours": actual_subject_hours,
                "Predicted_Sleep_Hours_MeanAgg": pred_mean,
                "Predicted_Sleep_Hours_MedianAgg": pred_median,
                "Num_Stimuli": len(test_df),
            }
        )
        dummy_subject_predictions.append(
            {
                "Fold": fold,
                "Subject_ID": test_subject,
                "Actual_Sleep_Hours": actual_subject_hours,
                "Predicted_Sleep_Hours_DummyMean": dummy_mean,
                "Num_Stimuli": len(test_df),
            }
        )

        print(
            f"Fold {fold}/{len(unique_subjects)}: subject={test_subject}, "
            f"stimuli={len(test_df)}, actual={actual_subject_hours:.2f}, "
            f"pred_mean={pred_mean:.2f}, pred_median={pred_median:.2f}"
        )

    stimulus_df = pd.DataFrame(stimulus_predictions)
    subject_df = pd.DataFrame(subject_predictions)
    dummy_df = pd.DataFrame(dummy_subject_predictions)

    actual = subject_df["Actual_Sleep_Hours"].to_numpy()
    pred_mean = subject_df["Predicted_Sleep_Hours_MeanAgg"].to_numpy()
    pred_median = subject_df["Predicted_Sleep_Hours_MedianAgg"].to_numpy()
    pred_dummy = dummy_df["Predicted_Sleep_Hours_DummyMean"].to_numpy()

    rmse_mean = rmse(actual, pred_mean)
    mae_mean = float(mean_absolute_error(actual, pred_mean))
    r2_mean = float(r2_score(actual, pred_mean))

    rmse_median = rmse(actual, pred_median)
    mae_median = float(mean_absolute_error(actual, pred_median))
    r2_median = float(r2_score(actual, pred_median))

    rmse_dummy = rmse(actual, pred_dummy)
    mae_dummy = float(mean_absolute_error(actual, pred_dummy))
    r2_dummy = float(r2_score(actual, pred_dummy))

    stimulus_df.to_csv(OUT_DIR / "stimulus_level_regression_predictions.csv", index=False)
    subject_df.to_csv(OUT_DIR / "subject_level_regression_predictions.csv", index=False)
    dummy_df.to_csv(OUT_DIR / "subject_level_dummy_predictions.csv", index=False)

    metrics_text = (
        f"Input file: {INPUT_FILE}\n"
        f"Subjects: {df['Subject_ID'].nunique()}\n"
        f"Stimulus rows: {len(df)}\n"
        f"Tasks: {df['Task'].nunique()}\n"
        f"Features: {len(feature_cols)}\n"
        f"RandomForest n_estimators: {N_ESTIMATORS}\n"
        f"Evaluation: Leave-One-Subject-Out CV\n"
        f"Aggregation: stimulus predictions -> subject mean or median\n\n"
        f"Dummy baseline (subject aggregated):\n"
        f"  RMSE: {rmse_dummy:.4f}\n"
        f"  MAE : {mae_dummy:.4f}\n"
        f"  R²  : {r2_dummy:.4f}\n\n"
        f"RandomForest subject mean aggregation:\n"
        f"  RMSE: {rmse_mean:.4f}\n"
        f"  MAE : {mae_mean:.4f}\n"
        f"  R²  : {r2_mean:.4f}\n\n"
        f"RandomForest subject median aggregation:\n"
        f"  RMSE: {rmse_median:.4f}\n"
        f"  MAE : {mae_median:.4f}\n"
        f"  R²  : {r2_median:.4f}\n"
    )
    (OUT_DIR / "stimulus_regression_metrics.txt").write_text(metrics_text)

    save_scatter(
        subject_df.rename(columns={"Predicted_Sleep_Hours_MeanAgg": "Predicted_Sleep_Hours"}),
        "Stimulus-Level Regression\nSubject Mean Aggregation",
        "subject_regression_mean_scatter.png",
        rmse_mean,
        r2_mean,
    )
    save_scatter(
        subject_df.rename(columns={"Predicted_Sleep_Hours_MedianAgg": "Predicted_Sleep_Hours"}),
        "Stimulus-Level Regression\nSubject Median Aggregation",
        "subject_regression_median_scatter.png",
        rmse_median,
        r2_median,
    )

    print("\nSubject-level regression results")
    print(f"Dummy baseline RMSE : {rmse_dummy:.4f}")
    print(f"RF mean-agg RMSE    : {rmse_mean:.4f}")
    print(f"RF mean-agg MAE     : {mae_mean:.4f}")
    print(f"RF mean-agg R²      : {r2_mean:.4f}")
    print(f"RF median-agg RMSE  : {rmse_median:.4f}")
    print(f"RF median-agg MAE   : {mae_median:.4f}")
    print(f"RF median-agg R²    : {r2_median:.4f}")
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
