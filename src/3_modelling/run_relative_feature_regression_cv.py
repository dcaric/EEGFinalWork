"""
Try stronger regression features from the stimulus-level EEG table.

This experiment is designed for the current 83-subject dataset, where each
subject has many stimulus rows but only one sleep-hours label. To avoid
leakage, evaluation is always subject-level leave-one-subject-out CV.

Input:
  enhanced_preparation/all_task_stimulus_median_ready.csv

Feature views compared:
  1. raw_summary
     Per-subject summaries of the raw stimulus-median EEG/PM features.

  2. baseline_relative_summary
     Per-subject summaries after subtracting the subject's baseline_talk row.

  3. task_family_relative
     Baseline-relative features grouped by task family, e.g. frontal, gyrus,
     image, temporal, valdo.

  4. task_wide_relative
     One feature per task x EEG/PM column, expressed relative to baseline_talk.

Why this script exists:
  The earlier RandomForest regression on stimulus medians had R2 near zero.
  This script tests the most defensible next idea: use within-subject relative
  EEG response patterns instead of only absolute EEG power values.

Saved outputs:
  results/relative_feature_regression_cv/
    relative_feature_regression_metrics.csv
    relative_feature_regression_predictions.csv
    relative_feature_regression_summary.txt
    relative_feature_regression_r2_bar.png
    best_relative_feature_regression_scatter.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_INPUT = BASE_DIR / "enhanced_preparation" / "all_task_stimulus_median_ready.csv"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "relative_feature_regression_cv"

BASELINE_TASK = "baseline_talk"
RANDOM_STATE = 42


@dataclass(frozen=True)
class FeatureView:
    name: str
    frame: pd.DataFrame


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def signed_log1p(values: pd.DataFrame | pd.Series | np.ndarray) -> pd.DataFrame | pd.Series | np.ndarray:
    """Compress skewed positive power values while still being safe for negatives."""
    return np.sign(values) * np.log1p(np.abs(values))


def iqr(series_or_frame: pd.DataFrame) -> pd.Series:
    return series_or_frame.quantile(0.75) - series_or_frame.quantile(0.25)


def task_family(task: str) -> str:
    if task == BASELINE_TASK:
        return "baseline"
    if task.startswith("frontal"):
        return "frontal"
    if task.startswith("gyrus"):
        return "gyrus"
    if task.startswith("image"):
        return "image"
    if task.startswith("imagine"):
        return "imagine"
    if task.startswith("occipital"):
        return "occipital"
    if task.startswith("parietal"):
        return "parietal"
    if task.startswith("prefrontal"):
        return "prefrontal"
    if task.startswith("saying_words"):
        return "verbal"
    if task.startswith("temporal"):
        return "temporal"
    if task.startswith("valdo"):
        return "valdo"
    return "other"


def add_prefixed_values(target: dict[str, float], prefix: str, values: pd.Series) -> None:
    for col, value in values.items():
        target[f"{prefix}__{col}"] = float(value)


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    targets = (
        df.groupby("Subject_ID", as_index=True)
        .agg(Sleep_Hours=("Sleep_Hours", "first"), Num_Stimuli=("Task", "nunique"))
        .sort_index()
    )
    return targets


def subject_baseline(sub_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    baseline_rows = sub_df.loc[sub_df["Task"] == BASELINE_TASK, feature_cols]
    if not baseline_rows.empty:
        return baseline_rows.iloc[0].astype(float)
    return sub_df[feature_cols].median(numeric_only=True)


def build_raw_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for subject_id, sub_df in df.groupby("Subject_ID", sort=True):
        values = signed_log1p(sub_df[feature_cols].astype(float))
        row: dict[str, float | str] = {"Subject_ID": subject_id}
        add_prefixed_values(row, "raw_median", values.median())
        add_prefixed_values(row, "raw_mean", values.mean())
        add_prefixed_values(row, "raw_std", values.std(ddof=0))
        add_prefixed_values(row, "raw_iqr", iqr(values))
        row["meta_num_stimuli"] = float(sub_df["Task"].nunique())
        rows.append(row)
    return pd.DataFrame(rows).set_index("Subject_ID").sort_index()


def build_baseline_relative_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for subject_id, sub_df in df.groupby("Subject_ID", sort=True):
        baseline = subject_baseline(sub_df, feature_cols)
        non_baseline = sub_df.loc[sub_df["Task"] != BASELINE_TASK, feature_cols].astype(float)
        if non_baseline.empty:
            non_baseline = sub_df[feature_cols].astype(float)

        delta = non_baseline - baseline
        log_delta = signed_log1p(non_baseline) - signed_log1p(baseline)

        row: dict[str, float | str] = {"Subject_ID": subject_id}
        add_prefixed_values(row, "baseline_log", signed_log1p(baseline))
        add_prefixed_values(row, "delta_median", delta.median())
        add_prefixed_values(row, "delta_mean", delta.mean())
        add_prefixed_values(row, "delta_std", delta.std(ddof=0))
        add_prefixed_values(row, "delta_iqr", iqr(delta))
        add_prefixed_values(row, "log_delta_median", log_delta.median())
        add_prefixed_values(row, "log_delta_mean", log_delta.mean())
        add_prefixed_values(row, "log_delta_std", log_delta.std(ddof=0))
        add_prefixed_values(row, "log_delta_iqr", iqr(log_delta))
        row["meta_num_stimuli"] = float(sub_df["Task"].nunique())
        rows.append(row)
    return pd.DataFrame(rows).set_index("Subject_ID").sort_index()


def build_task_family_relative(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["Task_Family"] = df["Task"].map(task_family)

    for subject_id, sub_df in df.groupby("Subject_ID", sort=True):
        baseline = subject_baseline(sub_df, feature_cols)
        row: dict[str, float | str] = {"Subject_ID": subject_id}

        for family, family_df in sub_df.groupby("Task_Family", sort=True):
            if family == "baseline":
                continue
            values = family_df[feature_cols].astype(float)
            delta = values - baseline
            log_delta = signed_log1p(values) - signed_log1p(baseline)
            add_prefixed_values(row, f"family_{family}_delta_median", delta.median())
            add_prefixed_values(row, f"family_{family}_log_delta_median", log_delta.median())

        row["meta_num_stimuli"] = float(sub_df["Task"].nunique())
        rows.append(row)

    return pd.DataFrame(rows).set_index("Subject_ID").sort_index()


def build_task_wide_relative(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for subject_id, sub_df in df.groupby("Subject_ID", sort=True):
        baseline = subject_baseline(sub_df, feature_cols)
        row: dict[str, float | str] = {"Subject_ID": subject_id}

        for _, task_row in sub_df.sort_values("Task").iterrows():
            task = str(task_row["Task"])
            if task == BASELINE_TASK:
                continue
            values = task_row[feature_cols].astype(float)
            log_delta = signed_log1p(values) - signed_log1p(baseline)
            safe_task = task.replace(" ", "_")
            add_prefixed_values(row, f"task_{safe_task}_log_delta", log_delta)

        row["meta_num_stimuli"] = float(sub_df["Task"].nunique())
        rows.append(row)

    return pd.DataFrame(rows).set_index("Subject_ID").sort_index()


def build_feature_views(df: pd.DataFrame, feature_cols: list[str]) -> list[FeatureView]:
    raw_summary = build_raw_summary(df, feature_cols)
    baseline_relative = build_baseline_relative_summary(df, feature_cols)
    family_relative = build_task_family_relative(df, feature_cols)
    task_wide_relative = build_task_wide_relative(df, feature_cols)

    combined_relative = pd.concat(
        [
            baseline_relative.add_prefix("summary__"),
            family_relative.add_prefix("family__"),
        ],
        axis=1,
    )

    return [
        FeatureView("raw_summary", raw_summary),
        FeatureView("baseline_relative_summary", baseline_relative),
        FeatureView("task_family_relative", family_relative),
        FeatureView("task_wide_relative", task_wide_relative),
        FeatureView("combined_relative", combined_relative),
    ]


def model_factory(name: str, n_features: int, rf_trees: int) -> Callable[[], object]:
    alphas = np.logspace(-4, 4, 25)

    def dummy() -> DummyRegressor:
        return DummyRegressor(strategy="mean")

    def ridge() -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=alphas)),
            ]
        )

    def elasticnet() -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.8, 1.0],
                        alphas=np.logspace(-4, 2, 30),
                        cv=5,
                        max_iter=50000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    def elasticnet_fast() -> GridSearchCV:
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        return GridSearchCV(
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        ElasticNet(
                            max_iter=20000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            param_grid={
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            n_jobs=-1,
        )

    def pls() -> GridSearchCV:
        max_components = int(max(1, min(10, n_features, 20)))
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        return GridSearchCV(
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", PLSRegression(scale=False)),
                ]
            ),
            param_grid={"model__n_components": list(range(1, max_components + 1))},
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            n_jobs=-1,
        )

    def randomforest() -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=rf_trees,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    factories: dict[str, Callable[[], object]] = {
        "DummyMean": dummy,
        "RidgeCV": ridge,
        "ElasticNetCV": elasticnet,
        "ElasticNetFast": elasticnet_fast,
        "PLSRegression": pls,
        "RandomForest": randomforest,
    }
    if name not in factories:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(factories)}")
    return factories[name]


def extract_model_details(model: object) -> str:
    if isinstance(model, GridSearchCV):
        return str(model.best_params_)
    if isinstance(model, Pipeline):
        final_model = model.named_steps.get("model")
        details = []
        if hasattr(final_model, "alpha_"):
            details.append(f"alpha={final_model.alpha_:.6g}")
        if hasattr(final_model, "l1_ratio_"):
            details.append(f"l1_ratio={final_model.l1_ratio_:.6g}")
        return ", ".join(details)
    return ""


def evaluate_view(
    view: FeatureView,
    targets: pd.DataFrame,
    model_names: list[str],
    rf_trees: int,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    aligned = targets.join(view.frame, how="inner")
    y = aligned["Sleep_Hours"].to_numpy(dtype=float)
    subjects = aligned.index.astype(str).to_numpy()
    X_df = aligned.drop(columns=["Sleep_Hours", "Num_Stimuli"], errors="ignore")
    X = X_df.to_numpy(dtype=float)

    if len(subjects) < 3:
        raise ValueError("At least 3 subjects are required for this evaluation.")

    logo = LeaveOneGroupOut()
    metrics = []
    predictions = []

    for model_name in model_names:
        factory = model_factory(model_name, n_features=X.shape[1], rf_trees=rf_trees)
        y_pred = np.zeros_like(y, dtype=float)
        fold_details = []

        print(
            f"Evaluating feature_view={view.name:<28} model={model_name:<14} "
            f"subjects={len(subjects)} features={X.shape[1]}"
        )

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, subjects), start=1):
            model = factory()
            model.fit(X[train_idx], y[train_idx])
            pred = np.asarray(model.predict(X[test_idx])).ravel()
            y_pred[test_idx] = pred
            details = extract_model_details(model)
            if details:
                fold_details.append(details)

            predictions.append(
                {
                    "Feature_View": view.name,
                    "Model": model_name,
                    "Fold": fold,
                    "Subject_ID": subjects[test_idx][0],
                    "Actual_Sleep_Hours": float(y[test_idx][0]),
                    "Predicted_Sleep_Hours": float(pred[0]),
                }
            )

        r2 = float(r2_score(y, y_pred))
        result = {
            "Feature_View": view.name,
            "Model": model_name,
            "Subjects": int(len(subjects)),
            "Features": int(X.shape[1]),
            "RMSE": rmse(y, y_pred),
            "MAE": float(mean_absolute_error(y, y_pred)),
            "R2": r2,
            "PearsonR": float(np.corrcoef(y, y_pred)[0, 1]) if np.std(y_pred) > 0 else np.nan,
            "Prediction_Min": float(np.min(y_pred)),
            "Prediction_Max": float(np.max(y_pred)),
            "Fold_Model_Details_Example": fold_details[0] if fold_details else "",
        }
        metrics.append(result)

    return metrics, predictions


def save_r2_bar(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = metrics_df.sort_values("R2", ascending=True).copy()
    labels = plot_df["Feature_View"] + " / " + plot_df["Model"]

    fig_height = max(5, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = ["#2b8cbe" if value >= 0 else "#de6b48" for value in plot_df["R2"]]
    ax.barh(labels, plot_df["R2"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Subject-Level R²")
    ax.set_title("Relative Feature Regression: Leave-One-Subject-Out CV")
    fig.tight_layout()
    fig.savefig(out_dir / "relative_feature_regression_r2_bar.png", dpi=160)
    plt.close(fig)


def save_best_scatter(metrics_df: pd.DataFrame, predictions_df: pd.DataFrame, out_dir: Path) -> None:
    best = metrics_df.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0]
    pred_df = predictions_df[
        (predictions_df["Feature_View"] == best["Feature_View"]) & (predictions_df["Model"] == best["Model"])
    ].copy()

    actual = pred_df["Actual_Sleep_Hours"].to_numpy()
    pred = pred_df["Predicted_Sleep_Hours"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(actual, pred, s=70, alpha=0.78, edgecolors="#24516b", facecolors="#9ecae1")
    mn = min(actual.min(), pred.min())
    mx = max(actual.max(), pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Sleep Hours")
    ax.set_ylabel("Predicted Sleep Hours")
    ax.set_title(
        f"Best Relative-Feature Regression\n"
        f"{best['Feature_View']} / {best['Model']} | RMSE={best['RMSE']:.3f}, R²={best['R2']:.3f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "best_relative_feature_regression_scatter.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline-relative and task-relative regression features for sleep-hours prediction."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Stimulus-median CSV input file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for saved outputs.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["DummyMean", "RidgeCV", "PLSRegression", "RandomForest"],
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--feature-views",
        nargs="+",
        default=[
            "raw_summary",
            "baseline_relative_summary",
            "task_family_relative",
            "task_wide_relative",
            "combined_relative",
        ],
        help="Feature views to evaluate.",
    )
    parser.add_argument("--rf-trees", type=int, default=300, help="RandomForest trees.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    feature_cols = [c for c in df.columns if c.startswith("POW.") or c.startswith("PM.")]
    if not feature_cols:
        raise ValueError(f"No POW./PM. feature columns found in {args.input}")
    required = {"Subject_ID", "Task", "Sleep_Hours"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    targets = build_targets(df)
    views = build_feature_views(df, feature_cols)
    selected_views = [view for view in views if view.name in set(args.feature_views)]
    if not selected_views:
        raise ValueError(f"No selected feature views matched: {args.feature_views}")

    print(f"Loaded stimulus-level table: {df.shape}")
    print(f"Subjects: {targets.shape[0]} | Stimulus rows: {len(df)} | Tasks: {df['Task'].nunique()}")
    print(f"Base EEG/PM features: {len(feature_cols)}")
    print(f"Sleep mean={targets['Sleep_Hours'].mean():.2f}, std={targets['Sleep_Hours'].std():.2f}, "
          f"range=[{targets['Sleep_Hours'].min()}, {targets['Sleep_Hours'].max()}]")
    print()

    all_metrics = []
    all_predictions = []
    for view in selected_views:
        metrics, predictions = evaluate_view(view, targets, args.models, args.rf_trees)
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

    metrics_df = pd.DataFrame(all_metrics).sort_values(["R2", "RMSE"], ascending=[False, True])
    predictions_df = pd.DataFrame(all_predictions)

    metrics_file = args.out_dir / "relative_feature_regression_metrics.csv"
    predictions_file = args.out_dir / "relative_feature_regression_predictions.csv"
    summary_file = args.out_dir / "relative_feature_regression_summary.txt"

    metrics_df.to_csv(metrics_file, index=False)
    predictions_df.to_csv(predictions_file, index=False)
    save_r2_bar(metrics_df, args.out_dir)
    save_best_scatter(metrics_df, predictions_df, args.out_dir)

    best = metrics_df.iloc[0]
    summary = (
        "Relative Feature Regression CV\n"
        "==============================\n\n"
        f"Input file: {args.input}\n"
        f"Subjects: {targets.shape[0]}\n"
        f"Stimulus rows: {len(df)}\n"
        f"Tasks: {df['Task'].nunique()}\n"
        f"Base EEG/PM features: {len(feature_cols)}\n"
        "Evaluation: Leave-One-Subject-Out CV at subject level\n\n"
        "Best result:\n"
        f"  Feature view: {best['Feature_View']}\n"
        f"  Model       : {best['Model']}\n"
        f"  RMSE        : {best['RMSE']:.4f}\n"
        f"  MAE         : {best['MAE']:.4f}\n"
        f"  R2          : {best['R2']:.4f}\n"
        f"  Pearson r   : {best['PearsonR']:.4f}\n\n"
        "Important interpretation:\n"
        "  Positive R2 means the model improved over predicting the training-set mean.\n"
        "  R2 near 0 means the model is only about as useful as the mean baseline.\n"
        "  Negative R2 means the model is worse than the mean baseline.\n"
        "  If a positive result appears here, it should still be treated as exploratory\n"
        "  until checked with a permutation test or a fresh independent cohort.\n\n"
        "All results sorted by R2:\n"
        f"{metrics_df[['Feature_View', 'Model', 'RMSE', 'MAE', 'R2', 'PearsonR', 'Features']].to_string(index=False)}\n"
    )
    summary_file.write_text(summary)

    print("\nTop results")
    print(metrics_df[["Feature_View", "Model", "RMSE", "MAE", "R2", "PearsonR", "Features"]].head(12).to_string(index=False))
    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
