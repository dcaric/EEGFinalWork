"""
Use all rows from pilot_files/all_task_ready.csv for:
  1. Supervised regression of Sleep_Hours with grouped CV by Subject_ID
  2. Subject-level clustering to see whether EEG-defined groups differ in sleep

Why grouped CV matters:
  Each subject appears in many task rows. We must split by Subject_ID, not by row,
  otherwise the model would leak subject-specific patterns from train to test.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).parent.parent.parent
DATA_FILE = BASE_DIR / "pilot_files" / "all_task_ready.csv"
OUT_DIR = BASE_DIR / "results" / "all_task_supervised_and_clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 5
N_CLUSTERS = 3
RANDOM_STATE = 42


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["Sleep_Hours", "Subject_ID", "Task"]).copy()

    eeg_cols = [c for c in df.columns if c.startswith("POW.")]
    pm_cols = [c for c in df.columns if c.startswith("PM.")]
    numeric_cols = eeg_cols + pm_cols

    # Remove rows with missing predictor values to keep the grouped evaluation simple.
    df = df.dropna(subset=numeric_cols).copy()

    X = df[numeric_cols + ["Task"]]
    y = df["Sleep_Hours"].to_numpy()
    groups = df["Subject_ID"].to_numpy()

    print(f"Loaded all_task_ready.csv: {df.shape}")
    print(f"Subjects: {df['Subject_ID'].nunique()} | Tasks: {df['Task'].nunique()} | Rows: {len(df)}")
    print(f"EEG features: {len(eeg_cols)} | PM features: {len(pm_cols)}")
    print(
        f"Sleep hours — mean={y.mean():.2f}, std={y.std():.2f}, "
        f"range=[{y.min():.1f}, {y.max():.1f}]"
    )

    print_header(f"SUPERVISED REGRESSION ({K}-fold GroupKFold by Subject_ID)")

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("task", OneHotEncoder(handle_unknown="ignore"), ["Task"]),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("prep", preprocess),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    dummy_model = Pipeline(
        steps=[
            ("prep", preprocess),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )

    cv = GroupKFold(n_splits=K)

    rf_pred = cross_val_predict(rf_model, X, y, cv=cv, groups=groups, n_jobs=1)
    dummy_pred = cross_val_predict(dummy_model, X, y, cv=cv, groups=groups, n_jobs=1)

    rf_rmse = np.sqrt(mean_squared_error(y, rf_pred))
    rf_mae = mean_absolute_error(y, rf_pred)
    rf_r2 = r2_score(y, rf_pred)
    dummy_rmse = np.sqrt(mean_squared_error(y, dummy_pred))
    dummy_r2 = r2_score(y, dummy_pred)

    print(f"Dummy RMSE : {dummy_rmse:.4f}")
    print(f"RF RMSE    : {rf_rmse:.4f}")
    print(f"RF MAE     : {rf_mae:.4f}")
    print(f"Dummy R²   : {dummy_r2:.4f}")
    print(f"RF R²      : {rf_r2:.4f}")

    pred_df = df[["Subject_ID", "Task", "Sleep_Hours"]].copy()
    pred_df["Predicted_RF"] = rf_pred
    pred_df["Predicted_Dummy"] = dummy_pred
    pred_df.to_csv(OUT_DIR / "all_task_grouped_regression_predictions.csv", index=False)

    print_header(f"SUBJECT-LEVEL CLUSTERING (KMeans, k={N_CLUSTERS})")

    subject_df = (
        df.groupby("Subject_ID")
        .agg(
            {
                **{c: "mean" for c in numeric_cols},
                "Sleep_Hours": "first",
            }
        )
        .reset_index()
    )

    cluster_features = subject_df[numeric_cols].to_numpy()
    scaled = StandardScaler().fit_transform(cluster_features)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    subject_df["Cluster"] = kmeans.fit_predict(scaled)

    cluster_summary = (
        subject_df.groupby("Cluster")
        .agg(
            N_Subjects=("Subject_ID", "count"),
            Mean_Sleep_Hours=("Sleep_Hours", "mean"),
            Std_Sleep_Hours=("Sleep_Hours", "std"),
            Min_Sleep_Hours=("Sleep_Hours", "min"),
            Max_Sleep_Hours=("Sleep_Hours", "max"),
        )
        .reset_index()
    )
    print(cluster_summary.to_string(index=False))

    subject_df.to_csv(OUT_DIR / "subject_clusters.csv", index=False)
    cluster_summary.to_csv(OUT_DIR / "cluster_sleep_summary.csv", index=False)

    summary_text = (
        f"Input file: {DATA_FILE}\n"
        f"Rows used: {len(df)}\n"
        f"Subjects used: {df['Subject_ID'].nunique()}\n"
        f"Tasks used: {df['Task'].nunique()}\n"
        f"Numeric features used: {len(numeric_cols)}\n\n"
        f"Grouped regression results ({K}-fold GroupKFold by Subject_ID):\n"
        f"Dummy RMSE: {dummy_rmse:.4f}\n"
        f"RF RMSE: {rf_rmse:.4f}\n"
        f"RF MAE: {rf_mae:.4f}\n"
        f"Dummy R2: {dummy_r2:.4f}\n"
        f"RF R2: {rf_r2:.4f}\n\n"
        f"Clustering summary (k={N_CLUSTERS}):\n"
        f"{cluster_summary.to_string(index=False)}\n"
    )
    (OUT_DIR / "summary.txt").write_text(summary_text)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, rf_pred, alpha=0.25, s=18)
    mn, mx = min(y.min(), rf_pred.min()), max(y.max(), rf_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
    ax.set_xlabel("Actual Sleep Hours")
    ax.set_ylabel("Predicted Sleep Hours")
    ax.set_title(f"All-task grouped RF\nRMSE={rf_rmse:.3f}h  R²={rf_r2:.3f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grouped_regression_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    order = cluster_summary["Cluster"].tolist()
    means = cluster_summary["Mean_Sleep_Hours"].to_numpy()
    errs = cluster_summary["Std_Sleep_Hours"].fillna(0).to_numpy()
    ax.bar([str(c) for c in order], means, yerr=errs, color="steelblue", capsize=4)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean Sleep Hours")
    ax.set_title("Subject clusters vs sleep hours")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cluster_sleep_hours.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
