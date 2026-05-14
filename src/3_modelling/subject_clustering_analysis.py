"""
Subject-level unsupervised clustering on all_task_ready.csv.

This script does not try to predict Sleep_Hours directly.
Instead, it asks:
  - Do subjects naturally cluster based on their EEG/task summary features?
  - If clusters exist, do they differ in Sleep_Hours?

Models used:
  - KMeans
  - AgglomerativeClustering

Outputs:
  results/subject_clustering_analysis/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).parent.parent.parent
DATA_FILE = BASE_DIR / "pilot_files" / "all_task_ready.csv"
OUT_DIR = BASE_DIR / "results" / "subject_clustering_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 3
RANDOM_STATE = 42
USE_PM_FEATURES = True


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def build_subject_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    eeg_cols = [c for c in df.columns if c.startswith("POW.")]
    pm_cols = [c for c in df.columns if c.startswith("PM.")]
    numeric_cols = eeg_cols + (pm_cols if USE_PM_FEATURES else [])

    work = df.dropna(subset=["Subject_ID", "Sleep_Hours"] + numeric_cols).copy()
    subject_df = (
        work.groupby("Subject_ID")
        .agg(
            {
                **{c: "mean" for c in numeric_cols},
                "Sleep_Hours": "first",
                "Gender": "first",
            }
        )
        .reset_index()
    )
    return subject_df, numeric_cols


def cluster_summary(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(label_col)
        .agg(
            N_Subjects=("Subject_ID", "count"),
            Mean_Sleep_Hours=("Sleep_Hours", "mean"),
            Std_Sleep_Hours=("Sleep_Hours", "std"),
            Min_Sleep_Hours=("Sleep_Hours", "min"),
            Max_Sleep_Hours=("Sleep_Hours", "max"),
        )
        .reset_index()
    )
    return summary


def save_pca_plot(df: pd.DataFrame, label_col: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster_id in sorted(df[label_col].unique()):
        sub = df[df[label_col] == cluster_id]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=55,
            alpha=0.8,
            label=f"Cluster {cluster_id}",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)


def save_sleep_bar(summary: pd.DataFrame, cluster_col: str, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = summary[cluster_col].astype(str).tolist()
    means = summary["Mean_Sleep_Hours"].to_numpy()
    errs = summary["Std_Sleep_Hours"].fillna(0).to_numpy()
    ax.bar(labels, means, yerr=errs, capsize=4, color="steelblue")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean Sleep Hours")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    raw_df = pd.read_csv(DATA_FILE)
    subject_df, feature_cols = build_subject_table(raw_df)

    X = subject_df[feature_cols].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_scaled)
    subject_df["PC1"] = pcs[:, 0]
    subject_df["PC2"] = pcs[:, 1]

    print(f"Loaded {DATA_FILE.name}")
    print(f"Subjects used: {len(subject_df)}")
    print(f"Features used: {len(feature_cols)}")
    print(
        f"Sleep hours — mean={subject_df['Sleep_Hours'].mean():.2f}, "
        f"std={subject_df['Sleep_Hours'].std():.2f}, "
        f"range=[{subject_df['Sleep_Hours'].min():.1f}, {subject_df['Sleep_Hours'].max():.1f}]"
    )

    print_header(f"KMEANS CLUSTERING (k={N_CLUSTERS})")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    subject_df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)
    kmeans_silhouette = silhouette_score(X_scaled, subject_df["KMeans_Cluster"])
    kmeans_summary = cluster_summary(subject_df, "KMeans_Cluster")
    print(f"Silhouette score: {kmeans_silhouette:.4f}")
    print(kmeans_summary.to_string(index=False))

    print_header(f"AGGLOMERATIVE CLUSTERING (k={N_CLUSTERS})")
    agg = AgglomerativeClustering(n_clusters=N_CLUSTERS)
    subject_df["Agglomerative_Cluster"] = agg.fit_predict(X_scaled)
    agg_silhouette = silhouette_score(X_scaled, subject_df["Agglomerative_Cluster"])
    agg_summary = cluster_summary(subject_df, "Agglomerative_Cluster")
    print(f"Silhouette score: {agg_silhouette:.4f}")
    print(agg_summary.to_string(index=False))

    subject_df.to_csv(OUT_DIR / "subject_cluster_assignments.csv", index=False)
    kmeans_summary.to_csv(OUT_DIR / "kmeans_cluster_summary.csv", index=False)
    agg_summary.to_csv(OUT_DIR / "agglomerative_cluster_summary.csv", index=False)

    save_pca_plot(
        subject_df,
        "KMeans_Cluster",
        f"KMeans clusters on subject EEG summaries (k={N_CLUSTERS})",
        "kmeans_pca.png",
    )
    save_pca_plot(
        subject_df,
        "Agglomerative_Cluster",
        f"Agglomerative clusters on subject EEG summaries (k={N_CLUSTERS})",
        "agglomerative_pca.png",
    )
    save_sleep_bar(
        kmeans_summary,
        "KMeans_Cluster",
        "kmeans_sleep_hours.png",
        "KMeans cluster mean sleep hours",
    )
    save_sleep_bar(
        agg_summary,
        "Agglomerative_Cluster",
        "agglomerative_sleep_hours.png",
        "Agglomerative cluster mean sleep hours",
    )

    summary_text = (
        f"Input file: {DATA_FILE}\n"
        f"Subjects used: {len(subject_df)}\n"
        f"Features used: {len(feature_cols)}\n"
        f"Clustering target interpretation only: Sleep_Hours\n\n"
        f"KMeans silhouette: {kmeans_silhouette:.4f}\n"
        f"{kmeans_summary.to_string(index=False)}\n\n"
        f"Agglomerative silhouette: {agg_silhouette:.4f}\n"
        f"{agg_summary.to_string(index=False)}\n"
    )
    (OUT_DIR / "summary.txt").write_text(summary_text)

    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
