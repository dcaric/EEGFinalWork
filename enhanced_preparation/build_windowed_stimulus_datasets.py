import os
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_PATH = PROJECT_ROOT / "DatasetSub_shorten.csv"
REFERENCE_TASKS_PATH = PROJECT_ROOT / "pilot_files" / "all_task_ready.csv"

WINDOW_OUTPUT = SCRIPT_DIR / "all_task_window_ready.csv"
MEDIAN_OUTPUT = SCRIPT_DIR / "all_task_stimulus_median_ready.csv"
DIAG_OUTPUT = SCRIPT_DIR / "all_task_window_ready_diagnostics.csv"

MIN_OVERALL_CQ = 0.7
MIN_CHANNEL_CQ = 2


def quality_mask(data: pd.DataFrame) -> pd.Series:
    mask = data["EEG.Interpolated"] == 0

    overall_cq = pd.to_numeric(data["CQ.Overall"], errors="coerce")
    if not overall_cq.dropna().empty and overall_cq.dropna().max() <= 1.0:
        mask &= overall_cq >= MIN_OVERALL_CQ
    else:
        mask &= overall_cq >= (MIN_OVERALL_CQ * 100)

    channel_cq_cols = [c for c in data.columns if c.startswith("CQ.") and c != "CQ.Overall"]
    channel_cq = data[channel_cq_cols].apply(pd.to_numeric, errors="coerce")
    mask &= (channel_cq >= MIN_CHANNEL_CQ).all(axis=1)
    return mask


def load_subject_metadata() -> pd.DataFrame:
    metadata_df = pd.read_csv(METADATA_PATH, sep=";", header=1, decimal=",")
    metadata_df = metadata_df.dropna(subset=["ID"]).copy()
    metadata_df["ID_upper"] = metadata_df["ID"].astype(str).str.strip().str.upper()
    metadata_df["Folder_Name"] = metadata_df["ID_upper"].str.lower()
    metadata_df["Folder_Exists"] = metadata_df["Folder_Name"].apply(
        lambda folder: (DATA_DIR / folder).is_dir()
    )
    metadata_df["Sleep_Hours_Num"] = pd.to_numeric(
        metadata_df["Average Sleep Hours per Night"], errors="coerce"
    )
    return metadata_df[metadata_df["Folder_Exists"]].copy()


def load_allowed_tasks() -> set[str]:
    reference_df = pd.read_csv(REFERENCE_TASKS_PATH)
    return set(reference_df["Task"].astype(str).str.strip().str.lower())


def main() -> None:
    print("--- Building window-level and stimulus-level datasets ---")
    available_df = load_subject_metadata()
    allowed_tasks = load_allowed_tasks()

    print(f"Subjects with local folders: {len(available_df)}")
    print(f"Allowed tasks inferred from pilot_files/all_task_ready.csv: {len(allowed_tasks)}")

    all_window_rows = []
    diagnostics = []

    for _, meta_row in available_df.iterrows():
        subject_id = str(meta_row["ID"]).strip().upper()
        folder_name = str(meta_row["Folder_Name"]).strip()
        data_path = DATA_DIR / folder_name / "data.csv"
        marker_path = DATA_DIR / folder_name / "marker.csv"

        if not data_path.exists() or not marker_path.exists():
            continue

        print(f"\nProcessing {subject_id}...")
        data = pd.read_csv(data_path, header=1, low_memory=False)
        markers = pd.read_csv(marker_path)

        raw_rows = len(data)
        clean_data = data[quality_mask(data)].copy()
        clean_rows = len(clean_data)

        pow_cols = [c for c in clean_data.columns if c.startswith("POW.")]
        pm_cols = [c for c in clean_data.columns if c.startswith("PM.") and c.endswith(".Scaled")]
        metrics_to_keep = pow_cols + pm_cols

        clean_data["MarkerIndex"] = pd.to_numeric(clean_data["MarkerIndex"], errors="coerce")
        clean_data["Timestamp"] = pd.to_numeric(clean_data["Timestamp"], errors="coerce")
        clean_data = clean_data.sort_values("Timestamp").reset_index(drop=True)

        id_to_task = dict(zip(markers["marker_id"], markers["type"]))
        kept_tasks = 0
        kept_windows = 0

        for marker_id, task_type in id_to_task.items():
            task_clean = str(task_type).strip().lower()

            if task_clean not in allowed_tasks:
                diagnostics.append(
                    {
                        "Subject_ID": subject_id,
                        "Task": task_type,
                        "marker_id": marker_id,
                        "status": "not_allowed_by_reference_tasks",
                        "raw_rows": raw_rows,
                        "clean_rows": clean_rows,
                    }
                )
                continue

            start_rows = clean_data[clean_data["MarkerIndex"] == marker_id]
            end_rows = clean_data[clean_data["MarkerIndex"] == -marker_id]

            if start_rows.empty or end_rows.empty:
                diagnostics.append(
                    {
                        "Subject_ID": subject_id,
                        "Task": task_type,
                        "marker_id": marker_id,
                        "status": "missing_start_or_end",
                        "raw_rows": raw_rows,
                        "clean_rows": clean_rows,
                    }
                )
                continue

            t_start = start_rows["Timestamp"].iloc[0]
            t_end = end_rows["Timestamp"].iloc[0]

            if pd.isna(t_start) or pd.isna(t_end) or t_end <= t_start:
                diagnostics.append(
                    {
                        "Subject_ID": subject_id,
                        "Task": task_type,
                        "marker_id": marker_id,
                        "status": "bad_timestamps",
                        "raw_rows": raw_rows,
                        "clean_rows": clean_rows,
                        "t_start": t_start,
                        "t_end": t_end,
                    }
                )
                continue

            task_segment = clean_data[
                (clean_data["Timestamp"] >= t_start) & (clean_data["Timestamp"] <= t_end)
            ].copy()

            if task_segment.empty:
                diagnostics.append(
                    {
                        "Subject_ID": subject_id,
                        "Task": task_type,
                        "marker_id": marker_id,
                        "status": "empty_segment",
                        "raw_rows": raw_rows,
                        "clean_rows": clean_rows,
                        "t_start": t_start,
                        "t_end": t_end,
                    }
                )
                continue

            if "POW.AF3.Theta" in task_segment.columns and task_segment["POW.AF3.Theta"].isna().all():
                diagnostics.append(
                    {
                        "Subject_ID": subject_id,
                        "Task": task_type,
                        "marker_id": marker_id,
                        "status": "all_reference_fft_nan",
                        "raw_rows": raw_rows,
                        "clean_rows": clean_rows,
                        "segment_rows": len(task_segment),
                    }
                )
                continue

            task_segment = task_segment.reset_index(drop=True)
            task_segment["Window_Index"] = range(1, len(task_segment) + 1)
            task_segment["Subject_ID"] = subject_id
            task_segment["Task"] = task_type
            task_segment["Sleep_Hours"] = float(meta_row["Sleep_Hours_Num"])
            task_segment["Gender"] = meta_row["Gender"]
            task_segment["Stimulus_Subject_ID"] = (
                task_segment["Task"].astype(str) + "-" + task_segment["Subject_ID"].astype(str)
            )
            task_segment["Stimulus_Subject_Window_ID"] = task_segment.apply(
                lambda r: f"{r['Task']}-{r['Subject_ID']}-window{int(r['Window_Index'])}",
                axis=1,
            )

            keep_cols = [
                "Stimulus_Subject_Window_ID",
                "Stimulus_Subject_ID",
                "Subject_ID",
                "Task",
                "Window_Index",
                "Timestamp",
                *metrics_to_keep,
                "Sleep_Hours",
                "Gender",
            ]
            all_window_rows.append(task_segment[keep_cols])

            diagnostics.append(
                {
                    "Subject_ID": subject_id,
                    "Task": task_type,
                    "marker_id": marker_id,
                    "status": "ok",
                    "raw_rows": raw_rows,
                    "clean_rows": clean_rows,
                    "segment_rows": len(task_segment),
                    "t_start": t_start,
                    "t_end": t_end,
                    "duration_ticks": t_end - t_start,
                }
            )
            kept_tasks += 1
            kept_windows += len(task_segment)

        print(f"  kept tasks: {kept_tasks}")
        print(f"  kept windows: {kept_windows}")

    if not all_window_rows:
        raise RuntimeError("No window-level rows were generated. Check data folders and filters.")

    window_df = pd.concat(all_window_rows, ignore_index=True)
    if "POW.AF3.Theta" in window_df.columns:
        window_df = window_df.dropna(subset=["POW.AF3.Theta"]).copy()

    metric_cols = [c for c in window_df.columns if c.startswith("POW.") or c.startswith("PM.")]
    median_df = (
        window_df.groupby(["Subject_ID", "Task"], as_index=False)[metric_cols]
        .median()
        .merge(
            window_df.groupby(["Subject_ID", "Task"], as_index=False).agg(
                Sleep_Hours=("Sleep_Hours", "first"),
                Gender=("Gender", "first"),
                Num_Windows=("Window_Index", "count"),
            ),
            on=["Subject_ID", "Task"],
            how="left",
        )
    )
    median_df["Stimulus_Subject_ID"] = median_df["Task"].astype(str) + "-" + median_df["Subject_ID"].astype(str)

    diag_df = pd.DataFrame(diagnostics)

    window_df.to_csv(WINDOW_OUTPUT, index=False)
    median_df.to_csv(MEDIAN_OUTPUT, index=False)
    diag_df.to_csv(DIAG_OUTPUT, index=False)

    print("\nSaved files:")
    print(f"  Window-level dataset: {WINDOW_OUTPUT}")
    print(f"  Stimulus-median dataset: {MEDIAN_OUTPUT}")
    print(f"  Diagnostics: {DIAG_OUTPUT}")
    print(
        f"Window rows: {len(window_df)} | Subjects: {window_df['Subject_ID'].nunique()} | "
        f"Stimulus-subject pairs: {window_df['Stimulus_Subject_ID'].nunique()}"
    )
    print(
        f"Median rows: {len(median_df)} | Subjects: {median_df['Subject_ID'].nunique()} | "
        f"Unique tasks: {median_df['Task'].nunique()}"
    )


if __name__ == "__main__":
    main()
