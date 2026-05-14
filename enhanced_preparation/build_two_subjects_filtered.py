import os

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

METADATA_PATH = os.path.join(PROJECT_ROOT, "DatasetSub_shorten.csv")
REFERENCE_TASKS_PATH = os.path.join(PROJECT_ROOT, "pilot_files", "all_task_ready.csv")

OUTPUT_DATA = os.path.join(SCRIPT_DIR, "all_task_ready_2subjects_filtered.csv")
OUTPUT_DIAG = os.path.join(SCRIPT_DIR, "all_task_ready_2subjects_filtered_diagnostics.csv")

SUBJECT_IDS = ["BA9", "BA39"]
MIN_OVERALL_CQ = 0.7
MIN_CHANNEL_CQ = 2


def quality_mask(data: pd.DataFrame) -> pd.Series:
    mask = data["EEG.Interpolated"] == 0

    overall_cq = pd.to_numeric(data["CQ.Overall"], errors="coerce")
    if overall_cq.dropna().max() is not None and overall_cq.dropna().max() <= 1.0:
        mask &= overall_cq >= MIN_OVERALL_CQ
    else:
        mask &= overall_cq >= (MIN_OVERALL_CQ * 100)

    channel_cq_cols = [c for c in data.columns if c.startswith("CQ.") and c != "CQ.Overall"]
    channel_cq = data[channel_cq_cols].apply(pd.to_numeric, errors="coerce")
    mask &= (channel_cq >= MIN_CHANNEL_CQ).all(axis=1)

    return mask


def main() -> None:
    print("--- Loading metadata and reference tasks ---")
    metadata_df = pd.read_csv(METADATA_PATH, sep=";", header=1, decimal=",")
    metadata_df = metadata_df.dropna(subset=["ID"]).copy()
    metadata_df["ID_upper"] = metadata_df["ID"].astype(str).str.strip().str.upper()
    selected_meta = metadata_df[metadata_df["ID_upper"].isin(SUBJECT_IDS)].copy()

    reference_df = pd.read_csv(REFERENCE_TASKS_PATH)
    allowed_tasks = set(reference_df["Task"].astype(str).str.strip().str.lower())

    print(selected_meta[["ID", "Average Sleep Hours per Night", "Gender"]].to_string(index=False))
    print(f"Allowed tasks inferred from pilot_files/all_task_ready.csv: {len(allowed_tasks)}")

    all_rows = []
    diagnostics = []

    for _, row in selected_meta.iterrows():
        subject_id = str(row["ID"]).strip()
        folder_name = subject_id.lower()
        data_path = os.path.join(DATA_DIR, folder_name, "data.csv")
        marker_path = os.path.join(DATA_DIR, folder_name, "marker.csv")

        print(f"\nProcessing {subject_id}...")
        data = pd.read_csv(data_path, header=1, low_memory=False)
        markers = pd.read_csv(marker_path)

        raw_rows = len(data)
        clean_data = data[quality_mask(data)].copy()
        clean_rows = len(clean_data)

        pow_cols = [c for c in clean_data.columns if "POW." in c]
        pm_cols = [c for c in clean_data.columns if "PM." in c and ".Scaled" in c]
        metrics_to_avg = pow_cols + pm_cols

        clean_data["MarkerIndex"] = pd.to_numeric(clean_data["MarkerIndex"], errors="coerce")
        clean_data["Timestamp"] = pd.to_numeric(clean_data["Timestamp"], errors="coerce")

        id_to_task = dict(zip(markers["marker_id"], markers["type"]))

        kept = 0
        skipped_not_allowed = 0

        for marker_id, task_type in id_to_task.items():
            task_clean = str(task_type).strip().lower()

            if task_clean not in allowed_tasks:
                skipped_not_allowed += 1
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
                (clean_data["Timestamp"] >= t_start) &
                (clean_data["Timestamp"] <= t_end)
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

            task_averages = task_segment[metrics_to_avg].mean().to_dict()
            task_averages.update(
                {
                    "Subject_ID": subject_id,
                    "Task": task_type,
                    "Sleep_Hours": float(row["Average Sleep Hours per Night"]),
                    "Gender": row["Gender"],
                }
            )
            all_rows.append(task_averages)

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
            kept += 1

        print(f"  kept tasks: {kept}")
        print(f"  skipped by stimuli filter: {skipped_not_allowed}")

    final_df = pd.DataFrame(all_rows)
    diag_df = pd.DataFrame(diagnostics)

    if "POW.AF3.Theta" in final_df.columns:
        final_df = final_df.dropna(subset=["POW.AF3.Theta"])

    final_df.to_csv(OUTPUT_DATA, index=False)
    diag_df.to_csv(OUTPUT_DIAG, index=False)

    print(f"\nSaved filtered dataset: {OUTPUT_DATA}")
    print(f"Saved diagnostics: {OUTPUT_DIAG}")
    print(
        f"Rows: {len(final_df)} | Subjects: {final_df['Subject_ID'].nunique()} | "
        f"Unique tasks: {final_df['Task'].nunique()}"
    )


if __name__ == "__main__":
    main()
