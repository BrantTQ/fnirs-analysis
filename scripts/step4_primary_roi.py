#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats

from analysis_exclusions import exclusion_table_rows, load_participant_exclusions, participant_exclusion_reason


ROOT = Path(__file__).resolve().parents[1]

STEP2_TRIAL_PATH = ROOT / "data_clean" / "step2" / "06_behavior_trial_clean.csv"
STEP3_TRIGGER_PATH = ROOT / "data_clean" / "step3" / "06_fnirs_trigger_harmonized.csv"
STEP3_STATUS_PATH = ROOT / "data_clean" / "step3" / "07_fnirs_session_status.csv"
STEP3_CHANNEL_QC_PATH = ROOT / "data_clean" / "step3" / "08_fnirs_channel_qc.csv"
STEP3_PREPROC_PATH = ROOT / "data_clean" / "step3" / "12_preprocessed_file_manifest.csv"

CLEAN_DIR = ROOT / "data_clean" / "step4"
FIGURES_DIR = ROOT / "figures" / "step4"
REPORTS_DIR = ROOT / "reports" / "step4"
TABLES_DIR = REPORTS_DIR / "tables"
TEXT_DIR = REPORTS_DIR / "text"

ALPHA = 0.05
BASELINE_SEC = 2.0
WAVEFORM_PRE_SEC = 2.0
WAVEFORM_POST_SEC = 20.0
WAVEFORM_STEP_SEC = 0.1

SHORT_PAIRS = ["S1_D8", "S2_D9", "S3_D10", "S4_D11", "S5_D12", "S6_D13", "S7_D14", "S8_D15"]
FRONT_PAIRS = ["S2_D6", "S2_D4", "S1_D6", "S1_D3", "S5_D6", "S5_D4", "S5_D3"]
BACK_PAIRS = ["S4_D2", "S3_D1", "S6_D2", "S6_D1", "S6_D5", "S7_D2", "S7_D5", "S8_D1", "S8_D5"]
TRANSITION_PAIRS = ["S5_D7", "S4_D4", "S4_D7", "S3_D3", "S3_D7", "S6_D7"]
ALL_LONG_PAIRS = FRONT_PAIRS + BACK_PAIRS + TRANSITION_PAIRS
STEP4_MANUAL_INCLUDE_SESSIONS = {
    ("PID025", "2025-11-25_002"): "User-directed Step 4 override added on 2026-03-20: include this session despite Step 3 REVIEW_BEFORE_STEP4 status.",
    ("PID034", "2025-11-27_003"): "User-directed Step 4 override added on 2026-03-20: include this session despite Step 3 REVIEW_BEFORE_STEP4 status.",
}


def ensure_directories() -> None:
    for path in [CLEAN_DIR, FIGURES_DIR, REPORTS_DIR, TABLES_DIR, TEXT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def pair_to_channel(pair_id: str, chromophore: str) -> str:
    return f"{pair_id} {chromophore}"


def pair_list_to_channels(pair_ids: list[str], chromophore: str) -> list[str]:
    return [pair_to_channel(pair_id, chromophore) for pair_id in pair_ids]


FRONT_HBO_CHANNELS = pair_list_to_channels(FRONT_PAIRS, "hbo")
BACK_HBO_CHANNELS = pair_list_to_channels(BACK_PAIRS, "hbo")
FRONT_HBR_CHANNELS = pair_list_to_channels(FRONT_PAIRS, "hbr")
BACK_HBR_CHANNELS = pair_list_to_channels(BACK_PAIRS, "hbr")
LONG_HBO_CHANNELS = pair_list_to_channels(ALL_LONG_PAIRS, "hbo")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def sanitize_for_tex(text: Any) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    value = str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped: list[str] = []
    for char in value:
        if char == "\\":
            escaped.append(r"\textbackslash{}")
        else:
            escaped.append(replacements.get(char, char))
    return "".join(escaped)


def make_latex_table(rows: list[tuple[str, Any]], column_spec: str = r"p{0.34\linewidth}p{0.56\linewidth}") -> str:
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule", r"Metric & Value\\", r"\midrule"]
    for key, value in rows:
        lines.append(f"{sanitize_for_tex(key)} & {sanitize_for_tex(value)}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def make_latex_grid(headers: list[str], rows: list[list[Any]], column_spec: str) -> str:
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule", " & ".join(sanitize_for_tex(item) for item in headers) + r"\\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(sanitize_for_tex(item) for item in row) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def compile_report(report_path: Path) -> None:
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", report_path.name],
            cwd=report_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", report_path.name],
            cwd=report_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        pass


def format_float(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def format_p_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = float(value)
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.6f}"


def normalize_question_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = re.match(r"^\s*(\d+(?:\.\d+)?)_([A-Za-z]+)_(\d+)\s*$", text)
    if match:
        year = str(int(float(match.group(1))))
        return f"{year}_{match.group(2).upper()}_{int(match.group(3))}"
    return text


def holm_correction(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [math.nan] * len(p_values)
    running_max = 0.0
    n_tests = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed, start=1):
        adjusted_value = min(1.0, (n_tests - rank + 1) * p_value)
        running_max = max(running_max, adjusted_value)
        adjusted[idx] = running_max
    return adjusted


def run_one_sample_test(values: pd.Series, label: str) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(clean)
    if n == 0:
        return {
            "analysis_label": label,
            "n": 0,
            "mean": math.nan,
            "sd": math.nan,
            "t_stat": math.nan,
            "df": math.nan,
            "p_value": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "cohens_d": math.nan,
        }

    mean_value = float(np.mean(clean))
    sd_value = float(np.std(clean, ddof=1)) if n > 1 else 0.0
    if n == 1:
        return {
            "analysis_label": label,
            "n": n,
            "mean": mean_value,
            "sd": sd_value,
            "t_stat": math.nan,
            "df": 0,
            "p_value": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "cohens_d": math.nan,
        }

    if math.isclose(sd_value, 0.0):
        t_stat = 0.0 if math.isclose(mean_value, 0.0) else math.copysign(math.inf, mean_value)
        p_value = 1.0 if math.isclose(mean_value, 0.0) else 0.0
        ci_low = mean_value
        ci_high = mean_value
        cohens_d = 0.0 if math.isclose(mean_value, 0.0) else math.copysign(math.inf, mean_value)
    else:
        test = stats.ttest_1samp(clean, popmean=0.0)
        t_stat = float(test.statistic)
        p_value = float(test.pvalue)
        sem = sd_value / math.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_low = mean_value - t_crit * sem
        ci_high = mean_value + t_crit * sem
        cohens_d = mean_value / sd_value

    return {
        "analysis_label": label,
        "n": n,
        "mean": mean_value,
        "sd": sd_value,
        "t_stat": t_stat,
        "df": n - 1,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohens_d": cohens_d,
    }


def available_channels(raw: mne.io.BaseRaw, requested_channels: list[str]) -> list[str]:
    bads = set(raw.info["bads"])
    names = set(raw.ch_names)
    return [channel for channel in requested_channels if channel in names and channel not in bads]


def summary_reasons(*parts: str) -> str:
    return " | ".join(part for part in parts if part)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    trial_df = pd.read_csv(STEP2_TRIAL_PATH)
    trigger_df = pd.read_csv(STEP3_TRIGGER_PATH)
    status_df = pd.read_csv(STEP3_STATUS_PATH)
    channel_qc_df = pd.read_csv(STEP3_CHANNEL_QC_PATH)
    preproc_df = pd.read_csv(STEP3_PREPROC_PATH)
    exclusions = load_participant_exclusions()

    trial_df["question_id"] = trial_df["question_id"].map(normalize_question_id)
    trigger_df["question_id"] = trigger_df["question_id"].map(normalize_question_id)
    return trial_df, trigger_df, status_df, channel_qc_df, preproc_df, exclusions


def manual_override_reason(participant_id: str, session_id: str) -> str:
    return STEP4_MANUAL_INCLUDE_SESSIONS.get((participant_id, session_id), "")


def build_trial_frame(trial_df: pd.DataFrame, trigger_df: pd.DataFrame, status_df: pd.DataFrame, preproc_df: pd.DataFrame) -> pd.DataFrame:
    status_cols = [
        "participant_id",
        "session_id",
        "psychopy_file",
        "final_status",
        "preprocessing_success",
        "preprocessed_file",
        "global_analysis_excluded",
        "global_exclusion_reason",
    ]
    session_df = status_df[status_cols].merge(
        preproc_df[["participant_id", "session_id", "preprocessed_file"]].drop_duplicates(),
        on=["participant_id", "session_id"],
        how="left",
        suffixes=("", "_manifest"),
    )
    session_df["preprocessed_file"] = session_df["preprocessed_file"].fillna(session_df["preprocessed_file_manifest"])
    session_df = session_df.drop(columns=["preprocessed_file_manifest"])

    trial_join_df = trial_df.merge(
        session_df,
        left_on=["participant_id", "source_log_full_path"],
        right_on=["participant_id", "psychopy_file"],
        how="left",
    )

    question_start_df = trigger_df[trigger_df["event_type"] == "question_start"][
        ["participant_id", "session_id", "block", "trial_idx_in_block", "event_time_final", "question_id"]
    ].rename(columns={"event_time_final": "question_start_time", "question_id": "trigger_question_id"})
    answer_df = trigger_df[trigger_df["event_type"] == "answer"][
        ["participant_id", "session_id", "block", "trial_idx_in_block", "event_time_final"]
    ].rename(columns={"event_time_final": "answer_time"})

    merged_df = trial_join_df.merge(
        question_start_df,
        on=["participant_id", "session_id", "block", "trial_idx_in_block"],
        how="left",
    ).merge(
        answer_df,
        on=["participant_id", "session_id", "block", "trial_idx_in_block"],
        how="left",
    )
    merged_df["question_id_matches_trigger"] = (
        merged_df["question_id"].fillna("").astype(str) == merged_df["trigger_question_id"].fillna("").astype(str)
    )
    merged_df["question_window_duration_sec"] = merged_df["answer_time"] - merged_df["question_start_time"]
    return merged_df


def extract_trial_response(
    data: np.ndarray,
    times: np.ndarray,
    start_time: float,
    end_time: float,
    channel_indices: list[int],
) -> tuple[np.ndarray | None, str | None]:
    if not channel_indices:
        return None, "no_channels"
    baseline_start = start_time - BASELINE_SEC
    if baseline_start < times[0] - 1e-9:
        return None, "baseline_unavailable"
    if end_time > times[-1] + 1e-9:
        return None, "task_window_out_of_bounds"

    baseline_mask = (times >= baseline_start) & (times <= start_time)
    task_mask = (times >= start_time) & (times <= end_time)
    if not baseline_mask.any():
        return None, "baseline_unavailable"
    if not task_mask.any():
        return None, "task_window_out_of_bounds"

    baseline_mean = data[np.ix_(channel_indices, baseline_mask)].mean(axis=1)
    task_mean = data[np.ix_(channel_indices, task_mask)].mean(axis=1)
    return task_mean - baseline_mean, None


def extract_waveform(
    data: np.ndarray,
    times: np.ndarray,
    start_time: float,
    channel_indices: list[int],
    waveform_grid: np.ndarray,
) -> np.ndarray | None:
    if not channel_indices:
        return None
    baseline_start = start_time - WAVEFORM_PRE_SEC
    waveform_end = start_time + WAVEFORM_POST_SEC
    if baseline_start < times[0] - 1e-9 or waveform_end > times[-1] + 1e-9:
        return None

    baseline_mask = (times >= baseline_start) & (times <= start_time)
    waveform_mask = (times >= baseline_start) & (times <= waveform_end)
    if not baseline_mask.any() or not waveform_mask.any():
        return None

    baseline_mean = data[np.ix_(channel_indices, baseline_mask)].mean(axis=1, keepdims=True)
    roi_series = (data[np.ix_(channel_indices, waveform_mask)] - baseline_mean).mean(axis=0)
    rel_times = times[waveform_mask] - start_time
    return np.interp(waveform_grid, rel_times, roi_series)


def build_step4_outputs(
    merged_trial_df: pd.DataFrame,
    status_df: pd.DataFrame,
    channel_qc_df: pd.DataFrame,
    exclusions: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    waveform_grid = np.arange(-WAVEFORM_PRE_SEC, WAVEFORM_POST_SEC + 1e-9, WAVEFORM_STEP_SEC)
    trial_hbo_rows: list[dict[str, Any]] = []
    trial_hbr_rows: list[dict[str, Any]] = []
    channel_trial_rows: list[dict[str, Any]] = []
    waveform_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    inclusion_rows: list[dict[str, Any]] = []
    session_channel_meta: dict[tuple[str, str], dict[str, Any]] = {}
    channel_position_lookup: dict[str, tuple[float, float]] = {}

    manual_override_mask = status_df.apply(
        lambda row: (row["participant_id"], row["session_id"]) in STEP4_MANUAL_INCLUDE_SESSIONS,
        axis=1,
    )
    prelim_status_mask = (
        (status_df["final_status"].isin(["READY_FOR_STEP4", "READY_FOR_STEP4_WITH_WARNINGS"]) | manual_override_mask)
        & (~status_df["global_analysis_excluded"])
        & status_df["preprocessing_success"]
        & status_df["preprocessed_file"].fillna("").astype(str).ne("")
    )
    prelim_session_keys = set(
        zip(
            status_df.loc[prelim_status_mask, "participant_id"],
            status_df.loc[prelim_status_mask, "session_id"],
        )
    )

    for row in status_df.itertuples(index=False):
        session_key = (row.participant_id, row.session_id)
        row_manual_override = session_key in STEP4_MANUAL_INCLUDE_SESSIONS
        row_global_excluded = bool(row.global_analysis_excluded) if pd.notna(row.global_analysis_excluded) else False
        row_preprocessing_success = bool(row.preprocessing_success) if pd.notna(row.preprocessing_success) else False
        if row_global_excluded:
            exclusion_rows.append(
                {
                    "level": "session",
                    "participant_id": row.participant_id,
                    "session_id": row.session_id,
                    "block": "",
                    "trial_idx_in_block": "",
                    "question_id": "",
                    "reason_code": "GLOBAL_ANALYSIS_EXCLUSION",
                    "reason_detail": participant_exclusion_reason(exclusions, row.participant_id),
                }
            )
        elif row.final_status not in {"READY_FOR_STEP4", "READY_FOR_STEP4_WITH_WARNINGS"} and not row_manual_override:
            exclusion_rows.append(
                {
                    "level": "session",
                    "participant_id": row.participant_id,
                    "session_id": row.session_id,
                    "block": "",
                    "trial_idx_in_block": "",
                    "question_id": "",
                    "reason_code": "STEP3_STATUS_NOT_READY",
                    "reason_detail": f"Step 3 final status is {row.final_status}.",
                }
            )
        elif not row_preprocessing_success:
            exclusion_rows.append(
                {
                    "level": "session",
                    "participant_id": row.participant_id,
                    "session_id": row.session_id,
                    "block": "",
                    "trial_idx_in_block": "",
                    "question_id": "",
                    "reason_code": "PREPROCESSING_NOT_AVAILABLE",
                    "reason_detail": "Step 3 preprocessing did not produce a usable preprocessed FIF file.",
                }
            )

        if session_key not in prelim_session_keys:
            continue

        raw = mne.io.read_raw_fif(str(row.preprocessed_file), preload=True, verbose="ERROR")
        session_data = raw.get_data()
        session_times = raw.times.copy()
        name_to_index = {name: idx for idx, name in enumerate(raw.ch_names)}

        front_hbo_good = available_channels(raw, FRONT_HBO_CHANNELS)
        back_hbo_good = available_channels(raw, BACK_HBO_CHANNELS)
        front_hbr_good = available_channels(raw, FRONT_HBR_CHANNELS)
        back_hbr_good = available_channels(raw, BACK_HBR_CHANNELS)
        long_hbo_good = available_channels(raw, LONG_HBO_CHANNELS)

        if not channel_position_lookup:
            for channel_name in LONG_HBO_CHANNELS:
                if channel_name in name_to_index:
                    ch = raw.info["chs"][name_to_index[channel_name]]
                    channel_position_lookup[channel_name] = (float(ch["loc"][0]), float(ch["loc"][1]))

        session_channel_meta[session_key] = {
            "front_hbo_channels": front_hbo_good,
            "back_hbo_channels": back_hbo_good,
            "front_hbr_channels": front_hbr_good,
            "back_hbr_channels": back_hbr_good,
            "long_hbo_channels": long_hbo_good,
        }

        session_trials = merged_trial_df[
            (merged_trial_df["participant_id"] == row.participant_id) & (merged_trial_df["session_id"] == row.session_id)
        ].copy()

        for trial in session_trials.itertuples(index=False):
            trial_reason_code = ""
            trial_reason_detail = ""
            question_type = str(trial.metadata_question_type).strip() if pd.notna(trial.metadata_question_type) else ""
            if pd.isna(trial.question_start_time):
                trial_reason_code = "MISSING_QUESTION_START"
                trial_reason_detail = "No harmonized question_start trigger was available."
            elif pd.isna(trial.answer_time):
                trial_reason_code = "MISSING_ANSWER"
                trial_reason_detail = "No harmonized answer trigger was available."
            elif question_type not in {"Abstract", "Concrete"}:
                trial_reason_code = "UNKNOWN_CONDITION"
                trial_reason_detail = "Question type label is missing or invalid."
            elif float(trial.answer_time) <= float(trial.question_start_time):
                trial_reason_code = "NONPOSITIVE_WINDOW"
                trial_reason_detail = "Question window duration is not positive."
            elif not front_hbo_good:
                trial_reason_code = "NO_FRONT_HBO_CHANNELS"
                trial_reason_detail = "No valid front ROI HbO channels remain after bad-channel exclusion."
            elif not back_hbo_good:
                trial_reason_code = "NO_BACK_HBO_CHANNELS"
                trial_reason_detail = "No valid back ROI HbO channels remain after bad-channel exclusion."

            if trial_reason_code:
                exclusion_rows.append(
                    {
                        "level": "trial",
                        "participant_id": trial.participant_id,
                        "session_id": trial.session_id,
                        "block": trial.block,
                        "trial_idx_in_block": int(trial.trial_idx_in_block),
                        "question_id": trial.question_id,
                        "reason_code": trial_reason_code,
                        "reason_detail": trial_reason_detail,
                    }
                )
                continue

            front_hbo_response, front_hbo_issue = extract_trial_response(
                session_data,
                session_times,
                float(trial.question_start_time),
                float(trial.answer_time),
                [name_to_index[item] for item in front_hbo_good],
            )
            back_hbo_response, back_hbo_issue = extract_trial_response(
                session_data,
                session_times,
                float(trial.question_start_time),
                float(trial.answer_time),
                [name_to_index[item] for item in back_hbo_good],
            )
            if front_hbo_issue or back_hbo_issue:
                issue = front_hbo_issue or back_hbo_issue or "unknown_trial_issue"
                reason_text = {
                    "baseline_unavailable": "The full 2-second baseline before question onset was unavailable.",
                    "task_window_out_of_bounds": "The question_start to answer interval extended beyond the available preprocessed signal.",
                    "no_channels": "No ROI channels were available for this trial.",
                }.get(issue, "Trial could not be summarized from the preprocessed signal.")
                exclusion_rows.append(
                    {
                        "level": "trial",
                        "participant_id": trial.participant_id,
                        "session_id": trial.session_id,
                        "block": trial.block,
                        "trial_idx_in_block": int(trial.trial_idx_in_block),
                        "question_id": trial.question_id,
                        "reason_code": issue.upper(),
                        "reason_detail": reason_text,
                    }
                )
                continue

            front_hbr_response, _ = extract_trial_response(
                session_data,
                session_times,
                float(trial.question_start_time),
                float(trial.answer_time),
                [name_to_index[item] for item in front_hbr_good],
            )
            back_hbr_response, _ = extract_trial_response(
                session_data,
                session_times,
                float(trial.question_start_time),
                float(trial.answer_time),
                [name_to_index[item] for item in back_hbr_good],
            )

            baseline_start = float(trial.question_start_time) - BASELINE_SEC
            common_row = {
                "participant_id": trial.participant_id,
                "session_id": trial.session_id,
                "block": trial.block,
                "trial_idx_in_block": int(trial.trial_idx_in_block),
                "trial_idx_global": int(trial.trial_idx_global),
                "question_id": trial.question_id,
                "question_type": question_type,
                "question_start_time": float(trial.question_start_time),
                "answer_time": float(trial.answer_time),
                "question_window_duration_sec": float(trial.question_window_duration_sec),
                "baseline_start_time": baseline_start,
                "baseline_end_time": float(trial.question_start_time),
                "source_log": trial.source_log,
                "session_primary_candidate": True,
            }

            trial_hbo_rows.append(
                {
                    **common_row,
                    "front_roi_mean": float(np.mean(front_hbo_response)),
                    "back_roi_mean": float(np.mean(back_hbo_response)),
                    "front_n_channels": len(front_hbo_good),
                    "back_n_channels": len(back_hbo_good),
                    "front_channels_used": ";".join(front_hbo_good),
                    "back_channels_used": ";".join(back_hbo_good),
                }
            )
            trial_hbr_rows.append(
                {
                    **common_row,
                    "front_roi_mean": float(np.mean(front_hbr_response)) if front_hbr_response is not None else math.nan,
                    "back_roi_mean": float(np.mean(back_hbr_response)) if back_hbr_response is not None else math.nan,
                    "front_n_channels": len(front_hbr_good),
                    "back_n_channels": len(back_hbr_good),
                    "front_channels_used": ";".join(front_hbr_good),
                    "back_channels_used": ";".join(back_hbr_good),
                }
            )

            for channel_name in long_hbo_good:
                channel_response, channel_issue = extract_trial_response(
                    session_data,
                    session_times,
                    float(trial.question_start_time),
                    float(trial.answer_time),
                    [name_to_index[channel_name]],
                )
                if channel_issue or channel_response is None:
                    continue
                channel_trial_rows.append(
                    {
                        "participant_id": trial.participant_id,
                        "session_id": trial.session_id,
                        "question_type": question_type,
                        "question_id": trial.question_id,
                        "channel_name": channel_name,
                        "pair_id": channel_name.replace(" hbo", ""),
                        "trial_value": float(channel_response[0]),
                    }
                )

            front_waveform = extract_waveform(
                session_data,
                session_times,
                float(trial.question_start_time),
                [name_to_index[item] for item in front_hbo_good],
                waveform_grid,
            )
            back_waveform = extract_waveform(
                session_data,
                session_times,
                float(trial.question_start_time),
                [name_to_index[item] for item in back_hbo_good],
                waveform_grid,
            )
            if front_waveform is not None:
                waveform_rows.append(
                    {
                        "participant_id": trial.participant_id,
                        "session_id": trial.session_id,
                        "roi": "Front",
                        "question_type": question_type,
                        "waveform": front_waveform,
                    }
                )
            if back_waveform is not None:
                waveform_rows.append(
                    {
                        "participant_id": trial.participant_id,
                        "session_id": trial.session_id,
                        "roi": "Back",
                        "question_type": question_type,
                        "waveform": back_waveform,
                    }
                )

    trial_hbo_df = pd.DataFrame(trial_hbo_rows).sort_values(["participant_id", "session_id", "trial_idx_global"]).reset_index(drop=True)
    trial_hbr_df = pd.DataFrame(trial_hbr_rows).sort_values(["participant_id", "session_id", "trial_idx_global"]).reset_index(drop=True)
    channel_trial_df = pd.DataFrame(channel_trial_rows).sort_values(["participant_id", "session_id", "channel_name", "question_type"]).reset_index(drop=True)

    participant_summary_rows: list[dict[str, Any]] = []
    valid_session_keys: set[tuple[str, str]] = set()
    for (participant_id, session_id), session_df in trial_hbo_df.groupby(["participant_id", "session_id"], sort=True):
        abstract_df = session_df[session_df["question_type"] == "Abstract"]
        concrete_df = session_df[session_df["question_type"] == "Concrete"]
        meta = session_channel_meta[(participant_id, session_id)]

        front_abstract = float(abstract_df["front_roi_mean"].mean()) if not abstract_df.empty else math.nan
        front_concrete = float(concrete_df["front_roi_mean"].mean()) if not concrete_df.empty else math.nan
        back_abstract = float(abstract_df["back_roi_mean"].mean()) if not abstract_df.empty else math.nan
        back_concrete = float(concrete_df["back_roi_mean"].mean()) if not concrete_df.empty else math.nan

        front_hbr_df = trial_hbr_df[(trial_hbr_df["participant_id"] == participant_id) & (trial_hbr_df["session_id"] == session_id)]
        hbr_abstract_df = front_hbr_df[front_hbr_df["question_type"] == "Abstract"]
        hbr_concrete_df = front_hbr_df[front_hbr_df["question_type"] == "Concrete"]

        included_primary = (
            len(abstract_df) >= 2
            and len(concrete_df) >= 2
            and len(meta["front_hbo_channels"]) >= 1
            and len(meta["back_hbo_channels"]) >= 1
        )
        inclusion_reason = "Included in primary ROI analysis."
        if not included_primary:
            detail_parts: list[str] = []
            if len(abstract_df) < 2:
                detail_parts.append(f"Only {len(abstract_df)} valid abstract trial(s) remained.")
            if len(concrete_df) < 2:
                detail_parts.append(f"Only {len(concrete_df)} valid concrete trial(s) remained.")
            if len(meta["front_hbo_channels"]) < 1:
                detail_parts.append("No valid front ROI HbO channels remained.")
            if len(meta["back_hbo_channels"]) < 1:
                detail_parts.append("No valid back ROI HbO channels remained.")
            inclusion_reason = " | ".join(detail_parts)
            exclusion_rows.append(
                {
                    "level": "session",
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "block": "",
                    "trial_idx_in_block": "",
                    "question_id": "",
                    "reason_code": "FAILED_STEP4_PRIMARY_RULES",
                    "reason_detail": inclusion_reason,
                }
            )
        else:
            valid_session_keys.add((participant_id, session_id))

        participant_summary_rows.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "manual_step4_override": (participant_id, session_id) in STEP4_MANUAL_INCLUDE_SESSIONS,
                "manual_step4_override_reason": manual_override_reason(participant_id, session_id),
                "included_primary_analysis": included_primary,
                "step4_inclusion_note": (
                    summary_reasons(inclusion_reason, manual_override_reason(participant_id, session_id))
                    if included_primary
                    else summary_reasons(inclusion_reason, manual_override_reason(participant_id, session_id))
                ),
                "n_valid_abstract_trials": len(abstract_df),
                "n_valid_concrete_trials": len(concrete_df),
                "front_hbo_n_channels": len(meta["front_hbo_channels"]),
                "back_hbo_n_channels": len(meta["back_hbo_channels"]),
                "front_hbr_n_channels": len(meta["front_hbr_channels"]),
                "back_hbr_n_channels": len(meta["back_hbr_channels"]),
                "front_hbo_channels": ";".join(meta["front_hbo_channels"]),
                "back_hbo_channels": ";".join(meta["back_hbo_channels"]),
                "front_hbr_channels": ";".join(meta["front_hbr_channels"]),
                "back_hbr_channels": ";".join(meta["back_hbr_channels"]),
                "front_abstract_hbo_mean": front_abstract,
                "front_concrete_hbo_mean": front_concrete,
                "back_abstract_hbo_mean": back_abstract,
                "back_concrete_hbo_mean": back_concrete,
                "dissociation_hbo": (front_abstract - front_concrete) - (back_abstract - back_concrete),
                "front_abstract_hbr_mean": float(hbr_abstract_df["front_roi_mean"].mean()) if not hbr_abstract_df.empty else math.nan,
                "front_concrete_hbr_mean": float(hbr_concrete_df["front_roi_mean"].mean()) if not hbr_concrete_df.empty else math.nan,
                "back_abstract_hbr_mean": float(hbr_abstract_df["back_roi_mean"].mean()) if not hbr_abstract_df.empty else math.nan,
                "back_concrete_hbr_mean": float(hbr_concrete_df["back_roi_mean"].mean()) if not hbr_concrete_df.empty else math.nan,
                "dissociation_hbr": (
                    float(hbr_abstract_df["front_roi_mean"].mean()) - float(hbr_concrete_df["front_roi_mean"].mean())
                    - float(hbr_abstract_df["back_roi_mean"].mean()) + float(hbr_concrete_df["back_roi_mean"].mean())
                )
                if not hbr_abstract_df.empty and not hbr_concrete_df.empty
                else math.nan,
            }
        )

    participant_summary_df = pd.DataFrame(participant_summary_rows).sort_values(["participant_id", "session_id"]).reset_index(drop=True)
    trial_hbo_df["included_primary_analysis"] = list(
        zip(trial_hbo_df["participant_id"], trial_hbo_df["session_id"])
    )
    trial_hbo_df["included_primary_analysis"] = trial_hbo_df["included_primary_analysis"].isin(valid_session_keys)
    trial_hbr_df["included_primary_analysis"] = list(
        zip(trial_hbr_df["participant_id"], trial_hbr_df["session_id"])
    )
    trial_hbr_df["included_primary_analysis"] = trial_hbr_df["included_primary_analysis"].isin(valid_session_keys)

    participant_summary_lookup = {
        (row["participant_id"], row["session_id"]): row for _, row in participant_summary_df.iterrows()
    }
    for row in status_df.itertuples(index=False):
        session_key = (row.participant_id, row.session_id)
        session_manual_override = session_key in STEP4_MANUAL_INCLUDE_SESSIONS
        summary_row = participant_summary_lookup.get(session_key, {})
        inclusion_rows.append(
            {
                "participant_id": row.participant_id,
                "session_id": row.session_id,
                "step3_final_status": row.final_status,
                "manual_step4_override": session_manual_override,
                "manual_step4_override_reason": manual_override_reason(row.participant_id, row.session_id),
                "global_analysis_excluded": bool(row.global_analysis_excluded),
                "global_exclusion_reason": participant_exclusion_reason(exclusions, row.participant_id) if bool(row.global_analysis_excluded) else "",
                "preprocessing_success": bool(row.preprocessing_success),
                "front_hbo_n_channels": summary_row.get("front_hbo_n_channels", 0),
                "back_hbo_n_channels": summary_row.get("back_hbo_n_channels", 0),
                "n_valid_abstract_trials": summary_row.get("n_valid_abstract_trials", 0),
                "n_valid_concrete_trials": summary_row.get("n_valid_concrete_trials", 0),
                "included_primary_analysis": bool(summary_row.get("included_primary_analysis", False)),
                "step4_inclusion_note": summary_row.get("step4_inclusion_note", ""),
            }
        )

    if "PID037" in exclusions and "PID037" not in set(status_df["participant_id"]):
        exclusion_rows.append(
            {
                "level": "participant",
                "participant_id": "PID037",
                "session_id": "",
                "block": "",
                "trial_idx_in_block": "",
                "question_id": "",
                "reason_code": "GLOBAL_ANALYSIS_EXCLUSION",
                "reason_detail": participant_exclusion_reason(exclusions, "PID037"),
            }
        )

    inclusion_df = pd.DataFrame(inclusion_rows).sort_values(["participant_id", "session_id"]).reset_index(drop=True)
    exclusion_df = pd.DataFrame(exclusion_rows).sort_values(
        ["participant_id", "session_id", "level", "block", "trial_idx_in_block", "reason_code"]
    ).reset_index(drop=True)
    diagnostics = {
        "waveform_grid": waveform_grid,
        "waveform_rows": waveform_rows,
        "channel_position_lookup": channel_position_lookup,
        "valid_session_keys": valid_session_keys,
        "channel_qc_df": channel_qc_df.copy(),
        "session_channel_meta": session_channel_meta,
        "channel_trial_df": channel_trial_df,
    }
    return inclusion_df, trial_hbo_df, trial_hbr_df, participant_summary_df, exclusion_df, trial_hbo_rows, trial_hbr_rows, diagnostics


def build_primary_results(participant_summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    included_df = participant_summary_df[participant_summary_df["included_primary_analysis"]].copy()
    primary_test = run_one_sample_test(included_df["dissociation_hbo"], "Primary HbO dissociation")
    primary_rows = [
        {
            "analysis_label": "Primary HbO dissociation",
            "n_included_participants": int(included_df["participant_id"].nunique()),
            "front_abstract_mean": included_df["front_abstract_hbo_mean"].mean(),
            "front_concrete_mean": included_df["front_concrete_hbo_mean"].mean(),
            "back_abstract_mean": included_df["back_abstract_hbo_mean"].mean(),
            "back_concrete_mean": included_df["back_concrete_hbo_mean"].mean(),
            "mean_dissociation": primary_test["mean"],
            "t_stat": primary_test["t_stat"],
            "df": primary_test["df"],
            "p_value": primary_test["p_value"],
            "ci_low": primary_test["ci_low"],
            "ci_high": primary_test["ci_high"],
            "cohens_d": primary_test["cohens_d"],
            "alpha": ALPHA,
        }
    ]
    primary_df = pd.DataFrame(primary_rows)

    followup_specs = [
        ("Front ROI planned follow-up", included_df["front_abstract_hbo_mean"] - included_df["front_concrete_hbo_mean"], "Front"),
        ("Back ROI planned follow-up", included_df["back_abstract_hbo_mean"] - included_df["back_concrete_hbo_mean"], "Back"),
    ]
    followup_rows = []
    raw_ps = []
    test_payloads = []
    for label, diff_series, roi_name in followup_specs:
        payload = run_one_sample_test(diff_series, label)
        raw_ps.append(payload["p_value"] if pd.notna(payload["p_value"]) else 1.0)
        test_payloads.append((payload, roi_name))
    adjusted_ps = holm_correction(raw_ps)
    for (payload, roi_name), p_holm in zip(test_payloads, adjusted_ps):
        followup_rows.append(
            {
                "analysis_label": payload["analysis_label"],
                "roi": roi_name,
                "n_included_participants": int(included_df["participant_id"].nunique()),
                "mean_difference_abstract_minus_concrete": payload["mean"],
                "t_stat": payload["t_stat"],
                "df": payload["df"],
                "p_value": payload["p_value"],
                "p_value_holm": p_holm,
                "ci_low": payload["ci_low"],
                "ci_high": payload["ci_high"],
                "cohens_dz": payload["cohens_d"],
            }
        )
    followup_df = pd.DataFrame(followup_rows)

    hbr_rows = []
    hbr_payload = run_one_sample_test(included_df["dissociation_hbr"], "Secondary HbR dissociation")
    hbr_rows.append(
        {
            "analysis_label": "Secondary HbR dissociation",
            "test_type": "primary-style secondary",
            "n_included_participants": int(included_df["participant_id"].nunique()),
            "mean_difference": hbr_payload["mean"],
            "t_stat": hbr_payload["t_stat"],
            "df": hbr_payload["df"],
            "p_value": hbr_payload["p_value"],
            "ci_low": hbr_payload["ci_low"],
            "ci_high": hbr_payload["ci_high"],
            "cohens_d": hbr_payload["cohens_d"],
        }
    )
    for label, diff_series in [
        ("Secondary HbR front follow-up", included_df["front_abstract_hbr_mean"] - included_df["front_concrete_hbr_mean"]),
        ("Secondary HbR back follow-up", included_df["back_abstract_hbr_mean"] - included_df["back_concrete_hbr_mean"]),
    ]:
        payload = run_one_sample_test(diff_series, label)
        hbr_rows.append(
            {
                "analysis_label": label,
                "test_type": "planned-style secondary",
                "n_included_participants": int(included_df["participant_id"].nunique()),
                "mean_difference": payload["mean"],
                "t_stat": payload["t_stat"],
                "df": payload["df"],
                "p_value": payload["p_value"],
                "ci_low": payload["ci_low"],
                "ci_high": payload["ci_high"],
                "cohens_d": payload["cohens_d"],
            }
        )
    hbr_df = pd.DataFrame(hbr_rows)
    return primary_df, followup_df, hbr_df


def build_channel_results(channel_trial_df: pd.DataFrame, valid_session_keys: set[tuple[str, str]]) -> pd.DataFrame:
    filtered_df = channel_trial_df[channel_trial_df.apply(lambda row: (row["participant_id"], row["session_id"]) in valid_session_keys, axis=1)].copy()
    participant_condition_df = (
        filtered_df.groupby(["participant_id", "session_id", "channel_name", "pair_id", "question_type"], sort=True)["trial_value"]
        .mean()
        .reset_index()
    )
    pivot_df = (
        participant_condition_df.pivot_table(
            index=["participant_id", "session_id", "channel_name", "pair_id"],
            columns="question_type",
            values="trial_value",
        )
        .reset_index()
    )
    pivot_df.columns.name = None
    pivot_df["abstract_minus_concrete"] = pivot_df["Abstract"] - pivot_df["Concrete"]

    result_rows = []
    for channel_name, channel_df in pivot_df.groupby("channel_name", sort=True):
        payload = run_one_sample_test(channel_df["abstract_minus_concrete"], f"Exploratory {channel_name}")
        result_rows.append(
            {
                "channel_name": channel_name,
                "pair_id": channel_df["pair_id"].iloc[0],
                "n_included_participants": payload["n"],
                "mean_abstract": channel_df["Abstract"].mean(),
                "mean_concrete": channel_df["Concrete"].mean(),
                "mean_difference_abstract_minus_concrete": payload["mean"],
                "t_stat": payload["t_stat"],
                "df": payload["df"],
                "p_value": payload["p_value"],
                "ci_low": payload["ci_low"],
                "ci_high": payload["ci_high"],
                "cohens_dz": payload["cohens_d"],
            }
        )
    channel_df = pd.DataFrame(result_rows).sort_values("channel_name").reset_index(drop=True)
    if not channel_df.empty:
        channel_df["p_value_holm"] = holm_correction([float(value) for value in channel_df["p_value"]])
    else:
        channel_df["p_value_holm"] = pd.Series(dtype=float)
    return channel_df


def plot_four_cell(participant_summary_df: pd.DataFrame) -> None:
    included_df = participant_summary_df[participant_summary_df["included_primary_analysis"]].copy()
    order = [
        ("Front-Abstract", "front_abstract_hbo_mean"),
        ("Front-Concrete", "front_concrete_hbo_mean"),
        ("Back-Abstract", "back_abstract_hbo_mean"),
        ("Back-Concrete", "back_concrete_hbo_mean"),
    ]
    x_positions = np.arange(len(order))
    plt.figure(figsize=(9, 6))
    for row in included_df.itertuples(index=False):
        values = [getattr(row, column_name) for _, column_name in order]
        plt.plot(x_positions, values, color="#B0B0B0", alpha=0.5, linewidth=1)
    means = [included_df[column_name].mean() for _, column_name in order]
    plt.plot(x_positions, means, color="#1F4E79", marker="o", linewidth=2.5)
    plt.xticks(x_positions, [label for label, _ in order], rotation=15)
    plt.ylabel("Baseline-corrected HbO mean")
    plt.title("Front/back ROI by condition")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "front_back_fourcell_plot.png", dpi=150)
    plt.close()


def plot_dissociation_distribution(participant_summary_df: pd.DataFrame) -> None:
    included_df = participant_summary_df[participant_summary_df["included_primary_analysis"]].copy()
    values = included_df["dissociation_hbo"].dropna().to_numpy(dtype=float)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=min(12, max(6, len(values))), color="#4C956C", edgecolor="white")
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.axvline(values.mean(), color="#C0392B", linestyle="-", linewidth=2)
    plt.xlabel("Participant dissociation score")
    plt.ylabel("Count")
    plt.title("Primary dissociation score distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "primary_dissociation_distribution.png", dpi=150)
    plt.close()


def plot_roi_pair(participant_summary_df: pd.DataFrame, roi_name: str, abstract_col: str, concrete_col: str, output_name: str) -> None:
    included_df = participant_summary_df[participant_summary_df["included_primary_analysis"]].copy()
    plt.figure(figsize=(6.5, 5.5))
    for row in included_df.itertuples(index=False):
        values = [getattr(row, abstract_col), getattr(row, concrete_col)]
        plt.plot([0, 1], values, color="#B0B0B0", alpha=0.6, linewidth=1)
    means = [included_df[abstract_col].mean(), included_df[concrete_col].mean()]
    plt.plot([0, 1], means, color="#8E44AD" if roi_name == "Front" else "#2874A6", marker="o", linewidth=2.5)
    plt.xticks([0, 1], ["Abstract", "Concrete"])
    plt.ylabel("Baseline-corrected HbO mean")
    plt.title(f"{roi_name} ROI: Abstract versus Concrete")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=150)
    plt.close()


def plot_waveforms(waveform_rows: list[dict[str, Any]], valid_session_keys: set[tuple[str, str]], waveform_grid: np.ndarray) -> None:
    waveform_df = pd.DataFrame(waveform_rows)
    if waveform_df.empty:
        return
    waveform_df["valid_session"] = waveform_df.apply(lambda row: (row["participant_id"], row["session_id"]) in valid_session_keys, axis=1)
    waveform_df = waveform_df[waveform_df["valid_session"]].copy()
    if waveform_df.empty:
        return

    grouped = (
        waveform_df.groupby(["participant_id", "session_id", "roi", "question_type"], sort=True)["waveform"]
        .apply(lambda items: np.vstack(items).mean(axis=0))
        .reset_index()
    )

    for roi_name, output_name in [("Front", "front_hbo_waveform.png"), ("Back", "back_hbo_waveform.png")]:
        plt.figure(figsize=(8, 5))
        for condition, color in [("Abstract", "#C0392B"), ("Concrete", "#1F4E79")]:
            subset = grouped[(grouped["roi"] == roi_name) & (grouped["question_type"] == condition)]
            if subset.empty:
                continue
            waveforms = np.vstack(subset["waveform"].to_list())
            mean_waveform = waveforms.mean(axis=0)
            sem_waveform = waveforms.std(axis=0, ddof=1) / math.sqrt(waveforms.shape[0]) if waveforms.shape[0] > 1 else np.zeros_like(mean_waveform)
            plt.plot(waveform_grid, mean_waveform, color=color, label=condition)
            plt.fill_between(waveform_grid, mean_waveform - sem_waveform, mean_waveform + sem_waveform, color=color, alpha=0.2)
        plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
        plt.axhline(0.0, color="gray", linestyle=":", linewidth=1)
        plt.xlabel("Seconds from question onset")
        plt.ylabel("Baseline-corrected HbO")
        plt.title(f"{roi_name} ROI descriptive HbO waveform")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / output_name, dpi=150)
        plt.close()


def plot_channel_map(channel_results_df: pd.DataFrame, channel_position_lookup: dict[str, tuple[float, float]]) -> None:
    if channel_results_df.empty or not channel_position_lookup:
        return
    x_vals = []
    y_vals = []
    color_vals = []
    labels = []
    for row in channel_results_df.itertuples(index=False):
        position = channel_position_lookup.get(row.channel_name)
        if position is None:
            continue
        x_vals.append(position[0])
        y_vals.append(position[1])
        color_vals.append(row.mean_difference_abstract_minus_concrete)
        labels.append(row.pair_id)
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(x_vals, y_vals, c=color_vals, cmap="coolwarm", s=250, edgecolor="black")
    for x_val, y_val, label in zip(x_vals, y_vals, labels):
        plt.text(x_val, y_val, label, fontsize=8, ha="center", va="center")
    plt.colorbar(scatter, label="Abstract - Concrete HbO")
    plt.title("Exploratory long-channel abstract-minus-concrete map")
    plt.xlabel("MNE x position")
    plt.ylabel("MNE y position")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "channel_exploratory_map.png", dpi=150)
    plt.close()


def write_report_artifacts(
    inclusion_df: pd.DataFrame,
    trial_hbo_df: pd.DataFrame,
    participant_summary_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    followup_df: pd.DataFrame,
    hbr_df: pd.DataFrame,
    channel_results_df: pd.DataFrame,
    exclusion_df: pd.DataFrame,
    exclusions: dict[str, dict[str, str]],
) -> pd.DataFrame:
    primary_row = primary_df.iloc[0]
    conclusion_significant = bool(primary_row["p_value"] < ALPHA) if pd.notna(primary_row["p_value"]) else False
    followup_lookup = {row["roi"]: row for _, row in followup_df.iterrows()}
    front_followup = followup_lookup.get("Front", {})
    back_followup = followup_lookup.get("Back", {})

    if conclusion_significant:
        followup_bits = []
        if front_followup:
            followup_bits.append(
                f"the front ROI abstract-minus-concrete contrast was {format_float(front_followup.get('mean_difference_abstract_minus_concrete'), 4)} (Holm-adjusted p={format_p_value(front_followup.get('p_value_holm'))})"
            )
        if back_followup:
            followup_bits.append(
                f"the back ROI abstract-minus-concrete contrast was {format_float(back_followup.get('mean_difference_abstract_minus_concrete'), 4)} (Holm-adjusted p={format_p_value(back_followup.get('p_value_holm'))})"
            )
        followup_text = "; ".join(followup_bits)
        conclusion_text = (
            "The pre-specified primary analysis showed a statistically significant front-versus-back dissociation "
            f"between abstract and concrete questions (t({int(primary_row['df'])})={format_float(primary_row['t_stat'], 4)}, "
            f"p={format_p_value(primary_row['p_value'])}, mean dissociation={format_float(primary_row['mean_dissociation'], 4)}, "
            f"95% CI [{format_float(primary_row['ci_low'], 4)}, {format_float(primary_row['ci_high'], 4)}], "
            f"d={format_float(primary_row['cohens_d'], 4)}). "
            "This indicates that the abstract-versus-concrete contrast differed significantly between the front and back left-lateral ROIs. "
            f"Planned follow-up comparisons showed that {followup_text}."
        )
    else:
        conclusion_text = (
            "The pre-specified primary analysis did not show a statistically significant front-versus-back dissociation "
            f"between abstract and concrete questions (t({int(primary_row['df'])})={format_float(primary_row['t_stat'], 4)}, "
            f"p={format_p_value(primary_row['p_value'])}, mean dissociation={format_float(primary_row['mean_dissociation'], 4)}, "
            f"95% CI [{format_float(primary_row['ci_low'], 4)}, {format_float(primary_row['ci_high'], 4)}], "
            f"d={format_float(primary_row['cohens_d'], 4)}). "
            "In this dataset, the abstract-versus-concrete contrast was not significantly different between the front and back left-lateral ROIs."
        )

    conclusion_df = pd.DataFrame(
        [
            {
                "n_included_participants": int(participant_summary_df["included_primary_analysis"].sum()),
                "primary_p_value": primary_row["p_value"],
                "primary_significant": conclusion_significant,
                "conclusion_text": conclusion_text,
            }
        ]
    )

    primary_table = make_latex_grid(
        [
            "n",
            "Front A",
            "Front C",
            "Back A",
            "Back C",
            "Mean " + r"$\Delta$",
            "t",
            "df",
            "p",
            "95\\% CI",
            "d",
        ],
        [
            [
                int(primary_row["n_included_participants"]),
                format_float(primary_row["front_abstract_mean"], 4),
                format_float(primary_row["front_concrete_mean"], 4),
                format_float(primary_row["back_abstract_mean"], 4),
                format_float(primary_row["back_concrete_mean"], 4),
                format_float(primary_row["mean_dissociation"], 4),
                format_float(primary_row["t_stat"], 4),
                int(primary_row["df"]),
                format_p_value(primary_row["p_value"]),
                f"[{format_float(primary_row['ci_low'], 4)}, {format_float(primary_row['ci_high'], 4)}]",
                format_float(primary_row["cohens_d"], 4),
            ]
        ],
        r"rrrrrrrrrrr",
    )
    (TABLES_DIR / "primary_dissociation_table.tex").write_text(primary_table + "\n", encoding="utf-8")

    followup_rows = []
    for row in followup_df.itertuples(index=False):
        followup_rows.append(
            [
                row.roi,
                row.n_included_participants,
                format_float(row.mean_difference_abstract_minus_concrete, 4),
                format_float(row.t_stat, 4),
                int(row.df),
                format_p_value(row.p_value),
                format_p_value(row.p_value_holm),
                f"[{format_float(row.ci_low, 4)}, {format_float(row.ci_high, 4)}]",
                format_float(row.cohens_dz, 4),
            ]
        )
    (TABLES_DIR / "followup_roi_tests_table.tex").write_text(
        make_latex_grid(
            ["ROI", "n", "Mean diff", "t", "df", "p", "Holm p", "95\\% CI", "d$_z$"],
            followup_rows,
            r"lrrrrrrrr",
        )
        + "\n",
        encoding="utf-8",
    )
    (TEXT_DIR / "final_conclusion.tex").write_text(conclusion_text + "\n", encoding="utf-8")

    overview_rows = [
        ("Participant-sessions in Step 3 table", len(inclusion_df)),
        ("Primary-included participant-sessions", int(inclusion_df["included_primary_analysis"].sum())),
        ("Primary valid HbO trials", int(trial_hbo_df["included_primary_analysis"].sum())),
        ("Manual Step 4 inclusion overrides", int(inclusion_df["manual_step4_override"].fillna(False).sum())),
        ("Global excluded participants", len(exclusions)),
        ("Logged Step 4 exclusions", len(exclusion_df)),
        ("Exploratory long channels tested", len(channel_results_df)),
    ]

    secondary_rows = []
    if not hbr_df.empty:
        secondary_rows.append(
            (
                "Secondary HbR primary-style p-value",
                format_p_value(hbr_df.iloc[0]["p_value"]),
            )
        )
    if not channel_results_df.empty:
        secondary_rows.append(
            (
                "Smallest corrected exploratory channel p-value",
                format_p_value(channel_results_df["p_value_holm"].min()),
            )
        )

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{array}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\begin{document}",
        r"\section*{Step 4 Primary ROI Analysis}",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
    ]

    if exclusions:
        lines.extend(
            [
                r"\subsection*{Global Participant Exclusions}",
                r"The shared exclusion list from earlier steps remained active in Step~4. Files were retained for audit, but these participants were not used in the ROI analyses.",
                make_latex_table(exclusion_table_rows(exclusions)),
            ]
        )

    manual_override_rows = [
        (f"{row.participant_id} / {row.session_id}", row.manual_step4_override_reason)
        for row in inclusion_df.itertuples(index=False)
        if bool(row.manual_step4_override)
    ]
    if manual_override_rows:
        lines.extend(
            [
                r"\subsection*{Manual Step 4 Inclusions}",
                r"The following sessions were included in Step~4 by explicit analyst override despite Step~3 review status. This deviation was requested and logged in the Step~4 specification file.",
                make_latex_table(manual_override_rows),
            ]
        )

    lines.extend(
        [
            r"\subsection*{Primary Dissociation Test}",
            r"\input{tables/primary_dissociation_table.tex}",
            r"\begin{figure}[H]",
            r"\centering",
            r"\includegraphics[width=0.82\linewidth]{../../figures/step4/front_back_fourcell_plot.png}",
            r"\caption{Participant-level four-cell HbO summary for the front and back ROIs.}",
            r"\end{figure}",
            r"\begin{figure}[H]",
            r"\centering",
            r"\includegraphics[width=0.70\linewidth]{../../figures/step4/primary_dissociation_distribution.png}",
            r"\caption{Distribution of the participant-level primary dissociation scores.}",
            r"\end{figure}",
            r"\subsection*{Planned Follow-Up Tests}",
            r"\input{tables/followup_roi_tests_table.tex}",
            r"\begin{figure}[H]",
            r"\centering",
            r"\includegraphics[width=0.48\linewidth]{../../figures/step4/front_roi_abstract_vs_concrete.png}",
            r"\includegraphics[width=0.48\linewidth]{../../figures/step4/back_roi_abstract_vs_concrete.png}",
            r"\caption{Planned paired comparisons for the front and back ROIs.}",
            r"\end{figure}",
        ]
    )

    if secondary_rows:
        lines.extend([r"\subsection*{Secondary Analyses}", make_latex_table(secondary_rows)])
    lines.extend(
        [
            r"\begin{figure}[H]",
            r"\centering",
            r"\includegraphics[width=0.48\linewidth]{../../figures/step4/front_hbo_waveform.png}",
            r"\includegraphics[width=0.48\linewidth]{../../figures/step4/back_hbo_waveform.png}",
            r"\caption{Descriptive HbO waveforms aligned to question onset for the front and back ROIs.}",
            r"\end{figure}",
            r"\begin{figure}[H]",
            r"\centering",
            r"\includegraphics[width=0.70\linewidth]{../../figures/step4/channel_exploratory_map.png}",
            r"\caption{Exploratory long-channel abstract-minus-concrete HbO effects.}",
            r"\end{figure}",
            r"\subsection*{Final Conclusion}",
            r"\input{text/final_conclusion.tex}",
            r"\end{document}",
        ]
    )

    report_path = REPORTS_DIR / "step4_primary_analysis_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return conclusion_df


def main() -> None:
    ensure_directories()
    trial_df, trigger_df, status_df, channel_qc_df, preproc_df, exclusions = load_inputs()
    merged_trial_df = build_trial_frame(trial_df, trigger_df, status_df, preproc_df)

    inclusion_df, trial_hbo_df, trial_hbr_df, participant_summary_df, exclusion_df, _, _, diagnostics = build_step4_outputs(
        merged_trial_df,
        status_df,
        channel_qc_df,
        exclusions,
    )

    primary_df, followup_df, hbr_df = build_primary_results(participant_summary_df)
    channel_results_df = build_channel_results(diagnostics["channel_trial_df"], diagnostics["valid_session_keys"])

    save_dataframe(inclusion_df, CLEAN_DIR / "01_step4_inclusion_table.csv")
    save_dataframe(trial_hbo_df, CLEAN_DIR / "02_trial_fnirs_summary_hbo.csv")
    save_dataframe(trial_hbr_df, CLEAN_DIR / "03_trial_fnirs_summary_hbr.csv")
    save_dataframe(participant_summary_df, CLEAN_DIR / "04_participant_roi_condition_summary.csv")
    save_dataframe(primary_df, CLEAN_DIR / "05_primary_dissociation_test_results.csv")
    save_dataframe(followup_df, CLEAN_DIR / "06_followup_roi_tests.csv")
    save_dataframe(hbr_df, CLEAN_DIR / "07_secondary_hbr_results.csv")
    save_dataframe(channel_results_df, CLEAN_DIR / "08_channel_exploratory_results.csv")
    save_dataframe(exclusion_df, CLEAN_DIR / "09_step4_exclusion_log.csv")

    plot_four_cell(participant_summary_df)
    plot_dissociation_distribution(participant_summary_df)
    plot_roi_pair(participant_summary_df, "Front", "front_abstract_hbo_mean", "front_concrete_hbo_mean", "front_roi_abstract_vs_concrete.png")
    plot_roi_pair(participant_summary_df, "Back", "back_abstract_hbo_mean", "back_concrete_hbo_mean", "back_roi_abstract_vs_concrete.png")
    plot_waveforms(diagnostics["waveform_rows"], diagnostics["valid_session_keys"], diagnostics["waveform_grid"])
    plot_channel_map(channel_results_df, diagnostics["channel_position_lookup"])

    conclusion_df = write_report_artifacts(
        inclusion_df,
        trial_hbo_df,
        participant_summary_df,
        primary_df,
        followup_df,
        hbr_df,
        channel_results_df,
        exclusion_df,
        exclusions,
    )
    save_dataframe(conclusion_df, CLEAN_DIR / "10_step4_final_conclusion.csv")

    report_path = REPORTS_DIR / "step4_primary_analysis_report.tex"
    compile_report(report_path)

    included_sessions = participant_summary_df[participant_summary_df["included_primary_analysis"]]
    print("Step 4 outputs generated successfully.")
    print(f"Included participant-sessions: {len(included_sessions)}")
    print(f"Included unique participants: {included_sessions['participant_id'].nunique()}")
    print(f"Primary p-value: {format_p_value(primary_df.iloc[0]['p_value'])}")


if __name__ == "__main__":
    main()
