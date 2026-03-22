#!/usr/bin/env python3

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats

from analysis_exclusions import exclusion_table_rows, load_participant_exclusions


ROOT = Path(__file__).resolve().parents[1]

STEP2_TRIAL_PATH = ROOT / "data_clean" / "step2" / "06_behavior_trial_clean.csv"
STEP3_TRIGGER_PATH = ROOT / "data_clean" / "step3" / "06_fnirs_trigger_harmonized.csv"
STEP3_STATUS_PATH = ROOT / "data_clean" / "step3" / "07_fnirs_session_status.csv"
STEP4_INCLUSION_PATH = ROOT / "data_clean" / "step4" / "01_step4_inclusion_table.csv"

CLEAN_DIR = ROOT / "data_clean" / "step5"
FIGURES_DIR = ROOT / "figures" / "step5"
REPORTS_DIR = ROOT / "reports" / "step5"
TABLES_DIR = REPORTS_DIR / "tables"
TEXT_DIR = REPORTS_DIR / "text"

ALPHA = 0.05
EPOCH_TMIN = -3.5
EPOCH_TMAX = 13.2
BASELINE_START = -2.0
BASELINE_END = 0.0
PRIMARY_WINDOW = (7.0, 11.0)
SENSITIVITY_WINDOWS = [
    ("window_6_10", 6.0, 10.0),
    ("window_8_12", 8.0, 12.0),
]
SINGLE_TIME_SEC = 9.0
FAST_RESPONSE_MIN_SEC = PRIMARY_WINDOW[1]
WAVEFORM_GRID = np.arange(EPOCH_TMIN, EPOCH_TMAX + 1e-9, 0.1)

SHORT_PAIRS = ["S1_D8", "S2_D9", "S3_D10", "S4_D11", "S5_D12", "S6_D13", "S7_D14", "S8_D15"]
FRONT_PAIRS = ["S2_D6", "S2_D4", "S1_D6", "S1_D3", "S5_D6", "S5_D4", "S5_D3"]
BACK_PAIRS = ["S4_D2", "S3_D1", "S6_D2", "S6_D1", "S6_D5", "S7_D2", "S7_D5", "S8_D1", "S8_D5"]
TRANSITION_PAIRS = ["S5_D7", "S4_D4", "S4_D7", "S3_D3", "S3_D7", "S6_D7"]


def ensure_directories() -> None:
    for path in [CLEAN_DIR, FIGURES_DIR, REPORTS_DIR, TABLES_DIR, TEXT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


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


def coerce_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def pair_to_channel(pair_id: str, chromophore: str) -> str:
    return f"{pair_id} {chromophore}"


def pair_list_to_channels(pair_ids: list[str], chromophore: str) -> list[str]:
    return [pair_to_channel(pair_id, chromophore) for pair_id in pair_ids]


FRONT_HBO_CHANNELS = pair_list_to_channels(FRONT_PAIRS, "hbo")
BACK_HBO_CHANNELS = pair_list_to_channels(BACK_PAIRS, "hbo")
FRONT_HBR_CHANNELS = pair_list_to_channels(FRONT_PAIRS, "hbr")
BACK_HBR_CHANNELS = pair_list_to_channels(BACK_PAIRS, "hbr")


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


def normalize_condition(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    if text == "abstract":
        return "Abstract"
    if text == "concrete":
        return "Concrete"
    return ""


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


def make_latex_table(rows: list[tuple[str, Any]], column_spec: str = r"p{0.38\linewidth}p{0.52\linewidth}") -> str:
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule", r"Metric & Value\\", r"\midrule"]
    for key, value in rows:
        lines.append(f"{sanitize_for_tex(key)} & {sanitize_for_tex(value)}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def make_latex_grid(headers: list[str], rows: list[list[Any]], column_spec: str) -> str:
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule", " & ".join(headers) + r"\\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(sanitize_for_tex(item) for item in row) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def available_channels(raw: mne.io.BaseRaw, requested_channels: list[str]) -> list[str]:
    bads = set(raw.info["bads"])
    names = set(raw.ch_names)
    return [channel for channel in requested_channels if channel in names and channel not in bads]


def run_one_sample_test(values: pd.Series | np.ndarray, label: str) -> dict[str, Any]:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    trial_df = pd.read_csv(STEP2_TRIAL_PATH)
    trigger_df = pd.read_csv(STEP3_TRIGGER_PATH)
    status_df = pd.read_csv(STEP3_STATUS_PATH)
    step4_inclusion_df = pd.read_csv(STEP4_INCLUSION_PATH)
    exclusions = load_participant_exclusions()

    trial_df["question_id"] = trial_df["question_id"].map(normalize_question_id)
    trigger_df["question_id"] = trigger_df["question_id"].map(normalize_question_id)
    trial_df["metadata_question_type"] = trial_df["metadata_question_type"].map(normalize_condition)

    for col in ["manual_step4_override", "included_primary_analysis"]:
        if col in step4_inclusion_df.columns:
            step4_inclusion_df[col] = step4_inclusion_df[col].map(coerce_bool)
    for col in ["global_analysis_excluded", "preprocessing_success"]:
        if col in status_df.columns:
            status_df[col] = status_df[col].map(coerce_bool)

    return trial_df, trigger_df, status_df, step4_inclusion_df, exclusions


def build_trial_frame(
    trial_df: pd.DataFrame,
    trigger_df: pd.DataFrame,
    status_df: pd.DataFrame,
    step4_inclusion_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    step4_sessions = step4_inclusion_df[step4_inclusion_df["included_primary_analysis"]].copy()
    session_cols = [
        "participant_id",
        "session_id",
        "psychopy_file",
        "preprocessed_file",
    ]
    session_df = step4_sessions.merge(
        status_df[session_cols],
        on=["participant_id", "session_id"],
        how="left",
    )
    trial_join_df = trial_df.merge(
        session_df,
        left_on=["participant_id", "source_log_full_path"],
        right_on=["participant_id", "psychopy_file"],
        how="inner",
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
    merged_df["answer_latency_sec"] = merged_df["answer_time"] - merged_df["question_start_time"]
    session_df = session_df.sort_values(["participant_id", "session_id"]).reset_index(drop=True)
    merged_df = merged_df.sort_values(["participant_id", "session_id", "trial_idx_global"]).reset_index(drop=True)
    return session_df, merged_df


def extract_window_response(
    data: np.ndarray,
    times: np.ndarray,
    onset_time: float,
    channel_indices: list[int],
    window_start: float,
    window_end: float,
) -> tuple[np.ndarray | None, str | None]:
    if not channel_indices:
        return None, "no_channels"
    baseline_mask = (times >= onset_time + BASELINE_START) & (times <= onset_time + BASELINE_END)
    window_mask = (times >= onset_time + window_start) & (times <= onset_time + window_end)
    if not baseline_mask.any():
        return None, "baseline_unavailable"
    if not window_mask.any():
        return None, "window_unavailable"
    baseline_mean = data[np.ix_(channel_indices, baseline_mask)].mean(axis=1)
    window_mean = data[np.ix_(channel_indices, window_mask)].mean(axis=1)
    return window_mean - baseline_mean, None


def extract_timepoint_response(
    data: np.ndarray,
    times: np.ndarray,
    onset_time: float,
    channel_indices: list[int],
    timepoint_sec: float,
) -> tuple[np.ndarray | None, str | None]:
    if not channel_indices:
        return None, "no_channels"
    baseline_mask = (times >= onset_time + BASELINE_START) & (times <= onset_time + BASELINE_END)
    if not baseline_mask.any():
        return None, "baseline_unavailable"
    target_time = onset_time + timepoint_sec
    if target_time < times[0] - 1e-9 or target_time > times[-1] + 1e-9:
        return None, "window_unavailable"
    idx = int(np.argmin(np.abs(times - target_time)))
    baseline_mean = data[np.ix_(channel_indices, baseline_mask)].mean(axis=1)
    return data[channel_indices, idx] - baseline_mean, None


def extract_epoch_waveform(
    data: np.ndarray,
    times: np.ndarray,
    onset_time: float,
    channel_indices: list[int],
    waveform_grid: np.ndarray,
) -> np.ndarray | None:
    if not channel_indices:
        return None
    epoch_start = onset_time + EPOCH_TMIN
    epoch_end = onset_time + EPOCH_TMAX
    if epoch_start < times[0] - 1e-9 or epoch_end > times[-1] + 1e-9:
        return None
    baseline_mask = (times >= onset_time + BASELINE_START) & (times <= onset_time + BASELINE_END)
    epoch_mask = (times >= epoch_start) & (times <= epoch_end)
    if not baseline_mask.any() or not epoch_mask.any():
        return None
    baseline_mean = data[np.ix_(channel_indices, baseline_mask)].mean(axis=1, keepdims=True)
    roi_series = (data[np.ix_(channel_indices, epoch_mask)] - baseline_mean).mean(axis=0)
    rel_times = times[epoch_mask] - onset_time
    return np.interp(waveform_grid, rel_times, roi_series)


def build_step5_outputs(
    session_df: pd.DataFrame,
    merged_trial_df: pd.DataFrame,
    exclusions: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    epoch_manifest_rows: list[dict[str, Any]] = []
    trial_hbo_rows: list[dict[str, Any]] = []
    trial_hbr_rows: list[dict[str, Any]] = []
    participant_summary_rows: list[dict[str, Any]] = []
    inclusion_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    waveform_rows: list[dict[str, Any]] = []
    session_channel_meta: dict[tuple[str, str], dict[str, Any]] = {}
    included_session_keys: set[tuple[str, str]] = set()

    for session_row in session_df.itertuples(index=False):
        session_key = (session_row.participant_id, session_row.session_id)
        raw = mne.io.read_raw_fif(str(session_row.preprocessed_file), preload=True, verbose="ERROR")
        session_data = raw.get_data()
        session_times = raw.times.copy()
        name_to_index = {name: idx for idx, name in enumerate(raw.ch_names)}

        front_hbo_good = available_channels(raw, FRONT_HBO_CHANNELS)
        back_hbo_good = available_channels(raw, BACK_HBO_CHANNELS)
        front_hbr_good = available_channels(raw, FRONT_HBR_CHANNELS)
        back_hbr_good = available_channels(raw, BACK_HBR_CHANNELS)
        session_channel_meta[session_key] = {
            "front_hbo_channels": front_hbo_good,
            "back_hbo_channels": back_hbo_good,
            "front_hbr_channels": front_hbr_good,
            "back_hbr_channels": back_hbr_good,
        }

        session_trials = merged_trial_df[
            (merged_trial_df["participant_id"] == session_row.participant_id)
            & (merged_trial_df["session_id"] == session_row.session_id)
        ].copy()

        for trial in session_trials.itertuples(index=False):
            question_type = normalize_condition(trial.metadata_question_type)
            onset_time = float(trial.question_start_time) if pd.notna(trial.question_start_time) else math.nan
            answer_time = float(trial.answer_time) if pd.notna(trial.answer_time) else math.nan
            answer_latency = float(trial.answer_latency_sec) if pd.notna(trial.answer_latency_sec) else math.nan
            epoch_start_time = onset_time + EPOCH_TMIN if pd.notna(trial.question_start_time) else math.nan
            epoch_end_time = onset_time + EPOCH_TMAX if pd.notna(trial.question_start_time) else math.nan
            full_epoch_available = bool(
                pd.notna(trial.question_start_time)
                and epoch_start_time >= session_times[0] - 1e-9
                and epoch_end_time <= session_times[-1] + 1e-9
            )
            baseline_available = bool(
                pd.notna(trial.question_start_time)
                and onset_time + BASELINE_START >= session_times[0] - 1e-9
                and onset_time + BASELINE_END <= session_times[-1] + 1e-9
            )

            reason_code = ""
            reason_detail = ""
            if pd.isna(trial.question_start_time):
                reason_code = "MISSING_QUESTION_START"
                reason_detail = "No harmonized question_start trigger was available."
            elif question_type not in {"Abstract", "Concrete"}:
                reason_code = "UNKNOWN_CONDITION"
                reason_detail = "Question type label is missing or invalid."
            elif not full_epoch_available:
                reason_code = "EPOCH_WINDOW_OUT_OF_BOUNDS"
                reason_detail = "The full [-3.5, 13.2] s onset-locked epoch was unavailable."
            elif not baseline_available:
                reason_code = "BASELINE_UNAVAILABLE"
                reason_detail = "The [-2, 0] s baseline interval was unavailable."
            elif not front_hbo_good:
                reason_code = "NO_FRONT_HBO_CHANNELS"
                reason_detail = "No valid front ROI HbO channels remained after bad-channel exclusion."
            elif not back_hbo_good:
                reason_code = "NO_BACK_HBO_CHANNELS"
                reason_detail = "No valid back ROI HbO channels remained after bad-channel exclusion."

            included_trial = reason_code == ""
            epoch_manifest_rows.append(
                {
                    "participant_id": trial.participant_id,
                    "session_id": trial.session_id,
                    "block": trial.block,
                    "trial_idx_in_block": int(trial.trial_idx_in_block),
                    "trial_idx_global": int(trial.trial_idx_global),
                    "question_id": trial.question_id,
                    "question_type": question_type,
                    "question_id_matches_trigger": bool(trial.question_id_matches_trigger),
                    "question_start_time": onset_time,
                    "answer_time": answer_time,
                    "answer_latency_sec": answer_latency,
                    "epoch_start_time": epoch_start_time,
                    "epoch_end_time": epoch_end_time,
                    "baseline_start_time": onset_time + BASELINE_START if pd.notna(trial.question_start_time) else math.nan,
                    "baseline_end_time": onset_time + BASELINE_END if pd.notna(trial.question_start_time) else math.nan,
                    "full_epoch_available": full_epoch_available,
                    "baseline_available": baseline_available,
                    "front_hbo_n_channels": len(front_hbo_good),
                    "back_hbo_n_channels": len(back_hbo_good),
                    "front_hbr_n_channels": len(front_hbr_good),
                    "back_hbr_n_channels": len(back_hbr_good),
                    "included_step5_trial": included_trial,
                    "exclusion_reason_code": reason_code,
                    "exclusion_reason_detail": reason_detail,
                    "source_log": trial.source_log,
                }
            )

            if not included_trial:
                continue

            front_hbo_response, front_hbo_issue = extract_window_response(
                session_data,
                session_times,
                onset_time,
                [name_to_index[item] for item in front_hbo_good],
                PRIMARY_WINDOW[0],
                PRIMARY_WINDOW[1],
            )
            back_hbo_response, back_hbo_issue = extract_window_response(
                session_data,
                session_times,
                onset_time,
                [name_to_index[item] for item in back_hbo_good],
                PRIMARY_WINDOW[0],
                PRIMARY_WINDOW[1],
            )
            if front_hbo_issue or back_hbo_issue or front_hbo_response is None or back_hbo_response is None:
                continue

            front_hbr_response, _ = extract_window_response(
                session_data,
                session_times,
                onset_time,
                [name_to_index[item] for item in front_hbr_good],
                PRIMARY_WINDOW[0],
                PRIMARY_WINDOW[1],
            )
            back_hbr_response, _ = extract_window_response(
                session_data,
                session_times,
                onset_time,
                [name_to_index[item] for item in back_hbr_good],
                PRIMARY_WINDOW[0],
                PRIMARY_WINDOW[1],
            )

            common_row = {
                "participant_id": trial.participant_id,
                "session_id": trial.session_id,
                "block": trial.block,
                "trial_idx_in_block": int(trial.trial_idx_in_block),
                "trial_idx_global": int(trial.trial_idx_global),
                "question_id": trial.question_id,
                "question_type": question_type,
                "question_start_time": onset_time,
                "answer_time": answer_time,
                "answer_latency_sec": answer_latency,
                "epoch_start_time": epoch_start_time,
                "epoch_end_time": epoch_end_time,
                "baseline_start_time": onset_time + BASELINE_START,
                "baseline_end_time": onset_time + BASELINE_END,
                "analysis_window_start_sec": PRIMARY_WINDOW[0],
                "analysis_window_end_sec": PRIMARY_WINDOW[1],
                "source_log": trial.source_log,
                "included_step5_analysis": True,
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

            for chromophore, front_channels, back_channels in [
                ("hbo", front_hbo_good, back_hbo_good),
                ("hbr", front_hbr_good, back_hbr_good),
            ]:
                for window_label, start_sec, end_sec in SENSITIVITY_WINDOWS:
                    front_response, _ = extract_window_response(
                        session_data,
                        session_times,
                        onset_time,
                        [name_to_index[item] for item in front_channels],
                        start_sec,
                        end_sec,
                    )
                    back_response, _ = extract_window_response(
                        session_data,
                        session_times,
                        onset_time,
                        [name_to_index[item] for item in back_channels],
                        start_sec,
                        end_sec,
                    )
                    sensitivity_rows.append(
                        {
                            "participant_id": trial.participant_id,
                            "session_id": trial.session_id,
                            "question_type": question_type,
                            "question_id": trial.question_id,
                            "answer_latency_sec": answer_latency,
                            "window_label": window_label,
                            "window_start_sec": start_sec,
                            "window_end_sec": end_sec,
                            "timepoint_sec": math.nan,
                            "chromophore": chromophore,
                            "front_roi_mean": float(np.mean(front_response)) if front_response is not None else math.nan,
                            "back_roi_mean": float(np.mean(back_response)) if back_response is not None else math.nan,
                        }
                    )

                front_response, _ = extract_timepoint_response(
                    session_data,
                    session_times,
                    onset_time,
                    [name_to_index[item] for item in front_channels],
                    SINGLE_TIME_SEC,
                )
                back_response, _ = extract_timepoint_response(
                    session_data,
                    session_times,
                    onset_time,
                    [name_to_index[item] for item in back_channels],
                    SINGLE_TIME_SEC,
                )
                sensitivity_rows.append(
                    {
                        "participant_id": trial.participant_id,
                        "session_id": trial.session_id,
                        "question_type": question_type,
                        "question_id": trial.question_id,
                        "answer_latency_sec": answer_latency,
                        "window_label": "timepoint_9s",
                        "window_start_sec": math.nan,
                        "window_end_sec": math.nan,
                        "timepoint_sec": SINGLE_TIME_SEC,
                        "chromophore": chromophore,
                        "front_roi_mean": float(np.mean(front_response)) if front_response is not None else math.nan,
                        "back_roi_mean": float(np.mean(back_response)) if back_response is not None else math.nan,
                    }
                )

            front_waveform = extract_epoch_waveform(
                session_data,
                session_times,
                onset_time,
                [name_to_index[item] for item in front_hbo_good],
                WAVEFORM_GRID,
            )
            back_waveform = extract_epoch_waveform(
                session_data,
                session_times,
                onset_time,
                [name_to_index[item] for item in back_hbo_good],
                WAVEFORM_GRID,
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

        session_trial_hbo_df = pd.DataFrame(
            [row for row in trial_hbo_rows if row["participant_id"] == session_row.participant_id and row["session_id"] == session_row.session_id]
        )
        session_trial_hbr_df = pd.DataFrame(
            [row for row in trial_hbr_rows if row["participant_id"] == session_row.participant_id and row["session_id"] == session_row.session_id]
        )
        abstract_hbo_df = session_trial_hbo_df[session_trial_hbo_df["question_type"] == "Abstract"]
        concrete_hbo_df = session_trial_hbo_df[session_trial_hbo_df["question_type"] == "Concrete"]
        abstract_hbr_df = session_trial_hbr_df[session_trial_hbr_df["question_type"] == "Abstract"]
        concrete_hbr_df = session_trial_hbr_df[session_trial_hbr_df["question_type"] == "Concrete"]

        included_session = (
            len(abstract_hbo_df) >= 1
            and len(concrete_hbo_df) >= 1
            and len(front_hbo_good) >= 1
            and len(back_hbo_good) >= 1
        )
        if included_session:
            included_session_keys.add(session_key)
            inclusion_note = "Included in Step 5 fixed-window onset-locked ROI analysis."
        else:
            detail_parts: list[str] = []
            if len(abstract_hbo_df) < 1:
                detail_parts.append("No valid abstract trials remained after onset-locked epoch screening.")
            if len(concrete_hbo_df) < 1:
                detail_parts.append("No valid concrete trials remained after onset-locked epoch screening.")
            if len(front_hbo_good) < 1:
                detail_parts.append("No valid front ROI HbO channels remained.")
            if len(back_hbo_good) < 1:
                detail_parts.append("No valid back ROI HbO channels remained.")
            inclusion_note = " | ".join(detail_parts)

        participant_summary_rows.append(
            {
                "participant_id": session_row.participant_id,
                "session_id": session_row.session_id,
                "manual_step4_override": coerce_bool(session_row.manual_step4_override),
                "manual_step4_override_reason": session_row.manual_step4_override_reason,
                "included_step5_analysis": included_session,
                "step5_inclusion_note": inclusion_note,
                "n_valid_abstract_trials": len(abstract_hbo_df),
                "n_valid_concrete_trials": len(concrete_hbo_df),
                "front_hbo_n_channels": len(front_hbo_good),
                "back_hbo_n_channels": len(back_hbo_good),
                "front_hbr_n_channels": len(front_hbr_good),
                "back_hbr_n_channels": len(back_hbr_good),
                "front_hbo_channels": ";".join(front_hbo_good),
                "back_hbo_channels": ";".join(back_hbo_good),
                "front_hbr_channels": ";".join(front_hbr_good),
                "back_hbr_channels": ";".join(back_hbr_good),
                "front_abstract_hbo_mean": float(abstract_hbo_df["front_roi_mean"].mean()) if not abstract_hbo_df.empty else math.nan,
                "front_concrete_hbo_mean": float(concrete_hbo_df["front_roi_mean"].mean()) if not concrete_hbo_df.empty else math.nan,
                "back_abstract_hbo_mean": float(abstract_hbo_df["back_roi_mean"].mean()) if not abstract_hbo_df.empty else math.nan,
                "back_concrete_hbo_mean": float(concrete_hbo_df["back_roi_mean"].mean()) if not concrete_hbo_df.empty else math.nan,
                "dissociation_hbo": (
                    float(abstract_hbo_df["front_roi_mean"].mean())
                    - float(concrete_hbo_df["front_roi_mean"].mean())
                    - float(abstract_hbo_df["back_roi_mean"].mean())
                    + float(concrete_hbo_df["back_roi_mean"].mean())
                )
                if not abstract_hbo_df.empty and not concrete_hbo_df.empty
                else math.nan,
                "front_abstract_hbr_mean": float(abstract_hbr_df["front_roi_mean"].mean()) if not abstract_hbr_df.empty else math.nan,
                "front_concrete_hbr_mean": float(concrete_hbr_df["front_roi_mean"].mean()) if not concrete_hbr_df.empty else math.nan,
                "back_abstract_hbr_mean": float(abstract_hbr_df["back_roi_mean"].mean()) if not abstract_hbr_df.empty else math.nan,
                "back_concrete_hbr_mean": float(concrete_hbr_df["back_roi_mean"].mean()) if not concrete_hbr_df.empty else math.nan,
                "dissociation_hbr": (
                    float(abstract_hbr_df["front_roi_mean"].mean())
                    - float(concrete_hbr_df["front_roi_mean"].mean())
                    - float(abstract_hbr_df["back_roi_mean"].mean())
                    + float(concrete_hbr_df["back_roi_mean"].mean())
                )
                if not abstract_hbr_df.empty and not concrete_hbr_df.empty
                else math.nan,
            }
        )

        inclusion_rows.append(
            {
                "participant_id": session_row.participant_id,
                "session_id": session_row.session_id,
                "manual_step4_override": coerce_bool(session_row.manual_step4_override),
                "manual_step4_override_reason": session_row.manual_step4_override_reason,
                "step4_inclusion_note": session_row.step4_inclusion_note,
                "step3_final_status": session_row.step3_final_status,
                "preprocessing_success": coerce_bool(session_row.preprocessing_success),
                "front_hbo_n_channels": len(front_hbo_good),
                "back_hbo_n_channels": len(back_hbo_good),
                "n_valid_abstract_trials": len(abstract_hbo_df),
                "n_valid_concrete_trials": len(concrete_hbo_df),
                "included_step5_analysis": included_session,
                "step5_inclusion_note": inclusion_note,
            }
        )

    epoch_manifest_df = pd.DataFrame(epoch_manifest_rows).sort_values(
        ["participant_id", "session_id", "trial_idx_global"]
    ).reset_index(drop=True)
    trial_hbo_df = pd.DataFrame(trial_hbo_rows).sort_values(
        ["participant_id", "session_id", "trial_idx_global"]
    ).reset_index(drop=True)
    trial_hbr_df = pd.DataFrame(trial_hbr_rows).sort_values(
        ["participant_id", "session_id", "trial_idx_global"]
    ).reset_index(drop=True)
    participant_summary_df = pd.DataFrame(participant_summary_rows).sort_values(
        ["participant_id", "session_id"]
    ).reset_index(drop=True)
    inclusion_df = pd.DataFrame(inclusion_rows).sort_values(["participant_id", "session_id"]).reset_index(drop=True)
    sensitivity_df = pd.DataFrame(sensitivity_rows).sort_values(
        ["window_label", "chromophore", "participant_id", "session_id", "question_id"]
    ).reset_index(drop=True)
    diagnostics = {
        "waveform_rows": waveform_rows,
        "waveform_grid": WAVEFORM_GRID,
        "included_session_keys": included_session_keys,
        "session_channel_meta": session_channel_meta,
        "exclusions": exclusions,
    }
    return inclusion_df, epoch_manifest_df, trial_hbo_df, trial_hbr_df, participant_summary_df, sensitivity_df, waveform_rows, diagnostics


def build_condition_test_row(
    participant_df: pd.DataFrame,
    abstract_col: str,
    concrete_col: str,
    n_abstract_col: str,
    n_concrete_col: str,
    analysis_label: str,
    chromophore: str,
    roi: str,
    test_scope: str,
    window_label: str,
    window_start_sec: float | None = None,
    window_end_sec: float | None = None,
    timepoint_sec: float | None = None,
) -> dict[str, Any]:
    valid_df = participant_df.dropna(subset=[abstract_col, concrete_col]).copy()
    payload = run_one_sample_test(valid_df[abstract_col] - valid_df[concrete_col], analysis_label)
    return {
        "analysis_label": analysis_label,
        "chromophore": chromophore,
        "roi": roi,
        "test_scope": test_scope,
        "window_label": window_label,
        "window_start_sec": window_start_sec if window_start_sec is not None else math.nan,
        "window_end_sec": window_end_sec if window_end_sec is not None else math.nan,
        "timepoint_sec": timepoint_sec if timepoint_sec is not None else math.nan,
        "n_included_participants": payload["n"],
        "n_valid_abstract_trials": int(valid_df[n_abstract_col].sum()) if not valid_df.empty else 0,
        "n_valid_concrete_trials": int(valid_df[n_concrete_col].sum()) if not valid_df.empty else 0,
        "abstract_mean": float(valid_df[abstract_col].mean()) if not valid_df.empty else math.nan,
        "concrete_mean": float(valid_df[concrete_col].mean()) if not valid_df.empty else math.nan,
        "mean_difference_abstract_minus_concrete": payload["mean"],
        "t_stat": payload["t_stat"],
        "df": payload["df"],
        "p_value": payload["p_value"],
        "ci_low": payload["ci_low"],
        "ci_high": payload["ci_high"],
        "cohens_dz": payload["cohens_d"],
        "alpha": ALPHA,
    }


def build_dissociation_test_row(
    participant_df: pd.DataFrame,
    front_abstract_col: str,
    front_concrete_col: str,
    back_abstract_col: str,
    back_concrete_col: str,
    n_abstract_col: str,
    n_concrete_col: str,
    analysis_label: str,
    chromophore: str,
    test_scope: str,
    window_label: str,
    window_start_sec: float | None = None,
    window_end_sec: float | None = None,
    timepoint_sec: float | None = None,
) -> dict[str, Any]:
    valid_df = participant_df.dropna(
        subset=[front_abstract_col, front_concrete_col, back_abstract_col, back_concrete_col]
    ).copy()
    diff_values = (
        valid_df[front_abstract_col]
        - valid_df[front_concrete_col]
        - valid_df[back_abstract_col]
        + valid_df[back_concrete_col]
    )
    payload = run_one_sample_test(diff_values, analysis_label)
    return {
        "analysis_label": analysis_label,
        "chromophore": chromophore,
        "test_scope": test_scope,
        "window_label": window_label,
        "window_start_sec": window_start_sec if window_start_sec is not None else math.nan,
        "window_end_sec": window_end_sec if window_end_sec is not None else math.nan,
        "timepoint_sec": timepoint_sec if timepoint_sec is not None else math.nan,
        "n_included_participants": payload["n"],
        "n_valid_abstract_trials": int(valid_df[n_abstract_col].sum()) if not valid_df.empty else 0,
        "n_valid_concrete_trials": int(valid_df[n_concrete_col].sum()) if not valid_df.empty else 0,
        "front_abstract_mean": float(valid_df[front_abstract_col].mean()) if not valid_df.empty else math.nan,
        "front_concrete_mean": float(valid_df[front_concrete_col].mean()) if not valid_df.empty else math.nan,
        "back_abstract_mean": float(valid_df[back_abstract_col].mean()) if not valid_df.empty else math.nan,
        "back_concrete_mean": float(valid_df[back_concrete_col].mean()) if not valid_df.empty else math.nan,
        "mean_dissociation": payload["mean"],
        "t_stat": payload["t_stat"],
        "df": payload["df"],
        "p_value": payload["p_value"],
        "ci_low": payload["ci_low"],
        "ci_high": payload["ci_high"],
        "cohens_dz": payload["cohens_d"],
        "alpha": ALPHA,
    }


def build_primary_results(
    participant_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    included_df = participant_summary_df[participant_summary_df["included_step5_analysis"]].copy()

    front_rows = [
        build_condition_test_row(
            included_df,
            "front_abstract_hbo_mean",
            "front_concrete_hbo_mean",
            "n_valid_abstract_trials",
            "n_valid_concrete_trials",
            "Primary front ROI fixed-window HbO test",
            "hbo",
            "Front",
            "primary",
            "window_7_11",
            PRIMARY_WINDOW[0],
            PRIMARY_WINDOW[1],
        ),
        build_condition_test_row(
            included_df,
            "front_abstract_hbr_mean",
            "front_concrete_hbr_mean",
            "n_valid_abstract_trials",
            "n_valid_concrete_trials",
            "Secondary front ROI fixed-window HbR test",
            "hbr",
            "Front",
            "secondary_hbr",
            "window_7_11",
            PRIMARY_WINDOW[0],
            PRIMARY_WINDOW[1],
        ),
    ]
    back_rows = [
        build_condition_test_row(
            included_df,
            "back_abstract_hbo_mean",
            "back_concrete_hbo_mean",
            "n_valid_abstract_trials",
            "n_valid_concrete_trials",
            "Secondary back ROI fixed-window HbO test",
            "hbo",
            "Back",
            "secondary",
            "window_7_11",
            PRIMARY_WINDOW[0],
            PRIMARY_WINDOW[1],
        ),
        build_condition_test_row(
            included_df,
            "back_abstract_hbr_mean",
            "back_concrete_hbr_mean",
            "n_valid_abstract_trials",
            "n_valid_concrete_trials",
            "Secondary back ROI fixed-window HbR test",
            "hbr",
            "Back",
            "secondary_hbr",
            "window_7_11",
            PRIMARY_WINDOW[0],
            PRIMARY_WINDOW[1],
        ),
    ]
    diss_rows = [
        build_dissociation_test_row(
            included_df,
            "front_abstract_hbo_mean",
            "front_concrete_hbo_mean",
            "back_abstract_hbo_mean",
            "back_concrete_hbo_mean",
            "n_valid_abstract_trials",
            "n_valid_concrete_trials",
            "Secondary front-versus-back fixed-window HbO dissociation test",
            "hbo",
            "secondary",
            "window_7_11",
            PRIMARY_WINDOW[0],
            PRIMARY_WINDOW[1],
        ),
        build_dissociation_test_row(
            included_df,
            "front_abstract_hbr_mean",
            "front_concrete_hbr_mean",
            "back_abstract_hbr_mean",
            "back_concrete_hbr_mean",
            "n_valid_abstract_trials",
            "n_valid_concrete_trials",
            "Secondary front-versus-back fixed-window HbR dissociation test",
            "hbr",
            "secondary_hbr",
            "window_7_11",
            PRIMARY_WINDOW[0],
            PRIMARY_WINDOW[1],
        ),
    ]

    return pd.DataFrame(front_rows), pd.DataFrame(back_rows), pd.DataFrame(diss_rows)


def build_sensitivity_participant_summary(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    participant_rows: list[dict[str, Any]] = []
    group_cols = ["window_label", "chromophore", "participant_id", "session_id"]
    for keys, session_df in sensitivity_df.groupby(group_cols, sort=True):
        window_label, chromophore, participant_id, session_id = keys
        abstract_df = session_df[session_df["question_type"] == "Abstract"]
        concrete_df = session_df[session_df["question_type"] == "Concrete"]
        participant_rows.append(
            {
                "window_label": window_label,
                "chromophore": chromophore,
                "participant_id": participant_id,
                "session_id": session_id,
                "n_valid_abstract_trials": len(abstract_df),
                "n_valid_concrete_trials": len(concrete_df),
                "front_abstract_mean": float(abstract_df["front_roi_mean"].mean()) if not abstract_df.empty else math.nan,
                "front_concrete_mean": float(concrete_df["front_roi_mean"].mean()) if not concrete_df.empty else math.nan,
                "back_abstract_mean": float(abstract_df["back_roi_mean"].mean()) if not abstract_df.empty else math.nan,
                "back_concrete_mean": float(concrete_df["back_roi_mean"].mean()) if not concrete_df.empty else math.nan,
            }
        )
    return pd.DataFrame(participant_rows)


def build_sensitivity_results(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    participant_df = build_sensitivity_participant_summary(sensitivity_df)
    rows: list[dict[str, Any]] = []
    for (window_label, chromophore), subset in participant_df.groupby(["window_label", "chromophore"], sort=True):
        timepoint_sec = SINGLE_TIME_SEC if window_label == "timepoint_9s" else None
        if window_label == "window_6_10":
            window_start, window_end = 6.0, 10.0
        elif window_label == "window_8_12":
            window_start, window_end = 8.0, 12.0
        else:
            window_start, window_end = None, None

        rows.append(
            build_condition_test_row(
                subset,
                "front_abstract_mean",
                "front_concrete_mean",
                "n_valid_abstract_trials",
                "n_valid_concrete_trials",
                f"{chromophore.upper()} front ROI sensitivity: {window_label}",
                chromophore,
                "Front",
                "sensitivity",
                window_label,
                window_start,
                window_end,
                timepoint_sec,
            )
        )
        rows.append(
            build_condition_test_row(
                subset,
                "back_abstract_mean",
                "back_concrete_mean",
                "n_valid_abstract_trials",
                "n_valid_concrete_trials",
                f"{chromophore.upper()} back ROI sensitivity: {window_label}",
                chromophore,
                "Back",
                "sensitivity",
                window_label,
                window_start,
                window_end,
                timepoint_sec,
            )
        )
        rows.append(
            build_dissociation_test_row(
                subset,
                "front_abstract_mean",
                "front_concrete_mean",
                "back_abstract_mean",
                "back_concrete_mean",
                "n_valid_abstract_trials",
                "n_valid_concrete_trials",
                f"{chromophore.upper()} dissociation sensitivity: {window_label}",
                chromophore,
                "sensitivity",
                window_label,
                window_start,
                window_end,
                timepoint_sec,
            )
        )
    return pd.DataFrame(rows)


def build_fast_response_sensitivity(trial_hbo_df: pd.DataFrame, participant_summary_df: pd.DataFrame) -> pd.DataFrame:
    included_sessions = set(
        zip(
            participant_summary_df.loc[participant_summary_df["included_step5_analysis"], "participant_id"],
            participant_summary_df.loc[participant_summary_df["included_step5_analysis"], "session_id"],
        )
    )
    valid_trial_df = trial_hbo_df[
        trial_hbo_df.apply(lambda row: (row["participant_id"], row["session_id"]) in included_sessions, axis=1)
    ].copy()
    valid_trial_df["eligible_fast_sensitivity"] = (
        valid_trial_df["answer_latency_sec"].notna() & (valid_trial_df["answer_latency_sec"] >= FAST_RESPONSE_MIN_SEC)
    )
    filtered_df = valid_trial_df[valid_trial_df["eligible_fast_sensitivity"]].copy()

    participant_rows = []
    for (participant_id, session_id), session_df in filtered_df.groupby(["participant_id", "session_id"], sort=True):
        abstract_df = session_df[session_df["question_type"] == "Abstract"]
        concrete_df = session_df[session_df["question_type"] == "Concrete"]
        if abstract_df.empty or concrete_df.empty:
            continue
        participant_rows.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "n_valid_abstract_trials": len(abstract_df),
                "n_valid_concrete_trials": len(concrete_df),
                "front_abstract_mean": float(abstract_df["front_roi_mean"].mean()),
                "front_concrete_mean": float(concrete_df["front_roi_mean"].mean()),
            }
        )
    participant_df = pd.DataFrame(participant_rows)
    test_row = build_condition_test_row(
        participant_df,
        "front_abstract_mean",
        "front_concrete_mean",
        "n_valid_abstract_trials",
        "n_valid_concrete_trials",
        "Fast-response sensitivity front ROI HbO test",
        "hbo",
        "Front",
        "fast_response_sensitivity",
        "window_7_11_fast_response",
        PRIMARY_WINDOW[0],
        PRIMARY_WINDOW[1],
    )
    test_row["total_trials_before_filter"] = int(len(valid_trial_df))
    test_row["total_trials_after_filter"] = int(len(filtered_df))
    test_row["trials_removed_for_fast_answers"] = int((~valid_trial_df["eligible_fast_sensitivity"]).sum())
    return pd.DataFrame([test_row])


def plot_roi_pair(participant_summary_df: pd.DataFrame, roi_name: str, abstract_col: str, concrete_col: str, output_name: str) -> None:
    included_df = participant_summary_df[participant_summary_df["included_step5_analysis"]].copy()
    plt.figure(figsize=(6.5, 5.5))
    for row in included_df.itertuples(index=False):
        values = [getattr(row, abstract_col), getattr(row, concrete_col)]
        plt.plot([0, 1], values, color="#B0B0B0", alpha=0.6, linewidth=1)
    means = [included_df[abstract_col].mean(), included_df[concrete_col].mean()]
    color = "#B03A2E" if roi_name == "Front" else "#2471A3"
    plt.plot([0, 1], means, color=color, marker="o", linewidth=2.5)
    plt.xticks([0, 1], ["Abstract", "Concrete"])
    plt.ylabel("Baseline-corrected HbO mean")
    plt.title(f"{roi_name} ROI fixed-window HbO summary")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=150)
    plt.close()


def plot_waveforms(
    waveform_rows: list[dict[str, Any]],
    included_session_keys: set[tuple[str, str]],
    waveform_grid: np.ndarray,
    roi_name: str,
    output_name: str,
) -> None:
    waveform_df = pd.DataFrame(waveform_rows)
    if waveform_df.empty:
        return
    waveform_df["valid_session"] = waveform_df.apply(
        lambda row: (row["participant_id"], row["session_id"]) in included_session_keys,
        axis=1,
    )
    waveform_df = waveform_df[(waveform_df["valid_session"]) & (waveform_df["roi"] == roi_name)].copy()
    if waveform_df.empty:
        return
    grouped = (
        waveform_df.groupby(["participant_id", "session_id", "question_type"], sort=True)["waveform"]
        .apply(lambda items: np.vstack(items).mean(axis=0))
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    for condition, color in [("Abstract", "#C0392B"), ("Concrete", "#1F4E79")]:
        subset = grouped[grouped["question_type"] == condition]
        if subset.empty:
            continue
        waves = np.vstack(subset["waveform"].to_list())
        mean_wave = waves.mean(axis=0)
        sem_wave = waves.std(axis=0, ddof=1) / math.sqrt(waves.shape[0]) if waves.shape[0] > 1 else np.zeros_like(mean_wave)
        plt.plot(waveform_grid, mean_wave, color=color, label=condition)
        plt.fill_between(waveform_grid, mean_wave - sem_wave, mean_wave + sem_wave, color=color, alpha=0.2)
    plt.axvspan(PRIMARY_WINDOW[0], PRIMARY_WINDOW[1], color="#F4D03F", alpha=0.25, label="Primary 7-11 s window")
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Seconds from question onset")
    plt.ylabel("Baseline-corrected HbO")
    plt.title(f"{roi_name} ROI onset-locked HbO waveform")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=150)
    plt.close()


def plot_dissociation_distribution(participant_summary_df: pd.DataFrame) -> None:
    included_df = participant_summary_df[participant_summary_df["included_step5_analysis"]].copy()
    values = included_df["dissociation_hbo"].dropna().to_numpy(dtype=float)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=min(12, max(6, len(values))), color="#4C956C", edgecolor="white")
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    if len(values):
        plt.axvline(values.mean(), color="#922B21", linestyle="-", linewidth=2)
    plt.xlabel("Participant fixed-window dissociation score")
    plt.ylabel("Count")
    plt.title("Fixed-window front-versus-back dissociation distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dissociation_histogram_fixed_window.png", dpi=150)
    plt.close()


def build_session_evokeds(epoch_manifest_df: pd.DataFrame, session_df: pd.DataFrame) -> tuple[mne.Evoked, mne.Evoked, dict[str, Any]]:
    included_df = epoch_manifest_df[epoch_manifest_df["included_step5_trial"]].copy()
    status_lookup = {
        (row["participant_id"], row["session_id"]): row["preprocessed_file"]
        for _, row in session_df.iterrows()
    }
    abstract_arrays: list[np.ndarray] = []
    concrete_arrays: list[np.ndarray] = []
    template_info = None
    abstract_epochs = 0
    concrete_epochs = 0

    for (participant_id, session_id), trials_df in included_df.groupby(["participant_id", "session_id"], sort=True):
        raw = mne.io.read_raw_fif(status_lookup[(participant_id, session_id)], preload=True, verbose="ERROR")
        if template_info is None:
            template_info = raw.info.copy()
            template_info["bads"] = []
        event_rows = []
        for row in trials_df.itertuples(index=False):
            event_code = 1 if row.question_type == "Abstract" else 2
            sample = raw.time_as_index([row.question_start_time])[0]
            event_rows.append([sample, 0, event_code])
        if not event_rows:
            continue
        events = np.asarray(sorted(event_rows), dtype=int)
        epochs = mne.Epochs(
            raw,
            events,
            event_id={"Abstract": 1, "Concrete": 2},
            tmin=EPOCH_TMIN,
            tmax=EPOCH_TMAX,
            baseline=(BASELINE_START, BASELINE_END),
            preload=True,
            reject_by_annotation=False,
            verbose="ERROR",
        )
        if len(epochs["Abstract"]):
            abstract_epochs += len(epochs["Abstract"])
            data = epochs["Abstract"].get_data(copy=True).mean(axis=0)
            bad_idx = [epochs.ch_names.index(name) for name in raw.info["bads"] if name in epochs.ch_names]
            if bad_idx:
                data[bad_idx, :] = np.nan
            abstract_arrays.append(data)
        if len(epochs["Concrete"]):
            concrete_epochs += len(epochs["Concrete"])
            data = epochs["Concrete"].get_data(copy=True).mean(axis=0)
            bad_idx = [epochs.ch_names.index(name) for name in raw.info["bads"] if name in epochs.ch_names]
            if bad_idx:
                data[bad_idx, :] = np.nan
            concrete_arrays.append(data)

    if template_info is None:
        raise RuntimeError("No valid Step 5 sessions were available to construct evoked data.")

    abstract_grand = np.nan_to_num(np.nanmean(np.stack(abstract_arrays, axis=0), axis=0), nan=0.0)
    concrete_grand = np.nan_to_num(np.nanmean(np.stack(concrete_arrays, axis=0), axis=0), nan=0.0)
    evoked_abstract = mne.EvokedArray(abstract_grand, template_info, tmin=EPOCH_TMIN, comment="Abstract", nave=len(abstract_arrays))
    evoked_concrete = mne.EvokedArray(concrete_grand, template_info, tmin=EPOCH_TMIN, comment="Concrete", nave=len(concrete_arrays))
    metadata = {
        "n_included_sessions": int(included_df[["participant_id", "session_id"]].drop_duplicates().shape[0]),
        "n_abstract_epochs": abstract_epochs,
        "n_concrete_epochs": concrete_epochs,
    }
    return evoked_abstract, evoked_concrete, metadata


def plot_topography_9s(evoked_abstract: mne.Evoked, evoked_concrete: mne.Evoked) -> None:
    evoked_diff = mne.combine_evoked([evoked_abstract, evoked_concrete], weights=[1, -1])
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.08]), layout="constrained")
    topomap_args = dict(extrapolate="local")

    abs_vals = []
    for evoked in [evoked_abstract, evoked_concrete, evoked_diff]:
        picked = evoked.copy().pick("hbo")
        idx = picked.time_as_index(SINGLE_TIME_SEC)[0]
        abs_vals.append(np.abs(picked.data[:, idx]))
    vmax = float(np.max(np.concatenate(abs_vals)))
    vlim = (-vmax, vmax)

    evoked_abstract.plot_topomap(
        ch_type="hbo",
        times=SINGLE_TIME_SEC,
        axes=axes[0],
        colorbar=False,
        show=False,
        vlim=vlim,
        **topomap_args,
    )
    evoked_concrete.plot_topomap(
        ch_type="hbo",
        times=SINGLE_TIME_SEC,
        axes=axes[1],
        colorbar=False,
        show=False,
        vlim=vlim,
        **topomap_args,
    )
    evoked_diff.plot_topomap(
        ch_type="hbo",
        times=SINGLE_TIME_SEC,
        axes=axes[2:],
        colorbar=True,
        show=False,
        vlim=vlim,
        **topomap_args,
    )
    axes[0].set_title("HbO: Abstract")
    axes[1].set_title("HbO: Concrete")
    axes[2].set_title("HbO: Abstract-Concrete")
    fig.suptitle("Step 5 onset-locked HbO topography at 9.0 s")
    fig.savefig(FIGURES_DIR / "topography_9s.png", dpi=160)
    plt.close(fig)


def write_report_artifacts(
    inclusion_df: pd.DataFrame,
    epoch_manifest_df: pd.DataFrame,
    trial_hbo_df: pd.DataFrame,
    participant_summary_df: pd.DataFrame,
    front_df: pd.DataFrame,
    back_df: pd.DataFrame,
    diss_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    fast_df: pd.DataFrame,
    exclusions: dict[str, dict[str, str]],
) -> pd.DataFrame:
    primary_row = front_df[front_df["chromophore"] == "hbo"].iloc[0]
    back_hbo_row = back_df[back_df["chromophore"] == "hbo"].iloc[0]
    diss_hbo_row = diss_df[diss_df["chromophore"] == "hbo"].iloc[0]

    if pd.notna(primary_row["p_value"]) and primary_row["p_value"] < ALPHA and pd.notna(primary_row["mean_difference_abstract_minus_concrete"]):
        if primary_row["mean_difference_abstract_minus_concrete"] > 0:
            conclusion_text = (
                "In the targeted fixed-window onset-locked follow-up analysis, abstract questions elicited a "
                "significantly larger HbO response than concrete questions in the left frontal ROI during the "
                f"7-11 s post-onset window (t({int(primary_row['df'])})={format_float(primary_row['t_stat'], 4)}, "
                f"p={format_p_value(primary_row['p_value'])}, mean difference={format_float(primary_row['mean_difference_abstract_minus_concrete'], 4)}, "
                f"95% CI [{format_float(primary_row['ci_low'], 4)}, {format_float(primary_row['ci_high'], 4)}], "
                f"d_z={format_float(primary_row['cohens_dz'], 4)}). This follow-up pattern suggests that the "
                "abstract-versus-concrete contrast is more visible in a delayed onset-locked analysis than in the "
                "whole-question-window metric used in Step 4."
            )
        else:
            conclusion_text = (
                "In the targeted fixed-window onset-locked follow-up analysis, concrete questions elicited a "
                "significantly larger HbO response than abstract questions in the left frontal ROI during the "
                f"7-11 s post-onset window (t({int(primary_row['df'])})={format_float(primary_row['t_stat'], 4)}, "
                f"p={format_p_value(primary_row['p_value'])}, mean difference={format_float(primary_row['mean_difference_abstract_minus_concrete'], 4)}, "
                f"95% CI [{format_float(primary_row['ci_low'], 4)}, {format_float(primary_row['ci_high'], 4)}], "
                f"d_z={format_float(primary_row['cohens_dz'], 4)})."
            )
    else:
        conclusion_text = (
            "In the targeted fixed-window onset-locked follow-up analysis, the difference between abstract and "
            "concrete questions in the left frontal ROI during the 7-11 s post-onset window was not statistically "
            f"significant (t({int(primary_row['df'])})={format_float(primary_row['t_stat'], 4)}, "
            f"p={format_p_value(primary_row['p_value'])}, mean difference={format_float(primary_row['mean_difference_abstract_minus_concrete'], 4)}, "
            f"95% CI [{format_float(primary_row['ci_low'], 4)}, {format_float(primary_row['ci_high'], 4)}], "
            f"d_z={format_float(primary_row['cohens_dz'], 4)})."
        )

    conclusion_df = pd.DataFrame(
        [
            {
                "n_included_participants": int(primary_row["n_included_participants"]),
                "n_valid_abstract_trials": int(primary_row["n_valid_abstract_trials"]),
                "n_valid_concrete_trials": int(primary_row["n_valid_concrete_trials"]),
                "primary_p_value": primary_row["p_value"],
                "primary_significant": bool(pd.notna(primary_row["p_value"]) and primary_row["p_value"] < ALPHA),
                "conclusion_text": conclusion_text,
            }
        ]
    )

    primary_table = make_latex_grid(
        [
            "n",
            "Abstract",
            "Concrete",
            "Mean diff",
            "t",
            "df",
            "p",
            "95\\% CI",
            "d$_z$",
        ],
        [
            [
                int(primary_row["n_included_participants"]),
                format_float(primary_row["abstract_mean"], 4),
                format_float(primary_row["concrete_mean"], 4),
                format_float(primary_row["mean_difference_abstract_minus_concrete"], 4),
                format_float(primary_row["t_stat"], 4),
                int(primary_row["df"]),
                format_p_value(primary_row["p_value"]),
                f"[{format_float(primary_row['ci_low'], 4)}, {format_float(primary_row['ci_high'], 4)}]",
                format_float(primary_row["cohens_dz"], 4),
            ]
        ],
        r"rrrrrrrrr",
    )
    (TABLES_DIR / "primary_front_test_table.tex").write_text(primary_table + "\n", encoding="utf-8")

    secondary_rows = []
    for row in pd.concat([back_df, diss_df, sensitivity_df, fast_df], ignore_index=True).itertuples(index=False):
        if hasattr(row, "roi"):
            label = f"{str(row.chromophore).upper()} {row.roi} {row.window_label}"
            mean_value = getattr(row, "mean_difference_abstract_minus_concrete", math.nan)
        else:
            label = f"{str(row.chromophore).upper()} Dissociation {row.window_label}"
            mean_value = getattr(row, "mean_dissociation", math.nan)
        secondary_rows.append(
            [
                label,
                int(row.n_included_participants),
                format_float(mean_value, 4),
                format_float(row.t_stat, 4),
                int(row.df) if pd.notna(row.df) else "",
                format_p_value(row.p_value),
                f"[{format_float(row.ci_low, 4)}, {format_float(row.ci_high, 4)}]",
                format_float(row.cohens_dz, 4),
            ]
        )
    (TABLES_DIR / "secondary_tests_table.tex").write_text(
        make_latex_grid(
            ["Analysis", "n", "Mean", "t", "df", "p", "95\\% CI", "d$_z$"],
            secondary_rows,
            r"lrrrrrrr",
        )
        + "\n",
        encoding="utf-8",
    )
    (TEXT_DIR / "final_conclusion.tex").write_text(conclusion_text + "\n", encoding="utf-8")

    overview_rows = [
        ("Step 4 included participant-sessions reused", len(inclusion_df)),
        ("Step 5 included participant-sessions", int(inclusion_df["included_step5_analysis"].sum())),
        ("Valid onset-locked trials in Step 5", int(epoch_manifest_df["included_step5_trial"].sum())),
        ("Primary valid abstract trials", int(primary_row["n_valid_abstract_trials"])),
        ("Primary valid concrete trials", int(primary_row["n_valid_concrete_trials"])),
        ("Primary window", "7.0 s to 11.0 s after onset"),
        ("Sensitivity windows", "6.0-10.0 s, 8.0-12.0 s, and single time point at 9.0 s"),
        ("Fast-response sensitivity", "Exclude trials with answer latency < 11.0 s"),
    ]

    manual_rows = [
        (f"{row.participant_id} / {row.session_id}", row.manual_step4_override_reason)
        for row in inclusion_df.itertuples(index=False)
        if coerce_bool(row.manual_step4_override)
    ]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{array}",
        r"\begin{document}",
        r"\section*{Step 5 Fixed-Window Onset-Locked ROI Follow-Up}",
        r"This report documents the targeted onset-locked follow-up requested after the non-significant pre-specified Step~4 primary test. Step~4 remains the primary analysis; Step~5 is a focused follow-up using the same participant-session cohort and a fixed 7--11~s post-onset HbO window.",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
    ]

    if exclusions:
        lines.extend(
            [
                r"\subsection*{Shared Global Exclusions}",
                r"The shared exclusion list remained active in Step~5 and the excluded participants were not reintroduced.",
                make_latex_table(exclusion_table_rows(exclusions)),
            ]
        )

    if manual_rows:
        lines.extend(
            [
                r"\subsection*{Inherited Step 4 Manual Inclusions}",
                r"Step~5 reused the Step~4 analysis cohort, including the following explicitly approved manual inclusions.",
                make_latex_table(manual_rows),
            ]
        )

    lines.extend(
        [
            r"\subsection*{Primary Front ROI Fixed-Window Test}",
            r"\input{tables/primary_front_test_table.tex}",
            r"\begin{figure}[H]",
            r"\centering",
            r"\includegraphics[width=0.48\linewidth]{../../figures/step5/front_roi_window_plot.png}",
            r"\includegraphics[width=0.48\linewidth]{../../figures/step5/back_roi_window_plot.png}",
            r"\caption{Participant-level fixed-window HbO summaries for the front and back ROIs.}",
            r"\end{figure}",
            r"\begin{figure}[H]",
            r"\centering",
            r"\includegraphics[width=0.48\linewidth]{../../figures/step5/front_waveform_with_window.png}",
            r"\includegraphics[width=0.48\linewidth]{../../figures/step5/back_waveform_with_window.png}",
            r"\caption{Descriptive onset-locked HbO waveforms with the 7--11~s primary analysis window shaded.}",
            r"\end{figure}",
            r"\begin{figure}[H]",
            r"\centering",
            r"\includegraphics[width=0.62\linewidth]{../../figures/step5/dissociation_histogram_fixed_window.png}",
            r"\caption{Distribution of the participant-level fixed-window HbO dissociation scores.}",
            r"\end{figure}",
            r"\begin{figure}[H]",
            r"\centering",
            r"\includegraphics[width=0.88\linewidth]{../../figures/step5/topography_9s.png}",
            r"\caption{HbO topographic comparison at 9.0~s after question onset for Abstract, Concrete, and Abstract-minus-Concrete.}",
            r"\end{figure}",
            r"\subsection*{Secondary and Sensitivity Results}",
            r"\small",
            r"\input{tables/secondary_tests_table.tex}",
            r"\normalsize",
            r"\paragraph{Primary follow-up result.} "
            + sanitize_for_tex(
                f"Front HbO p={format_p_value(primary_row['p_value'])}; back HbO p={format_p_value(back_hbo_row['p_value'])}; dissociation HbO p={format_p_value(diss_hbo_row['p_value'])}."
            ),
            r"\subsection*{Final Conclusion}",
            r"\input{text/final_conclusion.tex}",
            r"\end{document}",
        ]
    )

    report_path = REPORTS_DIR / "step5_fixed_window_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return conclusion_df


def main() -> None:
    ensure_directories()
    trial_df, trigger_df, status_df, step4_inclusion_df, exclusions = load_inputs()
    session_df, merged_trial_df = build_trial_frame(trial_df, trigger_df, status_df, step4_inclusion_df)

    inclusion_df, epoch_manifest_df, trial_hbo_df, trial_hbr_df, participant_summary_df, sensitivity_trial_df, waveform_rows, diagnostics = build_step5_outputs(
        session_df,
        merged_trial_df,
        exclusions,
    )

    front_df, back_df, diss_df = build_primary_results(participant_summary_df)
    sensitivity_results_df = build_sensitivity_results(sensitivity_trial_df)
    fast_response_df = build_fast_response_sensitivity(trial_hbo_df, participant_summary_df)

    save_dataframe(inclusion_df, CLEAN_DIR / "01_step5_inclusion_table.csv")
    save_dataframe(epoch_manifest_df, CLEAN_DIR / "02_step5_epoch_manifest.csv")
    save_dataframe(trial_hbo_df, CLEAN_DIR / "03_step5_trial_roi_summary_hbo.csv")
    save_dataframe(trial_hbr_df, CLEAN_DIR / "04_step5_trial_roi_summary_hbr.csv")
    save_dataframe(participant_summary_df, CLEAN_DIR / "05_step5_participant_roi_summary.csv")
    save_dataframe(front_df, CLEAN_DIR / "06_step5_primary_front_test.csv")
    save_dataframe(back_df, CLEAN_DIR / "07_step5_secondary_back_test.csv")
    save_dataframe(diss_df, CLEAN_DIR / "08_step5_secondary_dissociation_test.csv")
    save_dataframe(sensitivity_results_df, CLEAN_DIR / "09_step5_sensitivity_windows.csv")
    save_dataframe(fast_response_df, CLEAN_DIR / "10_step5_fast_response_sensitivity.csv")

    plot_roi_pair(participant_summary_df, "Front", "front_abstract_hbo_mean", "front_concrete_hbo_mean", "front_roi_window_plot.png")
    plot_roi_pair(participant_summary_df, "Back", "back_abstract_hbo_mean", "back_concrete_hbo_mean", "back_roi_window_plot.png")
    plot_waveforms(waveform_rows, diagnostics["included_session_keys"], diagnostics["waveform_grid"], "Front", "front_waveform_with_window.png")
    plot_waveforms(waveform_rows, diagnostics["included_session_keys"], diagnostics["waveform_grid"], "Back", "back_waveform_with_window.png")
    plot_dissociation_distribution(participant_summary_df)

    evoked_abstract, evoked_concrete, _ = build_session_evokeds(epoch_manifest_df, session_df)
    plot_topography_9s(evoked_abstract, evoked_concrete)

    conclusion_df = write_report_artifacts(
        inclusion_df,
        epoch_manifest_df,
        trial_hbo_df,
        participant_summary_df,
        front_df,
        back_df,
        diss_df,
        sensitivity_results_df,
        fast_response_df,
        exclusions,
    )
    save_dataframe(conclusion_df, CLEAN_DIR / "11_step5_final_conclusion.csv")

    report_path = REPORTS_DIR / "step5_fixed_window_report.tex"
    compile_report(report_path)

    primary_row = front_df[front_df["chromophore"] == "hbo"].iloc[0]
    print("Step 5 outputs generated successfully.")
    print(f"Included participant-sessions: {int(inclusion_df['included_step5_analysis'].sum())}")
    print(f"Included valid trials: {int(epoch_manifest_df['included_step5_trial'].sum())}")
    print(f"Primary front HbO p-value: {format_p_value(primary_row['p_value'])}")


if __name__ == "__main__":
    main()
