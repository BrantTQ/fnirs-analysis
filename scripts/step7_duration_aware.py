#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats

from step5_fixed_window_roi import BACK_HBO_CHANNELS, FRONT_HBO_CHANNELS, available_channels
from step6_covariate_adjusted import (
    coerce_bool,
    compile_report,
    extract_model_row,
    fit_fixed_effect_fallback,
    fit_mixed_model,
    fixed_effect_covariance,
    format_effect,
    format_p_value,
    make_latex_grid,
    make_latex_table,
    sanitize_for_tex,
    save_dataframe,
    zscore_series,
)


ROOT = Path(__file__).resolve().parents[1]
STEP6_TRIAL_PATH = ROOT / "data_clean" / "step6" / "03_step6_trial_model_table.csv"
STEP6_PRIMARY_MODEL_PATH = ROOT / "data_clean" / "step6" / "04_step6_primary_front_wordcount_model.csv"
STEP3_STATUS_PATH = ROOT / "data_clean" / "step3" / "07_fnirs_session_status.csv"

CLEAN_DIR = ROOT / "data_clean" / "step7"
FIGURES_DIR = ROOT / "figures" / "step7"
REPORTS_DIR = ROOT / "reports" / "step7"
TABLES_DIR = REPORTS_DIR / "tables"
TEXT_DIR = REPORTS_DIR / "text"

ALPHA = 0.05
PRIMARY_LAG_SEC = 4.0
LAG_SENSITIVITY_VALUES = [3.0, 4.0, 5.0]
CAPPED_WINDOW_MAX_SEC = 16.0
BASELINE_DURATION_SEC = 2.0
NEXT_ONSET_BUFFER_SEC = 0.5
MIN_WINDOW_DURATION_SEC = 2.0
CONDITION_ORDER = ["Concrete", "Abstract"]
CONDITION_COLORS = {"Abstract": "#AA3377", "Concrete": "#228833"}


def ensure_directories() -> None:
    for path in [CLEAN_DIR, FIGURES_DIR, REPORTS_DIR, TABLES_DIR, TEXT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step6_trial_df = pd.read_csv(STEP6_TRIAL_PATH)
    step6_primary_df = pd.read_csv(STEP6_PRIMARY_MODEL_PATH)
    status_df = pd.read_csv(STEP3_STATUS_PATH)
    if "eligible_primary_model" in step6_trial_df.columns:
        step6_trial_df["eligible_primary_model"] = step6_trial_df["eligible_primary_model"].map(coerce_bool)
    return step6_trial_df, step6_primary_df, status_df


def extract_variable_window_mean(
    data: np.ndarray,
    times: np.ndarray,
    baseline_start: float,
    baseline_end: float,
    window_start: float,
    window_end: float,
    channel_indices: list[int],
) -> tuple[np.ndarray | None, str | None]:
    if not channel_indices:
        return None, "no_channels"
    if baseline_start < times[0] - 1e-9 or baseline_end > times[-1] + 1e-9:
        return None, "baseline_unavailable"
    if window_start < times[0] - 1e-9 or window_end > times[-1] + 1e-9:
        return None, "window_unavailable"
    baseline_mask = (times >= baseline_start) & (times <= baseline_end)
    window_mask = (times >= window_start) & (times <= window_end)
    if not baseline_mask.any():
        return None, "baseline_unavailable"
    if not window_mask.any():
        return None, "window_unavailable"
    baseline_mean = data[np.ix_(channel_indices, baseline_mask)].mean(axis=1)
    window_mean = data[np.ix_(channel_indices, window_mask)].mean(axis=1)
    return window_mean - baseline_mean, None


def create_window_definition(
    onset_time: float,
    answer_time: float,
    next_onset_time: float,
    recording_end_time: float,
    lag_sec: float,
    cap_end_sec: float | None = None,
) -> dict[str, Any]:
    raw_window_end = answer_time + lag_sec
    candidates = [raw_window_end, next_onset_time - NEXT_ONSET_BUFFER_SEC, recording_end_time]
    truncation_flags = {
        "truncated_by_next_onset": False,
        "truncated_by_recording_end": False,
        "truncated_by_cap": False,
    }
    if cap_end_sec is not None:
        cap_time = onset_time + cap_end_sec
        candidates.append(cap_time)
    final_end = min(candidates)
    if math.isclose(final_end, next_onset_time - NEXT_ONSET_BUFFER_SEC) or final_end < raw_window_end - 1e-9 and final_end <= next_onset_time - NEXT_ONSET_BUFFER_SEC + 1e-9:
        truncation_flags["truncated_by_next_onset"] = final_end < raw_window_end - 1e-9
    if final_end < raw_window_end - 1e-9 and final_end <= recording_end_time + 1e-9 and math.isclose(final_end, recording_end_time, abs_tol=1e-9):
        truncation_flags["truncated_by_recording_end"] = True
    if cap_end_sec is not None:
        cap_time = onset_time + cap_end_sec
        if final_end < raw_window_end - 1e-9 and final_end <= cap_time + 1e-9 and math.isclose(final_end, cap_time, abs_tol=1e-9):
            truncation_flags["truncated_by_cap"] = True

    window_start = onset_time + lag_sec
    duration_sec = final_end - window_start
    truncation_reason = ""
    if truncation_flags["truncated_by_next_onset"]:
        truncation_reason = "next_onset"
    elif truncation_flags["truncated_by_recording_end"]:
        truncation_reason = "recording_end"
    elif truncation_flags["truncated_by_cap"]:
        truncation_reason = "cap"

    return {
        "lag_sec": lag_sec,
        "window_start_time": window_start,
        "window_end_time": final_end,
        "window_duration_sec": duration_sec,
        "window_raw_end_time": raw_window_end,
        "window_truncated": bool(truncation_reason),
        "truncation_reason": truncation_reason,
        **truncation_flags,
    }


def build_trial_tables(step6_trial_df: pd.DataFrame, status_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_trial_df = step6_trial_df[step6_trial_df["eligible_primary_model"]].copy()
    base_trial_df = base_trial_df.sort_values(["participant_id", "session_id", "question_start_time"]).reset_index(drop=True)

    status_lookup = {
        (row["participant_id"], row["session_id"]): row["preprocessed_file"]
        for _, row in status_df.iterrows()
        if isinstance(row.get("preprocessed_file"), str) and row.get("preprocessed_file")
    }

    trial_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = [
        {
            "stage": "cohort",
            "level": "info",
            "reason_code": "STEP6_COHORT_REUSED",
            "n_rows": int(base_trial_df[["participant_id", "session_id"]].drop_duplicates().shape[0]),
            "detail": "Step 7 started from the Step 6 eligible participant-session cohort.",
        },
        {
            "stage": "input_type",
            "level": "info",
            "reason_code": "CONTINUOUS_PREPROCESSED_RAW",
            "n_rows": int(base_trial_df[["participant_id", "session_id"]].drop_duplicates().shape[0]),
            "detail": "Step 7 trial summaries were computed directly from the continuous preprocessed fNIRS files from Step 3.",
        },
    ]

    for (participant_id, session_id), session_df in base_trial_df.groupby(["participant_id", "session_id"], sort=True):
        preprocessed_file = status_lookup.get((participant_id, session_id))
        if not preprocessed_file:
            exclusion_rows.append(
                {
                    "stage": "session_loading",
                    "level": "warning",
                    "reason_code": "MISSING_PREPROCESSED_FILE",
                    "n_rows": 1,
                    "detail": f"{participant_id} / {session_id} had no preprocessed continuous file.",
                }
            )
            continue

        raw = mne.io.read_raw_fif(preprocessed_file, preload=True, verbose="ERROR")
        session_data = raw.get_data()
        session_times = raw.times.copy()
        name_to_index = {name: idx for idx, name in enumerate(raw.ch_names)}
        recording_end_time = float(session_times[-1])

        front_channels = available_channels(raw, FRONT_HBO_CHANNELS)
        back_channels = available_channels(raw, BACK_HBO_CHANNELS)
        session_trials = session_df.sort_values("question_start_time").reset_index(drop=True)
        session_trials["next_question_onset_time"] = session_trials["question_start_time"].shift(-1)
        session_trials["next_question_onset_time"] = session_trials["next_question_onset_time"].fillna(recording_end_time)

        for trial in session_trials.itertuples(index=False):
            onset_time = float(trial.question_start_time)
            answer_time = float(trial.answer_time)
            next_onset_time = float(trial.next_question_onset_time)
            baseline_start_time = onset_time - BASELINE_DURATION_SEC
            baseline_end_time = onset_time
            raw_duration_sec = answer_time - onset_time

            primary_window = create_window_definition(
                onset_time=onset_time,
                answer_time=answer_time,
                next_onset_time=next_onset_time,
                recording_end_time=recording_end_time,
                lag_sec=PRIMARY_LAG_SEC,
            )
            lag_windows = {
                f"lag_{int(lag)}": create_window_definition(
                    onset_time=onset_time,
                    answer_time=answer_time,
                    next_onset_time=next_onset_time,
                    recording_end_time=recording_end_time,
                    lag_sec=lag,
                )
                for lag in LAG_SENSITIVITY_VALUES
            }
            capped_window = create_window_definition(
                onset_time=onset_time,
                answer_time=answer_time,
                next_onset_time=next_onset_time,
                recording_end_time=recording_end_time,
                lag_sec=PRIMARY_LAG_SEC,
                cap_end_sec=CAPPED_WINDOW_MAX_SEC,
            )

            reason_code = ""
            reason_detail = ""
            if pd.isna(trial.question_start_time):
                reason_code = "MISSING_ONSET"
                reason_detail = "No valid harmonized question onset time was available."
            elif pd.isna(trial.answer_time):
                reason_code = "MISSING_ANSWER"
                reason_detail = "No valid harmonized answer time was available."
            elif raw_duration_sec <= 0:
                reason_code = "NONPOSITIVE_DURATION"
                reason_detail = "The raw question duration was not positive."
            elif baseline_start_time < session_times[0] - 1e-9 or baseline_end_time > session_times[-1] + 1e-9:
                reason_code = "BASELINE_UNAVAILABLE"
                reason_detail = "The full two-second baseline interval was unavailable."
            elif primary_window["window_start_time"] < session_times[0] - 1e-9 or primary_window["window_end_time"] > session_times[-1] + 1e-9:
                reason_code = "WINDOW_UNAVAILABLE"
                reason_detail = "The final variable window was not fully available in the continuous recording."
            elif primary_window["window_duration_sec"] < MIN_WINDOW_DURATION_SEC:
                reason_code = "WINDOW_TOO_SHORT"
                reason_detail = "The final lagged variable window was shorter than 2.0 s after truncation."
            elif not front_channels:
                reason_code = "NO_FRONT_CHANNELS"
                reason_detail = "No valid front ROI channels remained after bad-channel exclusion."

            included_primary_pre_session = reason_code == ""
            common_row = {
                "participant_id": participant_id,
                "session_id": session_id,
                "block": trial.block,
                "trial_idx_in_block": int(trial.trial_idx_in_block),
                "trial_idx_global": int(trial.trial_idx_global),
                "question_id": trial.question_id,
                "condition": trial.condition,
                "condition_abstract": float(trial.condition_abstract),
                "question_start_time": onset_time,
                "answer_time": answer_time,
                "next_question_onset_time": next_onset_time,
                "response_time_sec": float(trial.response_time_sec),
                "raw_duration_sec": raw_duration_sec,
                "baseline_start_time": baseline_start_time,
                "baseline_end_time": baseline_end_time,
                "front_hbo_n_channels": len(front_channels),
                "back_hbo_n_channels": len(back_channels),
                "front_hbo_channels": ";".join(front_channels),
                "back_hbo_channels": ";".join(back_channels),
                "total_word_count": float(trial.total_word_count),
                "sentence_count": float(trial.sentence_count),
                "sentence_length": float(trial.sentence_length),
                "enem_correctness": float(trial.enem_correctness),
                "participant_correct": float(trial.participant_correct),
                "z_log_total_word_count": float(trial.z_log_total_word_count),
                "z_sentence_count": float(trial.z_sentence_count),
                "z_sentence_length": float(trial.z_sentence_length),
                "z_enem_correctness": float(trial.z_enem_correctness),
                "z_log_response_time_sec": float(trial.z_log_response_time_sec),
                "participant_code": int(trial.participant_code),
                "question_code": int(trial.question_code),
                "primary_lag_sec": PRIMARY_LAG_SEC,
                "primary_window_start_time": primary_window["window_start_time"],
                "primary_window_end_time": primary_window["window_end_time"],
                "primary_window_duration_sec": primary_window["window_duration_sec"],
                "primary_window_raw_end_time": primary_window["window_raw_end_time"],
                "primary_window_truncated": primary_window["window_truncated"],
                "primary_truncation_reason": primary_window["truncation_reason"],
                "truncated_by_next_onset": primary_window["truncated_by_next_onset"],
                "truncated_by_recording_end": primary_window["truncated_by_recording_end"],
                "truncated_by_cap": primary_window["truncated_by_cap"],
                "included_primary_pre_session_filter": included_primary_pre_session,
                "primary_exclusion_reason_code": reason_code,
                "primary_exclusion_reason_detail": reason_detail,
            }

            for lag_label, lag_window in lag_windows.items():
                common_row[f"{lag_label}_window_start_time"] = lag_window["window_start_time"]
                common_row[f"{lag_label}_window_end_time"] = lag_window["window_end_time"]
                common_row[f"{lag_label}_window_duration_sec"] = lag_window["window_duration_sec"]
                common_row[f"{lag_label}_window_truncated"] = lag_window["window_truncated"]
                common_row[f"{lag_label}_truncation_reason"] = lag_window["truncation_reason"]

            common_row["cap16_window_start_time"] = capped_window["window_start_time"]
            common_row["cap16_window_end_time"] = capped_window["window_end_time"]
            common_row["cap16_window_duration_sec"] = capped_window["window_duration_sec"]
            common_row["cap16_window_truncated"] = capped_window["window_truncated"]
            common_row["cap16_truncation_reason"] = capped_window["truncation_reason"]

            if included_primary_pre_session:
                front_primary, front_issue = extract_variable_window_mean(
                    session_data,
                    session_times,
                    baseline_start_time,
                    baseline_end_time,
                    primary_window["window_start_time"],
                    primary_window["window_end_time"],
                    [name_to_index[name] for name in front_channels],
                )
                if front_issue or front_primary is None:
                    common_row["included_primary_pre_session_filter"] = False
                    common_row["primary_exclusion_reason_code"] = front_issue or "PRIMARY_WINDOW_EXTRACTION_FAILED"
                    common_row["primary_exclusion_reason_detail"] = "The front ROI variable-window summary could not be extracted."
                    front_primary = None

                back_primary = None
                if back_channels:
                    back_primary, _ = extract_variable_window_mean(
                        session_data,
                        session_times,
                        baseline_start_time,
                        baseline_end_time,
                        primary_window["window_start_time"],
                        primary_window["window_end_time"],
                        [name_to_index[name] for name in back_channels],
                    )

                common_row["front_roi_mean_variable_lag4"] = float(np.mean(front_primary)) if front_primary is not None else math.nan
                common_row["back_roi_mean_variable_lag4"] = float(np.mean(back_primary)) if back_primary is not None else math.nan
                common_row["dissociation_roi_mean_variable_lag4"] = (
                    float(np.mean(front_primary) - np.mean(back_primary))
                    if front_primary is not None and back_primary is not None
                    else math.nan
                )

                for lag_label, lag_window in lag_windows.items():
                    front_value, front_issue = extract_variable_window_mean(
                        session_data,
                        session_times,
                        baseline_start_time,
                        baseline_end_time,
                        lag_window["window_start_time"],
                        lag_window["window_end_time"],
                        [name_to_index[name] for name in front_channels],
                    )
                    common_row[f"front_roi_mean_variable_{lag_label}"] = (
                        float(np.mean(front_value)) if front_value is not None and front_issue is None else math.nan
                    )
                    common_row[f"included_{lag_label}_sensitivity"] = bool(
                        front_issue is None and lag_window["window_duration_sec"] >= MIN_WINDOW_DURATION_SEC
                    )

                front_cap, front_cap_issue = extract_variable_window_mean(
                    session_data,
                    session_times,
                    baseline_start_time,
                    baseline_end_time,
                    capped_window["window_start_time"],
                    capped_window["window_end_time"],
                    [name_to_index[name] for name in front_channels],
                )
                common_row["front_roi_mean_variable_cap16"] = (
                    float(np.mean(front_cap)) if front_cap is not None and front_cap_issue is None else math.nan
                )
                common_row["included_cap16_sensitivity"] = bool(
                    front_cap_issue is None and capped_window["window_duration_sec"] >= MIN_WINDOW_DURATION_SEC
                )
            else:
                common_row["front_roi_mean_variable_lag4"] = math.nan
                common_row["back_roi_mean_variable_lag4"] = math.nan
                common_row["dissociation_roi_mean_variable_lag4"] = math.nan
                for lag_value in LAG_SENSITIVITY_VALUES:
                    lag_label = f"lag_{int(lag_value)}"
                    common_row[f"front_roi_mean_variable_{lag_label}"] = math.nan
                    common_row[f"included_{lag_label}_sensitivity"] = False
                common_row["front_roi_mean_variable_cap16"] = math.nan
                common_row["included_cap16_sensitivity"] = False

            trial_rows.append(common_row)

    trial_timing_df = pd.DataFrame(trial_rows)

    included_pre_df = trial_timing_df[trial_timing_df["included_primary_pre_session_filter"]].copy()
    session_condition_counts = (
        included_pre_df.groupby(["participant_id", "session_id", "condition"]).size().unstack(fill_value=0).reset_index()
    )
    session_condition_counts["included_primary_session"] = (
        (session_condition_counts.get("Abstract", 0) >= 2) & (session_condition_counts.get("Concrete", 0) >= 2)
    )
    session_include_lookup = {
        (row["participant_id"], row["session_id"]): bool(row["included_primary_session"])
        for _, row in session_condition_counts.iterrows()
    }
    trial_timing_df["included_primary_session"] = trial_timing_df.apply(
        lambda row: session_include_lookup.get((row["participant_id"], row["session_id"]), False),
        axis=1,
    )
    trial_timing_df["included_primary_model"] = (
        trial_timing_df["included_primary_pre_session_filter"] & trial_timing_df["included_primary_session"]
    )

    for lag_value in LAG_SENSITIVITY_VALUES:
        lag_label = f"lag_{int(lag_value)}"
        trial_timing_df[f"included_{lag_label}_model"] = (
            trial_timing_df[f"included_{lag_label}_sensitivity"] & trial_timing_df["included_primary_session"]
        )
    trial_timing_df["included_cap16_model"] = (
        trial_timing_df["included_cap16_sensitivity"] & trial_timing_df["included_primary_session"]
    )

    excluded_sessions = int((~session_condition_counts["included_primary_session"]).sum()) if not session_condition_counts.empty else 0
    exclusion_rows.append(
        {
            "stage": "participant_session_inclusion",
            "level": "warning" if excluded_sessions else "info",
            "reason_code": "SESSION_MIN_TRIAL_COUNT",
            "n_rows": excluded_sessions,
            "detail": "Excluded participant-sessions that did not contribute at least two valid abstract and two valid concrete trials to the Step 7 primary analysis.",
        }
    )

    trial_summary_df = trial_timing_df.copy()
    trial_summary_df["benchmark_fixed_front_roi_mean_step6"] = pd.to_numeric(base_trial_df["front_roi_mean"], errors="coerce")

    exclusion_counts = (
        trial_timing_df.loc[~trial_timing_df["included_primary_pre_session_filter"], "primary_exclusion_reason_code"]
        .fillna("UNKNOWN")
        .value_counts()
    )
    for reason_code, count in exclusion_counts.items():
        exclusion_rows.append(
            {
                "stage": "primary_trial_exclusions",
                "level": "warning",
                "reason_code": reason_code,
                "n_rows": int(count),
                "detail": f"Primary duration-aware trial exclusion count for {reason_code}.",
            }
        )

    return trial_timing_df, trial_summary_df, pd.DataFrame(exclusion_rows)


def build_duration_diagnostics(trial_summary_df: pd.DataFrame) -> pd.DataFrame:
    included_df = trial_summary_df[trial_summary_df["included_primary_model"]].copy()
    rows: list[dict[str, Any]] = []

    for condition in CONDITION_ORDER:
        values = pd.to_numeric(
            included_df.loc[included_df["condition"] == condition, "response_time_sec"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "diagnostic_type": "response_time_by_condition",
                "label": condition,
                "n_trials": int(len(values)),
                "mean_value": float(values.mean()),
                "sd_value": float(values.std(ddof=1)),
                "median_value": float(values.median()),
                "min_value": float(values.min()),
                "max_value": float(values.max()),
                "correlation_r": math.nan,
                "correlation_p_value": math.nan,
            }
        )

    for variable, label in [
        ("total_word_count", "Response time vs total word count"),
        ("sentence_count", "Response time vs sentence count"),
        ("sentence_length", "Response time vs sentence length"),
        ("enem_correctness", "Response time vs ENEM item correctness"),
    ]:
        corr_df = included_df[["response_time_sec", variable]].copy()
        corr_df = corr_df.apply(pd.to_numeric, errors="coerce").dropna()
        corr = stats.pearsonr(corr_df["response_time_sec"], corr_df[variable])
        rows.append(
            {
                "diagnostic_type": "response_time_correlation",
                "label": label,
                "n_trials": int(len(corr_df)),
                "mean_value": math.nan,
                "sd_value": math.nan,
                "median_value": math.nan,
                "min_value": math.nan,
                "max_value": math.nan,
                "correlation_r": float(corr.statistic),
                "correlation_p_value": float(corr.pvalue),
            }
        )

    return pd.DataFrame(rows)


def prepare_model_data(
    trial_summary_df: pd.DataFrame,
    outcome: str,
    covariates: list[str],
    include_flag: str,
    duration_col: str,
) -> pd.DataFrame:
    needed = [
        "participant_id",
        "question_id",
        "participant_code",
        "question_code",
        "condition_abstract",
        duration_col,
        outcome,
        *covariates,
    ]
    model_df = trial_summary_df[trial_summary_df[include_flag]].copy()
    model_df = model_df[needed].copy()
    numeric_cols = ["condition_abstract", duration_col, outcome, *covariates, "participant_code", "question_code"]
    for col in numeric_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna(subset=["condition_abstract", outcome, *covariates]).reset_index(drop=True)
    model_df = model_df.rename(columns={duration_col: "analysis_window_duration_sec"})
    model_df["participant_code"] = model_df["participant_code"].astype(int)
    model_df["question_code"] = model_df["question_code"].astype(int)
    model_df["participant_id"] = model_df["participant_id"].astype(str)
    model_df["question_id"] = model_df["question_id"].astype(str)
    return model_df


def run_model_suite(
    trial_summary_df: pd.DataFrame,
    step6_primary_df: pd.DataFrame,
    exclusion_log_df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame]:
    model_specs = [
        {
            "model_id": "primary_front_variable_window",
            "analysis_label": "Primary front ROI duration-aware model adjusted for log word count and ENEM correctness",
            "analysis_scope": "primary",
            "outcome": "front_roi_mean_variable_lag4",
            "outcome_label": "Front ROI HbO variable window",
            "covariates": ["z_log_total_word_count", "z_enem_correctness"],
            "output_name": "04_step7_primary_front_variable_window_model.csv",
            "include_flag": "included_primary_model",
            "duration_col": "primary_window_duration_sec",
        },
        {
            "model_id": "front_sentencecount",
            "analysis_label": "Alternative front ROI duration-aware model adjusted for sentence count and ENEM correctness",
            "analysis_scope": "alternative",
            "outcome": "front_roi_mean_variable_lag4",
            "outcome_label": "Front ROI HbO variable window",
            "covariates": ["z_sentence_count", "z_enem_correctness"],
            "output_name": "05_step7_front_sentencecount_model.csv",
            "include_flag": "included_primary_model",
            "duration_col": "primary_window_duration_sec",
        },
        {
            "model_id": "front_sentencelength",
            "analysis_label": "Alternative front ROI duration-aware model adjusted for sentence length and ENEM correctness",
            "analysis_scope": "alternative",
            "outcome": "front_roi_mean_variable_lag4",
            "outcome_label": "Front ROI HbO variable window",
            "covariates": ["z_sentence_length", "z_enem_correctness"],
            "output_name": "06_step7_front_sentencelength_model.csv",
            "include_flag": "included_primary_model",
            "duration_col": "primary_window_duration_sec",
        },
        {
            "model_id": "back_variable_window",
            "analysis_label": "Secondary back ROI duration-aware model adjusted for log word count and ENEM correctness",
            "analysis_scope": "secondary",
            "outcome": "back_roi_mean_variable_lag4",
            "outcome_label": "Back ROI HbO variable window",
            "covariates": ["z_log_total_word_count", "z_enem_correctness"],
            "output_name": "07_step7_back_variable_window_model.csv",
            "include_flag": "included_primary_model",
            "duration_col": "primary_window_duration_sec",
        },
        {
            "model_id": "dissociation_variable_window",
            "analysis_label": "Secondary dissociation duration-aware model adjusted for log word count and ENEM correctness",
            "analysis_scope": "secondary",
            "outcome": "dissociation_roi_mean_variable_lag4",
            "outcome_label": "Front-minus-back HbO variable window",
            "covariates": ["z_log_total_word_count", "z_enem_correctness"],
            "output_name": "08_step7_dissociation_variable_window_model.csv",
            "include_flag": "included_primary_model",
            "duration_col": "primary_window_duration_sec",
        },
        {
            "model_id": "front_lag3",
            "analysis_label": "Lag sensitivity front ROI duration-aware model with 3 s lag",
            "analysis_scope": "sensitivity_lag",
            "outcome": "front_roi_mean_variable_lag_3",
            "outcome_label": "Front ROI HbO variable window",
            "covariates": ["z_log_total_word_count", "z_enem_correctness"],
            "output_name": "09_step7_lag_sensitivity.csv",
            "include_flag": "included_lag_3_model",
            "lag_sec": 3.0,
            "duration_col": "lag_3_window_duration_sec",
        },
        {
            "model_id": "front_lag4",
            "analysis_label": "Lag sensitivity front ROI duration-aware model with 4 s lag",
            "analysis_scope": "sensitivity_lag",
            "outcome": "front_roi_mean_variable_lag_4",
            "outcome_label": "Front ROI HbO variable window",
            "covariates": ["z_log_total_word_count", "z_enem_correctness"],
            "output_name": "09_step7_lag_sensitivity.csv",
            "include_flag": "included_lag_4_model",
            "lag_sec": 4.0,
            "duration_col": "lag_4_window_duration_sec",
        },
        {
            "model_id": "front_lag5",
            "analysis_label": "Lag sensitivity front ROI duration-aware model with 5 s lag",
            "analysis_scope": "sensitivity_lag",
            "outcome": "front_roi_mean_variable_lag_5",
            "outcome_label": "Front ROI HbO variable window",
            "covariates": ["z_log_total_word_count", "z_enem_correctness"],
            "output_name": "09_step7_lag_sensitivity.csv",
            "include_flag": "included_lag_5_model",
            "lag_sec": 5.0,
            "duration_col": "lag_5_window_duration_sec",
        },
        {
            "model_id": "front_cap16",
            "analysis_label": "Capped-window sensitivity front ROI duration-aware model",
            "analysis_scope": "sensitivity_cap",
            "outcome": "front_roi_mean_variable_cap16",
            "outcome_label": "Front ROI HbO variable window",
            "covariates": ["z_log_total_word_count", "z_enem_correctness"],
            "output_name": "10_step7_capped_window_sensitivity.csv",
            "include_flag": "included_cap16_model",
            "duration_col": "cap16_window_duration_sec",
        },
        {
            "model_id": "front_rt_sensitivity",
            "analysis_label": "Response-time sensitivity front ROI duration-aware model",
            "analysis_scope": "sensitivity_rt",
            "outcome": "front_roi_mean_variable_lag4",
            "outcome_label": "Front ROI HbO variable window",
            "covariates": ["z_log_total_word_count", "z_enem_correctness", "z_log_response_time_sec"],
            "output_name": "11_step7_rt_sensitivity.csv",
            "include_flag": "included_primary_model",
            "duration_col": "primary_window_duration_sec",
        },
    ]

    model_outputs: dict[str, pd.DataFrame] = {}
    fit_artifacts: dict[str, Any] = {}
    extra_log_rows: list[dict[str, Any]] = []

    for spec in model_specs:
        model_df = prepare_model_data(
            trial_summary_df,
            spec["outcome"],
            spec["covariates"],
            spec["include_flag"],
            spec["duration_col"],
        )
        mixed_result, optimizer, warnings_list, fit_error = fit_mixed_model(model_df, spec["outcome"], spec["covariates"])
        if mixed_result is not None and getattr(mixed_result, "converged", False):
            result = mixed_result
            model_type = "MixedLM"
            fallback_used = False
        else:
            result = fit_fixed_effect_fallback(model_df, spec["outcome"], spec["covariates"])
            model_type = "FixedEffectsOLS"
            fallback_used = True
            if fit_error:
                warnings_list = warnings_list + [fit_error]

        row = extract_model_row(
            result=result,
            model_df=model_df,
            model_id=spec["model_id"],
            analysis_label=spec["analysis_label"],
            analysis_scope=spec["analysis_scope"],
            outcome=spec["outcome"],
            outcome_label=spec["outcome_label"],
            covariates=spec["covariates"],
            model_type=model_type,
            optimizer=optimizer,
            warnings_list=warnings_list,
            fallback_used=fallback_used,
            converged=True,
        )
        row["mean_window_duration_sec"] = float(model_df["analysis_window_duration_sec"].mean())
        row["median_window_duration_sec"] = float(model_df["analysis_window_duration_sec"].median())
        row["min_window_duration_sec"] = float(model_df["analysis_window_duration_sec"].min())
        row["max_window_duration_sec"] = float(model_df["analysis_window_duration_sec"].max())
        if "lag_sec" in spec:
            row["lag_sec"] = spec["lag_sec"]

        output_name = spec["output_name"]
        if output_name not in model_outputs:
            model_outputs[output_name] = pd.DataFrame([row])
        else:
            model_outputs[output_name] = pd.concat([model_outputs[output_name], pd.DataFrame([row])], ignore_index=True)

        fit_artifacts[spec["model_id"]] = {"result": result, "data": model_df, "spec": spec, "row": row}
        extra_log_rows.append(
            {
                "stage": "model_fit",
                "level": "warning" if warnings_list or fallback_used else "info",
                "reason_code": f"MODEL_{spec['model_id'].upper()}",
                "n_rows": 1,
                "detail": f"Model type={model_type}; optimizer={optimizer or 'n/a'}; fallback_used={fallback_used}; warnings={' | '.join(warnings_list) if warnings_list else 'none'}",
            }
        )

    benchmark_rows = [
        {
            "model_label": "Step 6 fixed-window benchmark",
            "window_definition": "Fixed 7-11 s onset-locked window",
            "n_trials": int(step6_primary_df.iloc[0]["n_trials"]) if "n_trials" in step6_primary_df.columns else int(step6_primary_df.iloc[0]["n_valid_abstract_trials"] + step6_primary_df.iloc[0]["n_valid_concrete_trials"]),
            "condition_beta": float(step6_primary_df.iloc[0]["condition_beta"]) if "condition_beta" in step6_primary_df.columns else float(step6_primary_df.iloc[0]["mean_difference_abstract_minus_concrete"]),
            "condition_se": float(step6_primary_df.iloc[0]["condition_se"]) if "condition_se" in step6_primary_df.columns else math.nan,
            "condition_p_value": float(step6_primary_df.iloc[0]["condition_p_value"]) if "condition_p_value" in step6_primary_df.columns else float(step6_primary_df.iloc[0]["p_value"]),
            "condition_ci_low": float(step6_primary_df.iloc[0]["condition_ci_low"]) if "condition_ci_low" in step6_primary_df.columns else float(step6_primary_df.iloc[0]["ci_low"]),
            "condition_ci_high": float(step6_primary_df.iloc[0]["condition_ci_high"]) if "condition_ci_high" in step6_primary_df.columns else float(step6_primary_df.iloc[0]["ci_high"]),
        },
        {
            "model_label": "Step 7 duration-aware variable window",
            "window_definition": "Lagged onset-to-answer window with 4 s lag and next-onset truncation",
            "n_trials": int(model_outputs["04_step7_primary_front_variable_window_model.csv"].iloc[0]["n_trials"]),
            "condition_beta": float(model_outputs["04_step7_primary_front_variable_window_model.csv"].iloc[0]["condition_beta"]),
            "condition_se": float(model_outputs["04_step7_primary_front_variable_window_model.csv"].iloc[0]["condition_se"]),
            "condition_p_value": float(model_outputs["04_step7_primary_front_variable_window_model.csv"].iloc[0]["condition_p_value"]),
            "condition_ci_low": float(model_outputs["04_step7_primary_front_variable_window_model.csv"].iloc[0]["condition_ci_low"]),
            "condition_ci_high": float(model_outputs["04_step7_primary_front_variable_window_model.csv"].iloc[0]["condition_ci_high"]),
        },
    ]
    model_outputs["12_step7_fixed_vs_variable_benchmark.csv"] = pd.DataFrame(benchmark_rows)

    for output_name, df in model_outputs.items():
        model_outputs[output_name] = df.reset_index(drop=True)

    exclusion_log_df = pd.concat([exclusion_log_df, pd.DataFrame(extra_log_rows)], ignore_index=True)
    return model_outputs, fit_artifacts, exclusion_log_df


def plot_response_time_by_condition(trial_summary_df: pd.DataFrame, path: Path) -> None:
    included_df = trial_summary_df[trial_summary_df["included_primary_model"]].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
    rng = np.random.default_rng(20260321)

    for idx, condition in enumerate(CONDITION_ORDER, start=1):
        values = pd.to_numeric(included_df.loc[included_df["condition"] == condition, "response_time_sec"], errors="coerce").dropna()
        box = ax.boxplot(
            values,
            positions=[idx],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#222222", linewidth=1.5),
        )
        for patch in box["boxes"]:
            patch.set_facecolor(CONDITION_COLORS[condition])
            patch.set_alpha(0.35)
            patch.set_edgecolor(CONDITION_COLORS[condition])
        jitter = rng.normal(loc=idx, scale=0.05, size=len(values))
        ax.scatter(jitter, values, s=30, color=CONDITION_COLORS[condition], edgecolor="white", linewidth=0.5, alpha=0.6)

    ax.set_xticks([1, 2], CONDITION_ORDER)
    ax.set_xlabel("Question condition")
    ax.set_ylabel("Response time (s)")
    ax.set_title("Response time by condition")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_response_time_scatter(trial_summary_df: pd.DataFrame, x_col: str, x_label: str, path: Path) -> None:
    included_df = trial_summary_df[trial_summary_df["included_primary_model"]].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.8), layout="constrained")
    for condition in CONDITION_ORDER:
        subset = included_df[included_df["condition"] == condition]
        ax.scatter(
            subset[x_col],
            subset["response_time_sec"],
            color=CONDITION_COLORS[condition],
            alpha=0.35,
            s=28,
            label=condition,
        )
        if len(subset) >= 2:
            coeffs = np.polyfit(subset[x_col], subset["response_time_sec"], deg=1)
            x_grid = np.linspace(float(subset[x_col].min()), float(subset[x_col].max()), 100)
            ax.plot(x_grid, coeffs[0] * x_grid + coeffs[1], color=CONDITION_COLORS[condition], linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Response time (s)")
    ax.set_title(f"Response time versus {x_label.lower()}")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_variable_window_duration_histogram(trial_summary_df: pd.DataFrame, path: Path) -> None:
    included_df = trial_summary_df[trial_summary_df["included_primary_model"]].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.8), layout="constrained")
    ax.hist(included_df["primary_window_duration_sec"], bins=25, color="#4477AA", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Final variable-window duration (s)")
    ax.set_ylabel("Trial count")
    ax.set_title("Distribution of duration-aware window lengths")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_condition_effect(row: pd.Series, title: str, path: Path) -> None:
    effect = float(row["condition_beta"])
    ci_low = float(row["condition_ci_low"])
    ci_high = float(row["condition_ci_high"])
    fig, ax = plt.subplots(figsize=(6.2, 2.8), layout="constrained")
    ax.errorbar(
        x=effect * 1e6,
        y=0,
        xerr=np.array([[effect - ci_low], [ci_high - effect]]) * 1e6,
        fmt="o",
        color="#AA3377",
        ecolor="#AA3377",
        elinewidth=2,
        capsize=4,
        markersize=7,
    )
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks([0], ["Abstract - Concrete"])
    ax.set_xlabel(r"Adjusted condition effect ($\mu$M)")
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_fixed_vs_variable_benchmark(benchmark_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.5), layout="constrained")
    y_positions = np.arange(len(benchmark_df))
    effects = benchmark_df["condition_beta"].to_numpy(dtype=float) * 1e6
    low_errors = (benchmark_df["condition_beta"].to_numpy(dtype=float) - benchmark_df["condition_ci_low"].to_numpy(dtype=float)) * 1e6
    high_errors = (benchmark_df["condition_ci_high"].to_numpy(dtype=float) - benchmark_df["condition_beta"].to_numpy(dtype=float)) * 1e6
    ax.errorbar(
        effects,
        y_positions,
        xerr=np.vstack([low_errors, high_errors]),
        fmt="o",
        color="#336699",
        ecolor="#336699",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions, benchmark_df["model_label"].tolist())
    ax.set_xlabel(r"Condition coefficient ($\mu$M)")
    ax.set_title("Fixed-window versus variable-window benchmark")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_primary_residuals(primary_fit: dict[str, Any], path: Path) -> None:
    result = primary_fit["result"]
    fitted = np.asarray(result.fittedvalues, dtype=float) * 1e6
    residuals = np.asarray(result.resid, dtype=float) * 1e6
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.scatter(fitted, residuals, s=24, alpha=0.35, color="#336699")
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Fitted front ROI HbO ($\mu$M)")
    ax.set_ylabel(r"Residual ($\mu$M)")
    ax.set_title("Primary duration-aware model residual diagnostic")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_figures(trial_summary_df: pd.DataFrame, fit_artifacts: dict[str, Any], benchmark_df: pd.DataFrame) -> None:
    plot_response_time_by_condition(trial_summary_df, FIGURES_DIR / "response_time_by_condition.png")
    plot_response_time_scatter(trial_summary_df, "total_word_count", "Total word count", FIGURES_DIR / "response_time_vs_wordcount.png")
    plot_response_time_scatter(trial_summary_df, "enem_correctness", "ENEM item correctness", FIGURES_DIR / "response_time_vs_correctness.png")
    plot_variable_window_duration_histogram(trial_summary_df, FIGURES_DIR / "variable_window_duration_histogram.png")
    plot_condition_effect(
        pd.Series(fit_artifacts["primary_front_variable_window"]["row"]),
        "Primary front ROI duration-aware condition effect",
        FIGURES_DIR / "front_roi_partial_effect_variable_window.png",
    )
    plot_condition_effect(
        pd.Series(fit_artifacts["back_variable_window"]["row"]),
        "Secondary back ROI duration-aware condition effect",
        FIGURES_DIR / "back_roi_partial_effect_variable_window.png",
    )
    plot_condition_effect(
        pd.Series(fit_artifacts["dissociation_variable_window"]["row"]),
        "Secondary dissociation duration-aware condition effect",
        FIGURES_DIR / "dissociation_partial_effect_variable_window.png",
    )
    plot_fixed_vs_variable_benchmark(benchmark_df, FIGURES_DIR / "fixed_vs_variable_comparison.png")
    plot_primary_residuals(fit_artifacts["primary_front_variable_window"], FIGURES_DIR / "model_residuals_primary.png")


def build_final_conclusion(primary_model_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    row = primary_model_df.iloc[0]
    beta = float(row["condition_beta"])
    se = float(row["condition_se"])
    p_value = float(row["condition_p_value"])
    ci_low = float(row["condition_ci_low"])
    ci_high = float(row["condition_ci_high"])

    if p_value < ALPHA and beta > 0:
        conclusion_text = (
            "In the duration-aware mixed-effects analysis, abstract questions were associated with a significantly larger left frontal ROI HbO response than concrete questions after adjusting for log word count and ENEM item correctness "
            f"(beta={beta:.3e}, SE={se:.3e}, p={p_value:.6f}, 95% CI [{ci_low:.3e}, {ci_high:.3e}]). "
            "This indicates that the frontal condition effect remains when the analysis window is allowed to vary with the actual duration of each question."
        )
    elif p_value < ALPHA and beta < 0:
        conclusion_text = (
            "In the duration-aware mixed-effects analysis, concrete questions were associated with a significantly larger left frontal ROI HbO response than abstract questions after adjusting for log word count and ENEM item correctness "
            f"(beta={beta:.3e}, SE={se:.3e}, p={p_value:.6f}, 95% CI [{ci_low:.3e}, {ci_high:.3e}])."
        )
    else:
        conclusion_text = (
            "In the duration-aware mixed-effects analysis, the abstract-versus-concrete effect in the left frontal ROI was not statistically significant after adjusting for log word count and ENEM item correctness "
            f"(beta={beta:.3e}, SE={se:.3e}, p={p_value:.6f}, 95% CI [{ci_low:.3e}, {ci_high:.3e}]). "
            "This suggests that allowing the trial window to vary with question duration does not by itself yield statistically significant evidence for a frontal condition effect in the current dataset."
        )

    final_df = pd.DataFrame(
        [
            {
                "n_included_participants": int(row["n_participants"]),
                "n_included_questions": int(row["n_questions"]),
                "n_included_trials": int(row["n_trials"]),
                "mean_window_duration_sec": float(row["mean_window_duration_sec"]),
                "median_window_duration_sec": float(row["median_window_duration_sec"]),
                "condition_beta": beta,
                "condition_se": se,
                "condition_p_value": p_value,
                "condition_ci_low": ci_low,
                "condition_ci_high": ci_high,
                "conclusion_text": conclusion_text,
            }
        ]
    )
    return final_df, conclusion_text


def write_supporting_tex(
    primary_model_df: pd.DataFrame,
    secondary_models_df: pd.DataFrame,
    conclusion_text: str,
) -> None:
    primary = primary_model_df.iloc[0]
    primary_rows = [
        ("Model type", primary["model_type"]),
        ("Included participants", int(primary["n_participants"])),
        ("Included questions", int(primary["n_questions"])),
        ("Included trials", int(primary["n_trials"])),
        ("Mean variable-window duration (s)", f"{float(primary['mean_window_duration_sec']):.2f}"),
        ("Median variable-window duration (s)", f"{float(primary['median_window_duration_sec']):.2f}"),
        ("Condition beta", format_effect(primary["condition_beta"])),
        ("Condition SE", format_effect(primary["condition_se"])),
        ("Condition statistic", f"{float(primary['condition_stat']):.3f}"),
        ("Condition p-value", format_p_value(primary["condition_p_value"])),
        ("Condition 95% CI", f"[{format_effect(primary['condition_ci_low'])}, {format_effect(primary['condition_ci_high'])}]"),
        ("Word-count beta", format_effect(primary["covariate_1_beta"])),
        ("ENEM-correctness beta", format_effect(primary["covariate_2_beta"])),
        ("Participant random-intercept variance", format_effect(primary["participant_random_intercept_variance"])),
        ("Question random-intercept variance", format_effect(primary["question_random_intercept_variance"])),
    ]
    (TABLES_DIR / "primary_model_table.tex").write_text(make_latex_table(primary_rows) + "\n", encoding="utf-8")

    secondary_rows = []
    for row in secondary_models_df.itertuples(index=False):
        secondary_rows.append(
            [
                row.analysis_label.replace(" duration-aware model", ""),
                format_effect(row.condition_beta),
                format_p_value(row.condition_p_value),
                f"{float(row.mean_window_duration_sec):.2f}",
                "yes" if row.fallback_used else "no",
            ]
        )
    (TABLES_DIR / "secondary_model_table.tex").write_text(
        make_latex_grid(
            ["Model", "Cond. beta", "Cond. p", "Mean dur. (s)", "Fallback"],
            secondary_rows,
            r"p{0.52\linewidth}p{0.12\linewidth}p{0.10\linewidth}p{0.12\linewidth}p{0.08\linewidth}",
        )
        + "\n",
        encoding="utf-8",
    )
    (TEXT_DIR / "final_conclusion.tex").write_text(conclusion_text + "\n", encoding="utf-8")


def write_report(
    trial_summary_df: pd.DataFrame,
    primary_model_df: pd.DataFrame,
    secondary_models_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    exclusion_log_df: pd.DataFrame,
) -> Path:
    primary = primary_model_df.iloc[0]
    response_time_diag = diagnostics_df[diagnostics_df["diagnostic_type"] == "response_time_by_condition"].copy()
    corr_diag = diagnostics_df[diagnostics_df["diagnostic_type"] == "response_time_correlation"].copy()
    warning_rows = exclusion_log_df[exclusion_log_df["level"] == "warning"].copy()
    warning_text = (
        "; ".join(f"{row.reason_code}: {row.detail}" for row in warning_rows.itertuples(index=False))
        if not warning_rows.empty
        else "No Step 7 exclusions or warnings were triggered."
    )
    response_time_text = "; ".join(
        f"{row.label} mean={row.mean_value:.2f}s (median={row.median_value:.2f}s)" for row in response_time_diag.itertuples(index=False)
    )
    correlation_text = "; ".join(
        f"{row.label}: r={row.correlation_r:.3f}, p={row.correlation_p_value:.6f}" for row in corr_diag.itertuples(index=False)
    )

    overview_rows = [
        ("Included participants", int(primary["n_participants"])),
        ("Included questions", int(primary["n_questions"])),
        ("Included trials", int(primary["n_trials"])),
        ("Mean variable-window duration (s)", f"{float(primary['mean_window_duration_sec']):.2f}"),
        ("Primary condition p-value", format_p_value(primary["condition_p_value"])),
        ("Step 6 benchmark p-value", format_p_value(benchmark_df.iloc[0]["condition_p_value"])),
    ]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Step 7 Duration-Aware Variable-Window Report}",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
        r"\paragraph{Duration diagnostics.} " + sanitize_for_tex(response_time_text + ". " + correlation_text + "."),
        r"\paragraph{Warnings and QC.} " + sanitize_for_tex(warning_text),
        r"\subsection*{Primary Front ROI Model}",
        r"\input{tables/primary_model_table.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/response_time_by_condition.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/variable_window_duration_histogram.png}",
        r"\caption{Response-time distribution by condition and the distribution of final duration-aware window lengths.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/response_time_vs_wordcount.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/response_time_vs_correctness.png}",
        r"\caption{Response time plotted against total word count and ENEM item correctness.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/front_roi_partial_effect_variable_window.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/model_residuals_primary.png}",
        r"\caption{Primary duration-aware front ROI condition effect and residual diagnostic.}",
        r"\end{figure}",
        r"\subsection*{Secondary and Sensitivity Models}",
        r"\input{tables/secondary_model_table.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/back_roi_partial_effect_variable_window.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/dissociation_partial_effect_variable_window.png}",
        r"\caption{Secondary back ROI and dissociation condition-effect plots.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.75\linewidth]{../../figures/step7/fixed_vs_variable_comparison.png}",
        r"\caption{Direct benchmark comparison of the Step~6 fixed-window coefficient and the Step~7 duration-aware coefficient.}",
        r"\end{figure}",
        r"\subsection*{Final Conclusion}",
        r"\input{text/final_conclusion.tex}",
        r"\end{document}",
    ]
    report_path = REPORTS_DIR / "step7_variable_window_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()
    step6_trial_df, step6_primary_df, status_df = load_inputs()

    trial_timing_df, trial_summary_df, exclusion_log_df = build_trial_tables(step6_trial_df, status_df)
    diagnostics_df = build_duration_diagnostics(trial_summary_df)
    model_outputs, fit_artifacts, exclusion_log_df = run_model_suite(trial_summary_df, step6_primary_df, exclusion_log_df)

    primary_model_df = model_outputs["04_step7_primary_front_variable_window_model.csv"].copy()
    secondary_models_df = pd.concat(
        [
            model_outputs["05_step7_front_sentencecount_model.csv"],
            model_outputs["06_step7_front_sentencelength_model.csv"],
            model_outputs["07_step7_back_variable_window_model.csv"],
            model_outputs["08_step7_dissociation_variable_window_model.csv"],
            model_outputs["09_step7_lag_sensitivity.csv"],
            model_outputs["10_step7_capped_window_sensitivity.csv"],
            model_outputs["11_step7_rt_sensitivity.csv"],
        ],
        ignore_index=True,
    )
    benchmark_df = model_outputs["12_step7_fixed_vs_variable_benchmark.csv"].copy()
    final_conclusion_df, conclusion_text = build_final_conclusion(primary_model_df)

    save_dataframe(trial_timing_df, CLEAN_DIR / "01_step7_trial_timing_table.csv")
    save_dataframe(diagnostics_df, CLEAN_DIR / "02_step7_duration_diagnostics.csv")
    save_dataframe(trial_summary_df, CLEAN_DIR / "03_step7_trial_variable_window_summary.csv")
    for name, df in model_outputs.items():
        save_dataframe(df, CLEAN_DIR / name)
    save_dataframe(exclusion_log_df, CLEAN_DIR / "13_step7_exclusion_log.csv")
    save_dataframe(final_conclusion_df, CLEAN_DIR / "14_step7_final_conclusion.csv")

    generate_figures(trial_summary_df, fit_artifacts, benchmark_df)
    write_supporting_tex(primary_model_df, secondary_models_df, conclusion_text)
    report_path = write_report(trial_summary_df, primary_model_df, secondary_models_df, diagnostics_df, benchmark_df, exclusion_log_df)
    compile_report(report_path)

    print("Step 7 duration-aware analysis completed.")
    print(f"Included participants: {int(primary_model_df.iloc[0]['n_participants'])}")
    print(f"Included questions: {int(primary_model_df.iloc[0]['n_questions'])}")
    print(f"Included trials: {int(primary_model_df.iloc[0]['n_trials'])}")
    print(f"Primary condition p-value: {float(primary_model_df.iloc[0]['condition_p_value']):.6f}")


if __name__ == "__main__":
    main()
