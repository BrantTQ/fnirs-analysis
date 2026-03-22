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
from statsmodels.stats.multitest import multipletests

from step5_fixed_window_roi import BACK_HBO_CHANNELS
from step6_covariate_adjusted import (
    compile_report,
    format_effect,
    format_p_value,
    make_latex_grid,
    make_latex_table,
    sanitize_for_tex,
    save_dataframe,
)
from step7_duration_aware import extract_variable_window_mean


ROOT = Path(__file__).resolve().parents[1]
STEP7_TRIAL_PATH = ROOT / "data_clean" / "step7" / "01_step7_trial_timing_table.csv"
STEP3_STATUS_PATH = ROOT / "data_clean" / "step3" / "07_fnirs_session_status.csv"

CLEAN_DIR = ROOT / "data_clean" / "step8"
FIGURES_DIR = ROOT / "figures" / "step8"
REPORTS_DIR = ROOT / "reports" / "step8"
TABLES_DIR = REPORTS_DIR / "tables"
TEXT_DIR = REPORTS_DIR / "text"

MIN_PAIR_N = 10
MATCH_ABS_Z_THRESHOLD = 0.5
SELECTED_PAIR_LIMIT = 3
ALPHA = 0.05
CONCRETE_COLOR = "#228833"
ABSTRACT_COLOR = "#AA3377"
NEUTRAL_GREY = "#BFC9CA"


def ensure_directories() -> None:
    for path in [CLEAN_DIR, FIGURES_DIR, REPORTS_DIR, TABLES_DIR, TEXT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def coerce_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_df = pd.read_csv(STEP7_TRIAL_PATH)
    status_df = pd.read_csv(STEP3_STATUS_PATH)
    if "included_primary_model" in trial_df.columns:
        trial_df["included_primary_model"] = trial_df["included_primary_model"].map(coerce_bool)
    return trial_df, status_df


def build_trial_base(trial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    included_df = trial_df[trial_df["included_primary_model"]].copy()
    included_df = included_df.sort_values(["participant_id", "question_id"]).reset_index(drop=True)

    exclusion_rows: list[dict[str, Any]] = [
        {
            "stage": "cohort",
            "level": "info",
            "reason_code": "STEP7_COHORT_REUSED",
            "n_rows": int(included_df[["participant_id", "session_id"]].drop_duplicates().shape[0]),
            "detail": "Step 8 reused the Step 7 included participant-session cohort without modification.",
        }
    ]

    duplicate_counts = included_df.groupby(["participant_id", "question_id"]).size().reset_index(name="n_rows")
    duplicates = duplicate_counts[duplicate_counts["n_rows"] > 1].copy()
    if not duplicates.empty:
        exclusion_rows.append(
            {
                "stage": "participant_question_matrix",
                "level": "warning",
                "reason_code": "DUPLICATE_PARTICIPANT_QUESTION_ROWS",
                "n_rows": int(duplicates.shape[0]),
                "detail": "Multiple Step 7 rows existed for the same participant-question combination; the first sorted row was retained.",
            }
        )
        included_df = included_df.drop_duplicates(["participant_id", "question_id"], keep="first").copy()
    else:
        exclusion_rows.append(
            {
                "stage": "participant_question_matrix",
                "level": "info",
                "reason_code": "UNIQUE_PARTICIPANT_QUESTION_ROWS",
                "n_rows": int(included_df.shape[0]),
                "detail": "Each participant-question combination appeared exactly once in the Step 7 trial table.",
            }
        )

    question_meta = (
        included_df[
            [
                "question_id",
                "condition",
                "total_word_count",
                "sentence_count",
                "sentence_length",
                "enem_correctness",
                "z_log_total_word_count",
                "z_enem_correctness",
            ]
        ]
        .drop_duplicates("question_id")
        .sort_values(["condition", "question_id"])
        .reset_index(drop=True)
    )

    question_matrix = (
        included_df.pivot(index="participant_id", columns="question_id", values="back_roi_mean_variable_lag4")
        .reset_index()
        .rename_axis(columns=None)
    )
    participant_session_map = (
        included_df[["participant_id", "session_id"]].drop_duplicates("participant_id").sort_values("participant_id")
    )
    question_matrix = participant_session_map.merge(question_matrix, on="participant_id", how="left")

    return included_df, question_meta, pd.DataFrame(exclusion_rows), question_matrix


def build_pair_metadata(question_meta: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    concrete_questions = sorted(question_meta.loc[question_meta["condition"] == "Concrete", "question_id"].astype(str).tolist())
    abstract_questions = sorted(question_meta.loc[question_meta["condition"] == "Abstract", "question_id"].astype(str).tolist())
    meta_lookup = question_meta.set_index("question_id").to_dict(orient="index")

    pair_rows: list[dict[str, Any]] = []
    for concrete_id in concrete_questions:
        concrete_meta = meta_lookup[concrete_id]
        for abstract_id in abstract_questions:
            abstract_meta = meta_lookup[abstract_id]
            diff_word = float(concrete_meta["total_word_count"] - abstract_meta["total_word_count"])
            diff_correctness = float(concrete_meta["enem_correctness"] - abstract_meta["enem_correctness"])
            diff_z_log_word = float(concrete_meta["z_log_total_word_count"] - abstract_meta["z_log_total_word_count"])
            diff_z_correctness = float(concrete_meta["z_enem_correctness"] - abstract_meta["z_enem_correctness"])
            pair_rows.append(
                {
                    "concrete_question_id": concrete_id,
                    "abstract_question_id": abstract_id,
                    "word_count_difference": diff_word,
                    "enem_correctness_difference": diff_correctness,
                    "z_log_total_word_count_difference": diff_z_log_word,
                    "z_enem_correctness_difference": diff_z_correctness,
                    "matching_distance": abs(diff_z_log_word) + abs(diff_z_correctness),
                    "matched_pair_flag": abs(diff_z_log_word) <= MATCH_ABS_Z_THRESHOLD
                    and abs(diff_z_correctness) <= MATCH_ABS_Z_THRESHOLD,
                    "matched_pair_rule": "|z(log word count difference)| <= 0.5 and |z(ENEM correctness difference)| <= 0.5",
                }
            )

    return concrete_questions, abstract_questions, pd.DataFrame(pair_rows)


def run_pairwise_tests(
    included_df: pd.DataFrame,
    pair_meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trial_values = included_df[["participant_id", "question_id", "back_roi_mean_variable_lag4"]].copy()
    value_matrix = trial_values.pivot(index="participant_id", columns="question_id", values="back_roi_mean_variable_lag4")

    long_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []

    for pair in pair_meta_df.itertuples(index=False):
        pair_df = value_matrix[[pair.concrete_question_id, pair.abstract_question_id]].dropna().copy()
        pair_df = pair_df.rename(
            columns={
                pair.concrete_question_id: "concrete_value",
                pair.abstract_question_id: "abstract_value",
            }
        )
        pair_df["pair_difference"] = pair_df["concrete_value"] - pair_df["abstract_value"]
        pair_df = pair_df.reset_index()
        pair_df["concrete_question_id"] = pair.concrete_question_id
        pair_df["abstract_question_id"] = pair.abstract_question_id
        pair_df["word_count_difference"] = pair.word_count_difference
        pair_df["enem_correctness_difference"] = pair.enem_correctness_difference
        long_rows.extend(pair_df.to_dict(orient="records"))

        n_pair = int(pair_df.shape[0])
        tested_flag = n_pair >= MIN_PAIR_N
        mean_diff = float(pair_df["pair_difference"].mean()) if n_pair else math.nan
        sd_diff = float(pair_df["pair_difference"].std(ddof=1)) if n_pair >= 2 else math.nan
        if tested_flag:
            t_stat, p_value = stats.ttest_rel(pair_df["concrete_value"], pair_df["abstract_value"])
            t_stat = float(t_stat)
            p_value = float(p_value)
            effect_size = mean_diff / sd_diff if np.isfinite(sd_diff) and not math.isclose(sd_diff, 0.0) else math.nan
        else:
            t_stat = math.nan
            p_value = math.nan
            effect_size = math.nan

        result_rows.append(
            {
                "concrete_question_id": pair.concrete_question_id,
                "abstract_question_id": pair.abstract_question_id,
                "n_participants": n_pair,
                "mean_difference": mean_diff,
                "sd_difference": sd_diff,
                "paired_t_stat": t_stat,
                "raw_p_value": p_value,
                "cohens_dz": effect_size,
                "word_count_difference": pair.word_count_difference,
                "enem_correctness_difference": pair.enem_correctness_difference,
                "matching_distance": pair.matching_distance,
                "matched_pair_flag": bool(pair.matched_pair_flag),
                "tested_flag": tested_flag,
                "min_pair_n_threshold": MIN_PAIR_N,
            }
        )

        if not tested_flag:
            exclusion_rows.append(
                {
                    "stage": "pairwise_testing",
                    "level": "warning",
                    "reason_code": "PAIR_BELOW_MIN_N",
                    "n_rows": 1,
                    "detail": f"{pair.concrete_question_id} vs {pair.abstract_question_id} had only {n_pair} paired participants.",
                }
            )

    long_df = pd.DataFrame(long_rows)
    results_df = pd.DataFrame(result_rows)
    valid_mask = results_df["tested_flag"] & results_df["raw_p_value"].notna()
    results_df["fdr_q_value"] = math.nan
    results_df["fdr_reject"] = False
    if valid_mask.any():
        reject, q_values, _, _ = multipletests(results_df.loc[valid_mask, "raw_p_value"], alpha=ALPHA, method="fdr_bh")
        results_df.loc[valid_mask, "fdr_q_value"] = q_values
        results_df.loc[valid_mask, "fdr_reject"] = reject

    results_df = results_df.sort_values(["fdr_q_value", "raw_p_value", "concrete_question_id", "abstract_question_id"], na_position="last").reset_index(drop=True)
    fdr_df = results_df.copy()
    return long_df, results_df, fdr_df, pd.DataFrame(exclusion_rows)


def build_question_centered_ranking(included_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = included_df[["participant_id", "question_id", "condition", "back_roi_mean_variable_lag4"]].copy()
    ranking_df["participant_mean_back_roi"] = ranking_df.groupby("participant_id")["back_roi_mean_variable_lag4"].transform("mean")
    ranking_df["participant_centered_back_roi"] = ranking_df["back_roi_mean_variable_lag4"] - ranking_df["participant_mean_back_roi"]

    question_ranking = (
        ranking_df.groupby(["question_id", "condition"], as_index=False)
        .agg(
            n_participants=("participant_id", "nunique"),
            mean_centered_score=("participant_centered_back_roi", "mean"),
            sd_centered_score=("participant_centered_back_roi", "std"),
            mean_raw_back_roi=("back_roi_mean_variable_lag4", "mean"),
        )
        .sort_values("mean_centered_score", ascending=False)
        .reset_index(drop=True)
    )
    question_ranking["rank_overall"] = np.arange(1, len(question_ranking) + 1)
    question_ranking["rank_within_condition"] = question_ranking.groupby("condition")["mean_centered_score"].rank(
        method="first", ascending=False
    )
    return question_ranking


def build_matched_pair_results(pair_meta_df: pd.DataFrame, fdr_df: pd.DataFrame) -> pd.DataFrame:
    matched_df = fdr_df.merge(
        pair_meta_df[
            [
                "concrete_question_id",
                "abstract_question_id",
                "matched_pair_flag",
                "matching_distance",
                "matched_pair_rule",
            ]
        ],
        on=["concrete_question_id", "abstract_question_id"],
        how="left",
        suffixes=("", "_meta"),
    )
    matched_df["matched_pair_flag"] = matched_df["matched_pair_flag_meta"].fillna(matched_df["matched_pair_flag"])
    matched_df.drop(columns=["matched_pair_flag_meta"], inplace=True)
    matched_df = matched_df[matched_df["matched_pair_flag"]].copy().reset_index(drop=True)
    valid_mask = matched_df["tested_flag"] & matched_df["raw_p_value"].notna()
    matched_df["matched_subset_fdr_q_value"] = math.nan
    matched_df["matched_subset_fdr_reject"] = False
    if valid_mask.any():
        reject, q_values, _, _ = multipletests(matched_df.loc[valid_mask, "raw_p_value"], alpha=ALPHA, method="fdr_bh")
        matched_df.loc[valid_mask, "matched_subset_fdr_q_value"] = q_values
        matched_df.loc[valid_mask, "matched_subset_fdr_reject"] = reject
    return matched_df.sort_values(["matched_subset_fdr_q_value", "raw_p_value"], na_position="last").reset_index(drop=True)


def select_pairs_for_channel_followup(fdr_df: pd.DataFrame) -> pd.DataFrame:
    tested_df = fdr_df[fdr_df["tested_flag"]].copy()
    significant_df = tested_df[tested_df["fdr_reject"]].copy()
    if not significant_df.empty:
        selected = significant_df.sort_values(["fdr_q_value", "raw_p_value", "mean_difference"], ascending=[True, True, False]).copy()
        selected["selection_reason"] = "FDR significant pair"
        return selected.reset_index(drop=True)

    selected = tested_df.sort_values(
        ["fdr_q_value", "raw_p_value", "mean_difference"],
        ascending=[True, True, False],
        na_position="last",
    ).head(SELECTED_PAIR_LIMIT).copy()
    selected["selection_reason"] = "Top descriptive pair by lowest FDR q-value (none survived FDR)"
    return selected.reset_index(drop=True)


def compute_channel_followup(
    included_df: pd.DataFrame,
    status_df: pd.DataFrame,
    selected_pairs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected_pairs_df.empty:
        empty = pd.DataFrame(
            columns=[
                "concrete_question_id",
                "abstract_question_id",
                "selection_reason",
                "channel_name",
                "n_participants",
                "mean_difference",
                "paired_t_stat",
                "raw_p_value",
                "channel_fdr_q_value",
                "channel_fdr_reject",
                "cohens_dz",
            ]
        )
        return empty, empty

    selected_question_ids = sorted(
        set(selected_pairs_df["concrete_question_id"].astype(str)).union(selected_pairs_df["abstract_question_id"].astype(str))
    )
    selected_trials = included_df[included_df["question_id"].astype(str).isin(selected_question_ids)].copy()
    status_lookup = {
        (str(row["participant_id"]), str(row["session_id"])): row["preprocessed_file"]
        for _, row in status_df.iterrows()
        if isinstance(row.get("preprocessed_file"), str) and row.get("preprocessed_file")
    }

    trial_channel_rows: list[dict[str, Any]] = []
    for (participant_id, session_id), session_df in selected_trials.groupby(["participant_id", "session_id"], sort=True):
        preprocessed_file = status_lookup.get((str(participant_id), str(session_id)))
        if not preprocessed_file:
            continue
        raw = mne.io.read_raw_fif(preprocessed_file, preload=True, verbose="ERROR")
        session_data = raw.get_data()
        session_times = raw.times.copy()
        name_to_index = {name: idx for idx, name in enumerate(raw.ch_names)}

        for row in session_df.itertuples(index=False):
            available_back_channels = [
                name.strip()
                for name in str(row.back_hbo_channels).split(";")
                if name and isinstance(name, str) and name.strip() in name_to_index
            ]
            for channel_name in BACK_HBO_CHANNELS:
                if channel_name not in available_back_channels:
                    trial_channel_rows.append(
                        {
                            "participant_id": participant_id,
                            "question_id": row.question_id,
                            "condition": row.condition,
                            "channel_name": channel_name,
                            "channel_response": math.nan,
                        }
                    )
                    continue
                values, issue = extract_variable_window_mean(
                    session_data,
                    session_times,
                    float(row.baseline_start_time),
                    float(row.baseline_end_time),
                    float(row.primary_window_start_time),
                    float(row.primary_window_end_time),
                    [name_to_index[channel_name]],
                )
                response = float(values[0]) if values is not None and issue is None else math.nan
                trial_channel_rows.append(
                    {
                        "participant_id": participant_id,
                        "question_id": row.question_id,
                        "condition": row.condition,
                        "channel_name": channel_name,
                        "channel_response": response,
                    }
                )

    trial_channel_df = pd.DataFrame(trial_channel_rows)
    if trial_channel_df.empty:
        empty = pd.DataFrame(
            columns=[
                "concrete_question_id",
                "abstract_question_id",
                "selection_reason",
                "channel_name",
                "n_participants",
                "mean_difference",
                "paired_t_stat",
                "raw_p_value",
                "channel_fdr_q_value",
                "channel_fdr_reject",
                "cohens_dz",
            ]
        )
        return trial_channel_df, empty

    result_rows: list[dict[str, Any]] = []
    channel_matrix = trial_channel_df.pivot_table(
        index=["participant_id", "channel_name"],
        columns="question_id",
        values="channel_response",
        aggfunc="first",
    )

    for pair in selected_pairs_df.itertuples(index=False):
        pair_channel_rows: list[dict[str, Any]] = []
        for channel_name in BACK_HBO_CHANNELS:
            if (channel_matrix.index.get_level_values("channel_name") == channel_name).sum() == 0:
                pair_channel_rows.append(
                    {
                        "concrete_question_id": pair.concrete_question_id,
                        "abstract_question_id": pair.abstract_question_id,
                        "selection_reason": pair.selection_reason,
                        "channel_name": channel_name,
                        "n_participants": 0,
                        "mean_difference": math.nan,
                        "paired_t_stat": math.nan,
                        "raw_p_value": math.nan,
                        "cohens_dz": math.nan,
                    }
                )
                continue

            channel_df = channel_matrix.xs(channel_name, level="channel_name").copy()
            if pair.concrete_question_id not in channel_df.columns or pair.abstract_question_id not in channel_df.columns:
                pair_channel_rows.append(
                    {
                        "concrete_question_id": pair.concrete_question_id,
                        "abstract_question_id": pair.abstract_question_id,
                        "selection_reason": pair.selection_reason,
                        "channel_name": channel_name,
                        "n_participants": 0,
                        "mean_difference": math.nan,
                        "paired_t_stat": math.nan,
                        "raw_p_value": math.nan,
                        "cohens_dz": math.nan,
                    }
                )
                continue

            pair_df = channel_df[[pair.concrete_question_id, pair.abstract_question_id]].dropna().copy()
            n_pair = int(pair_df.shape[0])
            if n_pair >= 2:
                diff = pair_df[pair.concrete_question_id] - pair_df[pair.abstract_question_id]
                mean_diff = float(diff.mean())
                sd_diff = float(diff.std(ddof=1))
                t_stat, p_value = stats.ttest_rel(pair_df[pair.concrete_question_id], pair_df[pair.abstract_question_id])
                effect_size = mean_diff / sd_diff if np.isfinite(sd_diff) and not math.isclose(sd_diff, 0.0) else math.nan
                t_stat = float(t_stat)
                p_value = float(p_value)
            else:
                mean_diff = math.nan
                t_stat = math.nan
                p_value = math.nan
                effect_size = math.nan

            pair_channel_rows.append(
                {
                    "concrete_question_id": pair.concrete_question_id,
                    "abstract_question_id": pair.abstract_question_id,
                    "selection_reason": pair.selection_reason,
                    "channel_name": channel_name,
                    "n_participants": n_pair,
                    "mean_difference": mean_diff,
                    "paired_t_stat": t_stat,
                    "raw_p_value": p_value,
                    "cohens_dz": effect_size,
                }
            )

        pair_channel_df = pd.DataFrame(pair_channel_rows)
        valid_mask = pair_channel_df["raw_p_value"].notna()
        pair_channel_df["channel_fdr_q_value"] = math.nan
        pair_channel_df["channel_fdr_reject"] = False
        if valid_mask.any():
            reject, q_values, _, _ = multipletests(pair_channel_df.loc[valid_mask, "raw_p_value"], alpha=ALPHA, method="fdr_bh")
            pair_channel_df.loc[valid_mask, "channel_fdr_q_value"] = q_values
            pair_channel_df.loc[valid_mask, "channel_fdr_reject"] = reject
        result_rows.extend(pair_channel_df.to_dict(orient="records"))

    return trial_channel_df, pd.DataFrame(result_rows)


def pivot_pair_matrix(
    fdr_df: pd.DataFrame,
    value_col: str,
    concrete_questions: list[str],
    abstract_questions: list[str],
) -> pd.DataFrame:
    matrix = (
        fdr_df.pivot(index="concrete_question_id", columns="abstract_question_id", values=value_col)
        .reindex(index=concrete_questions, columns=abstract_questions)
    )
    return matrix


def plot_heatmap(
    matrix_df: pd.DataFrame,
    title: str,
    colorbar_label: str,
    path: Path,
    cmap: str = "RdBu_r",
    center_zero: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    values = matrix_df.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    if finite.size == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    if center_zero:
        vmax_auto = float(np.nanpercentile(np.abs(finite), 98))
        vmax = vmax if vmax is not None else max(vmax_auto, 1e-8)
        vmin = -vmax if vmin is None else vmin
    else:
        if vmin is None:
            vmin = float(np.nanmin(finite))
        if vmax is None:
            vmax = float(np.nanmax(finite))
        if math.isclose(vmin, vmax):
            vmin -= 1e-8
            vmax += 1e-8

    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index.tolist(), fontsize=8)
    ax.set_xlabel("Abstract question")
    ax.set_ylabel("Concrete question")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(colorbar_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_matched_pair_heatmap(
    fdr_df: pd.DataFrame,
    concrete_questions: list[str],
    abstract_questions: list[str],
    path: Path,
) -> None:
    matched_df = fdr_df[fdr_df["matched_pair_flag"]].copy()
    matrix_df = pivot_pair_matrix(matched_df, "mean_difference", concrete_questions, abstract_questions)
    plot_heatmap(
        matrix_df,
        "Matched-pair subset mean difference heatmap\nConcrete minus Abstract back-ROI response",
        r"Mean difference ($\mu$M)",
        path,
        cmap="RdBu_r",
        center_zero=True,
    )


def plot_question_centered_ranking(ranking_df: pd.DataFrame, path: Path) -> None:
    ranking_sorted = ranking_df.sort_values("mean_centered_score", ascending=True).reset_index(drop=True)
    colors = ranking_sorted["condition"].map({"Concrete": CONCRETE_COLOR, "Abstract": ABSTRACT_COLOR}).fillna(NEUTRAL_GREY)
    fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
    y_pos = np.arange(len(ranking_sorted))
    ax.barh(y_pos, ranking_sorted["mean_centered_score"].to_numpy(dtype=float) * 1e6, color=colors.tolist(), alpha=0.85)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{qid} ({cond[0]})" for qid, cond in ranking_sorted[["question_id", "condition"]].itertuples(index=False)],
        fontsize=8,
    )
    ax.set_xlabel(r"Participant-centered back ROI score ($\mu$M)")
    ax.set_ylabel("Question")
    ax.set_title("Question ranking by participant-centered back-ROI response")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_selected_pair_channel_profile(channel_followup_df: pd.DataFrame, path: Path) -> None:
    if channel_followup_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.text(0.5, 0.5, "No selected pair follow-up results available", ha="center", va="center", fontsize=13)
        ax.set_axis_off()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    pair_labels = (
        channel_followup_df["concrete_question_id"].astype(str)
        + " vs "
        + channel_followup_df["abstract_question_id"].astype(str)
    )
    matrix_df = (
        channel_followup_df.assign(pair_label=pair_labels)
        .pivot(index="pair_label", columns="channel_name", values="mean_difference")
        .reindex(columns=BACK_HBO_CHANNELS)
    )
    q_matrix = (
        channel_followup_df.assign(pair_label=pair_labels)
        .pivot(index="pair_label", columns="channel_name", values="channel_fdr_q_value")
        .reindex(columns=BACK_HBO_CHANNELS)
    )

    values = matrix_df.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1e-8
    vmax = max(vmax, 1e-8)

    fig, ax = plt.subplots(figsize=(12, max(4, 1.2 * len(matrix_df.index))), layout="constrained")
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index.tolist(), fontsize=9)
    ax.set_xlabel("Back ROI channel")
    ax.set_ylabel("Selected pair")
    ax.set_title("Selected pair channel profile\nConcrete minus Abstract back-ROI channel responses")
    for row_idx in range(len(matrix_df.index)):
        for col_idx in range(len(matrix_df.columns)):
            q_value = q_matrix.iloc[row_idx, col_idx]
            if pd.notna(q_value) and float(q_value) < ALPHA:
                ax.text(col_idx, row_idx, "*", ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"Mean difference ($\mu$M)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_final_conclusion(fdr_df: pd.DataFrame, matched_results_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    tested_df = fdr_df[fdr_df["tested_flag"]].copy()
    n_tested_pairs = int(tested_df.shape[0])
    n_fdr_significant = int(tested_df["fdr_reject"].sum()) if not tested_df.empty else 0
    top_pair = tested_df.nsmallest(1, ["fdr_q_value", "raw_p_value"])
    top_text = ""
    if not top_pair.empty:
        top_row = top_pair.iloc[0]
        top_text = (
            f"The lowest-q pair was {top_row['concrete_question_id']} versus {top_row['abstract_question_id']} "
            f"(mean difference={top_row['mean_difference']:.3e}, raw p={top_row['raw_p_value']:.6f}, q={top_row['fdr_q_value']:.6f})."
        )

    matched_significant = int(matched_results_df["matched_subset_fdr_reject"].sum()) if not matched_results_df.empty else 0
    if n_fdr_significant == 0:
        conclusion = (
            "The exploratory item-pair analysis did not identify any concrete-abstract question pairs with reliable back-ROI differences after Benjamini-Hochberg correction. "
            "This supports the view that the back ROI does not show robust item-specific discrimination in the current dataset. "
            + top_text
        )
    else:
        conclusion = (
            f"The exploratory item-pair analysis identified {n_fdr_significant} concrete-abstract question pairs with FDR-adjusted q-values below 0.05 in the back ROI. "
            "These results suggest item-specific posterior sensitivity, but they should be interpreted as exploratory findings rather than evidence of a broad back-ROI concreteness effect. "
            + top_text
        )
    if matched_results_df.empty:
        conclusion += " No matched-pair subset comparisons were available under the predeclared matching rule."
    elif matched_significant == 0:
        conclusion += " No matched-pair subset comparison survived FDR correction under the predeclared matching rule."
    else:
        conclusion += f" The matched-pair robustness subset retained {matched_significant} FDR-significant pair(s)."

    final_df = pd.DataFrame(
        [
            {
                "n_tested_pairs": n_tested_pairs,
                "n_fdr_significant_pairs": n_fdr_significant,
                "n_matched_pairs": int(matched_results_df.shape[0]),
                "n_matched_fdr_significant_pairs": matched_significant,
                "matching_rule": "|z(log word count difference)| <= 0.5 and |z(ENEM correctness difference)| <= 0.5",
                "conclusion_text": conclusion,
            }
        ]
    )
    return final_df, conclusion


def write_supporting_tex(fdr_df: pd.DataFrame, ranking_df: pd.DataFrame, conclusion_text: str) -> None:
    top_pairs = fdr_df[fdr_df["tested_flag"]].head(12).copy()
    pair_rows = []
    for row in top_pairs.itertuples(index=False):
        pair_rows.append(
            [
                row.concrete_question_id,
                row.abstract_question_id,
                int(row.n_participants),
                format_effect(row.mean_difference),
                format_p_value(row.raw_p_value),
                format_p_value(row.fdr_q_value),
                f"{float(row.cohens_dz):.3f}" if pd.notna(row.cohens_dz) else "",
            ]
        )
    (TABLES_DIR / "pairwise_main_table.tex").write_text(
        make_latex_grid(
            ["Concrete", "Abstract", "N", "Mean diff", "Raw p", "FDR q", "d_z"],
            pair_rows,
            r"p{0.18\linewidth}p{0.18\linewidth}p{0.06\linewidth}p{0.14\linewidth}p{0.12\linewidth}p{0.12\linewidth}p{0.08\linewidth}",
        )
        + "\n",
        encoding="utf-8",
    )

    top_ranked = ranking_df.head(12).copy()
    ranking_rows = []
    for row in top_ranked.itertuples(index=False):
        ranking_rows.append(
            [
                int(row.rank_overall),
                row.question_id,
                row.condition,
                f"{float(row.mean_centered_score) * 1e6:.3f}",
                int(row.n_participants),
            ]
        )
    (TABLES_DIR / "question_ranking_table.tex").write_text(
        make_latex_grid(
            ["Rank", "Question", "Cond.", "Centered score (uM)", "N"],
            ranking_rows,
            r"p{0.08\linewidth}p{0.28\linewidth}p{0.12\linewidth}p{0.22\linewidth}p{0.08\linewidth}",
        )
        + "\n",
        encoding="utf-8",
    )
    (TEXT_DIR / "final_conclusion.tex").write_text(conclusion_text + "\n", encoding="utf-8")


def write_report(
    question_matrix_df: pd.DataFrame,
    fdr_df: pd.DataFrame,
    matched_results_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    channel_followup_df: pd.DataFrame,
    conclusion_text: str,
) -> Path:
    tested_df = fdr_df[fdr_df["tested_flag"]].copy()
    overview_rows = [
        ("Included participant-sessions from Step 7", int(question_matrix_df.shape[0])),
        ("Concrete questions", int(fdr_df["concrete_question_id"].nunique())),
        ("Abstract questions", int(fdr_df["abstract_question_id"].nunique())),
        ("Tested concrete-abstract pairs", int(tested_df.shape[0])),
        ("Pairs surviving FDR", int(tested_df["fdr_reject"].sum())),
        ("Minimum paired N threshold", MIN_PAIR_N),
        ("Matched-pair rule", "|z(log word count difference)| <= 0.5 and |z(ENEM correctness difference)| <= 0.5"),
        ("Selected-pair follow-up rule", "All FDR-significant pairs, otherwise the top 3 lowest-q descriptive pairs"),
    ]

    top_pair_text = "No tested pairs were available."
    if not tested_df.empty:
        top_row = tested_df.iloc[0]
        top_pair_text = (
            f"Top pair: {top_row['concrete_question_id']} versus {top_row['abstract_question_id']} "
            f"(mean difference={top_row['mean_difference']:.3e}, raw p={top_row['raw_p_value']:.6f}, q={top_row['fdr_q_value']:.6f})."
        )

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Step 8 Exploratory Back-ROI Item-Pair Report}",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
        r"\paragraph{Exploratory status.} Step~8 was implemented exactly as an exploratory item-pair deviation from the main pipeline, using the frozen Step~7 cohort and the Step~7 duration-aware back-ROI outcome.",
        r"\paragraph{Top pair summary.} " + sanitize_for_tex(top_pair_text),
        r"\paragraph{Final conclusion.} \input{text/final_conclusion.tex}",
        r"\subsection*{Top Pairwise Results}",
        r"\input{tables/pairwise_main_table.tex}",
        r"\subsection*{Top Question Rankings}",
        r"\input{tables/question_ranking_table.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step8/pairwise_mean_difference_heatmap.png}",
        r"\caption{Mean back-ROI concrete-minus-abstract difference heatmap for all tested item pairs.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step8/pairwise_fdr_qvalue_heatmap.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step8/pairwise_effect_size_heatmap.png}",
        r"\caption{FDR-adjusted q-values and paired effect sizes for all tested item pairs.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step8/pairwise_raw_pvalue_heatmap.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step8/matched_pair_heatmap.png}",
        r"\caption{Raw p-values for all tested item pairs and mean differences restricted to the matched-pair robustness subset.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.80\linewidth]{../../figures/step8/question_centered_ranking_plot.png}",
        r"\caption{Question ranking by participant-centered back-ROI response.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step8/selected_pair_channel_profile.png}",
        r"\caption{Optional channel-level follow-up for the selected item pairs. Asterisks mark channels surviving within-pair FDR correction.}",
        r"\end{figure}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step8_back_roi_itempair_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def generate_figures(
    fdr_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    matched_results_df: pd.DataFrame,
    channel_followup_df: pd.DataFrame,
    concrete_questions: list[str],
    abstract_questions: list[str],
) -> None:
    plot_heatmap(
        pivot_pair_matrix(fdr_df, "mean_difference", concrete_questions, abstract_questions),
        "Mean difference heatmap\nConcrete minus Abstract back-ROI response",
        r"Mean difference ($\mu$M)",
        FIGURES_DIR / "pairwise_mean_difference_heatmap.png",
        cmap="RdBu_r",
        center_zero=True,
    )
    plot_heatmap(
        pivot_pair_matrix(fdr_df, "raw_p_value", concrete_questions, abstract_questions),
        "Raw p-value heatmap",
        "Raw p-value",
        FIGURES_DIR / "pairwise_raw_pvalue_heatmap.png",
        cmap="viridis_r",
        center_zero=False,
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap(
        pivot_pair_matrix(fdr_df, "fdr_q_value", concrete_questions, abstract_questions),
        "FDR-adjusted q-value heatmap",
        "FDR q-value",
        FIGURES_DIR / "pairwise_fdr_qvalue_heatmap.png",
        cmap="viridis_r",
        center_zero=False,
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap(
        pivot_pair_matrix(fdr_df, "cohens_dz", concrete_questions, abstract_questions),
        "Effect size heatmap\nPaired Cohen's d_z",
        "Cohen's d_z",
        FIGURES_DIR / "pairwise_effect_size_heatmap.png",
        cmap="RdBu_r",
        center_zero=True,
    )
    plot_question_centered_ranking(ranking_df, FIGURES_DIR / "question_centered_ranking_plot.png")
    plot_matched_pair_heatmap(matched_results_df, concrete_questions, abstract_questions, FIGURES_DIR / "matched_pair_heatmap.png")
    plot_selected_pair_channel_profile(channel_followup_df, FIGURES_DIR / "selected_pair_channel_profile.png")


def main() -> None:
    ensure_directories()
    trial_df, status_df = load_inputs()
    included_df, question_meta_df, exclusion_log_df, question_matrix_df = build_trial_base(trial_df)
    concrete_questions, abstract_questions, pair_meta_df = build_pair_metadata(question_meta_df)
    pairwise_long_df, pairwise_results_df, fdr_df, pairwise_exclusion_df = run_pairwise_tests(included_df, pair_meta_df)
    ranking_df = build_question_centered_ranking(included_df)
    matched_subset_df = pair_meta_df.copy()
    matched_results_df = build_matched_pair_results(pair_meta_df, fdr_df)
    selected_pairs_df = select_pairs_for_channel_followup(fdr_df)
    _, channel_followup_df = compute_channel_followup(included_df, status_df, selected_pairs_df)

    exclusion_log_df = pd.concat(
        [
            exclusion_log_df,
            pairwise_exclusion_df,
            pd.DataFrame(
                [
                    {
                        "stage": "matched_pairs",
                        "level": "info",
                        "reason_code": "MATCH_RULE_FIXED",
                        "n_rows": int(matched_subset_df["matched_pair_flag"].sum()),
                        "detail": "|z(log word count difference)| <= 0.5 and |z(ENEM correctness difference)| <= 0.5",
                    },
                    {
                        "stage": "channel_followup",
                        "level": "info",
                        "reason_code": "SELECTED_PAIR_RULE",
                        "n_rows": int(selected_pairs_df.shape[0]),
                        "detail": "Selected all FDR-significant pairs when available; otherwise selected the top 3 lowest-q descriptive pairs for optional channel follow-up.",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    final_conclusion_df, conclusion_text = build_final_conclusion(fdr_df, matched_results_df)
    write_supporting_tex(fdr_df, ranking_df, conclusion_text)
    generate_figures(fdr_df, ranking_df, matched_results_df, channel_followup_df, concrete_questions, abstract_questions)
    report_path = write_report(question_matrix_df, fdr_df, matched_results_df, ranking_df, channel_followup_df, conclusion_text)

    save_dataframe(question_matrix_df, CLEAN_DIR / "01_step8_back_roi_question_matrix.csv")
    save_dataframe(pairwise_long_df, CLEAN_DIR / "02_step8_pairwise_differences_long.csv")
    save_dataframe(pairwise_results_df, CLEAN_DIR / "03_step8_pairwise_test_results.csv")
    save_dataframe(fdr_df, CLEAN_DIR / "04_step8_pairwise_fdr_results.csv")
    save_dataframe(ranking_df, CLEAN_DIR / "05_step8_question_centered_ranking.csv")
    save_dataframe(matched_subset_df, CLEAN_DIR / "06_step8_matched_pair_subset.csv")
    save_dataframe(matched_results_df, CLEAN_DIR / "07_step8_matched_pair_results.csv")
    save_dataframe(channel_followup_df, CLEAN_DIR / "08_step8_selected_pair_channel_followup.csv")
    save_dataframe(exclusion_log_df, CLEAN_DIR / "09_step8_exclusion_log.csv")
    save_dataframe(final_conclusion_df, CLEAN_DIR / "10_step8_final_conclusion.csv")

    compile_report(report_path)

    print("Step 8 back ROI item-pair analysis completed.")
    print(f"Included participant-sessions: {question_matrix_df.shape[0]}")
    print(f"Tested pairs: {int(fdr_df['tested_flag'].sum())}")
    print(f"Pairs surviving FDR: {int(fdr_df['fdr_reject'].sum())}")


if __name__ == "__main__":
    main()
