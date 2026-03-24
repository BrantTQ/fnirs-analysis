#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from step5_fixed_window_roi import available_channels
from step6_covariate_adjusted import (
    compile_report,
    format_effect,
    format_p_value,
    make_latex_grid,
    make_latex_table,
    save_dataframe,
)
from step7_duration_aware import extract_variable_window_mean
from step11_continuous_abstractness import (
    add_term,
    apply_primary_robust_cov,
    build_question_score_table,
    build_trial_table,
    fe_formula,
    fit_base_ols,
)


ROOT = Path(__file__).resolve().parents[1]
STEP10_TABLE_PATH = ROOT / "data_clean" / "step10" / "01_step10_primary_fe_table.csv"
STEP7_TIMING_PATH = ROOT / "data_clean" / "step7" / "01_step7_trial_timing_table.csv"
STEP3_STATUS_PATH = ROOT / "data_clean" / "step3" / "07_fnirs_session_status.csv"
FILTERED_QUESTIONS_PATH = ROOT / "materials" / "filtered_questions.json"

CLEAN_DIR = ROOT / "data_clean" / "step12"
FIGURES_DIR = ROOT / "figures" / "step12"
REPORTS_DIR = ROOT / "reports" / "step12"
TABLES_DIR = REPORTS_DIR / "tables"
TEXT_DIR = REPORTS_DIR / "text"

ALPHA = 0.05
WAVEFORM_PRE_SEC = 2.0
WAVEFORM_POST_SEC = 20.0
WAVEFORM_STEP_SEC = 0.1
WAVEFORM_GRID = np.arange(-WAVEFORM_PRE_SEC, WAVEFORM_POST_SEC + 1e-9, WAVEFORM_STEP_SEC)
CONDITION_COLORS = {"Abstract": "#AA3377", "Concrete": "#228833"}

ROI_SPECS: list[dict[str, Any]] = [
    {
        "roi_slug": "anterior_dorsal",
        "roi_label": "Anterior Dorsal",
        "roi_description": "Control-network / superior frontal cortex candidate",
        "pairs": ["S1_D6", "S1_D3", "S3_D3", "S5_D3", "S5_D6"],
    },
    {
        "roi_slug": "anterior_ventral",
        "roi_label": "Anterior Ventral",
        "roi_description": "Inferior frontal / semantic processing candidate",
        "pairs": ["S2_D4", "S2_D6", "S5_D4", "S4_D4", "S5_D7"],
    },
    {
        "roi_slug": "posterior_dorsal",
        "roi_label": "Posterior Dorsal",
        "roi_description": "Superior parietal / visuospatial candidate",
        "pairs": ["S3_D7", "S3_D1", "S8_D1", "S8_D5", "S6_D1"],
    },
    {
        "roi_slug": "posterior_ventral",
        "roi_label": "Posterior Ventral",
        "roi_description": "Wernicke / semantic-multisensory candidate",
        "pairs": ["S7_D2", "S7_D5", "S6_D2", "S4_D2", "S4_D7", "S6_D7"],
    },
]


def ensure_directories() -> None:
    for path in [CLEAN_DIR, FIGURES_DIR, REPORTS_DIR, TABLES_DIR, TEXT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def pair_to_channel(pair_id: str) -> str:
    return f"{pair_id.replace('-', '_')} hbo"


def extract_waveform(
    data: np.ndarray,
    times: np.ndarray,
    start_time: float,
    channel_indices: list[int],
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
    return np.interp(WAVEFORM_GRID, rel_times, roi_series)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step10_trial_df = pd.read_csv(STEP10_TABLE_PATH)
    step7_timing_df = pd.read_csv(STEP7_TIMING_PATH)
    status_df = pd.read_csv(STEP3_STATUS_PATH)
    with open(FILTERED_QUESTIONS_PATH, "r", encoding="utf-8") as handle:
        question_df = pd.DataFrame(json.load(handle))
    return step10_trial_df, step7_timing_df, status_df, question_df


def build_base_trial_table(
    step10_trial_df: pd.DataFrame,
    step7_timing_df: pd.DataFrame,
    question_df: pd.DataFrame,
) -> pd.DataFrame:
    score_lookup_df, _, _ = build_question_score_table(question_df)
    trial_df = build_trial_table(step10_trial_df, score_lookup_df)

    timing_cols = [
        "participant_id",
        "session_id",
        "question_id",
        "question_start_time",
        "baseline_start_time",
        "baseline_end_time",
        "primary_window_start_time",
        "primary_window_end_time",
    ]
    timing_df = step7_timing_df[timing_cols].copy()
    merged_df = trial_df.merge(
        timing_df,
        on=["participant_id", "session_id", "question_id"],
        how="left",
        validate="one_to_one",
        indicator="timing_merge_status",
    )
    if (merged_df["timing_merge_status"] != "both").any():
        missing = (
            merged_df.loc[merged_df["timing_merge_status"] != "both", ["participant_id", "session_id", "question_id"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise RuntimeError(f"Step 12 timing merge failed for rows: {missing}")
    merged_df.drop(columns=["timing_merge_status"], inplace=True)
    return merged_df.sort_values(["participant_id", "session_id", "cumulative_trial_order"]).reset_index(drop=True)


def compute_roi_trial_outputs(
    base_trial_df: pd.DataFrame,
    status_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    preprocessed_lookup = {
        (row["participant_id"], row["session_id"]): row["preprocessed_file"]
        for _, row in status_df.iterrows()
        if isinstance(row.get("preprocessed_file"), str) and row.get("preprocessed_file")
    }

    session_channel_rows: list[dict[str, Any]] = []
    waveform_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    for (participant_id, session_id), session_df in base_trial_df.groupby(["participant_id", "session_id"], sort=True):
        preprocessed_file = preprocessed_lookup.get((participant_id, session_id))
        if not preprocessed_file:
            raise RuntimeError(f"Missing preprocessed file for {participant_id} / {session_id}")

        raw = mne.io.read_raw_fif(preprocessed_file, preload=True, verbose="ERROR")
        session_data = raw.get_data()
        session_times = raw.times.copy()
        channel_index_lookup = {name: idx for idx, name in enumerate(raw.ch_names)}

        roi_channels: dict[str, list[str]] = {}
        roi_indices: dict[str, list[int]] = {}
        for roi_spec in ROI_SPECS:
            requested_channels = [pair_to_channel(pair_id) for pair_id in roi_spec["pairs"]]
            present_channels = available_channels(raw, requested_channels)
            roi_channels[roi_spec["roi_slug"]] = present_channels
            roi_indices[roi_spec["roi_slug"]] = [channel_index_lookup[name] for name in present_channels]
            session_channel_rows.append(
                {
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "roi_slug": roi_spec["roi_slug"],
                    "roi_label": roi_spec["roi_label"],
                    "n_requested_channels": len(requested_channels),
                    "n_available_channels": len(present_channels),
                    "requested_channels": "; ".join(requested_channels),
                    "available_channels": "; ".join(present_channels),
                }
            )

        for trial in session_df.to_dict("records"):
            row = dict(trial)
            for roi_spec in ROI_SPECS:
                roi_slug = roi_spec["roi_slug"]
                channel_indices = roi_indices[roi_slug]
                values, reason = extract_variable_window_mean(
                    data=session_data,
                    times=session_times,
                    baseline_start=float(trial["baseline_start_time"]),
                    baseline_end=float(trial["baseline_end_time"]),
                    window_start=float(trial["primary_window_start_time"]),
                    window_end=float(trial["primary_window_end_time"]),
                    channel_indices=channel_indices,
                )
                outcome_col = f"{roi_slug}_mean_variable_lag4"
                n_channels_col = f"{roi_slug}_n_channels"
                row[n_channels_col] = len(channel_indices)
                row[outcome_col] = float(np.mean(values)) if values is not None else math.nan
                if values is None:
                    log_rows.append(
                        {
                            "stage": "roi_extraction",
                            "level": "warning",
                            "reason_code": "ROI_OUTCOME_MISSING",
                            "participant_id": participant_id,
                            "session_id": session_id,
                            "question_id": trial["question_id"],
                            "roi_slug": roi_slug,
                            "detail": reason or "unknown",
                        }
                    )

                waveform = extract_waveform(
                    data=session_data,
                    times=session_times,
                    start_time=float(trial["question_start_time"]),
                    channel_indices=channel_indices,
                )
                if waveform is not None:
                    waveform_rows.append(
                        {
                            "participant_id": participant_id,
                            "session_id": session_id,
                            "question_id": trial["question_id"],
                            "condition": trial["condition"],
                            "roi_slug": roi_slug,
                            "roi_label": roi_spec["roi_label"],
                            "waveform": waveform,
                        }
                    )
            trial_rows.append(row)

    wide_df = pd.DataFrame(trial_rows).sort_values(["participant_id", "session_id", "cumulative_trial_order"]).reset_index(drop=True)
    session_channel_df = pd.DataFrame(session_channel_rows).sort_values(["participant_id", "session_id", "roi_slug"]).reset_index(drop=True)
    log_df = pd.DataFrame(log_rows)
    return wide_df, session_channel_df, waveform_rows, log_df


def build_long_roi_table(wide_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "participant_id",
        "participant_code",
        "session_id",
        "question_id",
        "question_code",
        "condition",
        "condition_abstract",
        "abstracness",
        "z_abstracness",
        "global_block_order",
        "within_block_position",
        "z_log_total_word_count",
        "z_enem_correctness",
        "response_time_sec",
        "manual_override_session",
        "no_override_cohort",
        "question_start_time",
        "baseline_start_time",
        "baseline_end_time",
        "primary_window_start_time",
        "primary_window_end_time",
    ]
    value_cols = [f"{roi_spec['roi_slug']}_mean_variable_lag4" for roi_spec in ROI_SPECS]
    long_df = wide_df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="outcome_variable",
        value_name="roi_mean_variable_lag4",
    )
    slug_lookup = {f"{roi_spec['roi_slug']}_mean_variable_lag4": roi_spec["roi_slug"] for roi_spec in ROI_SPECS}
    label_lookup = {roi_spec["roi_slug"]: roi_spec["roi_label"] for roi_spec in ROI_SPECS}
    long_df["roi_slug"] = long_df["outcome_variable"].map(slug_lookup)
    long_df["roi_label"] = long_df["roi_slug"].map(label_lookup)
    return long_df.sort_values(["roi_slug", "participant_id", "session_id", "question_id"]).reset_index(drop=True)


def fit_roi_models(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for roi_spec in ROI_SPECS:
        roi_slug = roi_spec["roi_slug"]
        outcome = f"{roi_slug}_mean_variable_lag4"
        model_df = wide_df.copy()
        numeric_cols = [
            "participant_code",
            "question_code",
            "global_block_order",
            "within_block_position",
            "z_log_total_word_count",
            "z_enem_correctness",
            "z_abstracness",
            outcome,
        ]
        for col in numeric_cols:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        model_df = model_df.dropna(subset=numeric_cols).reset_index(drop=True)
        model_df["participant_code"] = model_df["participant_code"].astype(int)
        model_df["question_code"] = model_df["question_code"].astype(int)

        formula = fe_formula(outcome, ["z_abstracness"])
        base_result = fit_base_ols(model_df, formula)
        robust_result, robust_variant, robust_label = apply_primary_robust_cov(base_result, model_df)

        row: dict[str, Any] = {
            "roi_slug": roi_slug,
            "roi_label": roi_spec["roi_label"],
            "roi_description": roi_spec["roi_description"],
            "roi_pairs": "; ".join(roi_spec["pairs"]),
            "n_roi_pairs": len(roi_spec["pairs"]),
            "outcome_variable": outcome,
            "formula": formula,
            "robust_variant": robust_variant,
            "robust_se_label": robust_label,
            "n_trials": int(len(model_df)),
            "n_participants": int(model_df["participant_id"].nunique()),
            "n_questions": int(model_df["question_id"].nunique()),
            "mean_available_channels_per_trial": float(model_df[f"{roi_slug}_n_channels"].mean()),
            "min_available_channels_per_trial": int(model_df[f"{roi_slug}_n_channels"].min()),
            "max_available_channels_per_trial": int(model_df[f"{roi_slug}_n_channels"].max()),
        }
        add_term(row, robust_result, "abstractness", "z(Abstractness score)", "z_abstracness")
        add_term(row, robust_result, "global_block_order", "Global block order", "global_block_order")
        add_term(row, robust_result, "within_block_position", "Within-block trial position", "within_block_position")
        add_term(row, robust_result, "wordcount", "z(log total word count)", "z_log_total_word_count")
        add_term(row, robust_result, "correctness", "z(ENEM correctness)", "z_enem_correctness")
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("roi_slug").reset_index(drop=True)
    reject, holm_pvals, _, _ = multipletests(results_df["abstractness_p_value"].to_numpy(dtype=float), alpha=ALPHA, method="holm")
    results_df["holm_corrected_p_value"] = holm_pvals
    results_df["holm_reject_alpha_0_05"] = reject
    results_df["raw_p_rank"] = results_df["abstractness_p_value"].rank(method="dense").astype(int)
    return results_df


def build_result_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = results_df[
        [
            "roi_slug",
            "roi_label",
            "n_trials",
            "n_participants",
            "n_questions",
            "mean_available_channels_per_trial",
            "abstractness_beta",
            "abstractness_se",
            "abstractness_stat",
            "abstractness_p_value",
            "abstractness_ci_low",
            "abstractness_ci_high",
            "holm_corrected_p_value",
            "holm_reject_alpha_0_05",
        ]
    ].copy()
    return summary_df.sort_values("abstractness_p_value").reset_index(drop=True)


def plot_waveforms(waveform_rows: list[dict[str, Any]]) -> None:
    waveform_df = pd.DataFrame(waveform_rows)
    if waveform_df.empty:
        return

    grouped = (
        waveform_df.groupby(["participant_id", "session_id", "roi_slug", "roi_label", "condition"], sort=True)["waveform"]
        .apply(lambda items: np.vstack(items).mean(axis=0))
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, layout="constrained")
    for ax, roi_spec in zip(axes.flat, ROI_SPECS):
        for condition in ["Concrete", "Abstract"]:
            subset = grouped[(grouped["roi_slug"] == roi_spec["roi_slug"]) & (grouped["condition"] == condition)]
            if subset.empty:
                continue
            waves = np.vstack(subset["waveform"].to_list())
            mean_wave = waves.mean(axis=0)
            sem_wave = waves.std(axis=0, ddof=1) / math.sqrt(waves.shape[0]) if waves.shape[0] > 1 else np.zeros_like(mean_wave)
            color = CONDITION_COLORS[condition]
            ax.plot(WAVEFORM_GRID, mean_wave, color=color, label=condition)
            ax.fill_between(WAVEFORM_GRID, mean_wave - sem_wave, mean_wave + sem_wave, color=color, alpha=0.2)
        ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1)
        ax.axhline(0.0, color="#777777", linestyle=":", linewidth=1)
        ax.set_title(roi_spec["roi_label"])
        ax.set_xlabel("Seconds from question onset")
        ax.set_ylabel("Baseline-corrected HbO")
        ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIGURES_DIR / "step12_roi_waveforms.png", dpi=180)
    plt.close(fig)


def plot_roi_coefficients(results_df: pd.DataFrame) -> None:
    plot_df = results_df.sort_values("abstractness_p_value").reset_index(drop=True)
    effects = plot_df["abstractness_beta"].to_numpy(dtype=float) * 1e6
    ci_low = plot_df["abstractness_ci_low"].to_numpy(dtype=float) * 1e6
    ci_high = plot_df["abstractness_ci_high"].to_numpy(dtype=float) * 1e6
    y_positions = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    ax.errorbar(
        effects,
        y_positions,
        xerr=np.vstack([effects - ci_low, ci_high - effects]),
        fmt="o",
        color="#225588",
        ecolor="#225588",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions, plot_df["roi_label"].tolist())
    ax.set_xlabel(r"Abstractness coefficient ($\mu$M per 1 SD)")
    ax.set_title("Step 12 abstractness coefficients across the four ROIs")

    x_span = max(np.max(np.abs(ci_low)), np.max(np.abs(ci_high)), 1.0) * 1.1
    ax.set_xlim(min(ci_low.min(), 0.0) - 0.1 * x_span, max(ci_high.max(), 0.0) + 0.35 * x_span)
    for y_pos, (_, row) in enumerate(plot_df.iterrows()):
        label = f"raw p={float(row['abstractness_p_value']):.3f}, Holm p={float(row['holm_corrected_p_value']):.3f}"
        ax.text(ax.get_xlim()[1] - 0.02 * x_span, y_pos, label, va="center", ha="right", fontsize=9)

    fig.savefig(FIGURES_DIR / "step12_roi_coefficients.png", dpi=180)
    plt.close(fig)


def build_final_conclusion(results_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    sorted_df = results_df.sort_values("abstractness_p_value").reset_index(drop=True)
    best_row = sorted_df.iloc[0]
    significant_count = int(sorted_df["holm_reject_alpha_0_05"].sum())
    if significant_count > 0:
        conclusion_text = (
            f"Using the Step 10/11 trial table and the Step 11 continuous abstractness model separately within four new ROIs, "
            f"{significant_count} ROI test(s) survived Holm correction. The strongest ROI was {best_row['roi_label']} "
            f"(beta={float(best_row['abstractness_beta']):.3e}, SE={float(best_row['abstractness_se']):.3e}, "
            f"raw p={float(best_row['abstractness_p_value']):.6f}, Holm p={float(best_row['holm_corrected_p_value']):.6f})."
        )
    else:
        conclusion_text = (
            f"Using the Step 10/11 trial table and the Step 11 continuous abstractness model separately within four new ROIs, "
            f"none of the ROI-specific abstractness effects survived Holm correction across the four tests. "
            f"The smallest raw p-value was in {best_row['roi_label']} "
            f"(beta={float(best_row['abstractness_beta']):.3e}, SE={float(best_row['abstractness_se']):.3e}, "
            f"raw p={float(best_row['abstractness_p_value']):.6f}, Holm p={float(best_row['holm_corrected_p_value']):.6f})."
        )

    final_df = pd.DataFrame(
        [
            {
                "n_rois_tested": int(len(results_df)),
                "n_trials": int(results_df["n_trials"].max()),
                "n_participants": int(results_df["n_participants"].max()),
                "n_questions": int(results_df["n_questions"].max()),
                "multiple_testing_method": "Holm",
                "n_holm_significant_rois": significant_count,
                "best_raw_p_roi": best_row["roi_label"],
                "best_raw_p_value": float(best_row["abstractness_p_value"]),
                "best_holm_p_value": float(best_row["holm_corrected_p_value"]),
                "conclusion_text": conclusion_text,
            }
        ]
    )
    return final_df, conclusion_text


def write_supporting_tex(results_df: pd.DataFrame, session_channel_df: pd.DataFrame, conclusion_text: str) -> None:
    roi_rows = [
        [roi_spec["roi_label"], roi_spec["roi_description"], ", ".join(roi_spec["pairs"])]
        for roi_spec in ROI_SPECS
    ]
    (TABLES_DIR / "roi_definitions.tex").write_text(
        make_latex_grid(
            headers=["ROI", "Interpretive label", "HbO pairs"],
            rows=roi_rows,
            column_spec=r"p{0.18\linewidth}p{0.30\linewidth}p{0.42\linewidth}",
        )
        + "\n",
        encoding="utf-8",
    )

    result_rows = []
    for _, row in results_df.sort_values("abstractness_p_value").iterrows():
        result_rows.append(
            [
                row["roi_label"],
                int(row["n_trials"]),
                f"{float(row['mean_available_channels_per_trial']):.2f}",
                format_effect(row["abstractness_beta"]),
                format_effect(row["abstractness_se"]),
                format_p_value(row["abstractness_p_value"]),
                format_p_value(row["holm_corrected_p_value"]),
                f"[{format_effect(row['abstractness_ci_low'])}, {format_effect(row['abstractness_ci_high'])}]",
            ]
        )
    (TABLES_DIR / "roi_results.tex").write_text(
        make_latex_grid(
            headers=["ROI", "Trials", "Mean channels", "Beta", "SE", "Raw p", "Holm p", "95% CI"],
            rows=result_rows,
            column_spec=r"p{0.16\linewidth}ccccccp{0.18\linewidth}",
        )
        + "\n",
        encoding="utf-8",
    )

    channel_summary = (
        session_channel_df.groupby(["roi_slug", "roi_label"], as_index=False)
        .agg(
            mean_available_channels=("n_available_channels", "mean"),
            min_available_channels=("n_available_channels", "min"),
            max_available_channels=("n_available_channels", "max"),
        )
        .sort_values("roi_slug")
    )
    channel_rows = [
        [
            row["roi_label"],
            f"{float(row['mean_available_channels']):.2f}",
            int(row["min_available_channels"]),
            int(row["max_available_channels"]),
        ]
        for _, row in channel_summary.iterrows()
    ]
    (TABLES_DIR / "channel_summary.tex").write_text(
        make_latex_grid(
            headers=["ROI", "Mean channels", "Min", "Max"],
            rows=channel_rows,
            column_spec="lccc",
        )
        + "\n",
        encoding="utf-8",
    )

    (TEXT_DIR / "final_conclusion.tex").write_text(conclusion_text + "\n", encoding="utf-8")


def write_report() -> Path:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Step 12 Four-ROI Continuous Abstractness Analysis}",
        r"This side analysis reused the Step 10/11 cohort and trial table, recomputed the variable-window HbO outcome for four user-defined ROIs, and fit the Step 11 continuous abstractness model separately for each ROI. Holm correction was applied across the four primary ROI tests.",
        r"\subsection*{ROI Definitions}",
        r"\input{tables/roi_definitions.tex}",
        r"\input{tables/channel_summary.tex}",
        r"\subsection*{ROI Model Results}",
        r"\input{tables/roi_results.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.78\linewidth]{../../figures/step12/step12_roi_coefficients.png}",
        r"\caption{Continuous abstractness coefficients and 95\% confidence intervals for the four ROI-specific models.}",
        r"\end{figure}",
        r"\subsection*{Descriptive Waveforms}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.96\linewidth]{../../figures/step12/step12_roi_waveforms.png}",
        r"\caption{Onset-locked descriptive HbO waveforms for the four ROI definitions, averaged within participant-session and then across the Step 10/11 cohort.}",
        r"\end{figure}",
        r"\subsection*{Conclusion}",
        r"\input{text/final_conclusion.tex}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step12_multi_roi_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()
    step10_trial_df, step7_timing_df, status_df, question_df = load_inputs()
    base_trial_df = build_base_trial_table(step10_trial_df, step7_timing_df, question_df)
    wide_df, session_channel_df, waveform_rows, log_df = compute_roi_trial_outputs(base_trial_df, status_df)
    long_df = build_long_roi_table(wide_df)
    results_df = fit_roi_models(wide_df)
    summary_df = build_result_summary(results_df)
    final_df, conclusion_text = build_final_conclusion(results_df)

    save_dataframe(wide_df, CLEAN_DIR / "01_step12_trial_table_four_rois_wide.csv")
    save_dataframe(long_df, CLEAN_DIR / "02_step12_trial_table_four_rois_long.csv")
    save_dataframe(session_channel_df, CLEAN_DIR / "03_step12_session_channel_manifest.csv")
    save_dataframe(results_df, CLEAN_DIR / "04_step12_roi_model_results.csv")
    save_dataframe(summary_df, CLEAN_DIR / "05_step12_roi_result_summary.csv")
    save_dataframe(final_df, CLEAN_DIR / "06_step12_final_conclusion.csv")
    if not log_df.empty:
        save_dataframe(log_df, CLEAN_DIR / "07_step12_extraction_log.csv")

    plot_waveforms(waveform_rows)
    plot_roi_coefficients(results_df)
    write_supporting_tex(results_df, session_channel_df, conclusion_text)
    report_path = write_report()
    compile_report(report_path)


if __name__ == "__main__":
    main()
