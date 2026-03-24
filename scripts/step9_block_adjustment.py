#!/usr/bin/env python3

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from step6_covariate_adjusted import (
    coerce_bool,
    compile_report,
    format_effect,
    format_p_value,
    make_latex_grid,
    make_latex_table,
    sanitize_for_tex,
    save_dataframe,
)


ROOT = Path(__file__).resolve().parents[1]
STEP7_TRIAL_PATH = ROOT / "data_clean" / "step7" / "01_step7_trial_timing_table.csv"
STEP2_TRIAL_PATH = ROOT / "data_clean" / "step2" / "06_behavior_trial_clean.csv"
STEP6_TRIAL_PATH = ROOT / "data_clean" / "step6" / "03_step6_trial_model_table.csv"

CLEAN_DIR = ROOT / "data_clean" / "step9"
FIGURES_DIR = ROOT / "figures" / "step9"
REPORTS_DIR = ROOT / "reports" / "step9"
TABLES_DIR = REPORTS_DIR / "tables"
TEXT_DIR = REPORTS_DIR / "text"

ALPHA = 0.05
MODEL_OPTIMIZERS = ["lbfgs", "bfgs", "powell"]
CONDITION_ORDER = ["Concrete", "Abstract"]
CONDITION_COLORS = {"Abstract": "#AA3377", "Concrete": "#228833"}
ORDER_LABELS = {"ConcreteFirst": "Concrete First", "AbstractFirst": "Abstract First"}
BLOCK_HALF_LABELS = {0: "Early blocks (1-5)", 1: "Late blocks (6-10)"}


def ensure_directories() -> None:
    for path in [CLEAN_DIR, FIGURES_DIR, REPORTS_DIR, TABLES_DIR, TEXT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step7_trial_df = pd.read_csv(STEP7_TRIAL_PATH)
    step2_trial_df = pd.read_csv(STEP2_TRIAL_PATH)
    step6_trial_df = pd.read_csv(STEP6_TRIAL_PATH)
    if "included_primary_model" in step7_trial_df.columns:
        step7_trial_df["included_primary_model"] = step7_trial_df["included_primary_model"].map(coerce_bool)
    return step7_trial_df, step2_trial_df, step6_trial_df


def sem_or_nan(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 2:
        return math.nan
    return float(numeric.std(ddof=1) / math.sqrt(len(numeric)))


def primary_trial_columns() -> list[str]:
    return [
        "front_roi_mean_variable_lag4",
        "back_roi_mean_variable_lag4",
        "front_roi_mean",
        "global_block_order",
        "within_block_position",
        "condition_order_abstract_first",
        "block_half",
        "z_log_total_word_count",
        "z_enem_correctness",
        "condition_abstract",
        "response_time_sec",
    ]


def build_trial_block_table(
    step7_trial_df: pd.DataFrame,
    step2_trial_df: pd.DataFrame,
    step6_trial_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_df = step7_trial_df[step7_trial_df["included_primary_model"]].copy()
    trial_df = trial_df.sort_values(["participant_id", "session_id", "trial_idx_global"]).reset_index(drop=True)

    exclusion_rows: list[dict[str, Any]] = [
        {
            "stage": "cohort",
            "level": "info",
            "reason_code": "STEP7_COHORT_REUSED",
            "n_rows": int(trial_df[["participant_id", "session_id"]].drop_duplicates().shape[0]),
            "detail": "Step 9 reused the Step 7 included participant-session cohort without additional manual exclusions.",
        }
    ]

    step2_merge_df = step2_trial_df[
        [
            "participant_id",
            "block",
            "trial_idx_in_block",
            "question_id",
            "global_block_order",
            "order_condition",
            "trial_idx_global",
        ]
    ].copy()
    step6_merge_df = step6_trial_df[
        [
            "participant_id",
            "session_id",
            "block",
            "trial_idx_in_block",
            "question_id",
            "front_roi_mean",
        ]
    ].copy()

    trial_df = trial_df.merge(
        step2_merge_df,
        on=["participant_id", "block", "trial_idx_in_block", "question_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_step2"),
        indicator="step2_merge_status",
    )
    missing_step2 = int((trial_df["step2_merge_status"] != "both").sum())
    exclusion_rows.append(
        {
            "stage": "merge_step2",
            "level": "warning" if missing_step2 else "info",
            "reason_code": "STEP2_BLOCK_METADATA_MERGE",
            "n_rows": missing_step2,
            "detail": "Merged realized block order and condition-order metadata from the cleaned Step 2 behavioral table.",
        }
    )
    trial_df.drop(columns=["step2_merge_status"], inplace=True)

    trial_df = trial_df.merge(
        step6_merge_df,
        on=["participant_id", "session_id", "block", "trial_idx_in_block", "question_id"],
        how="left",
        validate="one_to_one",
        indicator="step6_merge_status",
    )
    missing_step6 = int((trial_df["step6_merge_status"] != "both").sum())
    exclusion_rows.append(
        {
            "stage": "merge_step6",
            "level": "warning" if missing_step6 else "info",
            "reason_code": "STEP6_FIXED_WINDOW_MERGE",
            "n_rows": missing_step6,
            "detail": "Merged the fixed-window front ROI outcome for the Step 9 benchmark sensitivity model.",
        }
    )
    trial_df.drop(columns=["step6_merge_status"], inplace=True)

    if "trial_idx_global_step2" in trial_df.columns:
        trial_df["trial_idx_global_step2"] = pd.to_numeric(trial_df["trial_idx_global_step2"], errors="coerce")
        mismatch = int(
            (
                trial_df["trial_idx_global_step2"].notna()
                & trial_df["trial_idx_global"].notna()
                & (pd.to_numeric(trial_df["trial_idx_global"], errors="coerce") != trial_df["trial_idx_global_step2"])
            ).sum()
        )
        exclusion_rows.append(
            {
                "stage": "qc",
                "level": "warning" if mismatch else "info",
                "reason_code": "TRIAL_INDEX_CHECK",
                "n_rows": mismatch,
                "detail": "Checked that the cumulative trial index agreed between Step 7 and Step 2.",
            }
        )

    trial_df["global_block_order"] = pd.to_numeric(trial_df["global_block_order"], errors="coerce")
    trial_df["within_block_position"] = pd.to_numeric(trial_df["trial_idx_in_block"], errors="coerce")
    trial_df["condition_order_abstract_first"] = (
        trial_df["order_condition"].astype(str).str.strip().map({"AbstractFirst": 1.0, "ConcreteFirst": 0.0})
    )
    trial_df["block_half"] = np.where(trial_df["global_block_order"] >= 6, 1.0, 0.0)
    trial_df["block_half_label"] = trial_df["block_half"].map({0.0: BLOCK_HALF_LABELS[0], 1.0: BLOCK_HALF_LABELS[1]})
    trial_df["cumulative_trial_order"] = pd.to_numeric(trial_df["trial_idx_global"], errors="coerce")
    trial_df["order_condition_label"] = trial_df["order_condition"].map(ORDER_LABELS).fillna(trial_df["order_condition"])

    required_cols = primary_trial_columns()
    trial_df["included_step9_model"] = trial_df[required_cols].notna().all(axis=1)
    excluded_n = int((~trial_df["included_step9_model"]).sum())
    exclusion_rows.append(
        {
            "stage": "qc",
            "level": "warning" if excluded_n else "info",
            "reason_code": "STEP9_MODEL_ELIGIBILITY",
            "n_rows": excluded_n,
            "detail": "Checked that the Step 9 trial table contained all required block variables, covariates, and benchmark outcomes.",
        }
    )

    order_variation = int(trial_df["condition_order_abstract_first"].dropna().nunique())
    exclusion_rows.append(
        {
            "stage": "qc",
            "level": "info" if order_variation > 1 else "warning",
            "reason_code": "CONDITION_ORDER_VARIATION",
            "n_rows": order_variation,
            "detail": "The participant-level condition order was informative and retained in the Step 9 models."
            if order_variation > 1
            else "Condition order was non-informative; it would need to be dropped from inference.",
        }
    )

    ordered_cols = [
        "participant_id",
        "session_id",
        "block",
        "global_block_order",
        "block_half",
        "block_half_label",
        "trial_idx_in_block",
        "within_block_position",
        "trial_idx_global",
        "cumulative_trial_order",
        "question_id",
        "condition",
        "condition_abstract",
        "order_condition",
        "order_condition_label",
        "condition_order_abstract_first",
        "response_time_sec",
        "front_roi_mean_variable_lag4",
        "back_roi_mean_variable_lag4",
        "front_roi_mean",
        "z_log_total_word_count",
        "z_enem_correctness",
        "included_step9_model",
    ]
    remaining_cols = [col for col in trial_df.columns if col not in ordered_cols]
    trial_df = trial_df[ordered_cols + remaining_cols]
    exclusion_log_df = pd.DataFrame(exclusion_rows)
    return trial_df, exclusion_log_df


def block_summary_frame(
    trial_df: pd.DataFrame,
    group_cols: list[str],
    summary_type: str,
) -> pd.DataFrame:
    summary_df = (
        trial_df.groupby(group_cols + ["condition"], dropna=False)
        .agg(
            n_trials=("question_id", "size"),
            n_participants=("participant_id", "nunique"),
            mean_front_roi_variable=("front_roi_mean_variable_lag4", "mean"),
            sem_front_roi_variable=("front_roi_mean_variable_lag4", sem_or_nan),
            mean_front_roi_fixed=("front_roi_mean", "mean"),
            sem_front_roi_fixed=("front_roi_mean", sem_or_nan),
            mean_response_time_sec=("response_time_sec", "mean"),
            sem_response_time_sec=("response_time_sec", sem_or_nan),
        )
        .reset_index()
    )
    summary_df["summary_type"] = summary_type
    return summary_df


def build_block_descriptive_summary(trial_df: pd.DataFrame) -> pd.DataFrame:
    model_df = trial_df[trial_df["included_step9_model"]].copy()
    summaries = [
        block_summary_frame(model_df, ["global_block_order"], "global_block_order"),
        block_summary_frame(model_df, ["within_block_position"], "within_block_position"),
        block_summary_frame(model_df, ["block_half", "block_half_label"], "block_half"),
        block_summary_frame(model_df, ["order_condition", "order_condition_label"], "condition_order"),
    ]
    descriptive_df = pd.concat(summaries, ignore_index=True, sort=False)
    sort_cols = [col for col in ["summary_type", "global_block_order", "within_block_position", "block_half", "order_condition", "condition"] if col in descriptive_df.columns]
    descriptive_df = descriptive_df.sort_values(sort_cols).reset_index(drop=True)
    return descriptive_df


def build_formula(terms: list[str], outcome: str) -> str:
    return outcome + " ~ " + " + ".join(terms)


def fit_mixed_model_formula(formula: str, model_df: pd.DataFrame) -> tuple[Any | None, str | None, list[str], str | None]:
    collected_warnings: list[str] = []
    best_result = None
    best_method = None
    last_error: str | None = None

    for method in MODEL_OPTIMIZERS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = smf.mixedlm(
                    formula,
                    data=model_df,
                    groups=model_df["participant_code"],
                    vc_formula={"question": "0 + C(question_code)"},
                    re_formula="1",
                )
                result = model.fit(reml=False, method=method, maxiter=2000, disp=False)
                warning_messages = [str(item.message) for item in caught]
                collected_warnings.extend(f"{method}: {message}" for message in warning_messages)
                if result.converged and np.isfinite(result.fe_params).all():
                    return result, method, collected_warnings, None
                best_result = result
                best_method = method
            except Exception as exc:
                last_error = f"{method}: {exc}"

    if best_result is not None and getattr(best_result, "converged", False):
        return best_result, best_method, collected_warnings, None
    return None, None, collected_warnings, last_error


def fit_fixed_effect_fallback_formula(formula: str, model_df: pd.DataFrame) -> Any:
    fallback_formula = formula + " + C(participant_code) + C(question_code)"
    return smf.ols(fallback_formula, data=model_df).fit()


def term_stats(result: Any, term_name: str) -> dict[str, Any]:
    if term_name not in result.params.index:
        return {
            "term": term_name,
            "beta": math.nan,
            "se": math.nan,
            "stat": math.nan,
            "p_value": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
        }

    beta = float(result.params[term_name])
    se = float(result.bse[term_name])
    stat = beta / se if se else math.nan
    ci_low, ci_high = [float(item) for item in result.conf_int().loc[term_name].tolist()]
    return {
        "term": term_name,
        "beta": beta,
        "se": se,
        "stat": stat,
        "p_value": float(result.pvalues[term_name]),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def add_term_to_row(row: dict[str, Any], prefix: str, label: str, result: Any, term_name: str) -> None:
    stats = term_stats(result, term_name)
    row[f"{prefix}_term"] = term_name
    row[f"{prefix}_label"] = label
    for key, value in stats.items():
        if key == "term":
            continue
        row[f"{prefix}_{key}"] = value


def extract_model_row(
    result: Any,
    model_df: pd.DataFrame,
    *,
    model_id: str,
    analysis_label: str,
    analysis_scope: str,
    outcome: str,
    outcome_label: str,
    formula: str,
    model_type: str,
    optimizer: str | None,
    warnings_list: list[str],
    fallback_used: bool,
    target_prefix: str,
    target_label: str,
    target_term: str,
    term_map: list[tuple[str, str, str]],
) -> dict[str, Any]:
    row = {
        "model_id": model_id,
        "analysis_label": analysis_label,
        "analysis_scope": analysis_scope,
        "outcome_variable": outcome,
        "outcome_label": outcome_label,
        "formula": formula,
        "model_type": model_type,
        "fallback_used": fallback_used,
        "converged": True,
        "optimizer": optimizer or "",
        "n_trials": int(len(model_df)),
        "n_participants": int(model_df["participant_id"].nunique()),
        "n_questions": int(model_df["question_id"].nunique()),
        "residual_variance": float(getattr(result, "scale", math.nan)),
        "participant_random_intercept_variance": math.nan,
        "question_random_intercept_variance": math.nan,
        "warning_count": len(warnings_list),
        "warnings": " | ".join(warnings_list),
    }

    if model_type == "MixedLM":
        if hasattr(result, "cov_re") and getattr(result.cov_re, "shape", (0, 0))[0] > 0:
            row["participant_random_intercept_variance"] = float(result.cov_re.iloc[0, 0])
        if hasattr(result, "vcomp") and len(result.vcomp):
            row["question_random_intercept_variance"] = float(result.vcomp[0])

    for prefix, term_name, label in term_map:
        add_term_to_row(row, prefix, label, result, term_name)

    if target_prefix in row:
        pass
    target_stats = term_stats(result, target_term)
    row["target_term"] = target_term
    row["target_label"] = target_label
    for key, value in target_stats.items():
        if key == "term":
            continue
        row[f"target_{key}"] = value
    return row


def prepare_model_data(trial_df: pd.DataFrame, outcome: str, covariate_cols: list[str]) -> pd.DataFrame:
    model_df = trial_df[trial_df["included_step9_model"]].copy()
    numeric_cols = [
        "participant_code",
        "question_code",
        "condition_abstract",
        "global_block_order",
        "within_block_position",
        "condition_order_abstract_first",
        "block_half",
        "z_log_total_word_count",
        "z_enem_correctness",
        outcome,
        *covariate_cols,
    ]
    for col in dict.fromkeys(numeric_cols):
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna(subset=list(dict.fromkeys(numeric_cols))).reset_index(drop=True)
    model_df["participant_code"] = model_df["participant_code"].astype(int)
    model_df["question_code"] = model_df["question_code"].astype(int)
    model_df["participant_id"] = model_df["participant_id"].astype(str)
    model_df["question_id"] = model_df["question_id"].astype(str)
    return model_df


def run_models(trial_df: pd.DataFrame, exclusion_log_df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame]:
    model_specs = [
        {
            "output_name": "03_step9_primary_front_block_adjusted_model.csv",
            "model_id": "primary_front_block_adjusted",
            "analysis_label": "Primary front ROI model adjusted for block structure and item covariates",
            "analysis_scope": "primary",
            "outcome": "front_roi_mean_variable_lag4",
            "outcome_label": "Front ROI HbO variable window",
            "terms": [
                "condition_abstract",
                "global_block_order",
                "within_block_position",
                "condition_order_abstract_first",
                "z_log_total_word_count",
                "z_enem_correctness",
            ],
            "target_term": "condition_abstract",
            "target_prefix": "condition",
            "target_label": "Adjusted abstract vs concrete effect",
            "term_map": [
                ("condition", "condition_abstract", "Adjusted abstract vs concrete effect"),
                ("global_block_order", "global_block_order", "Global block order"),
                ("within_block_position", "within_block_position", "Within-block trial position"),
                ("condition_order", "condition_order_abstract_first", "Abstract-first order"),
                ("wordcount", "z_log_total_word_count", "z(log total word count)"),
                ("correctness", "z_enem_correctness", "z(ENEM correctness)"),
            ],
        },
        {
            "output_name": "04_step9_condition_by_blockorder_model.csv",
            "model_id": "condition_by_block_order",
            "analysis_label": "Condition by global block order interaction model",
            "analysis_scope": "secondary_interaction",
            "outcome": "front_roi_mean_variable_lag4",
            "outcome_label": "Front ROI HbO variable window",
            "terms": [
                "condition_abstract",
                "global_block_order",
                "condition_abstract:global_block_order",
                "within_block_position",
                "condition_order_abstract_first",
                "z_log_total_word_count",
                "z_enem_correctness",
            ],
            "target_term": "condition_abstract:global_block_order",
            "target_prefix": "condition_x_block_order",
            "target_label": "Condition x global block order",
            "term_map": [
                ("condition", "condition_abstract", "Condition main effect"),
                ("global_block_order", "global_block_order", "Global block order"),
                ("condition_x_block_order", "condition_abstract:global_block_order", "Condition x global block order"),
                ("within_block_position", "within_block_position", "Within-block trial position"),
                ("condition_order", "condition_order_abstract_first", "Abstract-first order"),
                ("wordcount", "z_log_total_word_count", "z(log total word count)"),
                ("correctness", "z_enem_correctness", "z(ENEM correctness)"),
            ],
        },
        {
            "output_name": "05_step9_condition_by_position_model.csv",
            "model_id": "condition_by_position",
            "analysis_label": "Condition by within-block position interaction model",
            "analysis_scope": "secondary_interaction",
            "outcome": "front_roi_mean_variable_lag4",
            "outcome_label": "Front ROI HbO variable window",
            "terms": [
                "condition_abstract",
                "within_block_position",
                "condition_abstract:within_block_position",
                "global_block_order",
                "condition_order_abstract_first",
                "z_log_total_word_count",
                "z_enem_correctness",
            ],
            "target_term": "condition_abstract:within_block_position",
            "target_prefix": "condition_x_position",
            "target_label": "Condition x within-block position",
            "term_map": [
                ("condition", "condition_abstract", "Condition main effect"),
                ("within_block_position", "within_block_position", "Within-block trial position"),
                ("condition_x_position", "condition_abstract:within_block_position", "Condition x within-block position"),
                ("global_block_order", "global_block_order", "Global block order"),
                ("condition_order", "condition_order_abstract_first", "Abstract-first order"),
                ("wordcount", "z_log_total_word_count", "z(log total word count)"),
                ("correctness", "z_enem_correctness", "z(ENEM correctness)"),
            ],
        },
        {
            "output_name": "06_step9_condition_by_ordercondition_model.csv",
            "model_id": "condition_by_order_condition",
            "analysis_label": "Condition by condition-order interaction model",
            "analysis_scope": "secondary_interaction",
            "outcome": "front_roi_mean_variable_lag4",
            "outcome_label": "Front ROI HbO variable window",
            "terms": [
                "condition_abstract",
                "condition_order_abstract_first",
                "condition_abstract:condition_order_abstract_first",
                "global_block_order",
                "within_block_position",
                "z_log_total_word_count",
                "z_enem_correctness",
            ],
            "target_term": "condition_abstract:condition_order_abstract_first",
            "target_prefix": "condition_x_order",
            "target_label": "Condition x abstract-first order",
            "term_map": [
                ("condition", "condition_abstract", "Condition main effect"),
                ("condition_order", "condition_order_abstract_first", "Abstract-first order"),
                ("condition_x_order", "condition_abstract:condition_order_abstract_first", "Condition x abstract-first order"),
                ("global_block_order", "global_block_order", "Global block order"),
                ("within_block_position", "within_block_position", "Within-block trial position"),
                ("wordcount", "z_log_total_word_count", "z(log total word count)"),
                ("correctness", "z_enem_correctness", "z(ENEM correctness)"),
            ],
        },
        {
            "output_name": "07_step9_early_late_sensitivity_model.csv",
            "model_id": "early_vs_late_sensitivity",
            "analysis_label": "Early versus late block-half interaction model",
            "analysis_scope": "sensitivity",
            "outcome": "front_roi_mean_variable_lag4",
            "outcome_label": "Front ROI HbO variable window",
            "terms": [
                "condition_abstract",
                "block_half",
                "condition_abstract:block_half",
                "z_log_total_word_count",
                "z_enem_correctness",
            ],
            "target_term": "condition_abstract:block_half",
            "target_prefix": "condition_x_block_half",
            "target_label": "Condition x late-half",
            "term_map": [
                ("condition", "condition_abstract", "Condition main effect"),
                ("block_half", "block_half", "Late-session half"),
                ("condition_x_block_half", "condition_abstract:block_half", "Condition x late-session half"),
                ("wordcount", "z_log_total_word_count", "z(log total word count)"),
                ("correctness", "z_enem_correctness", "z(ENEM correctness)"),
            ],
        },
        {
            "output_name": "08_step9_back_block_adjusted_model.csv",
            "model_id": "back_block_adjusted",
            "analysis_label": "Secondary back ROI block-adjusted model",
            "analysis_scope": "secondary_benchmark",
            "outcome": "back_roi_mean_variable_lag4",
            "outcome_label": "Back ROI HbO variable window",
            "terms": [
                "condition_abstract",
                "global_block_order",
                "within_block_position",
                "condition_order_abstract_first",
                "z_log_total_word_count",
                "z_enem_correctness",
            ],
            "target_term": "condition_abstract",
            "target_prefix": "condition",
            "target_label": "Adjusted abstract vs concrete effect",
            "term_map": [
                ("condition", "condition_abstract", "Adjusted abstract vs concrete effect"),
                ("global_block_order", "global_block_order", "Global block order"),
                ("within_block_position", "within_block_position", "Within-block trial position"),
                ("condition_order", "condition_order_abstract_first", "Abstract-first order"),
                ("wordcount", "z_log_total_word_count", "z(log total word count)"),
                ("correctness", "z_enem_correctness", "z(ENEM correctness)"),
            ],
        },
        {
            "output_name": "09_step9_fixed_window_block_benchmark.csv",
            "model_id": "fixed_window_benchmark",
            "analysis_label": "Fixed-window front ROI benchmark model adjusted for block structure",
            "analysis_scope": "sensitivity_benchmark",
            "outcome": "front_roi_mean",
            "outcome_label": "Front ROI HbO fixed 7-11 s window",
            "terms": [
                "condition_abstract",
                "global_block_order",
                "within_block_position",
                "condition_order_abstract_first",
                "z_log_total_word_count",
                "z_enem_correctness",
            ],
            "target_term": "condition_abstract",
            "target_prefix": "condition",
            "target_label": "Adjusted abstract vs concrete effect",
            "term_map": [
                ("condition", "condition_abstract", "Adjusted abstract vs concrete effect"),
                ("global_block_order", "global_block_order", "Global block order"),
                ("within_block_position", "within_block_position", "Within-block trial position"),
                ("condition_order", "condition_order_abstract_first", "Abstract-first order"),
                ("wordcount", "z_log_total_word_count", "z(log total word count)"),
                ("correctness", "z_enem_correctness", "z(ENEM correctness)"),
            ],
        },
    ]

    outputs: dict[str, pd.DataFrame] = {}
    fit_artifacts: dict[str, Any] = {}
    log_rows = exclusion_log_df.to_dict("records")

    for spec in model_specs:
        rhs_terms = spec["terms"]
        formula = build_formula(rhs_terms, spec["outcome"])
        required_cols = list(
            dict.fromkeys(
                [
                    spec["outcome"],
                    "condition_abstract",
                    "global_block_order",
                    "within_block_position",
                    "condition_order_abstract_first",
                    "block_half",
                    "z_log_total_word_count",
                    "z_enem_correctness",
                ]
            )
        )
        model_df = prepare_model_data(trial_df, spec["outcome"], required_cols)
        mixed_result, optimizer, warnings_list, fit_error = fit_mixed_model_formula(formula, model_df)
        if mixed_result is not None and getattr(mixed_result, "converged", False):
            result = mixed_result
            model_type = "MixedLM"
            fallback_used = False
        else:
            result = fit_fixed_effect_fallback_formula(formula, model_df)
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
            formula=formula,
            model_type=model_type,
            optimizer=optimizer,
            warnings_list=warnings_list,
            fallback_used=fallback_used,
            target_prefix=spec["target_prefix"],
            target_label=spec["target_label"],
            target_term=spec["target_term"],
            term_map=spec["term_map"],
        )

        outputs[spec["output_name"]] = pd.DataFrame([row])
        fit_artifacts[spec["model_id"]] = {"result": result, "data": model_df, "row": row, "formula": formula}
        log_rows.append(
            {
                "stage": "model_fit",
                "level": "warning" if warnings_list or fallback_used else "info",
                "reason_code": f"MODEL_{spec['model_id'].upper()}",
                "n_rows": 1,
                "detail": f"Model type={model_type}; optimizer={optimizer or 'n/a'}; fallback_used={fallback_used}; warnings={' | '.join(warnings_list) if warnings_list else 'none'}",
            }
        )

    updated_log_df = pd.DataFrame(log_rows)
    return outputs, fit_artifacts, updated_log_df


def plot_line_summary(
    summary_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    yerr_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    for condition in CONDITION_ORDER:
        subset = summary_df[summary_df["condition"] == condition].sort_values(x_col)
        ax.errorbar(
            subset[x_col],
            subset[y_col] * 1e6 if "roi" in y_col else subset[y_col],
            yerr=subset[yerr_col] * 1e6 if "roi" in y_col else subset[yerr_col],
            marker="o",
            linewidth=2,
            capsize=3,
            color=CONDITION_COLORS[condition],
            label=condition,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_early_late_condition(trial_df: pd.DataFrame, path: Path) -> None:
    summary_df = block_summary_frame(trial_df, ["block_half", "block_half_label"], "block_half")
    order_labels = [BLOCK_HALF_LABELS[0], BLOCK_HALF_LABELS[1]]
    x_positions = np.arange(len(order_labels), dtype=float)
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    for offset, condition in zip([-width / 2, width / 2], CONDITION_ORDER):
        subset = summary_df[summary_df["condition"] == condition].copy()
        subset = subset.set_index("block_half_label").reindex(order_labels).reset_index()
        ax.bar(
            x_positions + offset,
            subset["mean_front_roi_variable"] * 1e6,
            width=width,
            color=CONDITION_COLORS[condition],
            alpha=0.65,
            label=condition,
            yerr=subset["sem_front_roi_variable"] * 1e6,
            capsize=4,
        )

    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xticks(x_positions, order_labels)
    ax.set_ylabel(r"Mean front ROI HbO ($\mu$M)")
    ax.set_xlabel("Session half")
    ax.set_title("Early versus late blocks by condition")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_primary_coefficients(primary_row: pd.Series, path: Path) -> None:
    labels = [
        "Condition",
        "Global block",
        "Within block",
        "Abstract first",
        "z(log word count)",
        "z(correctness)",
    ]
    prefixes = [
        "condition",
        "global_block_order",
        "within_block_position",
        "condition_order",
        "wordcount",
        "correctness",
    ]

    effects = [float(primary_row[f"{prefix}_beta"]) * 1e6 for prefix in prefixes]
    ci_low = [float(primary_row[f"{prefix}_ci_low"]) * 1e6 for prefix in prefixes]
    ci_high = [float(primary_row[f"{prefix}_ci_high"]) * 1e6 for prefix in prefixes]
    y_positions = np.arange(len(prefixes))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    ax.errorbar(
        effects,
        y_positions,
        xerr=np.vstack([np.array(effects) - np.array(ci_low), np.array(ci_high) - np.array(effects)]),
        fmt="o",
        color="#336699",
        ecolor="#336699",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel(r"Coefficient estimate ($\mu$M)")
    ax.set_title("Primary block-adjusted fixed effects")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_condition_by_block_order_interaction(
    descriptive_df: pd.DataFrame,
    interaction_row: pd.Series,
    path: Path,
) -> None:
    block_df = descriptive_df[descriptive_df["summary_type"] == "global_block_order"].copy()
    pivot_df = block_df.pivot(index="global_block_order", columns="condition", values="mean_front_roi_variable").reset_index()
    pivot_df["difference"] = pivot_df["Abstract"] - pivot_df["Concrete"]

    x = pivot_df["global_block_order"].to_numpy(dtype=float)
    observed = pivot_df["difference"].to_numpy(dtype=float) * 1e6
    intercept = float(interaction_row["condition_beta"])
    slope = float(interaction_row["condition_x_block_order_beta"])
    fitted = (intercept + slope * x) * 1e6

    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    ax.plot(x, observed, marker="o", color="#AA3377", linewidth=2, label="Observed Abstract - Concrete")
    ax.plot(x, fitted, linestyle="--", color="#225588", linewidth=2, label="Interaction-model trend")
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel("Global block order")
    ax.set_ylabel(r"Condition difference in front ROI HbO ($\mu$M)")
    ax.set_title("Condition difference across global block order")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_condition_by_order_condition(descriptive_df: pd.DataFrame, path: Path) -> None:
    order_df = descriptive_df[descriptive_df["summary_type"] == "condition_order"].copy()
    order_labels = [ORDER_LABELS["ConcreteFirst"], ORDER_LABELS["AbstractFirst"]]
    x_positions = np.arange(len(order_labels), dtype=float)
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    for offset, condition in zip([-width / 2, width / 2], CONDITION_ORDER):
        subset = order_df[order_df["condition"] == condition].copy()
        subset = subset.set_index("order_condition_label").reindex(order_labels).reset_index()
        ax.bar(
            x_positions + offset,
            subset["mean_front_roi_variable"] * 1e6,
            width=width,
            color=CONDITION_COLORS[condition],
            alpha=0.65,
            label=condition,
            yerr=subset["sem_front_roi_variable"] * 1e6,
            capsize=4,
        )

    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xticks(x_positions, order_labels)
    ax.set_ylabel(r"Mean front ROI HbO ($\mu$M)")
    ax.set_xlabel("Participant condition order")
    ax.set_title("Front ROI response by participant condition order")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_fixed_vs_variable_benchmark(variable_row: pd.Series, fixed_row: pd.Series, path: Path) -> None:
    labels = ["Duration-aware", "Fixed 7-11 s"]
    effects = np.array([float(variable_row["condition_beta"]), float(fixed_row["condition_beta"])]) * 1e6
    ci_low = np.array([float(variable_row["condition_ci_low"]), float(fixed_row["condition_ci_low"])]) * 1e6
    ci_high = np.array([float(variable_row["condition_ci_high"]), float(fixed_row["condition_ci_high"])]) * 1e6
    y_positions = np.arange(2)

    fig, ax = plt.subplots(figsize=(7.0, 3.8), layout="constrained")
    ax.errorbar(
        effects,
        y_positions,
        xerr=np.vstack([effects - ci_low, ci_high - effects]),
        fmt="o",
        color="#336699",
        ecolor="#336699",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel(r"Adjusted condition coefficient ($\mu$M)")
    ax.set_title("Variable-window versus fixed-window block-adjusted benchmark")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_figures(
    descriptive_df: pd.DataFrame,
    fit_artifacts: dict[str, Any],
) -> None:
    block_df = descriptive_df[descriptive_df["summary_type"] == "global_block_order"].copy()
    within_df = descriptive_df[descriptive_df["summary_type"] == "within_block_position"].copy()
    trial_df = fit_artifacts["primary_front_block_adjusted"]["data"]

    plot_line_summary(
        block_df,
        x_col="global_block_order",
        y_col="mean_front_roi_variable",
        yerr_col="sem_front_roi_variable",
        xlabel="Global block order",
        ylabel=r"Mean front ROI HbO ($\mu$M)",
        title="Front ROI response by global block order",
        path=FIGURES_DIR / "front_roi_by_global_block_order.png",
    )
    plot_line_summary(
        within_df,
        x_col="within_block_position",
        y_col="mean_front_roi_variable",
        yerr_col="sem_front_roi_variable",
        xlabel="Within-block trial position",
        ylabel=r"Mean front ROI HbO ($\mu$M)",
        title="Front ROI response by within-block trial position",
        path=FIGURES_DIR / "front_roi_by_withinblock_position.png",
    )
    plot_line_summary(
        block_df,
        x_col="global_block_order",
        y_col="mean_response_time_sec",
        yerr_col="sem_response_time_sec",
        xlabel="Global block order",
        ylabel="Mean response time (s)",
        title="Response time by global block order",
        path=FIGURES_DIR / "response_time_by_global_block_order.png",
    )
    plot_early_late_condition(trial_df, FIGURES_DIR / "early_vs_late_condition_plot.png")
    plot_primary_coefficients(
        pd.Series(fit_artifacts["primary_front_block_adjusted"]["row"]),
        FIGURES_DIR / "primary_block_adjusted_condition_effect.png",
    )
    plot_condition_by_block_order_interaction(
        descriptive_df,
        pd.Series(fit_artifacts["condition_by_block_order"]["row"]),
        FIGURES_DIR / "condition_by_blockorder_interaction_plot.png",
    )
    plot_condition_by_order_condition(
        descriptive_df,
        FIGURES_DIR / "condition_by_ordercondition_plot.png",
    )
    plot_fixed_vs_variable_benchmark(
        pd.Series(fit_artifacts["primary_front_block_adjusted"]["row"]),
        pd.Series(fit_artifacts["fixed_window_benchmark"]["row"]),
        FIGURES_DIR / "fixed_vs_variable_block_benchmark.png",
    )


def build_final_conclusion(
    primary_df: pd.DataFrame,
    interaction_dfs: list[pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    primary = primary_df.iloc[0]
    interaction_rows = [df.iloc[0] for df in interaction_dfs]

    significant_interactions = [
        row["analysis_label"]
        for row in interaction_rows
        if pd.notna(row["target_p_value"]) and float(row["target_p_value"]) < ALPHA
    ]

    beta = float(primary["condition_beta"])
    se = float(primary["condition_se"])
    p_value = float(primary["condition_p_value"])
    ci_low = float(primary["condition_ci_low"])
    ci_high = float(primary["condition_ci_high"])

    if p_value < ALPHA:
        conclusion_text = (
            "In the block-adjusted mixed-effects analysis, the abstract-versus-concrete effect in the left frontal ROI remained after controlling for global block order, within-block trial position, participant condition order, question length, and item difficulty "
            f"(beta={beta:.3e}, SE={se:.3e}, p={p_value:.6f}, 95% CI [{ci_low:.3e}, {ci_high:.3e}]). "
            "This indicates that the frontal condition effect persists after explicit adjustment for the blocked task structure."
        )
    else:
        conclusion_text = (
            "In the block-adjusted mixed-effects analysis, the abstract-versus-concrete effect in the left frontal ROI was reduced and was not statistically significant after controlling for global block order, within-block trial position, participant condition order, question length, and item difficulty "
            f"(beta={beta:.3e}, SE={se:.3e}, p={p_value:.6f}, 95% CI [{ci_low:.3e}, {ci_high:.3e}]). "
            "This suggests that the residual frontal tendency does not survive clear statistical support once block structure is modeled explicitly."
        )

    if significant_interactions:
        conclusion_text += " A secondary interaction term was significant, indicating that the condition effect varies across the blocked structure of the task."
    else:
        conclusion_text += " None of the predeclared condition-by-block interaction terms reached the alpha=0.05 threshold."

    final_df = pd.DataFrame(
        [
            {
                "n_included_participants": int(primary["n_participants"]),
                "n_included_questions": int(primary["n_questions"]),
                "n_included_trials": int(primary["n_trials"]),
                "condition_beta": beta,
                "condition_se": se,
                "condition_p_value": p_value,
                "condition_ci_low": ci_low,
                "condition_ci_high": ci_high,
                "significant_interactions": "; ".join(significant_interactions),
                "conclusion_text": conclusion_text,
            }
        ]
    )
    return final_df, conclusion_text


def write_supporting_tex(
    primary_df: pd.DataFrame,
    model_outputs: dict[str, pd.DataFrame],
    descriptive_df: pd.DataFrame,
    conclusion_text: str,
) -> None:
    primary = primary_df.iloc[0]
    primary_rows = [
        ("Included participants", int(primary["n_participants"])),
        ("Included questions", int(primary["n_questions"])),
        ("Included trials", int(primary["n_trials"])),
        ("Condition beta", format_effect(primary["condition_beta"])),
        ("Condition SE", format_effect(primary["condition_se"])),
        ("Condition z statistic", f"{float(primary['condition_stat']):.3f}"),
        ("Condition p-value", format_p_value(primary["condition_p_value"])),
        ("Condition 95% CI", f"[{format_effect(primary['condition_ci_low'])}, {format_effect(primary['condition_ci_high'])}]"),
        ("Global block-order beta", format_effect(primary["global_block_order_beta"])),
        ("Within-block-position beta", format_effect(primary["within_block_position_beta"])),
        ("Condition-order beta", format_effect(primary["condition_order_beta"])),
        ("Word-count beta", format_effect(primary["wordcount_beta"])),
        ("ENEM-correctness beta", format_effect(primary["correctness_beta"])),
        ("Participant random-intercept variance", format_effect(primary["participant_random_intercept_variance"])),
        ("Question random-intercept variance", format_effect(primary["question_random_intercept_variance"])),
    ]
    (TABLES_DIR / "primary_block_model_table.tex").write_text(
        make_latex_table(primary_rows) + "\n",
        encoding="utf-8",
    )

    interaction_table_rows: list[list[Any]] = []
    interaction_files = [
        "04_step9_condition_by_blockorder_model.csv",
        "05_step9_condition_by_position_model.csv",
        "06_step9_condition_by_ordercondition_model.csv",
        "07_step9_early_late_sensitivity_model.csv",
        "08_step9_back_block_adjusted_model.csv",
        "09_step9_fixed_window_block_benchmark.csv",
    ]
    for file_name in interaction_files:
        row = model_outputs[file_name].iloc[0]
        interaction_table_rows.append(
            [
                row["analysis_label"],
                row["target_label"],
                format_effect(row["target_beta"]),
                format_effect(row["target_se"]),
                format_p_value(row["target_p_value"]),
                f"[{format_effect(row['target_ci_low'])}, {format_effect(row['target_ci_high'])}]",
            ]
        )
    (TABLES_DIR / "interaction_models_table.tex").write_text(
        make_latex_grid(
            headers=["Model", "Target term", "Beta", "SE", "p", "95% CI"],
            rows=interaction_table_rows,
            column_spec=r"p{0.28\linewidth}p{0.22\linewidth}p{0.12\linewidth}p{0.10\linewidth}p{0.10\linewidth}p{0.14\linewidth}",
        )
        + "\n",
        encoding="utf-8",
    )

    block_counts = (
        descriptive_df[descriptive_df["summary_type"] == "global_block_order"][["global_block_order", "condition", "n_trials"]]
        .sort_values(["global_block_order", "condition"])
        .copy()
    )
    count_rows = [
        [int(row["global_block_order"]), row["condition"], int(row["n_trials"])]
        for _, row in block_counts.iterrows()
    ]
    (TABLES_DIR / "block_count_table.tex").write_text(
        make_latex_grid(
            headers=["Global block", "Condition", "Valid trials"],
            rows=count_rows,
            column_spec="ccc",
        )
        + "\n",
        encoding="utf-8",
    )

    (TEXT_DIR / "final_conclusion.tex").write_text(conclusion_text + "\n", encoding="utf-8")


def write_report(
    primary_df: pd.DataFrame,
    descriptive_df: pd.DataFrame,
    exclusion_log_df: pd.DataFrame,
) -> Path:
    primary = primary_df.iloc[0]
    block_df = descriptive_df[descriptive_df["summary_type"] == "global_block_order"].copy()
    total_valid = int(primary["n_trials"])
    total_sessions = int(primary["n_participants"])
    total_questions = int(primary["n_questions"])
    block_count_summary = ", ".join(
        f"block {int(row['global_block_order'])}: {int(row['n_trials'])} {row['condition'].lower()} trials"
        for _, row in block_df.sort_values(["global_block_order", "condition"]).head(4).iterrows()
    )

    warning_lines = []
    warning_df = exclusion_log_df[exclusion_log_df["level"] == "warning"].copy()
    for _, row in warning_df.iterrows():
        warning_lines.append(
            f"\\item \\texttt{{{sanitize_for_tex(row['reason_code'])}}}: {sanitize_for_tex(row['detail'])}"
        )
    if not warning_lines:
        warning_lines.append(r"\item No Step 9-specific warnings were generated beyond standard boundary-fit notices logged in the model table.")

    report_lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{float}",
        r"\usepackage{caption}",
        r"\title{Step 9 Block-Structure, Order, and Adaptation Analysis}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Overview}",
        (
            "This report implements the Step 9 block-adjusted analysis using the same Step 7 participant-session cohort, "
            f"yielding {total_sessions} participant-sessions, {total_questions} questions, and {total_valid} valid trials."
        ),
        "The realized block metadata were merged from the cleaned behavioral table, so all diagnostics and models use the actual session order rather than an idealized task template.",
        rf"Descriptively, the valid-trial counts by early blocks began as {sanitize_for_tex(block_count_summary)}.",
        r"\section*{Primary Model}",
        r"\input{tables/primary_block_model_table.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\linewidth]{../../figures/step9/primary_block_adjusted_condition_effect.png}",
        r"\caption{Primary block-adjusted fixed-effect estimates for the front ROI variable-window model.}",
        r"\end{figure}",
        r"\section*{Descriptive Block Diagnostics}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\linewidth]{../../figures/step9/front_roi_by_global_block_order.png}",
        r"\caption{Mean front ROI response by realized global block order and condition.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\linewidth]{../../figures/step9/front_roi_by_withinblock_position.png}",
        r"\caption{Mean front ROI response by within-block question position and condition.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\linewidth]{../../figures/step9/response_time_by_global_block_order.png}",
        r"\caption{Mean response time by realized global block order and condition.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\linewidth]{../../figures/step9/early_vs_late_condition_plot.png}",
        r"\caption{Early-versus-late block-half descriptive comparison by condition.}",
        r"\end{figure}",
        r"\input{tables/block_count_table.tex}",
        r"\section*{Secondary Models And Benchmarks}",
        r"\input{tables/interaction_models_table.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\linewidth]{../../figures/step9/condition_by_blockorder_interaction_plot.png}",
        r"\caption{Observed condition difference across global block order with the interaction-model trend overlaid.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\linewidth]{../../figures/step9/condition_by_ordercondition_plot.png}",
        r"\caption{Front ROI response by participant-level condition order.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\linewidth]{../../figures/step9/fixed_vs_variable_block_benchmark.png}",
        r"\caption{Benchmark comparison between the duration-aware and fixed-window block-adjusted front ROI models.}",
        r"\end{figure}",
        r"\section*{Warnings And Notes}",
        r"\begin{itemize}",
        *warning_lines,
        r"\end{itemize}",
        r"\section*{Final Conclusion}",
        r"\input{text/final_conclusion.tex}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step9_block_analysis_report.tex"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()
    step7_trial_df, step2_trial_df, step6_trial_df = load_inputs()

    trial_df, exclusion_log_df = build_trial_block_table(step7_trial_df, step2_trial_df, step6_trial_df)
    descriptive_df = build_block_descriptive_summary(trial_df[trial_df["included_step9_model"]].copy())
    model_outputs, fit_artifacts, exclusion_log_df = run_models(trial_df, exclusion_log_df)
    generate_figures(descriptive_df, fit_artifacts)

    primary_df = model_outputs["03_step9_primary_front_block_adjusted_model.csv"]
    interaction_dfs = [
        model_outputs["04_step9_condition_by_blockorder_model.csv"],
        model_outputs["05_step9_condition_by_position_model.csv"],
        model_outputs["06_step9_condition_by_ordercondition_model.csv"],
    ]
    final_conclusion_df, conclusion_text = build_final_conclusion(primary_df, interaction_dfs)

    save_dataframe(trial_df, CLEAN_DIR / "01_step9_trial_block_table.csv")
    save_dataframe(descriptive_df, CLEAN_DIR / "02_step9_block_descriptive_summary.csv")
    for output_name, df in model_outputs.items():
        save_dataframe(df, CLEAN_DIR / output_name)
    save_dataframe(exclusion_log_df, CLEAN_DIR / "10_step9_exclusion_log.csv")
    save_dataframe(final_conclusion_df, CLEAN_DIR / "11_step9_final_conclusion.csv")

    write_supporting_tex(primary_df, model_outputs, descriptive_df, conclusion_text)
    report_path = write_report(primary_df, descriptive_df, exclusion_log_df)
    compile_report(report_path)


if __name__ == "__main__":
    main()
