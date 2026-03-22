#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import subprocess
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
STEP5_TRIAL_HBO_PATH = ROOT / "data_clean" / "step5" / "03_step5_trial_roi_summary_hbo.csv"
STEP5_SUMMARY_PATH = ROOT / "data_clean" / "step5" / "05_step5_participant_roi_summary.csv"
STEP2_TRIAL_PATH = ROOT / "data_clean" / "step2" / "06_behavior_trial_clean.csv"
FILTERED_QUESTIONS_PATH = ROOT / "materials" / "filtered_questions.json"

CLEAN_DIR = ROOT / "data_clean" / "step6"
FIGURES_DIR = ROOT / "figures" / "step6"
REPORTS_DIR = ROOT / "reports" / "step6"
TABLES_DIR = REPORTS_DIR / "tables"
TEXT_DIR = REPORTS_DIR / "text"

ALPHA = 0.05
CONDITION_ORDER = ["Concrete", "Abstract"]
CONDITION_COLORS = {"Abstract": "#AA3377", "Concrete": "#228833"}
MODEL_OPTIMIZERS = ["lbfgs", "bfgs", "powell"]
ITEM_COVARIATES = [
    ("total_word_count", "Total word count"),
    ("sentence_count", "Sentence count"),
    ("sentence_length", "Average sentence length"),
    ("enem_correctness", "ENEM item correctness"),
]


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


def make_latex_table(rows: list[tuple[str, Any]], column_spec: str = r"p{0.42\linewidth}p{0.50\linewidth}") -> str:
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


def coerce_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def zscore_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = float(numeric.std(ddof=0))
    if not np.isfinite(std) or math.isclose(std, 0.0):
        return pd.Series(np.zeros(len(numeric)), index=series.index, dtype=float)
    return (numeric - float(numeric.mean())) / std


def format_effect(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.{digits}e}"


def format_p_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = float(value)
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.6f}"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step5_trial_hbo_df = pd.read_csv(STEP5_TRIAL_HBO_PATH)
    step5_summary_df = pd.read_csv(STEP5_SUMMARY_PATH)
    step2_trial_df = pd.read_csv(STEP2_TRIAL_PATH)
    filtered_questions_df = pd.DataFrame(json.loads(FILTERED_QUESTIONS_PATH.read_text(encoding="utf-8")))
    return step5_trial_hbo_df, step5_summary_df, step2_trial_df, filtered_questions_df


def build_item_covariate_table(step5_trial_hbo_df: pd.DataFrame, filtered_questions_df: pd.DataFrame) -> pd.DataFrame:
    item_df = filtered_questions_df.copy()
    item_df["question_id"] = (
        item_df["year"].astype(str)
        + "_"
        + item_df["field"].astype(str)
        + "_"
        + pd.to_numeric(item_df["question_number"], errors="coerce").astype(int).astype(str)
    )
    item_df["condition"] = item_df["type"].astype(str).str.strip().str.lower().map(
        {"abstract": "Abstract", "concrete": "Concrete"}
    )
    numeric_cols = ["correctness", "total_word_count", "sentence_count", "sentence_length"]
    for col in numeric_cols:
        item_df[col] = pd.to_numeric(item_df[col], errors="coerce")

    step5_questions = sorted(step5_trial_hbo_df["question_id"].dropna().astype(str).unique().tolist())
    item_df = item_df[item_df["question_id"].isin(step5_questions)].copy()
    item_df = item_df.rename(columns={"correctness": "enem_correctness"})
    item_df["log_total_word_count"] = np.log(item_df["total_word_count"])
    item_df["z_log_total_word_count"] = zscore_series(item_df["log_total_word_count"])
    item_df["z_sentence_count"] = zscore_series(item_df["sentence_count"])
    item_df["z_sentence_length"] = zscore_series(item_df["sentence_length"])
    item_df["z_enem_correctness"] = zscore_series(item_df["enem_correctness"])
    item_df = item_df.sort_values(["condition", "question_id"]).reset_index(drop=True)
    return item_df


def build_trial_model_table(
    step5_trial_hbo_df: pd.DataFrame,
    step5_summary_df: pd.DataFrame,
    step2_trial_df: pd.DataFrame,
    item_covariate_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    exclusion_rows: list[dict[str, Any]] = []

    if "included_step5_analysis" in step5_summary_df.columns:
        step5_summary_df["included_step5_analysis"] = step5_summary_df["included_step5_analysis"].map(coerce_bool)
        step5_summary_df = step5_summary_df[step5_summary_df["included_step5_analysis"]].copy()

    step5_trial_df = step5_trial_hbo_df.copy()
    if "included_step5_analysis" in step5_trial_df.columns:
        step5_trial_df["included_step5_analysis"] = step5_trial_df["included_step5_analysis"].map(coerce_bool)
        step5_trial_df = step5_trial_df[step5_trial_df["included_step5_analysis"]].copy()

    step2_merge_cols = [
        "participant_id",
        "source_log",
        "block",
        "trial_idx_in_block",
        "question_id",
        "participant_correct",
        "rt_text_to_answer",
        "enem_correctness",
    ]
    step2_merge_df = step2_trial_df[step2_merge_cols].rename(columns={"enem_correctness": "enem_correctness_step2"})
    trial_df = step5_trial_df.merge(
        step2_merge_df,
        on=["participant_id", "source_log", "block", "trial_idx_in_block", "question_id"],
        how="left",
        validate="one_to_one",
        indicator="step2_merge_status",
    )
    missing_step2 = int((trial_df["step2_merge_status"] != "both").sum())
    exclusion_rows.append(
        {
            "stage": "merge_step2",
            "level": "warning" if missing_step2 else "info",
            "reason_code": "STEP2_TRIAL_MERGE",
            "n_rows": missing_step2,
            "detail": "Merged Step 5 trial rows onto Step 2 behavioral trial fields.",
        }
    )
    trial_df.drop(columns="step2_merge_status", inplace=True)

    item_merge_cols = [
        "question_id",
        "condition",
        "enem_correctness",
        "total_word_count",
        "sentence_count",
        "sentence_length",
        "log_total_word_count",
        "z_log_total_word_count",
        "z_sentence_count",
        "z_sentence_length",
        "z_enem_correctness",
    ]
    trial_df = trial_df.merge(
        item_covariate_df[item_merge_cols].rename(columns={"condition": "item_condition"}),
        on="question_id",
        how="left",
        validate="many_to_one",
        indicator="item_merge_status",
    )
    missing_item = int((trial_df["item_merge_status"] != "both").sum())
    exclusion_rows.append(
        {
            "stage": "merge_item_covariates",
            "level": "warning" if missing_item else "info",
            "reason_code": "ITEM_METADATA_MERGE",
            "n_rows": missing_item,
            "detail": "Merged Step 5 trials onto the filtered question-level covariates.",
        }
    )
    trial_df.drop(columns="item_merge_status", inplace=True)

    trial_df["participant_id"] = trial_df["participant_id"].astype(str)
    trial_df["session_id"] = trial_df["session_id"].astype(str)
    trial_df["question_id"] = trial_df["question_id"].astype(str)
    trial_df["condition"] = trial_df["question_type"].astype(str).str.strip().map(
        {"Abstract": "Abstract", "Concrete": "Concrete"}
    )
    trial_df["condition_abstract"] = trial_df["condition"].eq("Abstract").astype(float)
    trial_df["participant_correct"] = pd.to_numeric(trial_df["participant_correct"], errors="coerce")
    trial_df["rt_text_to_answer"] = pd.to_numeric(trial_df["rt_text_to_answer"], errors="coerce")
    trial_df["answer_latency_sec"] = pd.to_numeric(trial_df["answer_latency_sec"], errors="coerce")
    trial_df["response_time_sec"] = trial_df["answer_latency_sec"]
    trial_df["response_time_step2_sec"] = trial_df["rt_text_to_answer"]
    trial_df["response_time_abs_difference"] = np.abs(trial_df["response_time_sec"] - trial_df["response_time_step2_sec"])
    trial_df["enem_correctness_step2"] = pd.to_numeric(trial_df["enem_correctness_step2"], errors="coerce")

    numeric_cols = [
        "front_roi_mean",
        "back_roi_mean",
        "total_word_count",
        "sentence_count",
        "sentence_length",
        "enem_correctness",
        "participant_correct",
        "response_time_sec",
    ]
    for col in numeric_cols:
        trial_df[col] = pd.to_numeric(trial_df[col], errors="coerce")

    trial_df["log_total_word_count"] = np.log(trial_df["total_word_count"])
    trial_df["z_log_total_word_count"] = zscore_series(trial_df["log_total_word_count"])
    trial_df["z_sentence_count"] = zscore_series(trial_df["sentence_count"])
    trial_df["z_sentence_length"] = zscore_series(trial_df["sentence_length"])
    trial_df["z_enem_correctness"] = zscore_series(trial_df["enem_correctness"])
    trial_df["log_response_time_sec"] = np.log(trial_df["response_time_sec"])
    trial_df["z_log_response_time_sec"] = zscore_series(trial_df["log_response_time_sec"])

    participant_levels = sorted(trial_df["participant_id"].dropna().unique().tolist())
    question_levels = sorted(trial_df["question_id"].dropna().unique().tolist())
    participant_map = {participant_id: idx for idx, participant_id in enumerate(participant_levels)}
    question_map = {question_id: idx for idx, question_id in enumerate(question_levels)}
    trial_df["participant_code"] = trial_df["participant_id"].map(participant_map).astype(int)
    trial_df["question_code"] = trial_df["question_id"].map(question_map).astype(int)

    trial_df["item_condition_matches_trial_condition"] = trial_df["item_condition"] == trial_df["condition"]
    mismatch_count = int((~trial_df["item_condition_matches_trial_condition"]).sum())
    exclusion_rows.append(
        {
            "stage": "qc_condition_merge",
            "level": "warning" if mismatch_count else "info",
            "reason_code": "ITEM_CONDITION_MATCH",
            "n_rows": mismatch_count,
            "detail": "Checked that the filtered-question condition labels match the Step 5 trial condition labels.",
        }
    )

    correctness_diff = np.abs(trial_df["enem_correctness_step2"] - trial_df["enem_correctness"])
    correctness_mismatch = int((correctness_diff > 1e-12).fillna(False).sum())
    exclusion_rows.append(
        {
            "stage": "qc_item_correctness",
            "level": "warning" if correctness_mismatch else "info",
            "reason_code": "ITEM_CORRECTNESS_MATCH",
            "n_rows": correctness_mismatch,
            "detail": "Checked that ENEM item correctness from Step 2 matches the filtered question metadata.",
        }
    )

    primary_required = [
        "participant_id",
        "question_id",
        "condition",
        "front_roi_mean",
        "back_roi_mean",
        "total_word_count",
        "sentence_count",
        "sentence_length",
        "enem_correctness",
        "response_time_sec",
        "participant_correct",
        "z_log_total_word_count",
    ]
    trial_df["eligible_primary_model"] = True
    trial_df["primary_model_exclusion_reason"] = ""

    conditions = [
        (trial_df["condition"].isin(["Abstract", "Concrete"]), "invalid_condition"),
        (trial_df["total_word_count"] > 0, "nonpositive_word_count"),
        (trial_df["sentence_count"] > 0, "nonpositive_sentence_count"),
        (trial_df["sentence_length"] > 0, "nonpositive_sentence_length"),
        (trial_df["response_time_sec"] > 0, "nonpositive_response_time"),
        (trial_df["participant_correct"].isin([0, 1]), "invalid_participant_correct"),
    ]
    for mask, code in conditions:
        bad_mask = ~mask & trial_df["eligible_primary_model"]
        trial_df.loc[bad_mask, "eligible_primary_model"] = False
        trial_df.loc[bad_mask, "primary_model_exclusion_reason"] = code

    for col in primary_required:
        bad_mask = trial_df[col].isna() & trial_df["eligible_primary_model"]
        trial_df.loc[bad_mask, "eligible_primary_model"] = False
        trial_df.loc[bad_mask, "primary_model_exclusion_reason"] = f"missing_{col}"

    primary_excluded = int((~trial_df["eligible_primary_model"]).sum())
    exclusion_rows.append(
        {
            "stage": "qc_primary_model_table",
            "level": "warning" if primary_excluded else "info",
            "reason_code": "PRIMARY_MODEL_ELIGIBILITY",
            "n_rows": primary_excluded,
            "detail": "Excluded rows from the primary Step 6 model table after checking missing and impossible covariate values.",
        }
    )

    cohort_from_summary = {
        (row["participant_id"], row["session_id"])
        for _, row in step5_summary_df[["participant_id", "session_id"]].drop_duplicates().iterrows()
    }
    cohort_from_trial = {
        (row["participant_id"], row["session_id"])
        for _, row in trial_df[["participant_id", "session_id"]].drop_duplicates().iterrows()
    }
    cohort_delta = sorted(cohort_from_trial.symmetric_difference(cohort_from_summary))
    exclusion_rows.append(
        {
            "stage": "qc_step5_cohort",
            "level": "warning" if cohort_delta else "info",
            "reason_code": "STEP5_COHORT_MATCH",
            "n_rows": len(cohort_delta),
            "detail": "Checked that the Step 6 trial table uses the same participant-session cohort as Step 5.",
        }
    )

    trial_df = trial_df.sort_values(["participant_id", "session_id", "trial_idx_global"]).reset_index(drop=True)
    return trial_df, exclusion_rows


def summarize_item_balance(item_covariate_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for covariate, label in ITEM_COVARIATES:
        abstract_values = pd.to_numeric(
            item_covariate_df.loc[item_covariate_df["condition"] == "Abstract", covariate], errors="coerce"
        ).dropna()
        concrete_values = pd.to_numeric(
            item_covariate_df.loc[item_covariate_df["condition"] == "Concrete", covariate], errors="coerce"
        ).dropna()
        n_abstract = len(abstract_values)
        n_concrete = len(concrete_values)
        mean_abstract = float(abstract_values.mean())
        mean_concrete = float(concrete_values.mean())
        sd_abstract = float(abstract_values.std(ddof=1))
        sd_concrete = float(concrete_values.std(ddof=1))
        pooled_sd = math.sqrt(
            (((n_abstract - 1) * sd_abstract**2) + ((n_concrete - 1) * sd_concrete**2))
            / max(n_abstract + n_concrete - 2, 1)
        )
        smd = (mean_abstract - mean_concrete) / pooled_sd if pooled_sd > 0 else math.nan
        welch = stats.ttest_ind(abstract_values, concrete_values, equal_var=False)
        rows.append(
            {
                "covariate": covariate,
                "covariate_label": label,
                "n_abstract_items": n_abstract,
                "n_concrete_items": n_concrete,
                "abstract_mean": mean_abstract,
                "abstract_sd": sd_abstract,
                "abstract_median": float(abstract_values.median()),
                "abstract_min": float(abstract_values.min()),
                "abstract_max": float(abstract_values.max()),
                "concrete_mean": mean_concrete,
                "concrete_sd": sd_concrete,
                "concrete_median": float(concrete_values.median()),
                "concrete_min": float(concrete_values.min()),
                "concrete_max": float(concrete_values.max()),
                "standardized_mean_difference_abstract_minus_concrete": smd,
                "welch_t_stat": float(welch.statistic),
                "welch_p_value": float(welch.pvalue),
            }
        )
    return pd.DataFrame(rows)


def prepare_model_data(trial_model_df: pd.DataFrame, outcome: str, covariates: list[str]) -> pd.DataFrame:
    needed = [
        "participant_id",
        "question_id",
        "participant_code",
        "question_code",
        "condition_abstract",
        outcome,
        *covariates,
    ]
    model_df = trial_model_df[trial_model_df["eligible_primary_model"]].copy()
    model_df = model_df[needed].copy()
    numeric_cols = ["condition_abstract", outcome, *covariates, "participant_code", "question_code"]
    for col in numeric_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna(subset=["condition_abstract", outcome, *covariates]).reset_index(drop=True)
    model_df["participant_code"] = model_df["participant_code"].astype(int)
    model_df["question_code"] = model_df["question_code"].astype(int)
    model_df["participant_id"] = model_df["participant_id"].astype(str)
    model_df["question_id"] = model_df["question_id"].astype(str)
    return model_df


def fit_mixed_model(model_df: pd.DataFrame, outcome: str, covariates: list[str]) -> tuple[Any | None, str | None, list[str], str | None]:
    formula = outcome + " ~ condition_abstract"
    for covariate in covariates:
        formula += f" + {covariate}"

    collected_warnings: list[str] = []
    last_error: str | None = None
    best_result = None
    best_method = None

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

    if best_result is not None and best_result.converged:
        return best_result, best_method, collected_warnings, None
    return None, None, collected_warnings, last_error


def fit_fixed_effect_fallback(model_df: pd.DataFrame, outcome: str, covariates: list[str]) -> Any:
    formula = outcome + " ~ condition_abstract"
    for covariate in covariates:
        formula += f" + {covariate}"
    formula += " + C(participant_code) + C(question_code)"
    return smf.ols(formula, data=model_df).fit()


def fixed_effect_covariance(result: Any) -> pd.DataFrame:
    fe_params = result.fe_params if hasattr(result, "fe_params") else result.params
    fe_names = list(fe_params.index)
    cov = result.cov_params()
    if not isinstance(cov, pd.DataFrame):
        cov = pd.DataFrame(cov, index=result.params.index, columns=result.params.index)
    return cov.loc[fe_names, fe_names]


def extract_model_row(
    result: Any,
    model_df: pd.DataFrame,
    model_id: str,
    analysis_label: str,
    analysis_scope: str,
    outcome: str,
    outcome_label: str,
    covariates: list[str],
    model_type: str,
    optimizer: str | None,
    warnings_list: list[str],
    fallback_used: bool,
    converged: bool,
) -> dict[str, Any]:
    cov = fixed_effect_covariance(result)

    condition_beta = float(result.params["condition_abstract"])
    condition_se = float(result.bse["condition_abstract"])
    condition_stat = condition_beta / condition_se if condition_se else math.nan
    condition_ci_low, condition_ci_high = [
        float(item) for item in result.conf_int().loc["condition_abstract"].tolist()
    ]

    row = {
        "model_id": model_id,
        "analysis_label": analysis_label,
        "analysis_scope": analysis_scope,
        "outcome_variable": outcome,
        "outcome_label": outcome_label,
        "covariate_set": ";".join(covariates),
        "model_type": model_type,
        "fallback_used": fallback_used,
        "converged": converged,
        "optimizer": optimizer or "",
        "n_trials": int(len(model_df)),
        "n_participants": int(model_df["participant_id"].nunique()),
        "n_questions": int(model_df["question_id"].nunique()),
        "condition_term": "condition_abstract",
        "condition_beta": condition_beta,
        "condition_se": condition_se,
        "condition_stat": condition_stat,
        "condition_p_value": float(result.pvalues["condition_abstract"]),
        "condition_ci_low": condition_ci_low,
        "condition_ci_high": condition_ci_high,
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

    for idx in [1, 2]:
        row[f"covariate_{idx}_name"] = ""
        row[f"covariate_{idx}_beta"] = math.nan
        row[f"covariate_{idx}_se"] = math.nan
        row[f"covariate_{idx}_stat"] = math.nan
        row[f"covariate_{idx}_p_value"] = math.nan
        row[f"covariate_{idx}_ci_low"] = math.nan
        row[f"covariate_{idx}_ci_high"] = math.nan

    for idx, covariate in enumerate(covariates[:2], start=1):
        row[f"covariate_{idx}_name"] = covariate
        row[f"covariate_{idx}_beta"] = float(result.params[covariate])
        row[f"covariate_{idx}_se"] = float(result.bse[covariate])
        row[f"covariate_{idx}_stat"] = float(result.params[covariate] / result.bse[covariate]) if result.bse[covariate] else math.nan
        row[f"covariate_{idx}_p_value"] = float(result.pvalues[covariate])
        ci_low, ci_high = [float(item) for item in result.conf_int().loc[covariate].tolist()]
        row[f"covariate_{idx}_ci_low"] = ci_low
        row[f"covariate_{idx}_ci_high"] = ci_high

    return row


def run_model_suite(trial_model_df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[dict[str, Any]]]:
    model_specs = [
        {
            "model_id": "primary_front_wordcount",
            "analysis_label": "Primary front ROI mixed model adjusted for log word count",
            "analysis_scope": "primary",
            "outcome": "front_roi_mean",
            "outcome_label": "Front ROI HbO (7-11 s)",
            "covariates": ["z_log_total_word_count"],
            "output_name": "04_step6_primary_front_wordcount_model.csv",
        },
        {
            "model_id": "front_sentencecount",
            "analysis_label": "Alternative front ROI mixed model adjusted for sentence count",
            "analysis_scope": "alternative",
            "outcome": "front_roi_mean",
            "outcome_label": "Front ROI HbO (7-11 s)",
            "covariates": ["z_sentence_count"],
            "output_name": "05_step6_front_sentencecount_model.csv",
        },
        {
            "model_id": "front_sentencelength",
            "analysis_label": "Alternative front ROI mixed model adjusted for sentence length",
            "analysis_scope": "alternative",
            "outcome": "front_roi_mean",
            "outcome_label": "Front ROI HbO (7-11 s)",
            "covariates": ["z_sentence_length"],
            "output_name": "06_step6_front_sentencelength_model.csv",
        },
        {
            "model_id": "front_wordcount_difficulty",
            "analysis_label": "Sensitivity front ROI mixed model adjusted for log word count and ENEM correctness",
            "analysis_scope": "sensitivity",
            "outcome": "front_roi_mean",
            "outcome_label": "Front ROI HbO (7-11 s)",
            "covariates": ["z_log_total_word_count", "z_enem_correctness"],
            "output_name": "07_step6_front_wordcount_difficulty_model.csv",
        },
        {
            "model_id": "back_wordcount",
            "analysis_label": "Secondary back ROI mixed model adjusted for log word count",
            "analysis_scope": "secondary",
            "outcome": "back_roi_mean",
            "outcome_label": "Back ROI HbO (7-11 s)",
            "covariates": ["z_log_total_word_count"],
            "output_name": "08_step6_back_wordcount_model.csv",
        },
        {
            "model_id": "front_wordcount_rt",
            "analysis_label": "Sensitivity front ROI mixed model adjusted for log word count and log response time",
            "analysis_scope": "sensitivity_behavior",
            "outcome": "front_roi_mean",
            "outcome_label": "Front ROI HbO (7-11 s)",
            "covariates": ["z_log_total_word_count", "z_log_response_time_sec"],
            "output_name": "09_step6_behavior_sensitivity_models.csv",
        },
        {
            "model_id": "front_wordcount_participant_correct",
            "analysis_label": "Sensitivity front ROI mixed model adjusted for log word count and participant correctness",
            "analysis_scope": "sensitivity_behavior",
            "outcome": "front_roi_mean",
            "outcome_label": "Front ROI HbO (7-11 s)",
            "covariates": ["z_log_total_word_count", "participant_correct"],
            "output_name": "09_step6_behavior_sensitivity_models.csv",
        },
    ]

    model_outputs: dict[str, pd.DataFrame] = {}
    fit_artifacts: dict[str, Any] = {}
    exclusion_rows: list[dict[str, Any]] = []

    for spec in model_specs:
        model_df = prepare_model_data(trial_model_df, spec["outcome"], spec["covariates"])
        mixed_result, optimizer, warnings_list, fit_error = fit_mixed_model(model_df, spec["outcome"], spec["covariates"])

        if mixed_result is not None and getattr(mixed_result, "converged", False):
            result = mixed_result
            model_type = "MixedLM"
            fallback_used = False
            converged = True
        else:
            result = fit_fixed_effect_fallback(model_df, spec["outcome"], spec["covariates"])
            model_type = "FixedEffectsOLS"
            fallback_used = True
            converged = True
            warnings_list = warnings_list + ([fit_error] if fit_error else [])

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
            converged=converged,
        )

        output_name = spec["output_name"]
        if output_name not in model_outputs:
            model_outputs[output_name] = pd.DataFrame([row])
        else:
            model_outputs[output_name] = pd.concat([model_outputs[output_name], pd.DataFrame([row])], ignore_index=True)

        fit_artifacts[spec["model_id"]] = {"result": result, "data": model_df, "spec": spec, "row": row}
        exclusion_rows.append(
            {
                "stage": "model_fit",
                "level": "warning" if warnings_list or fallback_used else "info",
                "reason_code": f"MODEL_{spec['model_id'].upper()}",
                "n_rows": 1,
                "detail": f"Model type={model_type}; optimizer={optimizer or 'n/a'}; fallback_used={fallback_used}; warnings={' | '.join(warnings_list) if warnings_list else 'none'}",
            }
        )

    for output_name, df in model_outputs.items():
        model_outputs[output_name] = df.reset_index(drop=True)
    return model_outputs, fit_artifacts, exclusion_rows


def plot_balance_distribution(item_covariate_df: pd.DataFrame, covariate: str, label: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
    rng = np.random.default_rng(20260321)

    for idx, condition in enumerate(CONDITION_ORDER, start=1):
        values = pd.to_numeric(
            item_covariate_df.loc[item_covariate_df["condition"] == condition, covariate], errors="coerce"
        ).dropna()
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
        ax.scatter(
            jitter,
            values,
            s=48,
            color=CONDITION_COLORS[condition],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.95,
            zorder=3,
        )

    ax.set_xticks([1, 2], CONDITION_ORDER)
    ax.set_xlabel("Question condition")
    ax.set_ylabel(label)
    ax.set_title(f"{label} by condition")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_front_roi_vs_wordcount_scatter(primary_fit: dict[str, Any], path: Path) -> None:
    model_df = primary_fit["data"].copy()
    result = primary_fit["result"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")

    condition_labels = np.where(model_df["condition_abstract"] == 1.0, "Abstract", "Concrete")
    for condition in CONDITION_ORDER:
        mask = condition_labels == condition
        ax.scatter(
            model_df.loc[mask, "total_word_count"] if "total_word_count" in model_df.columns else np.exp(model_df.loc[mask, "z_log_total_word_count"]),
            model_df.loc[mask, "front_roi_mean"] * 1e6,
            s=28,
            alpha=0.35,
            color=CONDITION_COLORS[condition],
            label=condition,
        )

    fe_params = result.fe_params if hasattr(result, "fe_params") else result.params
    log_mean = primary_fit["full_trial_df"]["log_total_word_count"].mean()
    log_std = primary_fit["full_trial_df"]["log_total_word_count"].std(ddof=0)
    x_grid = np.linspace(
        float(primary_fit["full_trial_df"]["total_word_count"].min()),
        float(primary_fit["full_trial_df"]["total_word_count"].max()),
        200,
    )
    z_grid = (np.log(x_grid) - log_mean) / log_std
    intercept = float(fe_params["Intercept"])
    beta_condition = float(fe_params["condition_abstract"])
    beta_word = float(fe_params["z_log_total_word_count"])

    ax.plot(x_grid, (intercept + beta_word * z_grid) * 1e6, color=CONDITION_COLORS["Concrete"], linewidth=2.2)
    ax.plot(
        x_grid,
        (intercept + beta_condition + beta_word * z_grid) * 1e6,
        color=CONDITION_COLORS["Abstract"],
        linewidth=2.2,
    )

    ax.set_xlabel("Total word count")
    ax.set_ylabel(r"Front ROI HbO 7-11 s mean ($\mu$M)")
    ax.set_title("Front ROI HbO versus word count")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_primary_partial_effect(primary_fit: dict[str, Any], path: Path) -> None:
    result = primary_fit["result"]
    fe_params = result.fe_params if hasattr(result, "fe_params") else result.params
    cov = fixed_effect_covariance(result)
    effect = float(fe_params["condition_abstract"])
    se = float(math.sqrt(cov.loc["condition_abstract", "condition_abstract"]))
    ci_low = effect - 1.96 * se
    ci_high = effect + 1.96 * se

    fig, ax = plt.subplots(figsize=(6.2, 2.8), layout="constrained")
    ax.errorbar(
        x=effect * 1e6,
        y=0,
        xerr=np.array([[effect - ci_low], [ci_high - effect]]) * 1e6,
        fmt="o",
        color=CONDITION_COLORS["Abstract"],
        ecolor=CONDITION_COLORS["Abstract"],
        elinewidth=2,
        capsize=4,
        markersize=7,
    )
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks([0], ["Abstract - Concrete"])
    ax.set_xlabel(r"Adjusted condition effect on front ROI HbO ($\mu$M)")
    ax.set_title("Primary adjusted condition coefficient")
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
    ax.set_title("Primary model residual diagnostic")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_random_intercepts_proxy(primary_fit: dict[str, Any], path: Path) -> None:
    result = primary_fit["result"]
    model_df = primary_fit["data"].copy()
    fe_params = result.fe_params if hasattr(result, "fe_params") else result.params
    fixed_only = np.full(len(model_df), float(fe_params["Intercept"]))
    for term, beta in fe_params.items():
        if term == "Intercept":
            continue
        if term in model_df.columns:
            fixed_only += model_df[term].to_numpy(dtype=float) * float(beta)

    model_df["fixed_only_residual"] = model_df[primary_fit["spec"]["outcome"]].to_numpy(dtype=float) - fixed_only
    participant_offsets = model_df.groupby("participant_id")["fixed_only_residual"].mean().sort_values()
    question_offsets = model_df.groupby("question_id")["fixed_only_residual"].mean().sort_values()

    participant_sel = pd.concat([participant_offsets.head(8), participant_offsets.tail(8)])
    question_sel = pd.concat([question_offsets.head(8), question_offsets.tail(8)])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), layout="constrained")
    axes[0].barh(
        participant_sel.index,
        participant_sel.values * 1e6,
        color=["#4477AA" if value >= 0 else "#CC6677" for value in participant_sel.values],
    )
    axes[0].axvline(0.0, color="#444444", linewidth=1)
    axes[0].set_title("Participant intercept offsets")
    axes[0].set_xlabel(r"Fixed-effect residual mean ($\mu$M)")

    axes[1].barh(
        question_sel.index,
        question_sel.values * 1e6,
        color=["#4477AA" if value >= 0 else "#CC6677" for value in question_sel.values],
    )
    axes[1].axvline(0.0, color="#444444", linewidth=1)
    axes[1].set_title("Question intercept offsets")
    axes[1].set_xlabel(r"Fixed-effect residual mean ($\mu$M)")

    fig.suptitle("Primary model group-level intercept proxies", y=1.02)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_figures(item_covariate_df: pd.DataFrame, trial_model_df: pd.DataFrame, primary_fit: dict[str, Any]) -> None:
    trial_for_scatter = trial_model_df[trial_model_df["eligible_primary_model"]].copy()
    primary_fit["full_trial_df"] = trial_for_scatter
    primary_fit["data"] = primary_fit["data"].merge(
        trial_for_scatter[["participant_id", "question_id", "total_word_count"]],
        on=["participant_id", "question_id"],
        how="left",
    )

    plot_balance_distribution(
        item_covariate_df,
        "total_word_count",
        "Total word count",
        FIGURES_DIR / "wordcount_by_condition.png",
    )
    plot_balance_distribution(
        item_covariate_df,
        "sentence_count",
        "Sentence count",
        FIGURES_DIR / "sentencecount_by_condition.png",
    )
    plot_balance_distribution(
        item_covariate_df,
        "sentence_length",
        "Average sentence length",
        FIGURES_DIR / "sentencelength_by_condition.png",
    )
    plot_balance_distribution(
        item_covariate_df,
        "enem_correctness",
        "ENEM item correctness",
        FIGURES_DIR / "enem_correctness_by_condition.png",
    )
    plot_front_roi_vs_wordcount_scatter(primary_fit, FIGURES_DIR / "front_roi_vs_wordcount_scatter.png")
    plot_primary_partial_effect(primary_fit, FIGURES_DIR / "front_roi_partial_effect_condition.png")
    plot_primary_residuals(primary_fit, FIGURES_DIR / "model_residuals_primary.png")
    plot_random_intercepts_proxy(primary_fit, FIGURES_DIR / "model_random_intercepts.png")


def write_supporting_tex(
    balance_df: pd.DataFrame,
    primary_model_df: pd.DataFrame,
    secondary_models_df: pd.DataFrame,
    conclusion_text: str,
) -> None:
    balance_rows = []
    for row in balance_df.itertuples(index=False):
        balance_rows.append(
            [
                row.covariate_label,
                f"{row.abstract_mean:.2f} ({row.abstract_sd:.2f})",
                f"{row.concrete_mean:.2f} ({row.concrete_sd:.2f})",
                f"{row.standardized_mean_difference_abstract_minus_concrete:.3f}",
                format_p_value(row.welch_p_value),
            ]
        )
    (TABLES_DIR / "item_balance_table.tex").write_text(
        make_latex_grid(
            ["Covariate", "Abstract mean (SD)", "Concrete mean (SD)", "SMD", "Welch p"],
            balance_rows,
            r"p{0.28\linewidth}p{0.19\linewidth}p{0.19\linewidth}p{0.12\linewidth}p{0.12\linewidth}",
        )
        + "\n",
        encoding="utf-8",
    )

    primary = primary_model_df.iloc[0]
    primary_rows = [
        ("Model type", primary["model_type"]),
        ("Included participants", int(primary["n_participants"])),
        ("Included questions", int(primary["n_questions"])),
        ("Included trials", int(primary["n_trials"])),
        ("Condition beta", format_effect(primary["condition_beta"])),
        ("Condition SE", format_effect(primary["condition_se"])),
        ("Condition statistic", f"{float(primary['condition_stat']):.3f}"),
        ("Condition p-value", format_p_value(primary["condition_p_value"])),
        ("Condition 95% CI", f"[{format_effect(primary['condition_ci_low'])}, {format_effect(primary['condition_ci_high'])}]"),
        ("Word-count beta", format_effect(primary["covariate_1_beta"])),
        ("Residual variance", format_effect(primary["residual_variance"])),
        ("Participant random-intercept variance", format_effect(primary["participant_random_intercept_variance"])),
        ("Question random-intercept variance", format_effect(primary["question_random_intercept_variance"])),
    ]
    (TABLES_DIR / "primary_model_table.tex").write_text(make_latex_table(primary_rows) + "\n", encoding="utf-8")

    secondary_rows = []
    for row in secondary_models_df.itertuples(index=False):
        secondary_rows.append(
            [
                row.analysis_label.replace(" mixed model", ""),
                format_effect(row.condition_beta),
                format_p_value(row.condition_p_value),
                format_effect(row.participant_random_intercept_variance),
                format_effect(row.question_random_intercept_variance),
                "yes" if row.fallback_used else "no",
            ]
        )
    (TABLES_DIR / "secondary_model_table.tex").write_text(
        make_latex_grid(
            ["Model", "Cond. beta", "Cond. p", "Part. var", "Question var", "Fallback"],
            secondary_rows,
            r"p{0.40\linewidth}p{0.12\linewidth}p{0.10\linewidth}p{0.12\linewidth}p{0.14\linewidth}p{0.08\linewidth}",
        )
        + "\n",
        encoding="utf-8",
    )

    (TEXT_DIR / "final_conclusion.tex").write_text(conclusion_text + "\n", encoding="utf-8")


def write_report(
    trial_model_df: pd.DataFrame,
    item_covariate_df: pd.DataFrame,
    primary_model_df: pd.DataFrame,
    secondary_models_df: pd.DataFrame,
    exclusion_log_df: pd.DataFrame,
) -> Path:
    primary = primary_model_df.iloc[0]
    overview_rows = [
        ("Included participant-sessions", int(trial_model_df["session_id"].nunique())),
        ("Included participants", int(trial_model_df["participant_id"].nunique())),
        ("Included questions", int(item_covariate_df["question_id"].nunique())),
        ("Included trials", int(trial_model_df["eligible_primary_model"].sum())),
        ("Primary model type", primary["model_type"]),
        ("Primary model fallback used", "yes" if primary["fallback_used"] else "no"),
        ("Primary condition p-value", format_p_value(primary["condition_p_value"])),
    ]

    warning_rows = exclusion_log_df[exclusion_log_df["level"] == "warning"].copy()
    warning_text = (
        "; ".join(f"{row.reason_code}: {row.detail}" for row in warning_rows.itertuples(index=False))
        if not warning_rows.empty
        else "No Step 6 exclusions or warnings were triggered beyond model-fit boundary warnings captured in the model table."
    )

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Step 6 Covariate-Adjusted Trial-Level ROI Report}",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
        r"\paragraph{Primary question.} "
        r"Does the abstract-versus-concrete effect in the onset-locked left frontal ROI remain after adjusting for question length and participant/question variability?",
        r"\paragraph{Warnings and QC.} " + sanitize_for_tex(warning_text),
        r"\subsection*{Item-Level Balance Check}",
        r"\input{tables/item_balance_table.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/wordcount_by_condition.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/sentencecount_by_condition.png}",
        r"\caption{Item-level balance plots for total word count and sentence count by question condition.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/sentencelength_by_condition.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/enem_correctness_by_condition.png}",
        r"\caption{Item-level balance plots for average sentence length and ENEM item correctness by question condition.}",
        r"\end{figure}",
        r"\subsection*{Primary Front ROI Model}",
        r"\input{tables/primary_model_table.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/front_roi_vs_wordcount_scatter.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/front_roi_partial_effect_condition.png}",
        r"\caption{Front ROI HbO versus word count and the adjusted condition effect from the primary word-count model.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/model_residuals_primary.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/model_random_intercepts.png}",
        r"\caption{Residual diagnostic and group-level intercept proxies for the primary model.}",
        r"\end{figure}",
        r"\subsection*{Secondary and Sensitivity Models}",
        r"\input{tables/secondary_model_table.tex}",
        r"\subsection*{Final Conclusion}",
        r"\input{text/final_conclusion.tex}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step6_covariate_adjusted_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_final_conclusion(primary_model_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    row = primary_model_df.iloc[0]
    beta = float(row["condition_beta"])
    se = float(row["condition_se"])
    p_value = float(row["condition_p_value"])
    ci_low = float(row["condition_ci_low"])
    ci_high = float(row["condition_ci_high"])

    if p_value < ALPHA and beta > 0:
        conclusion_text = (
            "In the trial-level mixed-effects analysis adjusted for log word count, abstract questions "
            f"were associated with a significantly larger left frontal ROI HbO response than concrete questions "
            f"(beta={beta:.3e}, SE={se:.3e}, p={p_value:.6f}, 95% CI [{ci_low:.3e}, {ci_high:.3e}])."
        )
    elif p_value < ALPHA and beta < 0:
        conclusion_text = (
            "In the trial-level mixed-effects analysis adjusted for log word count, concrete questions "
            f"were associated with a significantly larger left frontal ROI HbO response than abstract questions "
            f"(beta={beta:.3e}, SE={se:.3e}, p={p_value:.6f}, 95% CI [{ci_low:.3e}, {ci_high:.3e}])."
        )
    else:
        conclusion_text = (
            "In the trial-level mixed-effects analysis adjusted for log word count, the abstract-versus-concrete "
            f"effect in the left frontal ROI was not statistically significant (beta={beta:.3e}, "
            f"SE={se:.3e}, p={p_value:.6f}, 95% CI [{ci_low:.3e}, {ci_high:.3e}]). "
            "This suggests that, after accounting for participant differences, question differences, and question length, "
            "the current data do not provide statistically significant evidence for a frontal condition effect."
        )

    final_df = pd.DataFrame(
        [
            {
                "n_included_participants": int(row["n_participants"]),
                "n_included_questions": int(row["n_questions"]),
                "n_included_trials": int(row["n_trials"]),
                "primary_model_type": row["model_type"],
                "primary_model_fallback_used": bool(row["fallback_used"]),
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


def main() -> None:
    ensure_directories()
    step5_trial_hbo_df, step5_summary_df, step2_trial_df, filtered_questions_df = load_inputs()

    item_covariate_df = build_item_covariate_table(step5_trial_hbo_df, filtered_questions_df)
    trial_model_df, exclusion_rows = build_trial_model_table(
        step5_trial_hbo_df=step5_trial_hbo_df,
        step5_summary_df=step5_summary_df,
        step2_trial_df=step2_trial_df,
        item_covariate_df=item_covariate_df,
    )
    balance_df = summarize_item_balance(item_covariate_df)
    model_outputs, fit_artifacts, model_exclusion_rows = run_model_suite(trial_model_df)
    exclusion_rows.extend(model_exclusion_rows)
    exclusion_log_df = pd.DataFrame(exclusion_rows)

    primary_model_df = model_outputs["04_step6_primary_front_wordcount_model.csv"].copy()
    secondary_model_frames = []
    for name, df in model_outputs.items():
        if name != "04_step6_primary_front_wordcount_model.csv":
            secondary_model_frames.append(df.copy())
    secondary_models_df = pd.concat(secondary_model_frames, ignore_index=True)
    final_conclusion_df, conclusion_text = build_final_conclusion(primary_model_df)

    save_dataframe(item_covariate_df, CLEAN_DIR / "01_step6_item_covariate_table.csv")
    save_dataframe(balance_df, CLEAN_DIR / "02_step6_item_balance_summary.csv")
    save_dataframe(trial_model_df, CLEAN_DIR / "03_step6_trial_model_table.csv")
    for name, df in model_outputs.items():
        save_dataframe(df, CLEAN_DIR / name)
    save_dataframe(exclusion_log_df, CLEAN_DIR / "10_step6_exclusion_log.csv")
    save_dataframe(final_conclusion_df, CLEAN_DIR / "11_step6_final_conclusion.csv")

    generate_figures(item_covariate_df, trial_model_df, fit_artifacts["primary_front_wordcount"])
    write_supporting_tex(balance_df, primary_model_df, secondary_models_df, conclusion_text)
    report_path = write_report(trial_model_df, item_covariate_df, primary_model_df, secondary_models_df, exclusion_log_df)
    compile_report(report_path)

    print("Step 6 covariate-adjusted analysis completed.")
    print(f"Included participants: {trial_model_df['participant_id'].nunique()}")
    print(f"Included questions: {item_covariate_df['question_id'].nunique()}")
    print(f"Included trials for primary model: {int(trial_model_df['eligible_primary_model'].sum())}")
    print(f"Primary condition p-value: {float(primary_model_df.iloc[0]['condition_p_value']):.6f}")


if __name__ == "__main__":
    main()
