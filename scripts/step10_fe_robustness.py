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
from scipy import stats
import statsmodels.formula.api as smf

from step6_covariate_adjusted import (
    compile_report,
    format_effect,
    format_p_value,
    make_latex_grid,
    make_latex_table,
    sanitize_for_tex,
    save_dataframe,
)


ROOT = Path(__file__).resolve().parents[1]
STEP9_TRIAL_PATH = ROOT / "data_clean" / "step9" / "01_step9_trial_block_table.csv"

CLEAN_DIR = ROOT / "data_clean" / "step10"
FIGURES_DIR = ROOT / "figures" / "step10"
REPORTS_DIR = ROOT / "reports" / "step10"
TABLES_DIR = REPORTS_DIR / "tables"
TEXT_DIR = REPORTS_DIR / "text"

ALPHA = 0.05
OVERRIDE_EXCLUDED_PIDS = {"PID025", "PID034"}
PRIMARY_ROBUST_VARIANTS = ["two_way_cluster", "participant_cluster"]
SENSITIVITY_ORDER = [
    ("02_step10_primary_fe_results.csv", "Primary FE"),
    ("03_step10_cluster_participant_results.csv", "Participant cluster"),
    ("04_step10_cluster_question_results.csv", "Question cluster"),
    ("05_step10_hc3_results.csv", "HC3"),
]


def ensure_directories() -> None:
    for path in [CLEAN_DIR, FIGURES_DIR, REPORTS_DIR, TABLES_DIR, TEXT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_trial_table() -> pd.DataFrame:
    trial_df = pd.read_csv(STEP9_TRIAL_PATH)
    if "included_step9_model" in trial_df.columns:
        trial_df["included_step9_model"] = trial_df["included_step9_model"].astype(bool)
    return trial_df


def build_primary_fe_table(step9_trial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_df = step9_trial_df[step9_trial_df["included_step9_model"]].copy()
    trial_df["participant_code"] = pd.to_numeric(trial_df["participant_code"], errors="coerce").astype(int)
    trial_df["question_code"] = pd.to_numeric(trial_df["question_code"], errors="coerce").astype(int)
    trial_df["manual_override_session"] = trial_df["participant_id"].isin(OVERRIDE_EXCLUDED_PIDS)
    trial_df["no_override_cohort"] = ~trial_df["manual_override_session"]

    keep_cols = [
        "participant_id",
        "participant_code",
        "session_id",
        "question_id",
        "question_code",
        "condition",
        "condition_abstract",
        "block",
        "global_block_order",
        "within_block_position",
        "cumulative_trial_order",
        "z_log_total_word_count",
        "z_enem_correctness",
        "response_time_sec",
        "front_roi_mean_variable_lag4",
        "front_roi_mean",
        "back_roi_mean_variable_lag4",
        "manual_override_session",
        "no_override_cohort",
    ]
    trial_df = trial_df[keep_cols].sort_values(["participant_id", "session_id", "cumulative_trial_order"]).reset_index(drop=True)

    log_rows = [
        {
            "stage": "input",
            "level": "info",
            "reason_code": "STEP9_TRIAL_TABLE_REUSED",
            "n_rows": int(len(trial_df)),
            "detail": "Step 10 started from the Step 9 included trial table without changing the duration-aware front ROI outcome.",
        },
        {
            "stage": "cohort",
            "level": "info",
            "reason_code": "MANUAL_OVERRIDE_FLAGGED",
            "n_rows": int(trial_df["manual_override_session"].sum()),
            "detail": "PID025 and PID034 were flagged for the no-override sensitivity but retained in the primary Step 10 cohort.",
        },
    ]
    return trial_df, pd.DataFrame(log_rows)


def fe_formula(outcome: str) -> str:
    return (
        f"{outcome} ~ condition_abstract + global_block_order + within_block_position + "
        f"z_log_total_word_count + z_enem_correctness + C(participant_code)"
    )


def prepare_model_data(trial_df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    model_df = trial_df.copy()
    numeric_cols = [
        "participant_code",
        "question_code",
        "condition_abstract",
        "global_block_order",
        "within_block_position",
        "z_log_total_word_count",
        "z_enem_correctness",
        outcome,
    ]
    for col in numeric_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna(subset=numeric_cols).reset_index(drop=True)
    model_df["participant_id"] = model_df["participant_id"].astype(str)
    model_df["question_id"] = model_df["question_id"].astype(str)
    model_df["participant_code"] = model_df["participant_code"].astype(int)
    model_df["question_code"] = model_df["question_code"].astype(int)
    return model_df


def fit_base_ols(model_df: pd.DataFrame, outcome: str) -> Any:
    return smf.ols(fe_formula(outcome), data=model_df).fit()


def apply_robust_covariance(base_result: Any, model_df: pd.DataFrame, variant: str) -> tuple[Any, str, list[str]]:
    warnings_list: list[str] = []
    question_groups = model_df["question_code"].to_numpy(dtype=int)
    participant_groups = model_df["participant_code"].to_numpy(dtype=int)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if variant == "two_way_cluster":
            groups = np.column_stack([participant_groups, question_groups])
            robust_result = base_result.get_robustcov_results(cov_type="cluster", groups=groups)
            se_label = "Two-way cluster (participant + question)"
        elif variant == "participant_cluster":
            robust_result = base_result.get_robustcov_results(cov_type="cluster", groups=participant_groups)
            se_label = "Participant-clustered"
        elif variant == "question_cluster":
            robust_result = base_result.get_robustcov_results(cov_type="cluster", groups=question_groups)
            se_label = "Question-clustered"
        elif variant == "hc3":
            robust_result = base_result.get_robustcov_results(cov_type="HC3")
            se_label = "HC3 heteroskedasticity-robust"
        else:
            raise ValueError(f"Unknown robust covariance variant: {variant}")

    warnings_list.extend(str(item.message) for item in caught)
    return robust_result, se_label, warnings_list


def fit_primary_or_fallback(base_result: Any, model_df: pd.DataFrame) -> tuple[Any, str, str, list[str]]:
    attempts: list[str] = []
    combined_warnings: list[str] = []
    last_error = ""

    for variant in PRIMARY_ROBUST_VARIANTS:
        try:
            robust_result, se_label, warnings_list = apply_robust_covariance(base_result, model_df, variant)
            stats_row = term_stats(robust_result, "condition_abstract")
            if np.isfinite(stats_row["se"]) and np.isfinite(stats_row["p_value"]):
                combined_warnings.extend(warnings_list)
                return robust_result, variant, se_label, combined_warnings
            attempts.append(f"{variant}: non-finite condition SE")
        except Exception as exc:
            attempts.append(f"{variant}: {exc}")
            last_error = str(exc)

    raise RuntimeError(f"Primary robust covariance failed. Attempts: {' | '.join(attempts)} | last_error={last_error}")


def model_names(result: Any) -> list[str]:
    return list(result.model.exog_names)


def term_stats(result: Any, term_name: str) -> dict[str, float]:
    names = model_names(result)
    idx = names.index(term_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        params = np.asarray(result.params, dtype=float)
        bse = np.asarray(result.bse, dtype=float)
        pvalues = np.asarray(result.pvalues, dtype=float)
        conf_int = np.asarray(result.conf_int(), dtype=float)
    beta = float(params[idx])
    se = float(bse[idx])
    stat = beta / se if se else math.nan
    return {
        "beta": beta,
        "se": se,
        "stat": stat,
        "p_value": float(pvalues[idx]),
        "ci_low": float(conf_int[idx, 0]),
        "ci_high": float(conf_int[idx, 1]),
    }


def add_term(row: dict[str, Any], result: Any, prefix: str, label: str, term_name: str) -> None:
    stats_row = term_stats(result, term_name)
    row[f"{prefix}_term"] = term_name
    row[f"{prefix}_label"] = label
    for key, value in stats_row.items():
        row[f"{prefix}_{key}"] = value


def build_result_row(
    robust_result: Any,
    model_df: pd.DataFrame,
    *,
    model_id: str,
    analysis_label: str,
    analysis_scope: str,
    outcome: str,
    outcome_label: str,
    robust_variant: str,
    robust_se_label: str,
    warnings_list: list[str],
) -> dict[str, Any]:
    row = {
        "model_id": model_id,
        "analysis_label": analysis_label,
        "analysis_scope": analysis_scope,
        "outcome_variable": outcome,
        "outcome_label": outcome_label,
        "formula": fe_formula(outcome),
        "robust_variant": robust_variant,
        "robust_se_label": robust_se_label,
        "n_trials": int(len(model_df)),
        "n_participants": int(model_df["participant_id"].nunique()),
        "n_questions": int(model_df["question_id"].nunique()),
        "warning_count": len(warnings_list),
        "warnings": " | ".join(warnings_list),
    }
    add_term(row, robust_result, "condition", "Adjusted abstract vs concrete effect", "condition_abstract")
    add_term(row, robust_result, "global_block_order", "Global block order", "global_block_order")
    add_term(row, robust_result, "within_block_position", "Within-block trial position", "within_block_position")
    add_term(row, robust_result, "wordcount", "z(log total word count)", "z_log_total_word_count")
    add_term(row, robust_result, "correctness", "z(ENEM correctness)", "z_enem_correctness")
    return row


def fit_analysis_model(
    trial_df: pd.DataFrame,
    *,
    outcome: str,
    model_id: str,
    analysis_label: str,
    analysis_scope: str,
    robust_variant: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    model_df = prepare_model_data(trial_df, outcome)
    base_result = fit_base_ols(model_df, outcome)
    log_rows: list[dict[str, Any]] = []

    if robust_variant is None:
        robust_result, chosen_variant, se_label, warnings_list = fit_primary_or_fallback(base_result, model_df)
    else:
        robust_result, se_label, warnings_list = apply_robust_covariance(base_result, model_df, robust_variant)
        chosen_variant = robust_variant

    row = build_result_row(
        robust_result,
        model_df,
        model_id=model_id,
        analysis_label=analysis_label,
        analysis_scope=analysis_scope,
        outcome=outcome,
        outcome_label="Front ROI HbO variable window" if outcome == "front_roi_mean_variable_lag4" else (
            "Front ROI HbO fixed 7-11 s window" if outcome == "front_roi_mean" else "Back ROI HbO variable window"
        ),
        robust_variant=chosen_variant,
        robust_se_label=se_label,
        warnings_list=warnings_list,
    )
    log_rows.append(
        {
            "stage": "model_fit",
            "level": "warning" if warnings_list else "info",
            "reason_code": f"MODEL_{model_id.upper()}",
            "n_rows": 1,
            "detail": f"Robust SE type={se_label}; warnings={' | '.join(warnings_list) if warnings_list else 'none'}",
        }
    )
    return pd.DataFrame([row]), {"base_result": base_result, "robust_result": robust_result, "data": model_df, "row": row}, log_rows


def run_leave_one_question_out(
    trial_df: pd.DataFrame,
    primary_variant: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    for question_id in sorted(trial_df["question_id"].astype(str).unique().tolist()):
        subset_df = trial_df[trial_df["question_id"].astype(str) != question_id].copy()
        model_df = prepare_model_data(subset_df, "front_roi_mean_variable_lag4")
        base_result = fit_base_ols(model_df, "front_roi_mean_variable_lag4")
        try:
            robust_result, se_label, warnings_list = apply_robust_covariance(base_result, model_df, primary_variant)
            robust_variant = primary_variant
        except Exception as exc:
            robust_result, se_label, warnings_list = apply_robust_covariance(base_result, model_df, "participant_cluster")
            warnings_list = warnings_list + [f"Primary variant fallback triggered: {exc}"]
            robust_variant = "participant_cluster"

        stats_row = term_stats(robust_result, "condition_abstract")
        rows.append(
            {
                "omitted_question_id": question_id,
                "robust_variant": robust_variant,
                "robust_se_label": se_label,
                "n_trials": int(len(model_df)),
                "n_participants": int(model_df["participant_id"].nunique()),
                "n_questions": int(model_df["question_id"].nunique()),
                "condition_beta": stats_row["beta"],
                "condition_se": stats_row["se"],
                "condition_stat": stats_row["stat"],
                "condition_p_value": stats_row["p_value"],
                "condition_ci_low": stats_row["ci_low"],
                "condition_ci_high": stats_row["ci_high"],
                "warning_count": len(warnings_list),
                "warnings": " | ".join(warnings_list),
            }
        )

    log_rows.append(
        {
            "stage": "influence",
            "level": "info",
            "reason_code": "LEAVE_ONE_QUESTION_OUT_COMPLETE",
            "n_rows": len(rows),
            "detail": "Completed one leave-one-question-out participant-fixed-effects refit per included question.",
        }
    )
    return pd.DataFrame(rows), log_rows


def run_leave_one_participant_out(
    trial_df: pd.DataFrame,
    primary_variant: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    for participant_id in sorted(trial_df["participant_id"].astype(str).unique().tolist()):
        subset_df = trial_df[trial_df["participant_id"].astype(str) != participant_id].copy()
        model_df = prepare_model_data(subset_df, "front_roi_mean_variable_lag4")
        base_result = fit_base_ols(model_df, "front_roi_mean_variable_lag4")
        try:
            robust_result, se_label, warnings_list = apply_robust_covariance(base_result, model_df, primary_variant)
            robust_variant = primary_variant
        except Exception as exc:
            robust_result, se_label, warnings_list = apply_robust_covariance(base_result, model_df, "participant_cluster")
            warnings_list = warnings_list + [f"Primary variant fallback triggered: {exc}"]
            robust_variant = "participant_cluster"

        stats_row = term_stats(robust_result, "condition_abstract")
        rows.append(
            {
                "omitted_participant_id": participant_id,
                "robust_variant": robust_variant,
                "robust_se_label": se_label,
                "n_trials": int(len(model_df)),
                "n_participants": int(model_df["participant_id"].nunique()),
                "n_questions": int(model_df["question_id"].nunique()),
                "condition_beta": stats_row["beta"],
                "condition_se": stats_row["se"],
                "condition_stat": stats_row["stat"],
                "condition_p_value": stats_row["p_value"],
                "condition_ci_low": stats_row["ci_low"],
                "condition_ci_high": stats_row["ci_high"],
                "warning_count": len(warnings_list),
                "warnings": " | ".join(warnings_list),
            }
        )

    log_rows.append(
        {
            "stage": "influence",
            "level": "info",
            "reason_code": "LEAVE_ONE_PARTICIPANT_OUT_COMPLETE",
            "n_rows": len(rows),
            "detail": "Completed one leave-one-participant-out participant-fixed-effects refit per included participant-session.",
        }
    )
    return pd.DataFrame(rows), log_rows


def plot_primary_coefficients(primary_row: pd.Series, path: Path) -> None:
    labels = [
        "Condition",
        "Global block",
        "Within block",
        "z(log word count)",
        "z(correctness)",
    ]
    prefixes = ["condition", "global_block_order", "within_block_position", "wordcount", "correctness"]
    effects = np.array([float(primary_row[f"{prefix}_beta"]) for prefix in prefixes]) * 1e6
    ci_low = np.array([float(primary_row[f"{prefix}_ci_low"]) for prefix in prefixes]) * 1e6
    ci_high = np.array([float(primary_row[f"{prefix}_ci_high"]) for prefix in prefixes]) * 1e6
    y_positions = np.arange(len(prefixes))

    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
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
    ax.set_xlabel(r"Coefficient estimate ($\mu$M)")
    ax.set_title("Primary participant-fixed-effects coefficients")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_coefficient_comparison(result_rows: list[pd.Series], labels: list[str], title: str, path: Path) -> None:
    effects = np.array([float(row["condition_beta"]) for row in result_rows]) * 1e6
    ci_low = np.array([float(row["condition_ci_low"]) for row in result_rows]) * 1e6
    ci_high = np.array([float(row["condition_ci_high"]) for row in result_rows]) * 1e6
    y_positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.0, max(3.4, 0.7 * len(labels) + 1.8)), layout="constrained")
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
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel(r"Condition coefficient ($\mu$M)")
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_leave_one_out(
    loo_df: pd.DataFrame,
    *,
    id_col: str,
    title: str,
    path: Path,
) -> None:
    plot_df = loo_df.sort_values("condition_beta").reset_index(drop=True).copy()
    labels = plot_df[id_col].astype(str).tolist()
    effects = plot_df["condition_beta"].to_numpy(dtype=float) * 1e6
    ci_low = plot_df["condition_ci_low"].to_numpy(dtype=float) * 1e6
    ci_high = plot_df["condition_ci_high"].to_numpy(dtype=float) * 1e6
    y_positions = np.arange(len(plot_df))

    fig_height = max(6.0, 0.28 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.0, fig_height), layout="constrained")
    ax.errorbar(
        effects,
        y_positions,
        xerr=np.vstack([effects - ci_low, ci_high - effects]),
        fmt="o",
        color="#336699",
        ecolor="#336699",
        elinewidth=1.5,
        capsize=3,
    )
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel(r"Condition coefficient ($\mu$M)")
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_final_conclusion(
    primary_df: pd.DataFrame,
    sensitivity_dfs: list[pd.DataFrame],
    loo_question_df: pd.DataFrame,
    loo_participant_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    primary = primary_df.iloc[0]
    primary_p = float(primary["condition_p_value"])
    primary_beta = float(primary["condition_beta"])
    primary_se = float(primary["condition_se"])
    primary_ci_low = float(primary["condition_ci_low"])
    primary_ci_high = float(primary["condition_ci_high"])

    sensitivity_ps = [float(df.iloc[0]["condition_p_value"]) for df in sensitivity_dfs]
    question_sign_change = bool(
        (loo_question_df["condition_beta"] < 0).any() and (loo_question_df["condition_beta"] > 0).any()
    )
    participant_sign_change = bool(
        (loo_participant_df["condition_beta"] < 0).any() and (loo_participant_df["condition_beta"] > 0).any()
    )
    influence_sensitive = question_sign_change or participant_sign_change

    if primary_p < ALPHA:
        conclusion_text = (
            "In the final participant-fixed-effects robustness analysis, the abstract-versus-concrete effect in the left frontal ROI remained after controlling for block order, within-block position, question length, and item difficulty "
            f"(beta={primary_beta:.3e}, SE={primary_se:.3e}, p={primary_p:.6f}, 95% CI [{primary_ci_low:.3e}, {primary_ci_high:.3e}]). "
            "This suggests that the frontal condition tendency survives a simpler regression framework with robust inference."
        )
    elif influence_sensitive:
        conclusion_text = (
            "In the final participant-fixed-effects robustness analysis, the abstract-versus-concrete effect in the left frontal ROI was not statistically significant "
            f"(beta={primary_beta:.3e}, SE={primary_se:.3e}, p={primary_p:.6f}, 95% CI [{primary_ci_low:.3e}, {primary_ci_high:.3e}]), "
            "and the condition coefficient changed sign across the leave-one-out influence checks. "
            "This indicates that the remaining frontal tendency is fragile and sensitive to a small subset of questions and/or participants."
        )
    else:
        conclusion_text = (
            "In the final participant-fixed-effects robustness analysis, the abstract-versus-concrete effect in the left frontal ROI was not statistically significant "
            f"(beta={primary_beta:.3e}, SE={primary_se:.3e}, p={primary_p:.6f}, 95% CI [{primary_ci_low:.3e}, {primary_ci_high:.3e}]). "
            "Together with the non-significant sensitivity models, this suggests that the frontal condition tendency remains descriptive but does not receive robust statistical support in the current dataset."
        )

    final_df = pd.DataFrame(
        [
            {
                "n_included_participants": int(primary["n_participants"]),
                "n_included_questions": int(primary["n_questions"]),
                "n_included_trials": int(primary["n_trials"]),
                "primary_robust_se_type": primary["robust_se_label"],
                "condition_beta": primary_beta,
                "condition_se": primary_se,
                "condition_p_value": primary_p,
                "condition_ci_low": primary_ci_low,
                "condition_ci_high": primary_ci_high,
                "all_sensitivity_p_values_ge_0_05": bool(all(p >= ALPHA for p in sensitivity_ps)),
                "leave_one_question_sign_change": question_sign_change,
                "leave_one_participant_sign_change": participant_sign_change,
                "conclusion_text": conclusion_text,
            }
        ]
    )
    return final_df, conclusion_text


def write_supporting_tex(
    primary_df: pd.DataFrame,
    sensitivity_rows: list[pd.Series],
    loo_question_df: pd.DataFrame,
    loo_participant_df: pd.DataFrame,
    conclusion_text: str,
) -> None:
    primary = primary_df.iloc[0]
    primary_rows = [
        ("Robust SE type", primary["robust_se_label"]),
        ("Included participants", int(primary["n_participants"])),
        ("Included questions", int(primary["n_questions"])),
        ("Included trials", int(primary["n_trials"])),
        ("Condition beta", format_effect(primary["condition_beta"])),
        ("Condition SE", format_effect(primary["condition_se"])),
        ("Condition test statistic", f"{float(primary['condition_stat']):.3f}"),
        ("Condition p-value", format_p_value(primary["condition_p_value"])),
        ("Condition 95% CI", f"[{format_effect(primary['condition_ci_low'])}, {format_effect(primary['condition_ci_high'])}]"),
        ("Global block-order beta", format_effect(primary["global_block_order_beta"])),
        ("Within-block-position beta", format_effect(primary["within_block_position_beta"])),
        ("Word-count beta", format_effect(primary["wordcount_beta"])),
        ("ENEM-correctness beta", format_effect(primary["correctness_beta"])),
    ]
    (TABLES_DIR / "primary_fe_table.tex").write_text(make_latex_table(primary_rows) + "\n", encoding="utf-8")

    sensitivity_table_rows = []
    for row in sensitivity_rows:
        sensitivity_table_rows.append(
            [
                row["analysis_label"],
                row["robust_se_label"],
                format_effect(row["condition_beta"]),
                format_effect(row["condition_se"]),
                format_p_value(row["condition_p_value"]),
                f"[{format_effect(row['condition_ci_low'])}, {format_effect(row['condition_ci_high'])}]",
            ]
        )
    question_range = (
        f"{loo_question_df['condition_beta'].min():.3e} to {loo_question_df['condition_beta'].max():.3e}; "
        f"p range {loo_question_df['condition_p_value'].min():.6f} to {loo_question_df['condition_p_value'].max():.6f}"
    )
    participant_range = (
        f"{loo_participant_df['condition_beta'].min():.3e} to {loo_participant_df['condition_beta'].max():.3e}; "
        f"p range {loo_participant_df['condition_p_value'].min():.6f} to {loo_participant_df['condition_p_value'].max():.6f}"
    )
    sensitivity_table_rows.append(["Leave-one-question-out range", "Primary robust SE", question_range, "", "", ""])
    sensitivity_table_rows.append(["Leave-one-participant-out range", "Primary robust SE", participant_range, "", "", ""])

    (TABLES_DIR / "sensitivity_table.tex").write_text(
        make_latex_grid(
            headers=["Model", "SE type", "Beta / range", "SE", "p", "95% CI"],
            rows=sensitivity_table_rows,
            column_spec=r"p{0.28\linewidth}p{0.20\linewidth}p{0.20\linewidth}p{0.10\linewidth}p{0.08\linewidth}p{0.12\linewidth}",
        )
        + "\n",
        encoding="utf-8",
    )
    (TEXT_DIR / "final_conclusion.tex").write_text(conclusion_text + "\n", encoding="utf-8")


def write_report(primary_df: pd.DataFrame, exclusion_log_df: pd.DataFrame) -> Path:
    primary = primary_df.iloc[0]
    warning_items = []
    warning_df = exclusion_log_df[exclusion_log_df["level"] == "warning"].copy()
    for _, row in warning_df.iterrows():
        warning_items.append(
            f"\\item \\texttt{{{sanitize_for_tex(row['reason_code'])}}}: {sanitize_for_tex(row['detail'])}"
        )
    if not warning_items:
        warning_items.append(r"\item No Step 10-specific warnings were generated.")

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{array}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Step 10 Final Robustness Report}",
        (
            "This report implements the final participant-fixed-effects robustness check using the same Step~9 duration-aware front ROI outcome and the same included cohort, "
            f"with {int(primary['n_participants'])} participant-sessions, {int(primary['n_questions'])} questions, and {int(primary['n_trials'])} trials."
        ),
        f"The primary inferential specification used {sanitize_for_tex(primary['robust_se_label'])}.",
        r"\subsection*{Primary Model}",
        r"\input{tables/primary_fe_table.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.78\linewidth]{../../figures/step10/primary_condition_coefficient_plot.png}",
        r"\caption{Coefficient plot for the primary participant-fixed-effects model.}",
        r"\end{figure}",
        r"\subsection*{Sensitivity Models}",
        r"\input{tables/sensitivity_table.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.78\linewidth]{../../figures/step10/robust_se_comparison.png}",
        r"\caption{Condition coefficient under the robust standard-error variants.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.78\linewidth]{../../figures/step10/fixed_vs_variable_fe_benchmark.png}",
        r"\caption{Final fixed-window versus duration-aware benchmark under participant fixed effects.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.78\linewidth]{../../figures/step10/no_override_comparison.png}",
        r"\caption{Primary cohort versus no-override sensitivity comparison.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\linewidth]{../../figures/step10/leave_one_question_out_plot.png}",
        r"\caption{Leave-one-question-out condition coefficients with 95\% confidence intervals.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\linewidth]{../../figures/step10/leave_one_participant_out_plot.png}",
        r"\caption{Leave-one-participant-out condition coefficients with 95\% confidence intervals.}",
        r"\end{figure}",
        r"\subsection*{Warnings And Notes}",
        r"\begin{itemize}",
        *warning_items,
        r"\end{itemize}",
        r"\subsection*{Final Conclusion}",
        r"\input{text/final_conclusion.tex}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step10_final_robustness_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()

    step9_trial_df = load_trial_table()
    primary_fe_df, exclusion_log_df = build_primary_fe_table(step9_trial_df)
    save_dataframe(primary_fe_df, CLEAN_DIR / "01_step10_primary_fe_table.csv")

    outputs: dict[str, pd.DataFrame] = {}
    fit_artifacts: dict[str, Any] = {}
    extra_logs: list[dict[str, Any]] = []

    primary_df, primary_artifact, log_rows = fit_analysis_model(
        primary_fe_df,
        outcome="front_roi_mean_variable_lag4",
        model_id="primary_fe",
        analysis_label="Primary participant-fixed-effects model",
        analysis_scope="primary",
        robust_variant=None,
    )
    outputs["02_step10_primary_fe_results.csv"] = primary_df
    fit_artifacts["primary"] = primary_artifact
    extra_logs.extend(log_rows)
    primary_variant = primary_df.iloc[0]["robust_variant"]

    participant_df, participant_artifact, log_rows = fit_analysis_model(
        primary_fe_df,
        outcome="front_roi_mean_variable_lag4",
        model_id="participant_cluster",
        analysis_label="Participant-clustered participant-fixed-effects model",
        analysis_scope="sensitivity",
        robust_variant="participant_cluster",
    )
    outputs["03_step10_cluster_participant_results.csv"] = participant_df
    fit_artifacts["participant_cluster"] = participant_artifact
    extra_logs.extend(log_rows)

    question_df, question_artifact, log_rows = fit_analysis_model(
        primary_fe_df,
        outcome="front_roi_mean_variable_lag4",
        model_id="question_cluster",
        analysis_label="Question-clustered participant-fixed-effects model",
        analysis_scope="sensitivity",
        robust_variant="question_cluster",
    )
    outputs["04_step10_cluster_question_results.csv"] = question_df
    fit_artifacts["question_cluster"] = question_artifact
    extra_logs.extend(log_rows)

    hc3_df, hc3_artifact, log_rows = fit_analysis_model(
        primary_fe_df,
        outcome="front_roi_mean_variable_lag4",
        model_id="hc3",
        analysis_label="HC3 participant-fixed-effects model",
        analysis_scope="sensitivity",
        robust_variant="hc3",
    )
    outputs["05_step10_hc3_results.csv"] = hc3_df
    fit_artifacts["hc3"] = hc3_artifact
    extra_logs.extend(log_rows)

    fixed_df, fixed_artifact, log_rows = fit_analysis_model(
        primary_fe_df,
        outcome="front_roi_mean",
        model_id="fixed_window_benchmark",
        analysis_label="Fixed-window front ROI benchmark under participant fixed effects",
        analysis_scope="benchmark",
        robust_variant=primary_variant,
    )
    outputs["06_step10_fixed_window_benchmark.csv"] = fixed_df
    fit_artifacts["fixed_window"] = fixed_artifact
    extra_logs.extend(log_rows)

    no_override_df, no_override_artifact, log_rows = fit_analysis_model(
        primary_fe_df[~primary_fe_df["manual_override_session"]].copy(),
        outcome="front_roi_mean_variable_lag4",
        model_id="no_override",
        analysis_label="No-override participant-fixed-effects model",
        analysis_scope="sensitivity",
        robust_variant=primary_variant,
    )
    outputs["07_step10_no_override_results.csv"] = no_override_df
    fit_artifacts["no_override"] = no_override_artifact
    extra_logs.extend(log_rows)
    extra_logs.append(
        {
            "stage": "sensitivity",
            "level": "info",
            "reason_code": "NO_OVERRIDE_REMOVAL",
            "n_rows": int(primary_fe_df["manual_override_session"].sum()),
            "detail": "The no-override sensitivity removed only PID025 and PID034, as specified.",
        }
    )

    loo_question_df, loo_q_logs = run_leave_one_question_out(primary_fe_df, primary_variant)
    outputs["08_step10_leave_one_question_out.csv"] = loo_question_df
    extra_logs.extend(loo_q_logs)

    loo_participant_df, loo_p_logs = run_leave_one_participant_out(primary_fe_df, primary_variant)
    outputs["09_step10_leave_one_participant_out.csv"] = loo_participant_df
    extra_logs.extend(loo_p_logs)

    back_df, back_artifact, log_rows = fit_analysis_model(
        primary_fe_df,
        outcome="back_roi_mean_variable_lag4",
        model_id="back_roi_benchmark",
        analysis_label="Back ROI benchmark under participant fixed effects",
        analysis_scope="benchmark_secondary",
        robust_variant=primary_variant,
    )
    outputs["10_step10_back_roi_benchmark.csv"] = back_df
    fit_artifacts["back"] = back_artifact
    extra_logs.extend(log_rows)

    exclusion_log_df = pd.concat([exclusion_log_df, pd.DataFrame(extra_logs)], ignore_index=True)
    save_dataframe(exclusion_log_df, CLEAN_DIR / "11_step10_exclusion_log.csv")

    sensitivity_dfs = [
        outputs["03_step10_cluster_participant_results.csv"],
        outputs["04_step10_cluster_question_results.csv"],
        outputs["05_step10_hc3_results.csv"],
        outputs["06_step10_fixed_window_benchmark.csv"],
        outputs["07_step10_no_override_results.csv"],
        outputs["10_step10_back_roi_benchmark.csv"],
    ]
    final_df, conclusion_text = build_final_conclusion(
        outputs["02_step10_primary_fe_results.csv"],
        sensitivity_dfs,
        loo_question_df,
        loo_participant_df,
    )
    outputs["12_step10_final_conclusion.csv"] = final_df

    for file_name, df in outputs.items():
        save_dataframe(df, CLEAN_DIR / file_name)

    plot_primary_coefficients(outputs["02_step10_primary_fe_results.csv"].iloc[0], FIGURES_DIR / "primary_condition_coefficient_plot.png")
    plot_coefficient_comparison(
        [outputs[file_name].iloc[0] for file_name, _ in SENSITIVITY_ORDER],
        [label for _, label in SENSITIVITY_ORDER],
        "Condition coefficient across robust SE variants",
        FIGURES_DIR / "robust_se_comparison.png",
    )
    plot_coefficient_comparison(
        [
            outputs["02_step10_primary_fe_results.csv"].iloc[0],
            outputs["06_step10_fixed_window_benchmark.csv"].iloc[0],
        ],
        ["Duration-aware", "Fixed 7-11 s"],
        "Duration-aware versus fixed-window FE benchmark",
        FIGURES_DIR / "fixed_vs_variable_fe_benchmark.png",
    )
    plot_coefficient_comparison(
        [
            outputs["02_step10_primary_fe_results.csv"].iloc[0],
            outputs["07_step10_no_override_results.csv"].iloc[0],
        ],
        ["Primary cohort", "No-override cohort"],
        "Primary versus no-override FE comparison",
        FIGURES_DIR / "no_override_comparison.png",
    )
    plot_leave_one_out(
        outputs["08_step10_leave_one_question_out.csv"],
        id_col="omitted_question_id",
        title="Leave-one-question-out condition coefficients",
        path=FIGURES_DIR / "leave_one_question_out_plot.png",
    )
    plot_leave_one_out(
        outputs["09_step10_leave_one_participant_out.csv"],
        id_col="omitted_participant_id",
        title="Leave-one-participant-out condition coefficients",
        path=FIGURES_DIR / "leave_one_participant_out_plot.png",
    )

    sensitivity_rows = [
        outputs["02_step10_primary_fe_results.csv"].iloc[0],
        outputs["03_step10_cluster_participant_results.csv"].iloc[0],
        outputs["04_step10_cluster_question_results.csv"].iloc[0],
        outputs["05_step10_hc3_results.csv"].iloc[0],
        outputs["06_step10_fixed_window_benchmark.csv"].iloc[0],
        outputs["07_step10_no_override_results.csv"].iloc[0],
        outputs["10_step10_back_roi_benchmark.csv"].iloc[0],
    ]
    write_supporting_tex(
        outputs["02_step10_primary_fe_results.csv"],
        sensitivity_rows,
        outputs["08_step10_leave_one_question_out.csv"],
        outputs["09_step10_leave_one_participant_out.csv"],
        conclusion_text,
    )
    report_path = write_report(outputs["02_step10_primary_fe_results.csv"], exclusion_log_df)
    compile_report(report_path)


if __name__ == "__main__":
    main()
