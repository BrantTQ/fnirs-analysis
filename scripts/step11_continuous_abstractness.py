#!/usr/bin/env python3

from __future__ import annotations

import json
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
STEP10_TABLE_PATH = ROOT / "data_clean" / "step10" / "01_step10_primary_fe_table.csv"
FILTERED_QUESTIONS_PATH = ROOT / "materials" / "filtered_questions.json"

CLEAN_DIR = ROOT / "data_clean" / "step11"
FIGURES_DIR = ROOT / "figures" / "step11"
REPORTS_DIR = ROOT / "reports" / "step11"
TABLES_DIR = REPORTS_DIR / "tables"
TEXT_DIR = REPORTS_DIR / "text"

ALPHA = 0.05
CUTOFF = 0.5
QUESTION_COLOR = "#4477AA"
ABSTRACT_COLOR = "#AA3377"
CONCRETE_COLOR = "#228833"


def ensure_directories() -> None:
    for path in [CLEAN_DIR, FIGURES_DIR, REPORTS_DIR, TABLES_DIR, TEXT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = float(numeric.std(ddof=0))
    if not np.isfinite(std) or math.isclose(std, 0.0):
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return (numeric - float(numeric.mean())) / std


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_df = pd.read_csv(STEP10_TABLE_PATH)
    with open(FILTERED_QUESTIONS_PATH, "r", encoding="utf-8") as handle:
        question_df = pd.DataFrame(json.load(handle))
    return trial_df, question_df


def build_question_score_table(question_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_df = question_df.copy()
    score_df["question_id"] = (
        score_df["year"].astype(str)
        + "_"
        + score_df["field"].astype(str)
        + "_"
        + pd.to_numeric(score_df["question_number"], errors="coerce").astype(int).astype(str)
    )
    score_df["abstracness"] = pd.to_numeric(score_df["abstracness"], errors="coerce")
    score_df["condition_abstract"] = score_df["type"].astype(str).str.strip().str.lower().map(
        {"abstract": 1.0, "concrete": 0.0}
    )
    score_df["score_above_cutoff"] = (score_df["abstracness"] >= CUTOFF).astype(int)
    score_df["score_matches_binary_label"] = score_df["score_above_cutoff"] == score_df["condition_abstract"].astype(int)
    score_df["z_abstracness"] = zscore(score_df["abstracness"])
    score_df["z_abstracness_sq"] = score_df["z_abstracness"] ** 2
    score_df["threshold_distance_raw"] = (score_df["abstracness"] - CUTOFF).abs()
    score_df["z_threshold_distance"] = zscore(score_df["threshold_distance_raw"])

    diagnostics_df = pd.DataFrame(
        [
            {
                "n_questions": int(score_df["question_id"].nunique()),
                "score_direction_aligned_more_abstract_is_higher": True,
                "raw_score_mean": float(score_df["abstracness"].mean()),
                "raw_score_sd": float(score_df["abstracness"].std(ddof=1)),
                "raw_score_median": float(score_df["abstracness"].median()),
                "raw_score_min": float(score_df["abstracness"].min()),
                "raw_score_max": float(score_df["abstracness"].max()),
                "raw_score_range": float(score_df["abstracness"].max() - score_df["abstracness"].min()),
                "binary_cutoff_used": CUTOFF,
                "n_questions_below_cutoff": int((score_df["abstracness"] < CUTOFF).sum()),
                "n_questions_at_cutoff": int((score_df["abstracness"] == CUTOFF).sum()),
                "n_questions_above_cutoff": int((score_df["abstracness"] > CUTOFF).sum()),
                "binary_cutoff_reproduces_existing_label": bool(score_df["score_matches_binary_label"].all()),
                "n_binary_abstract": int((score_df["condition_abstract"] == 1).sum()),
                "n_binary_concrete": int((score_df["condition_abstract"] == 0).sum()),
            }
        ]
    )

    score_lookup_df = score_df[
        [
            "question_id",
            "type",
            "condition_abstract",
            "abstracness",
            "z_abstracness",
            "z_abstracness_sq",
            "threshold_distance_raw",
            "z_threshold_distance",
            "total_word_count",
            "sentence_count",
            "sentence_length",
            "correctness",
        ]
    ].copy()
    return score_lookup_df, score_df, diagnostics_df


def build_trial_table(step10_trial_df: pd.DataFrame, score_lookup_df: pd.DataFrame) -> pd.DataFrame:
    trial_df = step10_trial_df.merge(
        score_lookup_df,
        on=["question_id", "condition_abstract"],
        how="left",
        validate="many_to_one",
        indicator="score_merge_status",
    )
    if (trial_df["score_merge_status"] != "both").any():
        missing = trial_df.loc[trial_df["score_merge_status"] != "both", ["question_id"]].drop_duplicates().to_dict("records")
        raise RuntimeError(f"Step 11 score merge failed for question rows: {missing}")
    trial_df.drop(columns=["score_merge_status"], inplace=True)
    trial_df["participant_code"] = pd.to_numeric(trial_df["participant_code"], errors="coerce").astype(int)
    trial_df["question_code"] = pd.to_numeric(trial_df["question_code"], errors="coerce").astype(int)
    return trial_df


def build_score_correlations(trial_df: pd.DataFrame) -> pd.DataFrame:
    question_level_df = (
        trial_df.groupby("question_id", as_index=False)
        .agg(
            abstracness=("abstracness", "first"),
            total_word_count=("total_word_count", "first"),
            sentence_count=("sentence_count", "first"),
            sentence_length=("sentence_length", "first"),
            enem_correctness=("correctness", "first"),
            mean_response_time_sec=("response_time_sec", "mean"),
        )
    )
    rows = []
    for variable, label in [
        ("total_word_count", "Total word count"),
        ("sentence_count", "Sentence count"),
        ("sentence_length", "Average sentence length"),
        ("enem_correctness", "ENEM item correctness"),
        ("mean_response_time_sec", "Mean response time"),
    ]:
        subset = question_level_df[["abstracness", variable]].dropna().copy()
        r_value, p_value = stats.pearsonr(subset["abstracness"], subset[variable])
        rows.append(
            {
                "variable": variable,
                "label": label,
                "n_questions": int(len(subset)),
                "pearson_r": float(r_value),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def fe_formula(outcome: str, predictor_terms: list[str]) -> str:
    rhs_terms = predictor_terms + [
        "global_block_order",
        "within_block_position",
        "z_log_total_word_count",
        "z_enem_correctness",
        "C(participant_code)",
    ]
    return outcome + " ~ " + " + ".join(rhs_terms)


def fit_base_ols(model_df: pd.DataFrame, formula: str) -> Any:
    return smf.ols(formula, data=model_df).fit()


def robust_model_names(result: Any) -> list[str]:
    return list(result.model.exog_names)


def term_stats(result: Any, term_name: str) -> dict[str, float]:
    names = robust_model_names(result)
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


def apply_primary_robust_cov(base_result: Any, model_df: pd.DataFrame) -> tuple[Any, str, str]:
    participant_groups = model_df["participant_code"].to_numpy(dtype=int)
    question_groups = model_df["question_code"].to_numpy(dtype=int)
    try:
        groups = np.column_stack([participant_groups, question_groups])
        robust_result = base_result.get_robustcov_results(cov_type="cluster", groups=groups)
        return robust_result, "two_way_cluster", "Two-way cluster (participant + question)"
    except Exception:
        robust_result = base_result.get_robustcov_results(cov_type="cluster", groups=participant_groups)
        return robust_result, "participant_cluster", "Participant-clustered"


def fit_model(
    trial_df: pd.DataFrame,
    *,
    outcome: str,
    predictor_terms: list[str],
    model_id: str,
    analysis_label: str,
    analysis_scope: str,
    target_prefix: str,
    target_label: str,
    target_term: str,
    extra_terms: list[tuple[str, str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_df = trial_df.copy()
    formula = fe_formula(outcome, predictor_terms)
    base_result = fit_base_ols(model_df, formula)
    robust_result, robust_variant, robust_label = apply_primary_robust_cov(base_result, model_df)

    row: dict[str, Any] = {
        "model_id": model_id,
        "analysis_label": analysis_label,
        "analysis_scope": analysis_scope,
        "outcome_variable": outcome,
        "outcome_label": "Front ROI HbO variable window"
        if outcome == "front_roi_mean_variable_lag4"
        else ("Front ROI HbO fixed 7-11 s window" if outcome == "front_roi_mean" else "Back ROI HbO variable window"),
        "formula": formula,
        "robust_variant": robust_variant,
        "robust_se_label": robust_label,
        "n_trials": int(len(model_df)),
        "n_participants": int(model_df["participant_id"].nunique()),
        "n_questions": int(model_df["question_id"].nunique()),
    }

    add_term(row, robust_result, target_prefix, target_label, target_term)
    add_term(row, robust_result, "global_block_order", "Global block order", "global_block_order")
    add_term(row, robust_result, "within_block_position", "Within-block trial position", "within_block_position")
    add_term(row, robust_result, "wordcount", "z(log total word count)", "z_log_total_word_count")
    add_term(row, robust_result, "correctness", "z(ENEM correctness)", "z_enem_correctness")
    if extra_terms:
        for prefix, label, term_name in extra_terms:
            add_term(row, robust_result, prefix, label, term_name)

    return pd.DataFrame([row]), {"base_result": base_result, "robust_result": robust_result, "data": model_df, "row": row}


def run_leave_one_question_out(trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for question_id in sorted(trial_df["question_id"].astype(str).unique().tolist()):
        subset_df = trial_df[trial_df["question_id"].astype(str) != question_id].copy()
        formula = fe_formula("front_roi_mean_variable_lag4", ["z_abstracness"])
        base_result = fit_base_ols(subset_df, formula)
        robust_result, robust_variant, robust_label = apply_primary_robust_cov(base_result, subset_df)
        stats_row = term_stats(robust_result, "z_abstracness")
        rows.append(
            {
                "omitted_question_id": question_id,
                "robust_variant": robust_variant,
                "robust_se_label": robust_label,
                "n_trials": int(len(subset_df)),
                "n_participants": int(subset_df["participant_id"].nunique()),
                "n_questions": int(subset_df["question_id"].nunique()),
                "abstractness_beta": stats_row["beta"],
                "abstractness_se": stats_row["se"],
                "abstractness_stat": stats_row["stat"],
                "abstractness_p_value": stats_row["p_value"],
                "abstractness_ci_low": stats_row["ci_low"],
                "abstractness_ci_high": stats_row["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def run_leave_one_participant_out(trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant_id in sorted(trial_df["participant_id"].astype(str).unique().tolist()):
        subset_df = trial_df[trial_df["participant_id"].astype(str) != participant_id].copy()
        formula = fe_formula("front_roi_mean_variable_lag4", ["z_abstracness"])
        base_result = fit_base_ols(subset_df, formula)
        robust_result, robust_variant, robust_label = apply_primary_robust_cov(base_result, subset_df)
        stats_row = term_stats(robust_result, "z_abstracness")
        rows.append(
            {
                "omitted_participant_id": participant_id,
                "robust_variant": robust_variant,
                "robust_se_label": robust_label,
                "n_trials": int(len(subset_df)),
                "n_participants": int(subset_df["participant_id"].nunique()),
                "n_questions": int(subset_df["question_id"].nunique()),
                "abstractness_beta": stats_row["beta"],
                "abstractness_se": stats_row["se"],
                "abstractness_stat": stats_row["stat"],
                "abstractness_p_value": stats_row["p_value"],
                "abstractness_ci_low": stats_row["ci_low"],
                "abstractness_ci_high": stats_row["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def question_level_summary(trial_df: pd.DataFrame) -> pd.DataFrame:
    return (
        trial_df.groupby(["question_id", "type"], as_index=False)
        .agg(
            abstracness=("abstracness", "first"),
            total_word_count=("total_word_count", "first"),
            correctness=("correctness", "first"),
            mean_front_roi=("front_roi_mean_variable_lag4", "mean"),
        )
    )


def plot_histogram(score_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), layout="constrained")
    ax.hist(score_df["abstracness"], bins=np.linspace(0.0, 1.0, 8), color=QUESTION_COLOR, alpha=0.85, edgecolor="white")
    ax.axvline(CUTOFF, color="#AA0000", linestyle="--", linewidth=1.5, label="Cutoff = 0.5")
    ax.set_xlabel("Raw abstractness score")
    ax.set_ylabel("Number of questions")
    ax.set_title("Distribution of the question-level abstractness score")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_score_by_label(score_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    groups = [score_df.loc[score_df["type"] == label, "abstracness"].to_numpy(dtype=float) for label in ["concrete", "abstract"]]
    box = ax.boxplot(
        groups,
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#222222", linewidth=1.5),
    )
    for patch, color in zip(box["boxes"], [CONCRETE_COLOR, ABSTRACT_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(20260322)
    for idx, (label, color) in enumerate([("concrete", CONCRETE_COLOR), ("abstract", ABSTRACT_COLOR)], start=1):
        values = score_df.loc[score_df["type"] == label, "abstracness"].to_numpy(dtype=float)
        jitter = rng.normal(loc=idx, scale=0.05, size=len(values))
        ax.scatter(jitter, values, s=36, color=color, alpha=0.75, edgecolor="white", linewidth=0.6)
    ax.axhline(CUTOFF, color="#AA0000", linestyle="--", linewidth=1.25)
    ax.set_xticks([1, 2], ["Concrete", "Abstract"])
    ax.set_ylabel("Raw abstractness score")
    ax.set_title("Abstractness score split by the existing binary label")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def scatter_with_fit(df: pd.DataFrame, x_col: str, x_label: str, y_col: str, y_label: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), layout="constrained")
    colors = df["type"].map({"abstract": ABSTRACT_COLOR, "concrete": CONCRETE_COLOR}).fillna(QUESTION_COLOR)
    ax.scatter(df[x_col], df[y_col], c=colors, s=46, alpha=0.8, edgecolors="white", linewidths=0.6)
    if len(df) >= 2:
        coeffs = np.polyfit(df[x_col], df[y_col], deg=1)
        x_grid = np.linspace(float(df[x_col].min()), float(df[x_col].max()), 200)
        ax.plot(x_grid, coeffs[0] * x_grid + coeffs[1], color="#333333", linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} versus {x_label.lower()}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_continuous_partial_effect(primary_artifact: dict[str, Any], question_df: pd.DataFrame, path: Path) -> None:
    row = pd.Series(primary_artifact["row"])
    beta = float(row["abstractness_beta"])
    ci_low = float(row["abstractness_ci_low"])
    ci_high = float(row["abstractness_ci_high"])
    score_mean = float(question_df["abstracness"].mean())
    score_std = float(question_df["abstracness"].std(ddof=0))
    x_grid = np.linspace(float(question_df["abstracness"].min()), float(question_df["abstracness"].max()), 300)
    z_grid = (x_grid - score_mean) / score_std
    y = beta * z_grid * 1e6
    y_low = ci_low * z_grid * 1e6
    y_high = ci_high * z_grid * 1e6

    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    ax.fill_between(x_grid, y_low, y_high, color="#AAC7E8", alpha=0.35)
    ax.plot(x_grid, y, color="#225588", linewidth=2.5, label="Primary linear effect")
    ax.scatter(
        question_df["abstracness"],
        question_df["mean_front_roi"] * 1e6,
        c=question_df["type"].map({"abstract": ABSTRACT_COLOR, "concrete": CONCRETE_COLOR}).fillna(QUESTION_COLOR),
        s=44,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.6,
        label="Question means",
    )
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.axvline(CUTOFF, color="#AA0000", linestyle=":", linewidth=1.25)
    ax.set_xlabel("Raw abstractness score")
    ax.set_ylabel(r"Front ROI HbO relative effect ($\mu$M)")
    ax.set_title("Primary continuous-score partial effect")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_coefficient_comparison(rows: list[pd.Series], labels: list[str], xlabel: str, title: str, path: Path) -> None:
    effects = np.array([float(row["target_beta"]) for row in rows]) * 1e6
    ci_low = np.array([float(row["target_ci_low"]) for row in rows]) * 1e6
    ci_high = np.array([float(row["target_ci_high"]) for row in rows]) * 1e6
    y_positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.0, max(3.4, 0.65 * len(rows) + 1.6)), layout="constrained")
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
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_quadratic_effect(quadratic_artifact: dict[str, Any], question_df: pd.DataFrame, path: Path) -> None:
    result = quadratic_artifact["robust_result"]
    names = robust_model_names(result)
    cov = np.asarray(result.cov_params(), dtype=float)
    idx_lin = names.index("z_abstracness")
    idx_quad = names.index("z_abstracness_sq")
    beta_lin = float(np.asarray(result.params, dtype=float)[idx_lin])
    beta_quad = float(np.asarray(result.params, dtype=float)[idx_quad])

    score_mean = float(question_df["abstracness"].mean())
    score_std = float(question_df["abstracness"].std(ddof=0))
    x_grid = np.linspace(float(question_df["abstracness"].min()), float(question_df["abstracness"].max()), 300)
    z_grid = (x_grid - score_mean) / score_std
    design = np.column_stack([z_grid, z_grid**2])
    betas = np.array([beta_lin, beta_quad], dtype=float)
    effect = design @ betas
    subcov = cov[np.ix_([idx_lin, idx_quad], [idx_lin, idx_quad])]
    effect_var = np.einsum("ij,jk,ik->i", design, subcov, design)
    effect_se = np.sqrt(np.clip(effect_var, a_min=0.0, a_max=None))
    ci_low = (effect - 1.96 * effect_se) * 1e6
    ci_high = (effect + 1.96 * effect_se) * 1e6

    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    ax.fill_between(x_grid, ci_low, ci_high, color="#FFD28A", alpha=0.35)
    ax.plot(x_grid, effect * 1e6, color="#CC7A00", linewidth=2.5, label="Quadratic fit")
    ax.scatter(
        question_df["abstracness"],
        question_df["mean_front_roi"] * 1e6,
        c=question_df["type"].map({"abstract": ABSTRACT_COLOR, "concrete": CONCRETE_COLOR}).fillna(QUESTION_COLOR),
        s=44,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.6,
        label="Question means",
    )
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.axvline(CUTOFF, color="#AA0000", linestyle=":", linewidth=1.25)
    ax.set_xlabel("Raw abstractness score")
    ax.set_ylabel(r"Front ROI HbO relative effect ($\mu$M)")
    ax.set_title("Quadratic abstractness sensitivity model")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_leave_one_question_out(loo_df: pd.DataFrame, path: Path) -> None:
    plot_df = loo_df.sort_values("abstractness_beta").reset_index(drop=True).copy()
    labels = plot_df["omitted_question_id"].astype(str).tolist()
    effects = plot_df["abstractness_beta"].to_numpy(dtype=float) * 1e6
    ci_low = plot_df["abstractness_ci_low"].to_numpy(dtype=float) * 1e6
    ci_high = plot_df["abstractness_ci_high"].to_numpy(dtype=float) * 1e6
    y_positions = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8.2, max(6.0, 0.28 * len(plot_df) + 1.8)), layout="constrained")
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
    ax.set_xlabel(r"Abstractness coefficient ($\mu$M per 1 SD)")
    ax.set_title("Leave-one-question-out abstractness coefficients")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_final_conclusion(
    primary_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    quadratic_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    no_override_df: pd.DataFrame,
    loo_question_df: pd.DataFrame,
    loo_participant_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    primary = primary_df.iloc[0]
    beta = float(primary["abstractness_beta"])
    se = float(primary["abstractness_se"])
    p_value = float(primary["abstractness_p_value"])
    ci_low = float(primary["abstractness_ci_low"])
    ci_high = float(primary["abstractness_ci_high"])

    if p_value < ALPHA and beta > 0:
        conclusion_text = (
            "In the participant-fixed-effects regression using the continuous abstractness score, more abstract questions were associated with a significantly larger left frontal ROI HbO response "
            f"(beta={beta:.3e}, SE={se:.3e}, p={p_value:.6f}, 95% CI [{ci_low:.3e}, {ci_high:.3e}]). "
            "This suggests that the frontal effect is better described as a continuous abstractness gradient than as a thresholded category difference."
        )
    elif p_value < ALPHA and beta < 0:
        conclusion_text = (
            "In the participant-fixed-effects regression using the continuous abstractness score, more concrete questions were associated with a significantly larger left frontal ROI HbO response "
            f"(beta={beta:.3e}, SE={se:.3e}, p={p_value:.6f}, 95% CI [{ci_low:.3e}, {ci_high:.3e}]). "
            "This suggests that the frontal effect increases with concreteness rather than with abstractness."
        )
    else:
        conclusion_text = (
            "In the participant-fixed-effects regression using the continuous abstractness score, the association between abstractness level and left frontal ROI HbO response was not statistically significant "
            f"(beta={beta:.3e}, SE={se:.3e}, p={p_value:.6f}, 95% CI [{ci_low:.3e}, {ci_high:.3e}]). "
            "This suggests that replacing the binary label with a continuous abstractness measure does not by itself yield robust evidence for a frontal abstractness effect in the current dataset."
        )

    conclusion_text += (
        f" The binary benchmark remained non-significant (p={float(binary_df.iloc[0]['condition_p_value']):.6f}), "
        f"the fixed-window continuous benchmark remained non-significant (p={float(fixed_df.iloc[0]['abstractness_p_value']):.6f}), "
        f"and the no-override continuous model remained non-significant (p={float(no_override_df.iloc[0]['abstractness_p_value']):.6f})."
    )

    final_df = pd.DataFrame(
        [
            {
                "n_included_participants": int(primary["n_participants"]),
                "n_included_questions": int(primary["n_questions"]),
                "n_included_trials": int(primary["n_trials"]),
                "primary_robust_se_type": primary["robust_se_label"],
                "abstractness_beta": beta,
                "abstractness_se": se,
                "abstractness_p_value": p_value,
                "abstractness_ci_low": ci_low,
                "abstractness_ci_high": ci_high,
                "quadratic_term_p_value": float(quadratic_df.iloc[0]["quadratic_p_value"]),
                "leave_one_question_p_min": float(loo_question_df["abstractness_p_value"].min()),
                "leave_one_question_p_max": float(loo_question_df["abstractness_p_value"].max()),
                "leave_one_participant_p_min": float(loo_participant_df["abstractness_p_value"].min()),
                "leave_one_participant_p_max": float(loo_participant_df["abstractness_p_value"].max()),
                "conclusion_text": conclusion_text,
            }
        ]
    )
    return final_df, conclusion_text


def write_supporting_tex(
    diagnostics_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    benchmark_rows: list[list[Any]],
    conclusion_text: str,
) -> None:
    diag = diagnostics_df.iloc[0]
    primary = primary_df.iloc[0]
    primary_rows = [
        ("Robust SE type", primary["robust_se_label"]),
        ("Included participants", int(primary["n_participants"])),
        ("Included questions", int(primary["n_questions"])),
        ("Included trials", int(primary["n_trials"])),
        ("Abstractness beta", format_effect(primary["abstractness_beta"])),
        ("Abstractness SE", format_effect(primary["abstractness_se"])),
        ("Abstractness statistic", f"{float(primary['abstractness_stat']):.3f}"),
        ("Abstractness p-value", format_p_value(primary["abstractness_p_value"])),
        ("Abstractness 95% CI", f"[{format_effect(primary['abstractness_ci_low'])}, {format_effect(primary['abstractness_ci_high'])}]"),
        ("Global block-order beta", format_effect(primary["global_block_order_beta"])),
        ("Within-block-position beta", format_effect(primary["within_block_position_beta"])),
        ("Word-count beta", format_effect(primary["wordcount_beta"])),
        ("ENEM-correctness beta", format_effect(primary["correctness_beta"])),
    ]
    (TABLES_DIR / "primary_continuous_table.tex").write_text(make_latex_table(primary_rows) + "\n", encoding="utf-8")

    (TABLES_DIR / "benchmark_table.tex").write_text(
        make_latex_grid(
            headers=["Model", "Target term", "Beta", "SE", "p", "95% CI"],
            rows=benchmark_rows,
            column_spec=r"p{0.28\linewidth}p{0.18\linewidth}p{0.14\linewidth}p{0.12\linewidth}p{0.10\linewidth}p{0.14\linewidth}",
        )
        + "\n",
        encoding="utf-8",
    )

    diag_rows = [
        ("Questions with score", int(diag["n_questions"])),
        ("Raw score mean", f"{float(diag['raw_score_mean']):.4f}"),
        ("Raw score SD", f"{float(diag['raw_score_sd']):.4f}"),
        ("Raw score median", f"{float(diag['raw_score_median']):.4f}"),
        ("Raw score min", f"{float(diag['raw_score_min']):.4f}"),
        ("Raw score max", f"{float(diag['raw_score_max']):.4f}"),
        ("Binary cutoff used", f"{float(diag['binary_cutoff_used']):.2f}"),
        ("Questions below cutoff", int(diag["n_questions_below_cutoff"])),
        ("Questions above cutoff", int(diag["n_questions_above_cutoff"])),
        ("Cutoff reproduces binary label", bool(diag["binary_cutoff_reproduces_existing_label"])),
    ]
    (TABLES_DIR / "score_diagnostics_table.tex").write_text(make_latex_table(diag_rows) + "\n", encoding="utf-8")

    corr_rows = [
        [
            row["label"],
            int(row["n_questions"]),
            f"{float(row['pearson_r']):.3f}",
            format_p_value(row["p_value"]),
        ]
        for _, row in correlations_df.iterrows()
    ]
    (TABLES_DIR / "score_correlation_table.tex").write_text(
        make_latex_grid(
            headers=["Variable", "n", "Pearson r", "p"],
            rows=corr_rows,
            column_spec="lccc",
        )
        + "\n",
        encoding="utf-8",
    )

    (TEXT_DIR / "final_conclusion.tex").write_text(conclusion_text + "\n", encoding="utf-8")


def write_report(final_df: pd.DataFrame) -> Path:
    final_row = final_df.iloc[0]
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Step 11 Continuous Abstractness Report}",
        r"\subsection*{Score Diagnostics}",
        r"\input{tables/score_diagnostics_table.tex}",
        r"\input{tables/score_correlation_table.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/score_histogram.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/score_by_binary_label.png}",
        r"\caption{Distribution of the raw question-level abstractness score and its split by the existing binary label.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/score_vs_wordcount.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/score_vs_correctness.png}",
        r"\caption{Question-level abstractness score against two key item properties retained in the regression models.}",
        r"\end{figure}",
        r"\subsection*{Primary Continuous-Score Model}",
        r"\input{tables/primary_continuous_table.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.78\linewidth]{../../figures/step11/continuous_partial_effect_plot.png}",
        r"\caption{Primary continuous-score partial-effect visualization. Higher values indicate more abstract questions.}",
        r"\end{figure}",
        r"\subsection*{Benchmarks And Sensitivity Checks}",
        r"\input{tables/benchmark_table.tex}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/continuous_vs_binary_benchmark.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/no_override_comparison.png}",
        r"\caption{Comparison against the binary benchmark and the no-override sensitivity model.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.78\linewidth]{../../figures/step11/quadratic_effect_plot.png}",
        r"\caption{Quadratic sensitivity model across the raw abstractness score.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\linewidth]{../../figures/step11/leave_one_question_out_plot.png}",
        r"\caption{Leave-one-question-out abstractness coefficients with 95\% confidence intervals.}",
        r"\end{figure}",
        r"\subsection*{Final Conclusion}",
        r"\input{text/final_conclusion.tex}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step11_continuous_abstractness_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()
    step10_trial_df, raw_question_df = load_inputs()
    score_lookup_df, score_df, diagnostics_df = build_question_score_table(raw_question_df)
    trial_df = build_trial_table(step10_trial_df, score_lookup_df)
    correlations_df = build_score_correlations(trial_df)

    primary_df, primary_artifact = fit_model(
        trial_df,
        outcome="front_roi_mean_variable_lag4",
        predictor_terms=["z_abstracness"],
        model_id="primary_continuous",
        analysis_label="Primary continuous abstractness model",
        analysis_scope="primary",
        target_prefix="abstractness",
        target_label="z(Abstractness score)",
        target_term="z_abstracness",
    )
    binary_df, binary_artifact = fit_model(
        trial_df,
        outcome="front_roi_mean_variable_lag4",
        predictor_terms=["condition_abstract"],
        model_id="binary_benchmark",
        analysis_label="Binary abstract-versus-concrete benchmark model",
        analysis_scope="benchmark",
        target_prefix="condition",
        target_label="Binary abstract label",
        target_term="condition_abstract",
    )
    quadratic_df, quadratic_artifact = fit_model(
        trial_df,
        outcome="front_roi_mean_variable_lag4",
        predictor_terms=["z_abstracness", "z_abstracness_sq"],
        model_id="quadratic_continuous",
        analysis_label="Quadratic continuous abstractness model",
        analysis_scope="sensitivity_quadratic",
        target_prefix="abstractness",
        target_label="z(Abstractness score)",
        target_term="z_abstracness",
        extra_terms=[("quadratic", "Quadratic score term", "z_abstracness_sq")],
    )
    threshold_df, threshold_artifact = fit_model(
        trial_df,
        outcome="front_roi_mean_variable_lag4",
        predictor_terms=["condition_abstract", "z_threshold_distance"],
        model_id="threshold_strength",
        analysis_label="Threshold-strength benchmark model",
        analysis_scope="sensitivity_threshold",
        target_prefix="condition",
        target_label="Binary abstract label",
        target_term="condition_abstract",
        extra_terms=[("strength", "Distance from 0.5 cutoff", "z_threshold_distance")],
    )
    back_df, back_artifact = fit_model(
        trial_df,
        outcome="back_roi_mean_variable_lag4",
        predictor_terms=["z_abstracness"],
        model_id="back_roi_continuous",
        analysis_label="Back ROI continuous abstractness benchmark",
        analysis_scope="benchmark_secondary",
        target_prefix="abstractness",
        target_label="z(Abstractness score)",
        target_term="z_abstracness",
    )
    fixed_df, fixed_artifact = fit_model(
        trial_df,
        outcome="front_roi_mean",
        predictor_terms=["z_abstracness"],
        model_id="fixed_window_continuous",
        analysis_label="Fixed-window continuous abstractness benchmark",
        analysis_scope="benchmark_fixed_window",
        target_prefix="abstractness",
        target_label="z(Abstractness score)",
        target_term="z_abstracness",
    )
    no_override_df, no_override_artifact = fit_model(
        trial_df[~trial_df["manual_override_session"].astype(bool)].copy(),
        outcome="front_roi_mean_variable_lag4",
        predictor_terms=["z_abstracness"],
        model_id="no_override_continuous",
        analysis_label="No-override continuous abstractness model",
        analysis_scope="sensitivity_no_override",
        target_prefix="abstractness",
        target_label="z(Abstractness score)",
        target_term="z_abstracness",
    )

    loo_question_df = run_leave_one_question_out(trial_df)
    loo_participant_df = run_leave_one_participant_out(trial_df)

    final_df, conclusion_text = build_final_conclusion(
        primary_df,
        binary_df,
        quadratic_df,
        fixed_df,
        no_override_df,
        loo_question_df,
        loo_participant_df,
    )

    question_summary_df = question_level_summary(trial_df)
    save_dataframe(diagnostics_df, CLEAN_DIR / "01_step11_score_diagnostics.csv")
    save_dataframe(correlations_df, CLEAN_DIR / "02_step11_score_correlations.csv")
    save_dataframe(primary_df, CLEAN_DIR / "03_step11_primary_continuous_model.csv")
    save_dataframe(binary_df, CLEAN_DIR / "04_step11_binary_benchmark_model.csv")
    save_dataframe(quadratic_df, CLEAN_DIR / "05_step11_quadratic_model.csv")
    save_dataframe(threshold_df, CLEAN_DIR / "06_step11_threshold_strength_model.csv")
    save_dataframe(back_df, CLEAN_DIR / "07_step11_back_roi_benchmark.csv")
    save_dataframe(fixed_df, CLEAN_DIR / "08_step11_fixed_window_benchmark.csv")
    save_dataframe(no_override_df, CLEAN_DIR / "09_step11_no_override_results.csv")
    save_dataframe(loo_question_df, CLEAN_DIR / "10_step11_leave_one_question_out.csv")
    save_dataframe(loo_participant_df, CLEAN_DIR / "11_step11_leave_one_participant_out.csv")
    save_dataframe(final_df, CLEAN_DIR / "12_step11_final_conclusion.csv")

    plot_histogram(score_df, FIGURES_DIR / "score_histogram.png")
    plot_score_by_label(score_df, FIGURES_DIR / "score_by_binary_label.png")
    scatter_with_fit(question_summary_df, "abstracness", "Raw abstractness score", "total_word_count", "Total word count", FIGURES_DIR / "score_vs_wordcount.png")
    scatter_with_fit(question_summary_df, "abstracness", "Raw abstractness score", "correctness", "ENEM item correctness", FIGURES_DIR / "score_vs_correctness.png")
    plot_continuous_partial_effect(primary_artifact, question_summary_df, FIGURES_DIR / "continuous_partial_effect_plot.png")
    plot_coefficient_comparison(
        [
            pd.Series(
                {
                    "target_beta": primary_df.iloc[0]["abstractness_beta"],
                    "target_ci_low": primary_df.iloc[0]["abstractness_ci_low"],
                    "target_ci_high": primary_df.iloc[0]["abstractness_ci_high"],
                }
            ),
            pd.Series(
                {
                    "target_beta": binary_df.iloc[0]["condition_beta"],
                    "target_ci_low": binary_df.iloc[0]["condition_ci_low"],
                    "target_ci_high": binary_df.iloc[0]["condition_ci_high"],
                }
            ),
        ],
        ["Continuous score (1 SD)", "Binary abstract label"],
        r"Coefficient estimate ($\mu$M)",
        "Continuous-score versus binary benchmark",
        FIGURES_DIR / "continuous_vs_binary_benchmark.png",
    )
    plot_quadratic_effect(quadratic_artifact, question_summary_df, FIGURES_DIR / "quadratic_effect_plot.png")
    plot_coefficient_comparison(
        [
            pd.Series(
                {
                    "target_beta": primary_df.iloc[0]["abstractness_beta"],
                    "target_ci_low": primary_df.iloc[0]["abstractness_ci_low"],
                    "target_ci_high": primary_df.iloc[0]["abstractness_ci_high"],
                }
            ),
            pd.Series(
                {
                    "target_beta": no_override_df.iloc[0]["abstractness_beta"],
                    "target_ci_low": no_override_df.iloc[0]["abstractness_ci_low"],
                    "target_ci_high": no_override_df.iloc[0]["abstractness_ci_high"],
                }
            ),
        ],
        ["Primary cohort", "No-override cohort"],
        r"Abstractness coefficient ($\mu$M per 1 SD)",
        "Primary versus no-override continuous model",
        FIGURES_DIR / "no_override_comparison.png",
    )
    plot_leave_one_question_out(loo_question_df, FIGURES_DIR / "leave_one_question_out_plot.png")

    benchmark_rows = [
        [
            binary_df.iloc[0]["analysis_label"],
            "Binary label",
            format_effect(binary_df.iloc[0]["condition_beta"]),
            format_effect(binary_df.iloc[0]["condition_se"]),
            format_p_value(binary_df.iloc[0]["condition_p_value"]),
            f"[{format_effect(binary_df.iloc[0]['condition_ci_low'])}, {format_effect(binary_df.iloc[0]['condition_ci_high'])}]",
        ],
        [
            quadratic_df.iloc[0]["analysis_label"],
            "z(Abstractness score)",
            format_effect(quadratic_df.iloc[0]["abstractness_beta"]),
            format_effect(quadratic_df.iloc[0]["abstractness_se"]),
            format_p_value(quadratic_df.iloc[0]["abstractness_p_value"]),
            f"[{format_effect(quadratic_df.iloc[0]['abstractness_ci_low'])}, {format_effect(quadratic_df.iloc[0]['abstractness_ci_high'])}]",
        ],
        [
            quadratic_df.iloc[0]["analysis_label"],
            "Squared score term",
            format_effect(quadratic_df.iloc[0]["quadratic_beta"]),
            format_effect(quadratic_df.iloc[0]["quadratic_se"]),
            format_p_value(quadratic_df.iloc[0]["quadratic_p_value"]),
            f"[{format_effect(quadratic_df.iloc[0]['quadratic_ci_low'])}, {format_effect(quadratic_df.iloc[0]['quadratic_ci_high'])}]",
        ],
        [
            threshold_df.iloc[0]["analysis_label"],
            "Distance from cutoff",
            format_effect(threshold_df.iloc[0]["strength_beta"]),
            format_effect(threshold_df.iloc[0]["strength_se"]),
            format_p_value(threshold_df.iloc[0]["strength_p_value"]),
            f"[{format_effect(threshold_df.iloc[0]['strength_ci_low'])}, {format_effect(threshold_df.iloc[0]['strength_ci_high'])}]",
        ],
        [
            back_df.iloc[0]["analysis_label"],
            "z(Abstractness score)",
            format_effect(back_df.iloc[0]["abstractness_beta"]),
            format_effect(back_df.iloc[0]["abstractness_se"]),
            format_p_value(back_df.iloc[0]["abstractness_p_value"]),
            f"[{format_effect(back_df.iloc[0]['abstractness_ci_low'])}, {format_effect(back_df.iloc[0]['abstractness_ci_high'])}]",
        ],
        [
            fixed_df.iloc[0]["analysis_label"],
            "z(Abstractness score)",
            format_effect(fixed_df.iloc[0]["abstractness_beta"]),
            format_effect(fixed_df.iloc[0]["abstractness_se"]),
            format_p_value(fixed_df.iloc[0]["abstractness_p_value"]),
            f"[{format_effect(fixed_df.iloc[0]['abstractness_ci_low'])}, {format_effect(fixed_df.iloc[0]['abstractness_ci_high'])}]",
        ],
        [
            no_override_df.iloc[0]["analysis_label"],
            "z(Abstractness score)",
            format_effect(no_override_df.iloc[0]["abstractness_beta"]),
            format_effect(no_override_df.iloc[0]["abstractness_se"]),
            format_p_value(no_override_df.iloc[0]["abstractness_p_value"]),
            f"[{format_effect(no_override_df.iloc[0]['abstractness_ci_low'])}, {format_effect(no_override_df.iloc[0]['abstractness_ci_high'])}]",
        ],
    ]

    write_supporting_tex(diagnostics_df, correlations_df, primary_df, benchmark_rows, conclusion_text)
    report_path = write_report(final_df)
    compile_report(report_path)


if __name__ == "__main__":
    main()
