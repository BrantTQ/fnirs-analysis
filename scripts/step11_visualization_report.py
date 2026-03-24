#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step6_covariate_adjusted import compile_report, make_latex_table, sanitize_for_tex


ROOT = Path(__file__).resolve().parents[1]
FILTERED_QUESTIONS_PATH = ROOT / "materials" / "filtered_questions.json"
STEP10_TABLE_PATH = ROOT / "data_clean" / "step10" / "01_step10_primary_fe_table.csv"

STEP11_DIAGNOSTICS_PATH = ROOT / "data_clean" / "step11" / "01_step11_score_diagnostics.csv"
STEP11_CORRELATIONS_PATH = ROOT / "data_clean" / "step11" / "02_step11_score_correlations.csv"
STEP11_PRIMARY_PATH = ROOT / "data_clean" / "step11" / "03_step11_primary_continuous_model.csv"
STEP11_BINARY_PATH = ROOT / "data_clean" / "step11" / "04_step11_binary_benchmark_model.csv"
STEP11_QUADRATIC_PATH = ROOT / "data_clean" / "step11" / "05_step11_quadratic_model.csv"
STEP11_THRESHOLD_PATH = ROOT / "data_clean" / "step11" / "06_step11_threshold_strength_model.csv"
STEP11_BACK_PATH = ROOT / "data_clean" / "step11" / "07_step11_back_roi_benchmark.csv"
STEP11_FIXED_PATH = ROOT / "data_clean" / "step11" / "08_step11_fixed_window_benchmark.csv"
STEP11_NO_OVERRIDE_PATH = ROOT / "data_clean" / "step11" / "09_step11_no_override_results.csv"
STEP11_LOO_QUESTION_PATH = ROOT / "data_clean" / "step11" / "10_step11_leave_one_question_out.csv"
STEP11_LOO_PARTICIPANT_PATH = ROOT / "data_clean" / "step11" / "11_step11_leave_one_participant_out.csv"
STEP11_FINAL_PATH = ROOT / "data_clean" / "step11" / "12_step11_final_conclusion.csv"

STEP11_FIGURES_DIR = ROOT / "figures" / "step11"
FIGURES_DIR = ROOT / "figures" / "step11_visualization"
REPORTS_DIR = ROOT / "reports" / "step11_visualization"

ABSTRACT_COLOR = "#AA3377"
CONCRETE_COLOR = "#228833"
BASE_COLOR = "#4477AA"


def ensure_directories() -> None:
    for path in [FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(FILTERED_QUESTIONS_PATH, "r", encoding="utf-8") as handle:
        question_df = pd.DataFrame(json.load(handle))
    step10_df = pd.read_csv(STEP10_TABLE_PATH)
    return (
        question_df,
        step10_df,
        pd.read_csv(STEP11_DIAGNOSTICS_PATH),
        pd.read_csv(STEP11_CORRELATIONS_PATH),
        pd.read_csv(STEP11_PRIMARY_PATH),
        pd.read_csv(STEP11_BINARY_PATH),
        pd.read_csv(STEP11_QUADRATIC_PATH),
        pd.read_csv(STEP11_NO_OVERRIDE_PATH),
        pd.read_csv(STEP11_LOO_QUESTION_PATH),
        pd.read_csv(STEP11_LOO_PARTICIPANT_PATH),
    )


def load_more_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(STEP11_THRESHOLD_PATH),
        pd.read_csv(STEP11_BACK_PATH),
        pd.read_csv(STEP11_FIXED_PATH),
        pd.read_csv(STEP11_FINAL_PATH),
    )


def build_question_summary(question_df: pd.DataFrame, step10_df: pd.DataFrame) -> pd.DataFrame:
    q = question_df.copy()
    q["question_id"] = (
        q["year"].astype(str)
        + "_"
        + q["field"].astype(str)
        + "_"
        + pd.to_numeric(q["question_number"], errors="coerce").astype(int).astype(str)
    )
    q["abstracness"] = pd.to_numeric(q["abstracness"], errors="coerce")
    q["sentence_count"] = pd.to_numeric(q["sentence_count"], errors="coerce")
    q["sentence_length"] = pd.to_numeric(q["sentence_length"], errors="coerce")
    q["correctness"] = pd.to_numeric(q["correctness"], errors="coerce")

    rt_df = (
        step10_df.groupby("question_id", as_index=False)
        .agg(mean_response_time_sec=("response_time_sec", "mean"))
    )
    summary_df = q.merge(rt_df, on="question_id", how="left", validate="one_to_one")
    return summary_df


def scatter_with_fit(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), layout="constrained")
    colors = df["type"].astype(str).str.lower().map({"abstract": ABSTRACT_COLOR, "concrete": CONCRETE_COLOR}).fillna(BASE_COLOR)
    ax.scatter(df[x_col], df[y_col], c=colors, s=46, alpha=0.8, edgecolors="white", linewidths=0.6)
    subset = df[[x_col, y_col]].dropna()
    if len(subset) >= 2:
        coeffs = np.polyfit(subset[x_col], subset[y_col], deg=1)
        x_grid = np.linspace(float(subset[x_col].min()), float(subset[x_col].max()), 200)
        ax.plot(x_grid, coeffs[0] * x_grid + coeffs[1], color="#333333", linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_leave_one_participant_out(loo_df: pd.DataFrame, path: Path) -> None:
    plot_df = loo_df.sort_values("abstractness_beta").reset_index(drop=True).copy()
    effects = plot_df["abstractness_beta"].to_numpy(dtype=float) * 1e6
    ci_low = plot_df["abstractness_ci_low"].to_numpy(dtype=float) * 1e6
    ci_high = plot_df["abstractness_ci_high"].to_numpy(dtype=float) * 1e6
    labels = plot_df["omitted_participant_id"].astype(str).tolist()
    y_positions = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(8.2, max(6.0, 0.25 * len(plot_df) + 1.8)), layout="constrained")
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
    ax.set_title("Leave-one-participant-out abstractness coefficients")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sensitivity_pvalues(rows: list[pd.Series], labels: list[str], path: Path) -> None:
    pvalues = np.array([float(row["target_p_value"]) for row in rows], dtype=float)
    colors = ["#225588", "#4477AA", "#66A5D3", "#88CCEE", "#CCBB44", "#CC6677"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    bars = ax.bar(x, pvalues, color=colors[: len(labels)])
    ax.axhline(0.05, color="#AA0000", linestyle="--", linewidth=1.5, label="alpha = 0.05")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylabel("Target-term p-value")
    ax.set_title("Step 11 benchmark and sensitivity-model p-values")
    ax.legend(frameon=False)
    for bar, p in zip(bars, pvalues):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.01, f"{p:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ranked_pvalues(loo_df: pd.DataFrame, id_col: str, title: str, path: Path) -> None:
    plot_df = loo_df.sort_values("abstractness_p_value").reset_index(drop=True).copy()
    y_positions = np.arange(len(plot_df))
    colors = np.where(plot_df["abstractness_p_value"] < 0.05, "#CC3311", "#4477AA")

    fig_height = max(4.8, 0.18 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(8.4, fig_height), layout="constrained")
    ax.scatter(plot_df["abstractness_p_value"], y_positions, c=colors, s=42)
    ax.axvline(0.05, color="#AA0000", linestyle="--", linewidth=1.5)
    ax.set_yticks(y_positions, plot_df[id_col].astype(str).tolist())
    ax.set_xlabel("Abstractness p-value")
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    diagnostics_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    quadratic_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    back_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    no_override_df: pd.DataFrame,
    loo_question_df: pd.DataFrame,
    loo_participant_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> Path:
    primary = primary_df.iloc[0]
    diag = diagnostics_df.iloc[0]
    final_row = final_df.iloc[0]

    overview_rows = [
        ("Included participant-sessions", int(primary["n_participants"])),
        ("Included questions", int(primary["n_questions"])),
        ("Included trials", int(primary["n_trials"])),
        ("Primary robust SE type", primary["robust_se_label"]),
        ("Primary continuous p-value", f"{float(primary['abstractness_p_value']):.6f}"),
        ("Binary benchmark p-value", f"{float(binary_df.iloc[0]['condition_p_value']):.6f}"),
        ("Quadratic linear-term p-value", f"{float(quadratic_df.iloc[0]['abstractness_p_value']):.6f}"),
        ("Quadratic squared-term p-value", f"{float(quadratic_df.iloc[0]['quadratic_p_value']):.6f}"),
        ("Threshold-strength p-value", f"{float(threshold_df.iloc[0]['strength_p_value']):.6f}"),
        ("Back ROI benchmark p-value", f"{float(back_df.iloc[0]['abstractness_p_value']):.6f}"),
        ("Fixed-window benchmark p-value", f"{float(fixed_df.iloc[0]['abstractness_p_value']):.6f}"),
        ("No-override p-value", f"{float(no_override_df.iloc[0]['abstractness_p_value']):.6f}"),
        ("Binary cutoff used", f"{float(diag['binary_cutoff_used']):.2f}"),
        ("Questions below / above cutoff", f"{int(diag['n_questions_below_cutoff'])} / {int(diag['n_questions_above_cutoff'])}"),
    ]

    corr_parts = []
    for _, row in correlations_df.iterrows():
        corr_parts.append(f"{row['label']}: r={float(row['pearson_r']):.3f}, p={float(row['p_value']):.6f}")

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Supplementary Step 11 Visualization Report}",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
        r"\paragraph{Interpretation note.} " + sanitize_for_tex(final_row["conclusion_text"]),
        r"\paragraph{Correlation note.} " + sanitize_for_tex("Score-diagnostic correlations were: " + "; ".join(corr_parts) + "."),
        r"\subsection*{Core Step 11 Figures}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/score_histogram.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/score_by_binary_label.png}",
        r"\caption{Distribution of the raw abstractness score and its split by the existing binary label.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/score_vs_wordcount.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/score_vs_correctness.png}",
        r"\caption{Abstractness score against total word count and ENEM item correctness.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/continuous_partial_effect_plot.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/continuous_vs_binary_benchmark.png}",
        r"\caption{Primary continuous-score partial effect and the comparison against the binary benchmark.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/quadratic_effect_plot.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11/no_override_comparison.png}",
        r"\caption{Quadratic sensitivity visualization and the no-override comparison.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\linewidth]{../../figures/step11/leave_one_question_out_plot.png}",
        r"\caption{Leave-one-question-out abstractness coefficients from the main Step~11 run.}",
        r"\end{figure}",
        r"\subsection*{Additional Step 11 Diagnostics}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.32\linewidth]{../../figures/step11_visualization/score_vs_sentence_count.png}",
        r"\includegraphics[width=0.32\linewidth]{../../figures/step11_visualization/score_vs_sentence_length.png}",
        r"\includegraphics[width=0.32\linewidth]{../../figures/step11_visualization/score_vs_response_time.png}",
        r"\caption{Additional score-diagnostic scatter plots for sentence count, sentence length, and mean response time.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11_visualization/leave_one_participant_out_plot.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11_visualization/sensitivity_pvalue_summary.png}",
        r"\caption{Leave-one-participant-out abstractness coefficients and the p-value summary across Step~11 benchmark/sensitivity models.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11_visualization/leave_one_question_out_pvalues.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step11_visualization/leave_one_participant_out_pvalues.png}",
        r"\caption{Ranked p-values from the leave-one-question-out and leave-one-participant-out analyses. The red line marks alpha=0.05.}",
        r"\end{figure}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step11_visualization_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()
    (
        question_df,
        step10_df,
        diagnostics_df,
        correlations_df,
        primary_df,
        binary_df,
        quadratic_df,
        no_override_df,
        loo_question_df,
        loo_participant_df,
    ) = load_inputs()
    threshold_df, back_df, fixed_df, final_df = load_more_outputs()

    question_summary_df = build_question_summary(question_df, step10_df)

    scatter_with_fit(
        question_summary_df,
        x_col="abstracness",
        y_col="sentence_count",
        x_label="Raw abstractness score",
        y_label="Sentence count",
        title="Sentence count versus abstractness score",
        path=FIGURES_DIR / "score_vs_sentence_count.png",
    )
    scatter_with_fit(
        question_summary_df,
        x_col="abstracness",
        y_col="sentence_length",
        x_label="Raw abstractness score",
        y_label="Average sentence length",
        title="Average sentence length versus abstractness score",
        path=FIGURES_DIR / "score_vs_sentence_length.png",
    )
    scatter_with_fit(
        question_summary_df,
        x_col="abstracness",
        y_col="mean_response_time_sec",
        x_label="Raw abstractness score",
        y_label="Mean response time (s)",
        title="Mean response time versus abstractness score",
        path=FIGURES_DIR / "score_vs_response_time.png",
    )

    plot_leave_one_participant_out(loo_participant_df, FIGURES_DIR / "leave_one_participant_out_plot.png")

    sensitivity_rows = [
        pd.Series({"target_p_value": primary_df.iloc[0]["abstractness_p_value"]}),
        pd.Series({"target_p_value": binary_df.iloc[0]["condition_p_value"]}),
        pd.Series({"target_p_value": quadratic_df.iloc[0]["quadratic_p_value"]}),
        pd.Series({"target_p_value": threshold_df.iloc[0]["strength_p_value"]}),
        pd.Series({"target_p_value": fixed_df.iloc[0]["abstractness_p_value"]}),
        pd.Series({"target_p_value": no_override_df.iloc[0]["abstractness_p_value"]}),
    ]
    plot_sensitivity_pvalues(
        sensitivity_rows,
        ["Continuous", "Binary", "Quadratic", "Threshold", "Fixed-window", "No-override"],
        FIGURES_DIR / "sensitivity_pvalue_summary.png",
    )
    plot_ranked_pvalues(
        loo_question_df,
        "omitted_question_id",
        "Ranked leave-one-question-out p-values",
        FIGURES_DIR / "leave_one_question_out_pvalues.png",
    )
    plot_ranked_pvalues(
        loo_participant_df,
        "omitted_participant_id",
        "Ranked leave-one-participant-out p-values",
        FIGURES_DIR / "leave_one_participant_out_pvalues.png",
    )

    report_path = write_report(
        diagnostics_df,
        correlations_df,
        primary_df,
        binary_df,
        quadratic_df,
        threshold_df,
        back_df,
        fixed_df,
        no_override_df,
        loo_question_df,
        loo_participant_df,
        final_df,
    )
    compile_report(report_path)

    print("Step 11 visualization report generated successfully.")
    print(f"Included participant-sessions: {int(primary_df.iloc[0]['n_participants'])}")
    print(f"Included questions: {int(primary_df.iloc[0]['n_questions'])}")
    print(f"Included trials: {int(primary_df.iloc[0]['n_trials'])}")


if __name__ == "__main__":
    main()
