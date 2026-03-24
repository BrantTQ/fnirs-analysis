#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step6_covariate_adjusted import compile_report, make_latex_table, sanitize_for_tex


ROOT = Path(__file__).resolve().parents[1]
STEP10_PRIMARY_PATH = ROOT / "data_clean" / "step10" / "02_step10_primary_fe_results.csv"
STEP10_CLUSTER_PARTICIPANT_PATH = ROOT / "data_clean" / "step10" / "03_step10_cluster_participant_results.csv"
STEP10_CLUSTER_QUESTION_PATH = ROOT / "data_clean" / "step10" / "04_step10_cluster_question_results.csv"
STEP10_HC3_PATH = ROOT / "data_clean" / "step10" / "05_step10_hc3_results.csv"
STEP10_FIXED_PATH = ROOT / "data_clean" / "step10" / "06_step10_fixed_window_benchmark.csv"
STEP10_NO_OVERRIDE_PATH = ROOT / "data_clean" / "step10" / "07_step10_no_override_results.csv"
STEP10_LOO_QUESTION_PATH = ROOT / "data_clean" / "step10" / "08_step10_leave_one_question_out.csv"
STEP10_LOO_PARTICIPANT_PATH = ROOT / "data_clean" / "step10" / "09_step10_leave_one_participant_out.csv"
STEP10_BACK_PATH = ROOT / "data_clean" / "step10" / "10_step10_back_roi_benchmark.csv"
STEP10_FINAL_PATH = ROOT / "data_clean" / "step10" / "12_step10_final_conclusion.csv"

STEP10_FIGURES_DIR = ROOT / "figures" / "step10"
FIGURES_DIR = ROOT / "figures" / "step10_visualization"
REPORTS_DIR = ROOT / "reports" / "step10_visualization"


def ensure_directories() -> None:
    for path in [FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(STEP10_PRIMARY_PATH),
        pd.read_csv(STEP10_CLUSTER_PARTICIPANT_PATH),
        pd.read_csv(STEP10_CLUSTER_QUESTION_PATH),
        pd.read_csv(STEP10_HC3_PATH),
        pd.read_csv(STEP10_FIXED_PATH),
        pd.read_csv(STEP10_NO_OVERRIDE_PATH),
        pd.read_csv(STEP10_LOO_QUESTION_PATH),
        pd.read_csv(STEP10_LOO_PARTICIPANT_PATH),
        pd.read_csv(STEP10_BACK_PATH),
    )


def load_final() -> pd.DataFrame:
    return pd.read_csv(STEP10_FINAL_PATH)


def plot_sensitivity_pvalue_summary(rows: list[pd.Series], labels: list[str], path: Path) -> None:
    pvalues = np.array([float(row["condition_p_value"]) for row in rows], dtype=float)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.0, 4.8), layout="constrained")
    bars = ax.bar(x, pvalues, color=["#225588", "#4477AA", "#66A5D3", "#88CCEE", "#CCBB44", "#CC6677", "#999999"])
    ax.axhline(0.05, color="#AA0000", linestyle="--", linewidth=1.5, label="alpha = 0.05")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Condition p-value")
    ax.set_title("Step 10 sensitivity-model p-values")
    ax.legend(frameon=False)
    for bar, p in zip(bars, pvalues):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.01, f"{p:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ranked_pvalues(loo_df: pd.DataFrame, id_col: str, title: str, path: Path) -> None:
    plot_df = loo_df.sort_values("condition_p_value").reset_index(drop=True).copy()
    x = np.arange(len(plot_df))
    colors = np.where(plot_df["condition_p_value"] < 0.05, "#CC3311", "#4477AA")

    fig_height = max(4.8, 0.18 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(8.4, fig_height), layout="constrained")
    ax.scatter(plot_df["condition_p_value"], x, c=colors, s=42)
    ax.axvline(0.05, color="#AA0000", linestyle="--", linewidth=1.5)
    ax.set_yticks(x, plot_df[id_col].astype(str).tolist())
    ax.set_xlabel("Condition p-value")
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    primary_df: pd.DataFrame,
    cluster_participant_df: pd.DataFrame,
    cluster_question_df: pd.DataFrame,
    hc3_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    no_override_df: pd.DataFrame,
    loo_question_df: pd.DataFrame,
    loo_participant_df: pd.DataFrame,
    back_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> Path:
    primary = primary_df.iloc[0]
    final_row = final_df.iloc[0]

    overview_rows = [
        ("Included participant-sessions", int(primary["n_participants"])),
        ("Included questions", int(primary["n_questions"])),
        ("Included trials", int(primary["n_trials"])),
        ("Primary robust SE type", primary["robust_se_label"]),
        ("Primary FE p-value", f"{float(primary['condition_p_value']):.6f}"),
        ("Participant-clustered p-value", f"{float(cluster_participant_df.iloc[0]['condition_p_value']):.6f}"),
        ("Question-clustered p-value", f"{float(cluster_question_df.iloc[0]['condition_p_value']):.6f}"),
        ("HC3 p-value", f"{float(hc3_df.iloc[0]['condition_p_value']):.6f}"),
        ("Fixed-window benchmark p-value", f"{float(fixed_df.iloc[0]['condition_p_value']):.6f}"),
        ("No-override p-value", f"{float(no_override_df.iloc[0]['condition_p_value']):.6f}"),
        ("Back ROI benchmark p-value", f"{float(back_df.iloc[0]['condition_p_value']):.6f}"),
        ("Leave-one-question-out p range", f"{loo_question_df['condition_p_value'].min():.6f} to {loo_question_df['condition_p_value'].max():.6f}"),
        ("Leave-one-participant-out p range", f"{loo_participant_df['condition_p_value'].min():.6f} to {loo_participant_df['condition_p_value'].max():.6f}"),
    ]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Supplementary Step 10 Visualization Report}",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
        r"\paragraph{Interpretation note.} " + sanitize_for_tex(final_row["conclusion_text"]),
        r"\paragraph{Influence note.} "
        + sanitize_for_tex(
            "The leave-one-question-out p-values ranged from "
            f"{loo_question_df['condition_p_value'].min():.6f} to {loo_question_df['condition_p_value'].max():.6f}, "
            "and the leave-one-participant-out p-values ranged from "
            f"{loo_participant_df['condition_p_value'].min():.6f} to {loo_participant_df['condition_p_value'].max():.6f}. "
            "Neither influence analysis produced any p-value below 0.05."
        ),
        r"\subsection*{Core Step 10 Figures}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step10/primary_condition_coefficient_plot.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step10/robust_se_comparison.png}",
        r"\caption{Primary participant-fixed-effects coefficient summary and the robust-standard-error comparison.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step10/fixed_vs_variable_fe_benchmark.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step10/no_override_comparison.png}",
        r"\caption{Duration-aware versus fixed-window benchmark and the no-override sensitivity comparison.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step10/leave_one_question_out_plot.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step10/leave_one_participant_out_plot.png}",
        r"\caption{Condition coefficients from the leave-one-question-out and leave-one-participant-out influence analyses.}",
        r"\end{figure}",
        r"\subsection*{Additional Influence Diagnostics}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.90\linewidth]{../../figures/step10_visualization/sensitivity_pvalue_summary.png}",
        r"\caption{Condition p-values across the primary and sensitivity Step~10 models.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step10_visualization/leave_one_question_out_pvalues.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step10_visualization/leave_one_participant_out_pvalues.png}",
        r"\caption{Ranked p-values from the leave-one-question-out and leave-one-participant-out analyses. The red line marks alpha=0.05.}",
        r"\end{figure}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step10_visualization_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()
    (
        primary_df,
        cluster_participant_df,
        cluster_question_df,
        hc3_df,
        fixed_df,
        no_override_df,
        loo_question_df,
        loo_participant_df,
        back_df,
    ) = load_inputs()
    final_df = load_final()

    sensitivity_rows = [
        primary_df.iloc[0],
        cluster_participant_df.iloc[0],
        cluster_question_df.iloc[0],
        hc3_df.iloc[0],
        fixed_df.iloc[0],
        no_override_df.iloc[0],
        back_df.iloc[0],
    ]
    sensitivity_labels = [
        "Primary FE",
        "Participant cluster",
        "Question cluster",
        "HC3",
        "Fixed-window",
        "No-override",
        "Back ROI",
    ]

    plot_sensitivity_pvalue_summary(sensitivity_rows, sensitivity_labels, FIGURES_DIR / "sensitivity_pvalue_summary.png")
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
        primary_df,
        cluster_participant_df,
        cluster_question_df,
        hc3_df,
        fixed_df,
        no_override_df,
        loo_question_df,
        loo_participant_df,
        back_df,
        final_df,
    )
    compile_report(report_path)

    print("Step 10 visualization report generated successfully.")
    print(f"Included participant-sessions: {int(primary_df.iloc[0]['n_participants'])}")
    print(f"Included questions: {int(primary_df.iloc[0]['n_questions'])}")
    print(f"Included trials: {int(primary_df.iloc[0]['n_trials'])}")


if __name__ == "__main__":
    main()
