#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step6_covariate_adjusted import compile_report, make_latex_table, sanitize_for_tex


ROOT = Path(__file__).resolve().parents[1]
STEP8_PAIRWISE_PATH = ROOT / "data_clean" / "step8" / "03_step8_pairwise_test_results.csv"
STEP8_RANKING_PATH = ROOT / "data_clean" / "step8" / "05_step8_question_centered_ranking.csv"
STEP8_MATCHED_PATH = ROOT / "data_clean" / "step8" / "07_step8_matched_pair_results.csv"
STEP8_CHANNEL_PATH = ROOT / "data_clean" / "step8" / "08_step8_selected_pair_channel_followup.csv"
STEP8_FINAL_PATH = ROOT / "data_clean" / "step8" / "10_step8_final_conclusion.csv"

STEP8_FIGURES_DIR = ROOT / "figures" / "step8"
FIGURES_DIR = ROOT / "figures" / "step8_visualization"
REPORTS_DIR = ROOT / "reports" / "step8_visualization"

CONCRETE_COLOR = "#228833"
ABSTRACT_COLOR = "#AA3377"
MATCHED_COLOR = "#1F77B4"
UNMATCHED_COLOR = "#BDC3C7"


def ensure_directories() -> None:
    for path in [FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(STEP8_PAIRWISE_PATH),
        pd.read_csv(STEP8_RANKING_PATH),
        pd.read_csv(STEP8_MATCHED_PATH),
        pd.read_csv(STEP8_CHANNEL_PATH),
        pd.read_csv(STEP8_FINAL_PATH),
    )


def pivot_pair_matrix(pairwise_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    concrete_questions = sorted(pairwise_df["concrete_question_id"].astype(str).unique().tolist())
    abstract_questions = sorted(pairwise_df["abstract_question_id"].astype(str).unique().tolist())
    return (
        pairwise_df.pivot(index="concrete_question_id", columns="abstract_question_id", values=value_col)
        .reindex(index=concrete_questions, columns=abstract_questions)
    )


def plot_pairwise_sample_size_heatmap(pairwise_df: pd.DataFrame, path: Path) -> None:
    matrix_df = pivot_pair_matrix(pairwise_df, "n_participants")
    values = matrix_df.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=np.nanmin(values), vmax=np.nanmax(values))
    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index.tolist(), fontsize=8)
    ax.set_xlabel("Abstract question")
    ax.set_ylabel("Concrete question")
    ax.set_title("Pairwise sample-size heatmap\nContributing participants per concrete-abstract pair")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Number of paired participants")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_top_pair_forest(pairwise_df: pd.DataFrame, path: Path, n_top: int = 15) -> None:
    tested_df = pairwise_df[pairwise_df["tested_flag"]].copy()
    tested_df = tested_df.sort_values(["fdr_q_value", "raw_p_value", "mean_difference"], na_position="last").head(n_top).copy()
    if tested_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.text(0.5, 0.5, "No tested pairs available", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    tested_df["pair_label"] = tested_df["concrete_question_id"].astype(str) + " vs " + tested_df["abstract_question_id"].astype(str)
    tested_df["se_difference"] = tested_df["sd_difference"] / np.sqrt(tested_df["n_participants"])
    tested_df["ci_low"] = tested_df["mean_difference"] - 1.96 * tested_df["se_difference"]
    tested_df["ci_high"] = tested_df["mean_difference"] + 1.96 * tested_df["se_difference"]
    tested_df = tested_df.sort_values("mean_difference", ascending=True).reset_index(drop=True)

    fig_height = max(6.5, 0.55 * len(tested_df) + 2.5)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    y_pos = np.arange(len(tested_df))
    colors = [CONCRETE_COLOR if value >= 0 else ABSTRACT_COLOR for value in tested_df["mean_difference"]]
    x = tested_df["mean_difference"].to_numpy(dtype=float) * 1e6
    xerr = np.vstack(
        [
            (tested_df["mean_difference"] - tested_df["ci_low"]).to_numpy(dtype=float) * 1e6,
            (tested_df["ci_high"] - tested_df["mean_difference"]).to_numpy(dtype=float) * 1e6,
        ]
    )
    ax.errorbar(x, y_pos, xerr=xerr, fmt="none", ecolor="#555555", elinewidth=1.6, capsize=3, zorder=1)
    ax.scatter(x, y_pos, color=colors, s=50, zorder=2)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tested_df["pair_label"].tolist(), fontsize=8)
    ax.set_xlabel(r"Mean concrete-minus-abstract difference ($\mu$M)")
    ax.set_ylabel("Item pair")
    ax.set_title("Top descriptive item pairs\nMean back-ROI difference with 95% CI")

    for idx, row in tested_df.iterrows():
        ax.text(
            x[idx] + (4 if x[idx] >= 0 else -4),
            idx,
            f"q={row['fdr_q_value']:.3f}",
            va="center",
            ha="left" if x[idx] >= 0 else "right",
            fontsize=7,
            color="#333333",
        )

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_matching_distance_scatter(pairwise_df: pd.DataFrame, path: Path) -> None:
    tested_df = pairwise_df[pairwise_df["tested_flag"]].copy()
    if tested_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.text(0.5, 0.5, "No tested pairs available", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    tested_df["neg_log10_p"] = -np.log10(np.clip(tested_df["raw_p_value"].to_numpy(dtype=float), 1e-12, 1.0))
    colors = np.where(tested_df["matched_pair_flag"], MATCHED_COLOR, UNMATCHED_COLOR)
    sizes = 24 + 18 * tested_df["neg_log10_p"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    ax.scatter(
        tested_df["matching_distance"],
        tested_df["mean_difference"] * 1e6,
        c=colors,
        s=sizes,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax.axvline(1.0, color="#777777", linestyle=":", linewidth=1)
    ax.set_xlabel("Matching distance")
    ax.set_ylabel(r"Mean concrete-minus-abstract difference ($\mu$M)")
    ax.set_title("Matching distance versus pairwise effect\nBlue = matched-pair subset")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MATCHED_COLOR, markersize=8, label="Matched pair"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=UNMATCHED_COLOR, markersize=8, label="Unmatched pair"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    pairwise_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> Path:
    final_row = final_df.iloc[0]
    tested_df = pairwise_df[pairwise_df["tested_flag"]].copy()
    top_pair = tested_df.nsmallest(1, ["fdr_q_value", "raw_p_value"])
    top_pair_text = "No tested item pairs were available."
    if not top_pair.empty:
        row = top_pair.iloc[0]
        top_pair_text = (
            f"Top descriptive pair: {row['concrete_question_id']} versus {row['abstract_question_id']} "
            f"(mean difference={row['mean_difference']:.3e}, raw p={row['raw_p_value']:.6f}, q={row['fdr_q_value']:.6f})."
        )

    overview_rows = [
        ("Tested concrete-abstract pairs", int(tested_df.shape[0])),
        ("Pairs surviving FDR", int(tested_df["fdr_reject"].sum())),
        ("Matched-pair subset size", int(matched_df.shape[0])),
        ("Matched subset FDR-significant pairs", int(matched_df["matched_subset_fdr_reject"].sum()) if not matched_df.empty else 0),
        ("Ranked questions", int(ranking_df.shape[0])),
        ("Selected channel-followup pairs", int(channel_df[["concrete_question_id", "abstract_question_id"]].drop_duplicates().shape[0]) if not channel_df.empty else 0),
    ]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Supplementary Step 8 Visualization Report}",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
        r"\paragraph{Interpretation note.} "
        + sanitize_for_tex(final_row["conclusion_text"]),
        r"\paragraph{Top pair summary.} "
        + sanitize_for_tex(top_pair_text),
        r"\subsection*{Core Step 8 Heatmaps}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step8/pairwise_mean_difference_heatmap.png}",
        r"\caption{Mean back-ROI concrete-minus-abstract differences for all item pairs.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step8/pairwise_raw_pvalue_heatmap.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step8/pairwise_fdr_qvalue_heatmap.png}",
        r"\caption{Raw p-values and FDR-adjusted q-values across all tested item pairs.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step8/pairwise_effect_size_heatmap.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step8_visualization/pairwise_sample_size_heatmap.png}",
        r"\caption{Paired effect sizes and the contributing sample size for each concrete-abstract pair.}",
        r"\end{figure}",
        r"\subsection*{Descriptive Ranking Views}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.90\linewidth]{../../figures/step8/question_centered_ranking_plot.png}",
        r"\caption{Participant-centered ranking of questions by back-ROI response.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.92\linewidth]{../../figures/step8_visualization/top_pair_forest_plot.png}",
        r"\caption{Top descriptive item pairs ranked by the strongest pairwise evidence, shown with mean differences and 95\% confidence intervals.}",
        r"\end{figure}",
        r"\subsection*{Matching And Channel Follow-up}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step8/matched_pair_heatmap.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step8_visualization/matching_distance_scatter.png}",
        r"\caption{Matched-pair subset heatmap and the relation between matching distance and pairwise mean effect.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step8/selected_pair_channel_profile.png}",
        r"\caption{Optional back-channel follow-up for the selected descriptive pairs.}",
        r"\end{figure}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step8_visualization_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()
    pairwise_df, ranking_df, matched_df, channel_df, final_df = load_inputs()
    plot_pairwise_sample_size_heatmap(pairwise_df, FIGURES_DIR / "pairwise_sample_size_heatmap.png")
    plot_top_pair_forest(pairwise_df, FIGURES_DIR / "top_pair_forest_plot.png")
    plot_matching_distance_scatter(pairwise_df, FIGURES_DIR / "matching_distance_scatter.png")
    report_path = write_report(pairwise_df, ranking_df, matched_df, channel_df, final_df)
    compile_report(report_path)
    print("Step 8 visualization report generated successfully.")
    print(f"Tested pairs: {int(pairwise_df['tested_flag'].sum())}")
    print(f"Matched-pair subset size: {int(matched_df.shape[0])}")


if __name__ == "__main__":
    main()
