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
STEP9_TRIAL_PATH = ROOT / "data_clean" / "step9" / "01_step9_trial_block_table.csv"
STEP9_DESCRIPTIVE_PATH = ROOT / "data_clean" / "step9" / "02_step9_block_descriptive_summary.csv"
STEP9_PRIMARY_PATH = ROOT / "data_clean" / "step9" / "03_step9_primary_front_block_adjusted_model.csv"
STEP9_BLOCK_INTERACTION_PATH = ROOT / "data_clean" / "step9" / "04_step9_condition_by_blockorder_model.csv"
STEP9_POSITION_INTERACTION_PATH = ROOT / "data_clean" / "step9" / "05_step9_condition_by_position_model.csv"
STEP9_ORDER_INTERACTION_PATH = ROOT / "data_clean" / "step9" / "06_step9_condition_by_ordercondition_model.csv"
STEP9_EARLY_LATE_PATH = ROOT / "data_clean" / "step9" / "07_step9_early_late_sensitivity_model.csv"
STEP9_BACK_PATH = ROOT / "data_clean" / "step9" / "08_step9_back_block_adjusted_model.csv"
STEP9_FIXED_PATH = ROOT / "data_clean" / "step9" / "09_step9_fixed_window_block_benchmark.csv"
STEP9_FINAL_PATH = ROOT / "data_clean" / "step9" / "11_step9_final_conclusion.csv"

STEP9_FIGURES_DIR = ROOT / "figures" / "step9"
FIGURES_DIR = ROOT / "figures" / "step9_visualization"
REPORTS_DIR = ROOT / "reports" / "step9_visualization"

CONDITION_ORDER = ["Concrete", "Abstract"]
CONDITION_COLORS = {"Abstract": "#AA3377", "Concrete": "#228833"}


def ensure_directories() -> None:
    for path in [FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(STEP9_TRIAL_PATH),
        pd.read_csv(STEP9_DESCRIPTIVE_PATH),
        pd.read_csv(STEP9_PRIMARY_PATH),
        pd.read_csv(STEP9_BLOCK_INTERACTION_PATH),
        pd.read_csv(STEP9_POSITION_INTERACTION_PATH),
        pd.read_csv(STEP9_ORDER_INTERACTION_PATH),
        pd.read_csv(STEP9_EARLY_LATE_PATH),
        pd.read_csv(STEP9_BACK_PATH),
        pd.read_csv(STEP9_FIXED_PATH),
    )


def load_final() -> pd.DataFrame:
    return pd.read_csv(STEP9_FINAL_PATH)


def plot_trial_count_by_block_and_condition(descriptive_df: pd.DataFrame, path: Path) -> None:
    block_df = descriptive_df[descriptive_df["summary_type"] == "global_block_order"].copy()
    block_df = block_df.sort_values(["global_block_order", "condition"])
    block_orders = sorted(block_df["global_block_order"].dropna().unique().tolist())
    x_positions = np.arange(len(block_orders), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.8, 4.8), layout="constrained")
    for offset, condition in zip([-width / 2, width / 2], CONDITION_ORDER):
        subset = block_df[block_df["condition"] == condition].set_index("global_block_order").reindex(block_orders).reset_index()
        ax.bar(
            x_positions + offset,
            subset["n_trials"],
            width=width,
            color=CONDITION_COLORS[condition],
            alpha=0.7,
            label=condition,
        )
    ax.set_xticks(x_positions, [str(int(value)) for value in block_orders])
    ax.set_xlabel("Global block order")
    ax.set_ylabel("Valid trial count")
    ax.set_title("Valid trials by global block order and condition")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_participant_block_half_difference(trial_df: pd.DataFrame, path: Path) -> None:
    included_df = trial_df[trial_df["included_step9_model"].astype(bool)].copy()
    participant_half_df = (
        included_df.groupby(["participant_id", "block_half_label", "condition"], dropna=False)["front_roi_mean_variable_lag4"]
        .mean()
        .reset_index()
    )
    diff_df = (
        participant_half_df.pivot(index=["participant_id", "block_half_label"], columns="condition", values="front_roi_mean_variable_lag4")
        .reset_index()
    )
    diff_df["abstract_minus_concrete"] = diff_df["Abstract"] - diff_df["Concrete"]
    order_labels = ["Early blocks (1-5)", "Late blocks (6-10)"]
    x_lookup = {label: idx for idx, label in enumerate(order_labels)}

    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    for participant_id, participant_df in diff_df.groupby("participant_id"):
        participant_df = participant_df.set_index("block_half_label").reindex(order_labels).reset_index()
        x = [x_lookup[label] for label in participant_df["block_half_label"]]
        y = participant_df["abstract_minus_concrete"].to_numpy(dtype=float) * 1e6
        ax.plot(x, y, color="#C0C0C0", linewidth=1.0, alpha=0.6)

    mean_df = (
        diff_df.groupby("block_half_label", dropna=False)["abstract_minus_concrete"]
        .mean()
        .reindex(order_labels)
        .reset_index()
    )
    ax.plot(
        [x_lookup[label] for label in mean_df["block_half_label"]],
        mean_df["abstract_minus_concrete"].to_numpy(dtype=float) * 1e6,
        color="#225588",
        linewidth=3,
        marker="o",
        markersize=7,
        label="Participant mean",
    )
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1], order_labels)
    ax.set_xlabel("Session half")
    ax.set_ylabel(r"Abstract - Concrete front ROI HbO ($\mu$M)")
    ax.set_title("Participant-level condition difference by session half")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    trial_df: pd.DataFrame,
    descriptive_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    block_df: pd.DataFrame,
    position_df: pd.DataFrame,
    order_df: pd.DataFrame,
    early_df: pd.DataFrame,
    back_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> Path:
    primary = primary_df.iloc[0]
    final_row = final_df.iloc[0]

    overview_rows = [
        ("Included participant-sessions", int(primary["n_participants"])),
        ("Included questions", int(primary["n_questions"])),
        ("Included trials", int(primary["n_trials"])),
        ("Primary block-adjusted condition p-value", f"{float(primary['condition_p_value']):.6f}"),
        ("Condition x block-order interaction p-value", f"{float(block_df.iloc[0]['target_p_value']):.6f}"),
        ("Condition x within-block-position interaction p-value", f"{float(position_df.iloc[0]['target_p_value']):.6f}"),
        ("Condition x condition-order interaction p-value", f"{float(order_df.iloc[0]['target_p_value']):.6f}"),
        ("Early-vs-late interaction p-value", f"{float(early_df.iloc[0]['target_p_value']):.6f}"),
        ("Back ROI benchmark p-value", f"{float(back_df.iloc[0]['condition_p_value']):.6f}"),
        ("Fixed-window benchmark p-value", f"{float(fixed_df.iloc[0]['condition_p_value']):.6f}"),
    ]

    block_counts = descriptive_df[descriptive_df["summary_type"] == "global_block_order"].copy()
    block_counts = block_counts.sort_values(["global_block_order", "condition"])
    early_mean = (
        trial_df[trial_df["included_step9_model"].astype(bool)]
        .groupby(["block_half_label", "condition"], dropna=False)["front_roi_mean_variable_lag4"]
        .mean()
        .reset_index()
    )
    early_text = []
    for _, row in early_mean.iterrows():
        early_text.append(
            f"{row['block_half_label']} / {row['condition']}: {row['front_roi_mean_variable_lag4']:.3e}"
        )

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Supplementary Step 9 Visualization Report}",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
        r"\paragraph{Interpretation note.} " + sanitize_for_tex(final_row["conclusion_text"]),
        r"\paragraph{Descriptive note.} "
        + sanitize_for_tex(
            "These figures focus on how the realized block structure, within-block position, and participant condition order relate to the front ROI outcome before and after block adjustment. "
            + "Mean front-ROI values by session half were: "
            + "; ".join(early_text)
            + "."
        ),
        r"\subsection*{Core Step 9 Diagnostic Figures}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step9/front_roi_by_global_block_order.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step9/front_roi_by_withinblock_position.png}",
        r"\caption{Front ROI response as a function of realized block order and within-block question position.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step9/response_time_by_global_block_order.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step9/early_vs_late_condition_plot.png}",
        r"\caption{Response-time progression across the session and the early-versus-late condition comparison.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step9/primary_block_adjusted_condition_effect.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step9/fixed_vs_variable_block_benchmark.png}",
        r"\caption{Primary block-adjusted coefficient summary and the duration-aware versus fixed-window benchmark.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step9/condition_by_blockorder_interaction_plot.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step9/condition_by_ordercondition_plot.png}",
        r"\caption{Secondary interaction-focused plots for global block order and participant condition order.}",
        r"\end{figure}",
        r"\subsection*{Additional Step 9 Visualization Diagnostics}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step9_visualization/trial_count_by_block_and_condition.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step9_visualization/participant_block_half_difference.png}",
        r"\caption{Additional descriptive views: valid trial counts by block and condition, and participant-level abstract-minus-concrete front-ROI differences from early to late session halves.}",
        r"\end{figure}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step9_visualization_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()
    (
        trial_df,
        descriptive_df,
        primary_df,
        block_df,
        position_df,
        order_df,
        early_df,
        back_df,
        fixed_df,
    ) = load_inputs()
    final_df = load_final()

    plot_trial_count_by_block_and_condition(descriptive_df, FIGURES_DIR / "trial_count_by_block_and_condition.png")
    plot_participant_block_half_difference(trial_df, FIGURES_DIR / "participant_block_half_difference.png")

    report_path = write_report(
        trial_df,
        descriptive_df,
        primary_df,
        block_df,
        position_df,
        order_df,
        early_df,
        back_df,
        fixed_df,
        final_df,
    )
    compile_report(report_path)

    print("Step 9 visualization report generated successfully.")
    print(f"Included participant-sessions: {int(primary_df.iloc[0]['n_participants'])}")
    print(f"Included questions: {int(primary_df.iloc[0]['n_questions'])}")
    print(f"Included trials: {int(primary_df.iloc[0]['n_trials'])}")


if __name__ == "__main__":
    main()
