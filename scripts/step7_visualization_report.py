#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import pandas as pd

import step6_visualization_report as base


ROOT = Path(__file__).resolve().parents[1]
STEP7_TRIAL_PATH = ROOT / "data_clean" / "step7" / "01_step7_trial_timing_table.csv"
STEP7_PRIMARY_MODEL_PATH = ROOT / "data_clean" / "step7" / "04_step7_primary_front_variable_window_model.csv"
STEP3_STATUS_PATH = ROOT / "data_clean" / "step3" / "07_fnirs_session_status.csv"

FIGURES_DIR = ROOT / "figures" / "step7_visualization"
REPORTS_DIR = ROOT / "reports" / "step7_visualization"


def ensure_directories() -> None:
    for path in [FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trial_df = pd.read_csv(STEP7_TRIAL_PATH)
    primary_model_df = pd.read_csv(STEP7_PRIMARY_MODEL_PATH)
    status_df = pd.read_csv(STEP3_STATUS_PATH)
    trial_df["eligible_primary_model"] = trial_df["included_primary_model"]
    return trial_df, primary_model_df, status_df


def write_report(representative_session: tuple[str, str], metadata: dict[str, Any], primary_p_value: float) -> Path:
    overview_rows = [
        ("Representative geometry session", f"{representative_session[0]} / {representative_session[1]}"),
        ("Included participant-sessions from Step 7", metadata["n_included_sessions"]),
        ("Included questions in Step 7 trial table", metadata["n_included_questions"]),
        ("Abstract onset-locked epochs used", metadata["n_abstract_epochs"]),
        ("Concrete onset-locked epochs used", metadata["n_concrete_epochs"]),
        ("Epoch window", f"{base.EPOCH_TMIN:.1f} s to {base.EPOCH_TMAX:.1f} s around question onset"),
        ("Baseline", f"{base.BASELINE[0]:.1f} s to {base.BASELINE[1]:.1f} s"),
        ("Long-channel montage used in descriptive plots", f"{metadata['n_long_pairs']} pairs ({metadata['n_long_channels_total']} chromophore channels)"),
        ("Primary Step 7 duration-aware p-value", f"{primary_p_value:.6f}"),
    ]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Supplementary Step 7 Visualization Report}",
        r"\subsection*{Overview}",
        base.make_latex_table(overview_rows),
        r"\paragraph{Adaptation note.} "
        r"This report mirrors the Step~6 visualization bundle but anchors it to the Step~7 duration-aware cohort and primary model results. "
        r"Because the Step~7 inferential model summarizes continuous fNIRS over variable-length windows, the descriptive topographic plots here remain onset-locked so that all trials share a common time base for direct visual comparison.",
        r"\paragraph{Rendering note.} "
        r"As in the previous supplementary visualization reports, the sensor-geometry figure uses the participant digitization point cloud plus source, detector, channel, and pair geometry rather than an fsaverage brain-surface rendering, because the local \texttt{pyvista}/\texttt{fsaverage} stack is not available in this environment.",
        r"\subsection*{Step 7 Main Figures Reused}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/response_time_by_condition.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/variable_window_duration_histogram.png}",
        r"\caption{Step~7 duration diagnostics for response time by condition and the distribution of final variable-window durations.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/response_time_vs_wordcount.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/response_time_vs_correctness.png}",
        r"\caption{Step~7 response-time associations with item length and ENEM item correctness.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/front_roi_partial_effect_variable_window.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/back_roi_partial_effect_variable_window.png}",
        r"\caption{Step~7 duration-aware condition-effect summaries for the front and back ROI models.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/dissociation_partial_effect_variable_window.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7/fixed_vs_variable_comparison.png}",
        r"\caption{Step~7 dissociation effect and the direct benchmark against the Step~6 fixed-window model.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.72\linewidth]{../../figures/step7/model_residuals_primary.png}",
        r"\caption{Residual diagnostic from the Step~7 primary duration-aware model.}",
        r"\end{figure}",
        r"\subsection*{Step 6-Style Supplementary Plots for Step 7}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step7_visualization/representative_sensor_geometry.png}",
        r"\caption{Representative optode, channel, and source-detector-pair geometry for one Step~7 included participant-session. Short pairs are shown with dashed orange lines.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7_visualization/abstract_trial_consistency.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7_visualization/concrete_trial_consistency.png}",
        r"\caption{Condition-adapted trial-consistency image plots for the Step~7 included trial set.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step7_visualization/channel_consistency_grid.png}",
        r"\caption{Grand-average long-channel response images across channels for the Abstract and Concrete conditions.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.92\linewidth]{../../figures/step7_visualization/standard_fnirs_compare.png}",
        r"\caption{Standard fNIRS waveform comparison across long channels for the Step~7 included trial set.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step7_visualization/abstract_hbo_joint.png}",
        r"\caption{Grand-average HbO topography over time for Abstract questions using the Step~7 included trial set.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step7_visualization/concrete_hbo_joint.png}",
        r"\caption{Grand-average HbO topography over time for Concrete questions using the Step~7 included trial set.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step7_visualization/abstract_minus_concrete_hbo_joint.png}",
        r"\caption{Grand-average Abstract-minus-Concrete HbO topography over time.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7_visualization/abstract_hbo_topomap_series.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7_visualization/concrete_hbo_topomap_series.png}",
        r"\caption{HbO topographic time series from 4.0~s to 10.0~s for the Abstract and Concrete conditions.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7_visualization/abstract_hbr_topomap_series.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step7_visualization/concrete_hbr_topomap_series.png}",
        r"\caption{HbR topographic time series from 4.0~s to 10.0~s for the Abstract and Concrete conditions.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step7_visualization/abstract_concrete_single_time_comparison.png}",
        r"\caption{Single-time comparison at 9.0~s for Abstract, Concrete, and Abstract-minus-Concrete topographies for both HbO and HbR.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step7_visualization/abstract_concrete_hbo_evoked_topo.png}",
        r"\caption{HbO waveforms by long channel, overlaid for Abstract and Concrete, to show what drives the descriptive topographic differences.}",
        r"\end{figure}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step7_visualization_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()

    base.FIGURES_DIR = FIGURES_DIR
    base.REPORTS_DIR = REPORTS_DIR
    base.STEP3_STATUS_PATH = STEP3_STATUS_PATH

    trial_df, primary_model_df, status_df = load_inputs()
    participant_id, session_id = base.get_representative_session(trial_df)
    representative_file = status_df.loc[
        (status_df["participant_id"] == participant_id) & (status_df["session_id"] == session_id),
        "preprocessed_file",
    ].iloc[0]
    representative_raw = mne.io.read_raw_fif(representative_file, preload=False, verbose="ERROR")
    base.plot_sensor_geometry(representative_raw, participant_id, session_id)

    dataset = base.build_visualization_dataset(trial_df, status_df)
    times = dataset["times"]
    condition_epochs = dataset["condition_epochs"]
    session_mean_traces = dataset["session_mean_traces"]
    grand_evokeds = dataset["grand_evokeds"]
    metadata = dataset["metadata"]

    base.plot_trial_consistency("Abstract", condition_epochs["Abstract"], times, FIGURES_DIR / "abstract_trial_consistency.png")
    base.plot_trial_consistency("Concrete", condition_epochs["Concrete"], times, FIGURES_DIR / "concrete_trial_consistency.png")
    base.plot_channel_consistency(grand_evokeds, times, FIGURES_DIR / "channel_consistency_grid.png")
    base.plot_standard_fnirs_response(session_mean_traces, times, FIGURES_DIR / "standard_fnirs_compare.png")

    base.save_joint_plot(grand_evokeds["Abstract"], "Abstract questions: HbO topography over time", FIGURES_DIR / "abstract_hbo_joint.png")
    base.save_joint_plot(grand_evokeds["Concrete"], "Concrete questions: HbO topography over time", FIGURES_DIR / "concrete_hbo_joint.png")
    base.save_difference_joint_plot(grand_evokeds["Abstract"], grand_evokeds["Concrete"], FIGURES_DIR / "abstract_minus_concrete_hbo_joint.png")

    hbo_series_vlim = base.compute_topomap_vlim([grand_evokeds["Abstract"], grand_evokeds["Concrete"]], "hbo", base.TOPOMAP_SERIES_TIMES)
    hbr_series_vlim = base.compute_topomap_vlim([grand_evokeds["Abstract"], grand_evokeds["Concrete"]], "hbr", base.TOPOMAP_SERIES_TIMES)
    base.save_topomap_series(grand_evokeds["Abstract"], "Abstract", "hbo", hbo_series_vlim, FIGURES_DIR / "abstract_hbo_topomap_series.png")
    base.save_topomap_series(grand_evokeds["Concrete"], "Concrete", "hbo", hbo_series_vlim, FIGURES_DIR / "concrete_hbo_topomap_series.png")
    base.save_topomap_series(grand_evokeds["Abstract"], "Abstract", "hbr", hbr_series_vlim, FIGURES_DIR / "abstract_hbr_topomap_series.png")
    base.save_topomap_series(grand_evokeds["Concrete"], "Concrete", "hbr", hbr_series_vlim, FIGURES_DIR / "concrete_hbr_topomap_series.png")

    base.save_single_time_comparison(
        grand_evokeds["Abstract"],
        grand_evokeds["Concrete"],
        FIGURES_DIR / "abstract_concrete_single_time_comparison.png",
    )
    base.save_evoked_topo_overlay(
        grand_evokeds["Abstract"],
        grand_evokeds["Concrete"],
        FIGURES_DIR / "abstract_concrete_hbo_evoked_topo.png",
    )

    primary_p_value = float(primary_model_df.iloc[0]["condition_p_value"])
    report_path = write_report((participant_id, session_id), metadata, primary_p_value)
    base.compile_report(report_path)

    print("Step 7 visualization report generated successfully.")
    print(f"Included participant-sessions: {metadata['n_included_sessions']}")
    print(f"Included questions: {metadata['n_included_questions']}")
    print(f"Abstract epochs: {metadata['n_abstract_epochs']}")
    print(f"Concrete epochs: {metadata['n_concrete_epochs']}")


if __name__ == "__main__":
    main()
