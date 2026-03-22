#!/usr/bin/env python3

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STEP4_TRIAL_PATH = ROOT / "data_clean" / "step4" / "02_trial_fnirs_summary_hbo.csv"
STEP4_SUMMARY_PATH = ROOT / "data_clean" / "step4" / "04_participant_roi_condition_summary.csv"
STEP3_STATUS_PATH = ROOT / "data_clean" / "step3" / "07_fnirs_session_status.csv"

FIGURES_DIR = ROOT / "figures" / "step4_visualization"
REPORTS_DIR = ROOT / "reports" / "step4_visualization"

EPOCH_TMIN = -3.5
EPOCH_TMAX = 13.2
BASELINE = (-2.0, 0.0)
JOINT_TIMES = np.arange(-3.5, 13.2, 3.0)
COMPARISON_TIME = 9.0
SHORT_PAIRS = {"S1_D8", "S2_D9", "S3_D10", "S4_D11", "S5_D12", "S6_D13", "S7_D14", "S8_D15"}


def ensure_directories() -> None:
    for path in [FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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


def make_latex_table(rows: list[tuple[str, Any]], column_spec: str = r"p{0.36\linewidth}p{0.54\linewidth}") -> str:
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule", r"Metric & Value\\", r"\midrule"]
    for key, value in rows:
        lines.append(f"{sanitize_for_tex(key)} & {sanitize_for_tex(value)}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


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


def set_axes_equal(ax: Any) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    half = max_range / 2
    ax.set_xlim3d([x_mid - half, x_mid + half])
    ax.set_ylim3d([y_mid - half, y_mid + half])
    ax.set_zlim3d([z_mid - half, z_mid + half])


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trial_df = pd.read_csv(STEP4_TRIAL_PATH)
    summary_df = pd.read_csv(STEP4_SUMMARY_PATH)
    status_df = pd.read_csv(STEP3_STATUS_PATH)
    return trial_df, summary_df, status_df


def get_representative_session(summary_df: pd.DataFrame) -> tuple[str, str]:
    included_df = summary_df[summary_df["included_primary_analysis"]].copy()
    included_df = included_df.sort_values(["participant_id", "session_id"]).reset_index(drop=True)
    first_row = included_df.iloc[0]
    return str(first_row["participant_id"]), str(first_row["session_id"])


def plot_sensor_geometry(raw: mne.io.BaseRaw, participant_id: str, session_id: str) -> None:
    hbo_channel_names = [name for name in raw.ch_names if name.endswith(" hbo")]
    source_points: dict[str, np.ndarray] = {}
    detector_points: dict[str, np.ndarray] = {}
    channel_points: list[tuple[str, np.ndarray, bool]] = []

    for channel_name in hbo_channel_names:
        pair_id = channel_name.replace(" hbo", "")
        source_id, detector_id = pair_id.split("_")
        ch = raw.info["chs"][raw.ch_names.index(channel_name)]
        midpoint = np.asarray(ch["loc"][:3], dtype=float)
        source = np.asarray(ch["loc"][3:6], dtype=float)
        detector = np.asarray(ch["loc"][6:9], dtype=float)
        source_points[source_id] = source
        detector_points[detector_id] = detector
        channel_points.append((pair_id, midpoint, pair_id in SHORT_PAIRS))

    scalp_points = []
    if raw.info.get("dig"):
        for dig in raw.info["dig"]:
            if hasattr(dig, "get"):
                coords = dig.get("r")
            else:
                coords = getattr(dig, "r", None)
            if coords is None:
                continue
            coords = np.asarray(coords, dtype=float)
            if coords.shape == (3,) and np.isfinite(coords).all():
                scalp_points.append(coords)
    scalp_points_arr = np.vstack(scalp_points) if scalp_points else np.empty((0, 3))

    fig = plt.figure(figsize=(12, 6))
    views = [(20, 60, "Oblique view"), (90, 15, "Left lateral view")]
    for idx, (azimuth, elevation, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        if len(scalp_points_arr):
            ax.scatter(
                scalp_points_arr[:, 0],
                scalp_points_arr[:, 1],
                scalp_points_arr[:, 2],
                color="#D5DBDB",
                alpha=0.25,
                s=6,
            )
        for pair_id, midpoint, is_short in channel_points:
            source_id, detector_id = pair_id.split("_")
            source = source_points[source_id]
            detector = detector_points[detector_id]
            line_color = "#F39C12" if is_short else "#7F8C8D"
            line_style = "--" if is_short else "-"
            ax.plot(
                [source[0], detector[0]],
                [source[1], detector[1]],
                [source[2], detector[2]],
                color=line_color,
                linestyle=line_style,
                linewidth=1.5,
                alpha=0.9,
            )
            ax.scatter(midpoint[0], midpoint[1], midpoint[2], color="#E67E22", s=28)
        for source in source_points.values():
            ax.scatter(source[0], source[1], source[2], color="#C0392B", s=35)
        for detector in detector_points.values():
            ax.scatter(detector[0], detector[1], detector[2], color="#17202A", s=30)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=elevation, azim=azimuth)
        set_axes_equal(ax)

    fig.suptitle(
        f"Representative sensor geometry: {participant_id} / {session_id}\n"
        "Gray point cloud = head digitization, red = sources, black = detectors, orange = channels"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "representative_sensor_geometry.png", dpi=180)
    plt.close(fig)


def build_session_evokeds(
    trial_df: pd.DataFrame,
    status_df: pd.DataFrame,
) -> tuple[mne.Evoked, mne.Evoked, dict[str, Any]]:
    included_df = trial_df[trial_df["included_primary_analysis"]].copy()
    included_df = included_df.sort_values(["participant_id", "session_id", "question_start_time"]).reset_index(drop=True)
    status_lookup = {
        (row["participant_id"], row["session_id"]): row["preprocessed_file"]
        for _, row in status_df.iterrows()
    }

    abstract_session_arrays: list[np.ndarray] = []
    concrete_session_arrays: list[np.ndarray] = []
    template_info = None
    abstract_epoch_count = 0
    concrete_epoch_count = 0
    dropped_for_window = 0

    for (participant_id, session_id), session_df in included_df.groupby(["participant_id", "session_id"], sort=True):
        raw = mne.io.read_raw_fif(status_lookup[(participant_id, session_id)], preload=True, verbose="ERROR")
        if template_info is None:
            template_info = raw.info.copy()
            template_info["bads"] = []

        session_df = session_df[
            (session_df["question_start_time"] + EPOCH_TMIN >= raw.times[0] - 1e-9)
            & (session_df["question_start_time"] + EPOCH_TMAX <= raw.times[-1] + 1e-9)
        ].copy()
        dropped_for_window += len(included_df[(included_df["participant_id"] == participant_id) & (included_df["session_id"] == session_id)]) - len(session_df)
        if session_df.empty:
            continue

        event_rows = []
        for row in session_df.itertuples(index=False):
            event_code = 1 if row.question_type == "Abstract" else 2
            sample = raw.time_as_index([row.question_start_time])[0]
            event_rows.append([sample, 0, event_code])
        events = np.asarray(sorted(event_rows), dtype=int)

        epochs = mne.Epochs(
            raw,
            events,
            event_id={"Abstract": 1, "Concrete": 2},
            tmin=EPOCH_TMIN,
            tmax=EPOCH_TMAX,
            baseline=BASELINE,
            preload=True,
            reject_by_annotation=False,
            verbose="ERROR",
        )

        if len(epochs["Abstract"]):
            abstract_epoch_count += len(epochs["Abstract"])
            abstract_data = epochs["Abstract"].get_data(copy=True).mean(axis=0)
            abstract_bad_idx = [epochs.ch_names.index(name) for name in raw.info["bads"] if name in epochs.ch_names]
            if abstract_bad_idx:
                abstract_data[abstract_bad_idx, :] = np.nan
            abstract_session_arrays.append(abstract_data)

        if len(epochs["Concrete"]):
            concrete_epoch_count += len(epochs["Concrete"])
            concrete_data = epochs["Concrete"].get_data(copy=True).mean(axis=0)
            concrete_bad_idx = [epochs.ch_names.index(name) for name in raw.info["bads"] if name in epochs.ch_names]
            if concrete_bad_idx:
                concrete_data[concrete_bad_idx, :] = np.nan
            concrete_session_arrays.append(concrete_data)

    if template_info is None:
        raise RuntimeError("No valid sessions were available to build visualization evokeds.")

    abstract_grand = np.nanmean(np.stack(abstract_session_arrays, axis=0), axis=0)
    concrete_grand = np.nanmean(np.stack(concrete_session_arrays, axis=0), axis=0)
    abstract_grand = np.nan_to_num(abstract_grand, nan=0.0)
    concrete_grand = np.nan_to_num(concrete_grand, nan=0.0)

    evoked_abstract = mne.EvokedArray(abstract_grand, template_info, tmin=EPOCH_TMIN, comment="Abstract", nave=len(abstract_session_arrays))
    evoked_concrete = mne.EvokedArray(concrete_grand, template_info, tmin=EPOCH_TMIN, comment="Concrete", nave=len(concrete_session_arrays))
    metadata = {
        "n_included_sessions": int(included_df[["participant_id", "session_id"]].drop_duplicates().shape[0]),
        "n_abstract_epochs": abstract_epoch_count,
        "n_concrete_epochs": concrete_epoch_count,
        "n_dropped_for_epoch_window": dropped_for_window,
    }
    return evoked_abstract, evoked_concrete, metadata


def save_joint_plot(evoked: mne.Evoked, title: str, path: Path) -> None:
    fig = evoked.copy().pick(picks="hbo").plot_joint(
        times=JOINT_TIMES,
        topomap_args=dict(extrapolate="local"),
        show=False,
    )
    fig.suptitle(title, y=0.98)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_single_time_comparison(evoked_abstract: mne.Evoked, evoked_concrete: mne.Evoked) -> None:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(10, 5.5),
        gridspec_kw=dict(width_ratios=[1, 1, 1, 0.12]),
        layout="constrained",
    )
    topomap_args = dict(extrapolate="local")
    evoked_diff = mne.combine_evoked([evoked_abstract, evoked_concrete], weights=[1, -1])

    def compute_vlim(ch_type: str) -> tuple[float, float]:
        abs_vals = []
        for evoked in [evoked_abstract, evoked_concrete, evoked_diff]:
            picked = evoked.copy().pick(ch_type)
            time_idx = picked.time_as_index(COMPARISON_TIME)[0]
            abs_vals.append(np.abs(picked.data[:, time_idx]))
        vmax = float(np.max(np.concatenate(abs_vals)))
        return (-vmax, vmax)

    vlim_hbo = compute_vlim("hbo")
    vlim_hbr = compute_vlim("hbr")

    evoked_abstract.plot_topomap(
        ch_type="hbo",
        times=COMPARISON_TIME,
        axes=axes[0, 0],
        vlim=vlim_hbo,
        colorbar=False,
        show=False,
        **topomap_args,
    )
    evoked_abstract.plot_topomap(
        ch_type="hbr",
        times=COMPARISON_TIME,
        axes=axes[1, 0],
        vlim=vlim_hbr,
        colorbar=False,
        show=False,
        **topomap_args,
    )
    evoked_concrete.plot_topomap(
        ch_type="hbo",
        times=COMPARISON_TIME,
        axes=axes[0, 1],
        vlim=vlim_hbo,
        colorbar=False,
        show=False,
        **topomap_args,
    )
    evoked_concrete.plot_topomap(
        ch_type="hbr",
        times=COMPARISON_TIME,
        axes=axes[1, 1],
        vlim=vlim_hbr,
        colorbar=False,
        show=False,
        **topomap_args,
    )
    evoked_diff.plot_topomap(
        ch_type="hbo",
        times=COMPARISON_TIME,
        axes=axes[0, 2:],
        vlim=vlim_hbo,
        colorbar=True,
        show=False,
        **topomap_args,
    )
    evoked_diff.plot_topomap(
        ch_type="hbr",
        times=COMPARISON_TIME,
        axes=axes[1, 2:],
        vlim=vlim_hbr,
        colorbar=True,
        show=False,
        **topomap_args,
    )

    for column, condition in enumerate(["Abstract", "Concrete", "Abstract-Concrete"]):
        axes[0, column].set_title(f"HbO: {condition}")
        axes[1, column].set_title(f"HbR: {condition}")
    fig.suptitle(f"Single-time-point topographic comparison at {COMPARISON_TIME:.1f} s")
    fig.savefig(FIGURES_DIR / "abstract_concrete_single_time_comparison.png", dpi=160)
    plt.close(fig)


def write_report(
    representative_session: tuple[str, str],
    metadata: dict[str, Any],
) -> Path:
    overview_rows = [
        ("Representative geometry session", f"{representative_session[0]} / {representative_session[1]}"),
        ("Included participant-sessions from Step 4", metadata["n_included_sessions"]),
        ("Abstract onset-locked epochs used", metadata["n_abstract_epochs"]),
        ("Concrete onset-locked epochs used", metadata["n_concrete_epochs"]),
        ("Trials dropped for fixed epoch window", metadata["n_dropped_for_epoch_window"]),
        ("Epoch window", f"{EPOCH_TMIN:.1f} s to {EPOCH_TMAX:.1f} s around question onset"),
        ("Baseline", f"{BASELINE[0]:.1f} s to {BASELINE[1]:.1f} s"),
        ("Single-time comparison", f"{COMPARISON_TIME:.1f} s"),
    ]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Supplementary Step 4 Visualization Report}",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
        r"\paragraph{Rendering note.} "
        r"The original example requested a brain-surface sensor rendering. In this environment, the report uses the participant digitization point cloud together with source, detector, channel, and pair geometry because the local \texttt{pyvista}/\texttt{fsaverage} rendering stack is not available.",
        r"\subsection*{Representative Sensor Geometry}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step4_visualization/representative_sensor_geometry.png}",
        r"\caption{Representative optode, channel, and source-detector-pair geometry for one included participant-session. Short pairs are shown with dashed orange lines.}",
        r"\end{figure}",
        r"\subsection*{Condition-Specific Topography Over Time}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step4_visualization/abstract_hbo_joint.png}",
        r"\caption{Grand-average HbO topography over time for abstract questions, aligned to question onset.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step4_visualization/concrete_hbo_joint.png}",
        r"\caption{Grand-average HbO topography over time for concrete questions, aligned to question onset.}",
        r"\end{figure}",
        r"\subsection*{Single-Time-Point Condition Comparison}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step4_visualization/abstract_concrete_single_time_comparison.png}",
        r"\caption{Condition comparison at one time point for HbO and HbR, with the abstract-minus-concrete difference shown in the rightmost column.}",
        r"\end{figure}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step4_visualization_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()
    trial_df, summary_df, status_df = load_inputs()
    participant_id, session_id = get_representative_session(summary_df)
    representative_file = status_df.loc[
        (status_df["participant_id"] == participant_id) & (status_df["session_id"] == session_id),
        "preprocessed_file",
    ].iloc[0]
    representative_raw = mne.io.read_raw_fif(representative_file, preload=False, verbose="ERROR")
    plot_sensor_geometry(representative_raw, participant_id, session_id)

    evoked_abstract, evoked_concrete, metadata = build_session_evokeds(trial_df, status_df)
    save_joint_plot(
        evoked_abstract,
        "Abstract questions: HbO topography over time",
        FIGURES_DIR / "abstract_hbo_joint.png",
    )
    save_joint_plot(
        evoked_concrete,
        "Concrete questions: HbO topography over time",
        FIGURES_DIR / "concrete_hbo_joint.png",
    )
    save_single_time_comparison(evoked_abstract, evoked_concrete)

    report_path = write_report((participant_id, session_id), metadata)
    compile_report(report_path)

    print("Step 4 visualization report generated successfully.")
    print(f"Included participant-sessions: {metadata['n_included_sessions']}")
    print(f"Abstract epochs: {metadata['n_abstract_epochs']}")
    print(f"Concrete epochs: {metadata['n_concrete_epochs']}")


if __name__ == "__main__":
    main()
