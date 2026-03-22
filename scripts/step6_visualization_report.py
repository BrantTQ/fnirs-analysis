#!/usr/bin/env python3

from __future__ import annotations

import math
import subprocess
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
STEP6_TRIAL_PATH = ROOT / "data_clean" / "step6" / "03_step6_trial_model_table.csv"
STEP6_PRIMARY_MODEL_PATH = ROOT / "data_clean" / "step6" / "04_step6_primary_front_wordcount_model.csv"
STEP3_STATUS_PATH = ROOT / "data_clean" / "step3" / "07_fnirs_session_status.csv"

FIGURES_DIR = ROOT / "figures" / "step6_visualization"
REPORTS_DIR = ROOT / "reports" / "step6_visualization"

CONDITIONS = ("Abstract", "Concrete")
CHROMOPHORES = ("hbo", "hbr")
EPOCH_TMIN = -3.5
EPOCH_TMAX = 13.2
BASELINE = (-2.0, 0.0)
JOINT_TIMES = np.arange(-3.5, 13.2, 3.0)
TOPOMAP_SERIES_TIMES = np.arange(4.0, 11.0, 1.0)
COMPARISON_TIME = 9.0
TOPOMAP_ARGS = dict(extrapolate="local")
SHORT_PAIRS = {"S1_D8", "S2_D9", "S3_D10", "S4_D11", "S5_D12", "S6_D13", "S7_D14", "S8_D15"}
FRONT_PAIRS = ["S2_D6", "S2_D4", "S1_D6", "S1_D3", "S5_D6", "S5_D4", "S5_D3"]
TRANSITION_PAIRS = ["S5_D7", "S4_D4", "S4_D7", "S3_D3", "S3_D7", "S6_D7"]
BACK_PAIRS = ["S4_D2", "S3_D1", "S6_D2", "S6_D1", "S6_D5", "S7_D2", "S7_D5", "S8_D1", "S8_D5"]
LONG_PAIRS = FRONT_PAIRS + TRANSITION_PAIRS + BACK_PAIRS
LONG_HBO_CHANNELS = [f"{pair_id} hbo" for pair_id in LONG_PAIRS]
LONG_HBR_CHANNELS = [f"{pair_id} hbr" for pair_id in LONG_PAIRS]
ALL_LONG_CHANNELS = LONG_HBO_CHANNELS + LONG_HBR_CHANNELS
HBO_SLICE = slice(0, len(LONG_HBO_CHANNELS))
HBR_SLICE = slice(len(LONG_HBO_CHANNELS), len(ALL_LONG_CHANNELS))


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


def nanmean_no_warning(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def compute_mean_and_ci(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_trace = nanmean_no_warning(traces, axis=0)
    counts = np.sum(np.isfinite(traces), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sd_trace = np.nanstd(traces, axis=0, ddof=1)
    sem_trace = np.divide(sd_trace, np.sqrt(counts), out=np.zeros_like(sd_trace), where=counts > 1)
    ci = 1.96 * sem_trace
    ci = np.where(counts > 1, ci, 0.0)
    return mean_trace, mean_trace - ci, mean_trace + ci


def symmetric_vlim(values: np.ndarray, percentile: float = 99.0, minimum: float = 0.05) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return minimum
    vmax = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        return minimum
    return max(vmax, minimum)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trial_df = pd.read_csv(STEP6_TRIAL_PATH)
    primary_model_df = pd.read_csv(STEP6_PRIMARY_MODEL_PATH)
    status_df = pd.read_csv(STEP3_STATUS_PATH)
    return trial_df, primary_model_df, status_df


def get_representative_session(trial_df: pd.DataFrame) -> tuple[str, str]:
    included_df = trial_df[trial_df["eligible_primary_model"]].copy()
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
            coords = dig.get("r") if hasattr(dig, "get") else getattr(dig, "r", None)
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
        f"Representative Step 6 sensor geometry: {participant_id} / {session_id}\n"
        "Gray point cloud = head digitization, red = sources, black = detectors, orange = channels"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "representative_sensor_geometry.png", dpi=180)
    plt.close(fig)


def build_visualization_dataset(trial_df: pd.DataFrame, status_df: pd.DataFrame) -> dict[str, Any]:
    included_df = trial_df[trial_df["eligible_primary_model"]].copy()
    included_df = included_df.sort_values(["participant_id", "session_id", "question_start_time"]).reset_index(drop=True)
    status_lookup = {
        (row["participant_id"], row["session_id"]): row["preprocessed_file"]
        for _, row in status_df.iterrows()
        if isinstance(row.get("preprocessed_file"), str) and row.get("preprocessed_file")
    }

    condition_epoch_chunks: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    condition_session_mean_chunks: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    session_mean_traces: dict[str, dict[str, list[np.ndarray]]] = {
        condition: {chroma: [] for chroma in CHROMOPHORES} for condition in CONDITIONS
    }
    template_info = None
    times = None
    included_sessions = 0

    for (participant_id, session_id), session_df in included_df.groupby(["participant_id", "session_id"], sort=True):
        preprocessed_file = status_lookup.get((participant_id, session_id))
        if not preprocessed_file:
            continue
        raw = mne.io.read_raw_fif(preprocessed_file, preload=True, verbose="ERROR")
        raw_long = raw.copy().pick(ALL_LONG_CHANNELS)
        if template_info is None:
            template_info = raw_long.info.copy()
            template_info["bads"] = []

        event_rows = []
        for row in session_df.itertuples(index=False):
            event_code = 1 if row.condition == "Abstract" else 2
            sample = raw_long.time_as_index([row.question_start_time])[0]
            event_rows.append([sample, 0, event_code])
        if not event_rows:
            continue

        epochs = mne.Epochs(
            raw_long,
            np.asarray(sorted(event_rows), dtype=int),
            event_id={"Abstract": 1, "Concrete": 2},
            tmin=EPOCH_TMIN,
            tmax=EPOCH_TMAX,
            baseline=BASELINE,
            preload=True,
            reject_by_annotation=False,
            verbose="ERROR",
        )
        if times is None:
            times = epochs.times.copy()

        data = epochs.get_data(copy=True)
        bad_idx = [epochs.ch_names.index(name) for name in raw_long.info["bads"] if name in epochs.ch_names]
        if bad_idx:
            data[:, bad_idx, :] = np.nan

        event_condition = np.where(epochs.events[:, 2] == 1, "Abstract", "Concrete")
        session_added = False
        for condition in CONDITIONS:
            condition_data = data[event_condition == condition]
            if condition_data.size == 0:
                continue
            session_added = True
            condition_epoch_chunks[condition].append(condition_data)
            session_mean = nanmean_no_warning(condition_data, axis=0)
            condition_session_mean_chunks[condition].append(session_mean)
            session_mean_traces[condition]["hbo"].append(nanmean_no_warning(session_mean[HBO_SLICE, :], axis=0))
            session_mean_traces[condition]["hbr"].append(nanmean_no_warning(session_mean[HBR_SLICE, :], axis=0))
        if session_added:
            included_sessions += 1

    if template_info is None or times is None:
        raise RuntimeError("No valid Step 6 sessions were available to build visualization plots.")

    condition_epochs = {
        condition: np.concatenate(chunks, axis=0) for condition, chunks in condition_epoch_chunks.items() if chunks
    }
    grand_evokeds: dict[str, mne.Evoked] = {}
    for condition, session_means in condition_session_mean_chunks.items():
        session_stack = np.stack(session_means, axis=0)
        grand_data = nanmean_no_warning(session_stack, axis=0)
        grand_evokeds[condition] = mne.EvokedArray(
            np.nan_to_num(grand_data, nan=0.0),
            template_info.copy(),
            tmin=EPOCH_TMIN,
            comment=condition,
            nave=len(session_means),
        )

    metadata = {
        "n_included_sessions": included_sessions,
        "n_included_questions": int(included_df["question_id"].nunique()),
        "n_abstract_epochs": int(condition_epochs["Abstract"].shape[0]),
        "n_concrete_epochs": int(condition_epochs["Concrete"].shape[0]),
        "n_long_pairs": len(LONG_PAIRS),
        "n_long_channels_total": len(ALL_LONG_CHANNELS),
    }
    return {
        "times": times,
        "condition_epochs": condition_epochs,
        "session_mean_traces": session_mean_traces,
        "grand_evokeds": grand_evokeds,
        "metadata": metadata,
    }


def plot_trial_consistency(condition: str, condition_epochs: np.ndarray, times: np.ndarray, path: Path) -> None:
    mean_hbo = nanmean_no_warning(condition_epochs[:, HBO_SLICE, :], axis=1) * 1e6
    mean_hbr = nanmean_no_warning(condition_epochs[:, HBR_SLICE, :], axis=1) * 1e6
    vmax = symmetric_vlim(np.concatenate([mean_hbo, mean_hbr], axis=0), percentile=99.0, minimum=0.1)

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[2.9, 1.5])
    row_specs = [
        ("HbO", mean_hbo, "#AA3377"),
        ("HbR", mean_hbr, "#004488"),
    ]

    for row_idx, (label, trial_matrix, color) in enumerate(row_specs):
        ax_img = fig.add_subplot(grid[row_idx, 0])
        ax_ts = fig.add_subplot(grid[row_idx, 1])

        image = ax_img.imshow(
            trial_matrix,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
            extent=[times[0], times[-1], 1, trial_matrix.shape[0]],
        )
        ax_img.axvline(0.0, color="k", linestyle="--", linewidth=1)
        ax_img.set_title(f"{condition}: {label} trial image (mean across {len(LONG_PAIRS)} long pairs)")
        ax_img.set_xlabel("Time from question onset (s)")
        ax_img.set_ylabel("Trial")
        fig.colorbar(image, ax=ax_img, fraction=0.046, pad=0.03, label=r"$\mu$M")

        mean_trace, low_trace, high_trace = compute_mean_and_ci(trial_matrix)
        ax_ts.fill_between(times, low_trace, high_trace, color=color, alpha=0.20)
        ax_ts.plot(times, mean_trace, color=color, linewidth=2)
        ax_ts.axhline(0.0, color="k", linewidth=1)
        ax_ts.axvline(0.0, color="k", linestyle="--", linewidth=1)
        ax_ts.set_title(f"{condition}: {label} mean with 95% CI")
        ax_ts.set_xlabel("Time from question onset (s)")
        ax_ts.set_ylabel(r"$\mu$M")
        ax_ts.set_ylim(-vmax * 1.05, vmax * 1.05)

    fig.suptitle(
        f"Step 6 response consistency across trials: {condition}\n"
        f"Same plot family used in the Step 5 supplementary visualization report"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_channel_consistency(grand_evokeds: dict[str, mne.Evoked], times: np.ndarray, path: Path) -> None:
    data = {
        ("Abstract", "hbo"): grand_evokeds["Abstract"].copy().pick(picks="hbo").data * 1e6,
        ("Abstract", "hbr"): grand_evokeds["Abstract"].copy().pick(picks="hbr").data * 1e6,
        ("Concrete", "hbo"): grand_evokeds["Concrete"].copy().pick(picks="hbo").data * 1e6,
        ("Concrete", "hbr"): grand_evokeds["Concrete"].copy().pick(picks="hbr").data * 1e6,
    }
    hbo_vmax = symmetric_vlim(np.concatenate([data[("Abstract", "hbo")], data[("Concrete", "hbo")]], axis=0), percentile=99.0, minimum=0.1)
    hbr_vmax = symmetric_vlim(np.concatenate([data[("Abstract", "hbr")], data[("Concrete", "hbr")]], axis=0), percentile=99.0, minimum=0.1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained", sharex=True)
    specs = [
        (0, 0, "Abstract", "hbo", hbo_vmax),
        (0, 1, "Concrete", "hbo", hbo_vmax),
        (1, 0, "Abstract", "hbr", hbr_vmax),
        (1, 1, "Concrete", "hbr", hbr_vmax),
    ]

    for row_idx, col_idx, condition, chroma, vmax in specs:
        ax = axes[row_idx, col_idx]
        image = ax.imshow(
            data[(condition, chroma)],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
            extent=[times[0], times[-1], 1, len(LONG_PAIRS)],
        )
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"{condition}: {chroma.upper()} ({len(LONG_PAIRS)} long pairs)")
        ax.set_xlabel("Time from question onset (s)")
        if col_idx == 0:
            ax.set_ylabel("Long-channel pair")
            ax.set_yticks(np.arange(1, len(LONG_PAIRS) + 1))
            ax.set_yticklabels(LONG_PAIRS, fontsize=7)
        else:
            ax.set_yticks(np.arange(1, len(LONG_PAIRS) + 1))
            ax.set_yticklabels([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label=r"$\mu$M")

    fig.suptitle(
        "Step 6 response consistency across channels\n"
        "Grand-average long-channel waveforms for Abstract and Concrete conditions"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_standard_fnirs_response(session_mean_traces: dict[str, dict[str, list[np.ndarray]]], times: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    styles = {
        ("Abstract", "hbo"): dict(color="#AA3377", linestyle="-"),
        ("Abstract", "hbr"): dict(color="#004488", linestyle="-"),
        ("Concrete", "hbo"): dict(color="#AA3377", linestyle="--"),
        ("Concrete", "hbr"): dict(color="#004488", linestyle="--"),
    }

    for condition in CONDITIONS:
        for chroma in CHROMOPHORES:
            traces = np.stack(session_mean_traces[condition][chroma], axis=0) * 1e6
            mean_trace, low_trace, high_trace = compute_mean_and_ci(traces)
            style = styles[(condition, chroma)]
            ax.fill_between(times, low_trace, high_trace, color=style["color"], alpha=0.12)
            ax.plot(
                times,
                mean_trace,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                label=f"{condition} {chroma.upper()}",
            )

    ax.axhline(0.0, color="k", linewidth=1)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_title("Standard fNIRS response summary across long channels")
    ax.set_xlabel("Time from question onset (s)")
    ax.set_ylabel(r"$\mu$M")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_joint_plot(evoked: mne.Evoked, title: str, path: Path) -> None:
    fig = evoked.copy().pick(picks="hbo").plot_joint(times=JOINT_TIMES, topomap_args=TOPOMAP_ARGS, show=False)
    fig.suptitle(title, y=0.98)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_difference_joint_plot(evoked_abstract: mne.Evoked, evoked_concrete: mne.Evoked, path: Path) -> None:
    evoked_diff = mne.combine_evoked([evoked_abstract, evoked_concrete], weights=[1, -1])
    fig = evoked_diff.copy().pick(picks="hbo").plot_joint(times=JOINT_TIMES, topomap_args=TOPOMAP_ARGS, show=False)
    fig.suptitle("Abstract-minus-Concrete: HbO topography over time", y=0.98)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compute_topomap_vlim(evokeds: list[mne.Evoked], chroma: str, times: np.ndarray) -> tuple[float, float]:
    all_values = []
    for evoked in evokeds:
        picked = evoked.copy().pick(picks=chroma)
        for time_value in np.atleast_1d(times):
            sample_idx = picked.time_as_index([float(time_value)])[0]
            all_values.append(np.abs(picked.data[:, sample_idx]))
    vmax = symmetric_vlim(np.concatenate(all_values), percentile=98.0, minimum=1e-7)
    return (-vmax, vmax)


def save_topomap_series(evoked: mne.Evoked, condition: str, chroma: str, vlim: tuple[float, float], path: Path) -> None:
    fig = evoked.copy().pick(picks=chroma).plot_topomap(
        times=TOPOMAP_SERIES_TIMES,
        vlim=vlim,
        colorbar=True,
        show=False,
        **TOPOMAP_ARGS,
    )
    fig.suptitle(f"{condition}: {chroma.upper()} topomaps from 4.0 s to 10.0 s", y=0.98)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_single_time_comparison(evoked_abstract: mne.Evoked, evoked_concrete: mne.Evoked, path: Path) -> None:
    evoked_diff = mne.combine_evoked([evoked_abstract, evoked_concrete], weights=[1, -1])
    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(10, 5),
        gridspec_kw=dict(width_ratios=[1, 1, 1, 0.08]),
        layout="constrained",
    )

    for row_idx, chroma in enumerate(CHROMOPHORES):
        vlim = compute_topomap_vlim([evoked_abstract, evoked_concrete, evoked_diff], chroma, np.array([COMPARISON_TIME]))
        evoked_abstract.plot_topomap(
            ch_type=chroma,
            times=COMPARISON_TIME,
            axes=axes[row_idx, 0],
            vlim=vlim,
            colorbar=False,
            show=False,
            **TOPOMAP_ARGS,
        )
        evoked_concrete.plot_topomap(
            ch_type=chroma,
            times=COMPARISON_TIME,
            axes=axes[row_idx, 1],
            vlim=vlim,
            colorbar=False,
            show=False,
            **TOPOMAP_ARGS,
        )
        evoked_diff.plot_topomap(
            ch_type=chroma,
            times=COMPARISON_TIME,
            axes=axes[row_idx, 2:],
            vlim=vlim,
            colorbar=True,
            show=False,
            **TOPOMAP_ARGS,
        )

    column_titles = ["Abstract", "Concrete", "Abstract-Concrete"]
    for column_idx, title in enumerate(column_titles):
        axes[0, column_idx].set_title(f"HbO: {title}")
        axes[1, column_idx].set_title(f"HbR: {title}")

    fig.suptitle(f"Single-time condition comparison at {COMPARISON_TIME:.1f} s", y=1.02)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_evoked_topo_overlay(evoked_abstract: mne.Evoked, evoked_concrete: mne.Evoked, path: Path) -> None:
    fig = mne.viz.plot_evoked_topo(
        [evoked_abstract.copy().pick(picks="hbo"), evoked_concrete.copy().pick(picks="hbo")],
        color=["#AA3377", "#228833"],
        legend=False,
        show=False,
    )
    if isinstance(fig, tuple):
        fig = fig[0]
    if isinstance(fig, list):
        fig = fig[0]
    handles = [
        Line2D([0], [0], color="#AA3377", linewidth=2, label="Abstract"),
        Line2D([0], [0], color="#228833", linewidth=2, label="Concrete"),
    ]
    fig.legend(handles=handles, loc="lower right", frameon=False)
    fig.suptitle("HbO waveforms by channel: Abstract versus Concrete", y=0.99)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(representative_session: tuple[str, str], metadata: dict[str, Any], primary_p_value: float) -> Path:
    overview_rows = [
        ("Representative geometry session", f"{representative_session[0]} / {representative_session[1]}"),
        ("Included participant-sessions from Step 6", metadata["n_included_sessions"]),
        ("Included questions in Step 6 trial table", metadata["n_included_questions"]),
        ("Abstract onset-locked epochs used", metadata["n_abstract_epochs"]),
        ("Concrete onset-locked epochs used", metadata["n_concrete_epochs"]),
        ("Epoch window", f"{EPOCH_TMIN:.1f} s to {EPOCH_TMAX:.1f} s around question onset"),
        ("Baseline", f"{BASELINE[0]:.1f} s to {BASELINE[1]:.1f} s"),
        ("Long-channel montage used in descriptive plots", f"{metadata['n_long_pairs']} pairs ({metadata['n_long_channels_total']} chromophore channels)"),
        ("Primary Step 6 adjusted p-value", f"{primary_p_value:.6f}"),
    ]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\begin{document}",
        r"\section*{Supplementary Step 6 Visualization Report}",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
        r"\paragraph{Adaptation note.} "
        r"This report reuses the same supplementary visualization family created for Step~5, but anchors it explicitly to the Step~6 trial-model table. "
        r"Because Step~6 freezes the Step~5 cohort and outcome definition, the onset-locked descriptive plots shown here are based on the same included trial set that feeds the covariate-adjusted mixed-effects models.",
        r"\paragraph{Rendering note.} "
        r"As in the earlier supplementary visualization reports, the sensor-geometry figure uses the participant digitization point cloud plus source, detector, channel, and pair geometry rather than an fsaverage brain-surface rendering, because the local \texttt{pyvista}/\texttt{fsaverage} stack is not available in this environment.",
        r"\subsection*{Step 6 Main Figures Reused}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/wordcount_by_condition.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/sentencecount_by_condition.png}",
        r"\caption{Step~6 item-level balance plots for total word count and sentence count.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/sentencelength_by_condition.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/enem_correctness_by_condition.png}",
        r"\caption{Step~6 item-level balance plots for sentence length and ENEM item correctness.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/front_roi_vs_wordcount_scatter.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/front_roi_partial_effect_condition.png}",
        r"\caption{Step~6 scatter and partial-effect figures from the primary covariate-adjusted model.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/model_residuals_primary.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6/model_random_intercepts.png}",
        r"\caption{Step~6 residual diagnostic and group-level intercept-proxy plots from the primary model.}",
        r"\end{figure}",
        r"\subsection*{Step 5-Style Supplementary Plots for Step 6}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step6_visualization/representative_sensor_geometry.png}",
        r"\caption{Representative optode, channel, and source-detector-pair geometry for one Step~6 included participant-session. Short pairs are shown with dashed orange lines.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6_visualization/abstract_trial_consistency.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6_visualization/concrete_trial_consistency.png}",
        r"\caption{Condition-adapted trial-consistency image plots for the Step~6 trial set.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step6_visualization/channel_consistency_grid.png}",
        r"\caption{Grand-average long-channel response images across channels for the Abstract and Concrete conditions.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.92\linewidth]{../../figures/step6_visualization/standard_fnirs_compare.png}",
        r"\caption{Standard fNIRS waveform comparison across long channels for the Step~6 trial set.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step6_visualization/abstract_hbo_joint.png}",
        r"\caption{Grand-average HbO topography over time for Abstract questions using the Step~6 included trial set.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step6_visualization/concrete_hbo_joint.png}",
        r"\caption{Grand-average HbO topography over time for Concrete questions using the Step~6 included trial set.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step6_visualization/abstract_minus_concrete_hbo_joint.png}",
        r"\caption{Grand-average Abstract-minus-Concrete HbO topography over time.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6_visualization/abstract_hbo_topomap_series.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6_visualization/concrete_hbo_topomap_series.png}",
        r"\caption{HbO topographic time series from 4.0~s to 10.0~s for the Abstract and Concrete conditions.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6_visualization/abstract_hbr_topomap_series.png}",
        r"\includegraphics[width=0.48\linewidth]{../../figures/step6_visualization/concrete_hbr_topomap_series.png}",
        r"\caption{HbR topographic time series from 4.0~s to 10.0~s for the Abstract and Concrete conditions.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step6_visualization/abstract_concrete_single_time_comparison.png}",
        r"\caption{Single-time comparison at 9.0~s for Abstract, Concrete, and Abstract-minus-Concrete topographies for both HbO and HbR.}",
        r"\end{figure}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step6_visualization/abstract_concrete_hbo_evoked_topo.png}",
        r"\caption{HbO waveforms by long channel, overlaid for Abstract and Concrete, to show what drives the descriptive topographic differences.}",
        r"\end{figure}",
        r"\end{document}",
    ]

    report_path = REPORTS_DIR / "step6_visualization_report.tex"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_directories()
    trial_df, primary_model_df, status_df = load_inputs()
    participant_id, session_id = get_representative_session(trial_df)
    representative_file = status_df.loc[
        (status_df["participant_id"] == participant_id) & (status_df["session_id"] == session_id),
        "preprocessed_file",
    ].iloc[0]
    representative_raw = mne.io.read_raw_fif(representative_file, preload=False, verbose="ERROR")
    plot_sensor_geometry(representative_raw, participant_id, session_id)

    dataset = build_visualization_dataset(trial_df, status_df)
    times = dataset["times"]
    condition_epochs = dataset["condition_epochs"]
    session_mean_traces = dataset["session_mean_traces"]
    grand_evokeds = dataset["grand_evokeds"]
    metadata = dataset["metadata"]

    plot_trial_consistency("Abstract", condition_epochs["Abstract"], times, FIGURES_DIR / "abstract_trial_consistency.png")
    plot_trial_consistency("Concrete", condition_epochs["Concrete"], times, FIGURES_DIR / "concrete_trial_consistency.png")
    plot_channel_consistency(grand_evokeds, times, FIGURES_DIR / "channel_consistency_grid.png")
    plot_standard_fnirs_response(session_mean_traces, times, FIGURES_DIR / "standard_fnirs_compare.png")

    save_joint_plot(grand_evokeds["Abstract"], "Abstract questions: HbO topography over time", FIGURES_DIR / "abstract_hbo_joint.png")
    save_joint_plot(grand_evokeds["Concrete"], "Concrete questions: HbO topography over time", FIGURES_DIR / "concrete_hbo_joint.png")
    save_difference_joint_plot(grand_evokeds["Abstract"], grand_evokeds["Concrete"], FIGURES_DIR / "abstract_minus_concrete_hbo_joint.png")

    hbo_series_vlim = compute_topomap_vlim([grand_evokeds["Abstract"], grand_evokeds["Concrete"]], "hbo", TOPOMAP_SERIES_TIMES)
    hbr_series_vlim = compute_topomap_vlim([grand_evokeds["Abstract"], grand_evokeds["Concrete"]], "hbr", TOPOMAP_SERIES_TIMES)
    save_topomap_series(grand_evokeds["Abstract"], "Abstract", "hbo", hbo_series_vlim, FIGURES_DIR / "abstract_hbo_topomap_series.png")
    save_topomap_series(grand_evokeds["Concrete"], "Concrete", "hbo", hbo_series_vlim, FIGURES_DIR / "concrete_hbo_topomap_series.png")
    save_topomap_series(grand_evokeds["Abstract"], "Abstract", "hbr", hbr_series_vlim, FIGURES_DIR / "abstract_hbr_topomap_series.png")
    save_topomap_series(grand_evokeds["Concrete"], "Concrete", "hbr", hbr_series_vlim, FIGURES_DIR / "concrete_hbr_topomap_series.png")

    save_single_time_comparison(
        grand_evokeds["Abstract"],
        grand_evokeds["Concrete"],
        FIGURES_DIR / "abstract_concrete_single_time_comparison.png",
    )
    save_evoked_topo_overlay(
        grand_evokeds["Abstract"],
        grand_evokeds["Concrete"],
        FIGURES_DIR / "abstract_concrete_hbo_evoked_topo.png",
    )

    primary_p_value = float(primary_model_df.iloc[0]["condition_p_value"])
    report_path = write_report((participant_id, session_id), metadata, primary_p_value)
    compile_report(report_path)

    print("Step 6 visualization report generated successfully.")
    print(f"Included participant-sessions: {metadata['n_included_sessions']}")
    print(f"Included questions: {metadata['n_included_questions']}")
    print(f"Abstract epochs: {metadata['n_abstract_epochs']}")
    print(f"Concrete epochs: {metadata['n_concrete_epochs']}")


if __name__ == "__main__":
    main()
