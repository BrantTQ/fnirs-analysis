#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
STEP5_SUMMARY_PATH = ROOT / "data_clean" / "step5" / "05_step5_participant_roi_summary.csv"
OUTPUT_DIR = ROOT / "data_clean" / "step5_leave_one_out"
ALPHA = 0.05

ANALYSIS_SPECS = [
    {
        "analysis_id": "front_hbo",
        "analysis_label": "Primary front ROI fixed-window HbO test",
        "chromophore": "hbo",
        "roi": "Front",
        "estimate_label": "abstract_minus_concrete",
        "abstract_col": "front_abstract_hbo_mean",
        "concrete_col": "front_concrete_hbo_mean",
        "difference_col": None,
    },
    {
        "analysis_id": "back_hbo",
        "analysis_label": "Secondary back ROI fixed-window HbO test",
        "chromophore": "hbo",
        "roi": "Back",
        "estimate_label": "abstract_minus_concrete",
        "abstract_col": "back_abstract_hbo_mean",
        "concrete_col": "back_concrete_hbo_mean",
        "difference_col": None,
    },
    {
        "analysis_id": "dissociation_hbo",
        "analysis_label": "Secondary front-versus-back fixed-window HbO dissociation test",
        "chromophore": "hbo",
        "roi": "FrontMinusBack",
        "estimate_label": "front_minus_back_condition_difference",
        "abstract_col": None,
        "concrete_col": None,
        "difference_col": "dissociation_hbo",
    },
    {
        "analysis_id": "front_hbr",
        "analysis_label": "Secondary front ROI fixed-window HbR test",
        "chromophore": "hbr",
        "roi": "Front",
        "estimate_label": "abstract_minus_concrete",
        "abstract_col": "front_abstract_hbr_mean",
        "concrete_col": "front_concrete_hbr_mean",
        "difference_col": None,
    },
    {
        "analysis_id": "back_hbr",
        "analysis_label": "Secondary back ROI fixed-window HbR test",
        "chromophore": "hbr",
        "roi": "Back",
        "estimate_label": "abstract_minus_concrete",
        "abstract_col": "back_abstract_hbr_mean",
        "concrete_col": "back_concrete_hbr_mean",
        "difference_col": None,
    },
    {
        "analysis_id": "dissociation_hbr",
        "analysis_label": "Secondary front-versus-back fixed-window HbR dissociation test",
        "chromophore": "hbr",
        "roi": "FrontMinusBack",
        "estimate_label": "front_minus_back_condition_difference",
        "abstract_col": None,
        "concrete_col": None,
        "difference_col": "dissociation_hbr",
    },
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def coerce_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def run_one_sample_test(values: pd.Series | np.ndarray, label: str) -> dict[str, Any]:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(clean)
    if n == 0:
        return {
            "analysis_label": label,
            "n": 0,
            "mean": math.nan,
            "sd": math.nan,
            "t_stat": math.nan,
            "df": math.nan,
            "p_value": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "cohens_d": math.nan,
        }
    mean_value = float(np.mean(clean))
    sd_value = float(np.std(clean, ddof=1)) if n > 1 else 0.0
    if n == 1:
        return {
            "analysis_label": label,
            "n": n,
            "mean": mean_value,
            "sd": sd_value,
            "t_stat": math.nan,
            "df": 0,
            "p_value": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "cohens_d": math.nan,
        }

    if math.isclose(sd_value, 0.0):
        t_stat = 0.0 if math.isclose(mean_value, 0.0) else math.copysign(math.inf, mean_value)
        p_value = 1.0 if math.isclose(mean_value, 0.0) else 0.0
        ci_low = mean_value
        ci_high = mean_value
        cohens_d = 0.0 if math.isclose(mean_value, 0.0) else math.copysign(math.inf, mean_value)
    else:
        test = stats.ttest_1samp(clean, popmean=0.0)
        t_stat = float(test.statistic)
        p_value = float(test.pvalue)
        sem = sd_value / math.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_low = mean_value - t_crit * sem
        ci_high = mean_value + t_crit * sem
        cohens_d = mean_value / sd_value

    return {
        "analysis_label": label,
        "n": n,
        "mean": mean_value,
        "sd": sd_value,
        "t_stat": t_stat,
        "df": n - 1,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohens_d": cohens_d,
    }


def load_summary() -> pd.DataFrame:
    summary_df = pd.read_csv(STEP5_SUMMARY_PATH)
    if "included_step5_analysis" in summary_df.columns:
        summary_df["included_step5_analysis"] = summary_df["included_step5_analysis"].map(coerce_bool)
        summary_df = summary_df[summary_df["included_step5_analysis"]].copy()
    summary_df = summary_df.sort_values(["participant_id", "session_id"]).reset_index(drop=True)
    return summary_df


def compute_difference(remaining_df: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    if spec["difference_col"]:
        return pd.to_numeric(remaining_df[spec["difference_col"]], errors="coerce")
    abstract_values = pd.to_numeric(remaining_df[spec["abstract_col"]], errors="coerce")
    concrete_values = pd.to_numeric(remaining_df[spec["concrete_col"]], errors="coerce")
    return abstract_values - concrete_values


def compute_analysis_result(remaining_df: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    differences = compute_difference(remaining_df, spec)
    result = run_one_sample_test(differences, spec["analysis_label"])
    row = {
        "analysis_id": spec["analysis_id"],
        "analysis_label": spec["analysis_label"],
        "chromophore": spec["chromophore"],
        "roi": spec["roi"],
        "estimate_label": spec["estimate_label"],
        "n_remaining_participants": int(result["n"]),
        "estimate_mean": result["mean"],
        "t_stat": result["t_stat"],
        "df": result["df"],
        "p_value": result["p_value"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "cohens_dz": result["cohens_d"],
        "significant_alpha_0_05": bool(pd.notna(result["p_value"]) and float(result["p_value"]) < ALPHA),
    }
    if spec["abstract_col"]:
        row["abstract_mean"] = float(pd.to_numeric(remaining_df[spec["abstract_col"]], errors="coerce").mean())
        row["concrete_mean"] = float(pd.to_numeric(remaining_df[spec["concrete_col"]], errors="coerce").mean())
    else:
        row["abstract_mean"] = math.nan
        row["concrete_mean"] = math.nan
    return row


def build_leave_one_out_tables(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for held_out in summary_df.itertuples(index=False):
        remaining_df = summary_df[
            ~(
                (summary_df["participant_id"] == held_out.participant_id)
                & (summary_df["session_id"] == held_out.session_id)
            )
        ].copy()

        wide_row = {
            "left_out_participant_id": held_out.participant_id,
            "left_out_session_id": held_out.session_id,
            "manual_step4_override": bool(held_out.manual_step4_override) if hasattr(held_out, "manual_step4_override") else False,
            "n_remaining_participants": int(len(remaining_df)),
            "n_remaining_abstract_trials": int(pd.to_numeric(remaining_df["n_valid_abstract_trials"], errors="coerce").sum()),
            "n_remaining_concrete_trials": int(pd.to_numeric(remaining_df["n_valid_concrete_trials"], errors="coerce").sum()),
        }

        for spec in ANALYSIS_SPECS:
            analysis_row = compute_analysis_result(remaining_df, spec)
            long_rows.append(
                {
                    "left_out_participant_id": held_out.participant_id,
                    "left_out_session_id": held_out.session_id,
                    **analysis_row,
                }
            )
            prefix = spec["analysis_id"]
            wide_row[f"{prefix}_estimate_mean"] = analysis_row["estimate_mean"]
            wide_row[f"{prefix}_t_stat"] = analysis_row["t_stat"]
            wide_row[f"{prefix}_df"] = analysis_row["df"]
            wide_row[f"{prefix}_p_value"] = analysis_row["p_value"]
            wide_row[f"{prefix}_ci_low"] = analysis_row["ci_low"]
            wide_row[f"{prefix}_ci_high"] = analysis_row["ci_high"]
            wide_row[f"{prefix}_cohens_dz"] = analysis_row["cohens_dz"]
            wide_row[f"{prefix}_significant_alpha_0_05"] = analysis_row["significant_alpha_0_05"]
            if pd.notna(analysis_row["abstract_mean"]):
                wide_row[f"{prefix}_abstract_mean"] = analysis_row["abstract_mean"]
                wide_row[f"{prefix}_concrete_mean"] = analysis_row["concrete_mean"]

        wide_rows.append(wide_row)

    wide_df = pd.DataFrame(wide_rows).sort_values("left_out_participant_id").reset_index(drop=True)
    long_df = pd.DataFrame(long_rows).sort_values(["left_out_participant_id", "analysis_id"]).reset_index(drop=True)
    return wide_df, long_df


def main() -> None:
    ensure_output_dir()
    summary_df = load_summary()
    wide_df, long_df = build_leave_one_out_tables(summary_df)

    wide_path = OUTPUT_DIR / "01_step5_leave_one_out_results_wide.csv"
    long_path = OUTPUT_DIR / "02_step5_leave_one_out_results_long.csv"
    wide_df.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)

    print("Step 5 leave-one-out side analysis completed.")
    print(f"Included Step 5 participant-sessions: {len(summary_df)}")
    print(f"Wide output: {wide_path}")
    print(f"Long output: {long_path}")


if __name__ == "__main__":
    main()
