#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import mne
import pandas as pd
from mne import Annotations
from mne.preprocessing.nirs import (
    beer_lambert_law,
    optical_density,
    scalp_coupling_index,
    short_channels,
    source_detector_distances,
    tddr,
)
from mne_nirs.signal_enhancement import short_channel_regression

from analysis_exclusions import (
    exclusion_table_rows,
    excluded_participant_ids,
    load_participant_exclusions,
    participant_exclusion_reason,
)


ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = ROOT / "sessions"

INTERMEDIATE_DIR = ROOT / "data_intermediate" / "step3"
CLEAN_DIR = ROOT / "data_clean" / "step3"
PREPROCESSED_DIR = CLEAN_DIR / "preprocessed"
REPORTS_DIR = ROOT / "reports" / "step3"
SESSION_NOTES_DIR = REPORTS_DIR / "session_notes"

SCI_THRESHOLD = 0.75
MIN_DURATION_SECONDS = 60.0
FILTER_L_FREQ = 0.01
FILTER_H_FREQ = 0.20
PPF = 6.0
SHORT_CHANNEL_MAX_DIST = 0.01

TRIGGER_MAP = {
    "EXP_START": 1,
    "EXP_END": 2,
    "PING": 3,
    "Q_TEXT_ON": 11,
    "Q_FULL_ON": 12,
    "BUTTON_CLICK": 13,
    "Q_STEM_ON": 14,
    "Q_OPTIONS_ON": 15,
    "ANS_A": 21,
    "ANS_B": 22,
    "ANS_C": 23,
    "ANS_D": 24,
    "ANS_E": 25,
    "BLK_ON": 91,
    "BLK_OFF": 92,
    "BLOCK_REST": 93,
    "ITI": 99,
    "QUESTIONNAIRE_ON": 71,
    "QUESTIONNAIRE_OFF": 72,
}

MIN_EVENT_TYPE_MAP = {
    "BLK_ON": "block_start",
    "Q_TEXT_ON": "question_start",
    "Q_FULL_ON": "question_start",
    "ANS_A": "answer",
    "ANS_B": "answer",
    "ANS_C": "answer",
    "ANS_D": "answer",
    "ANS_E": "answer",
    "BLK_OFF": "block_end",
}

RAW_MIN_CODE_MAP = {
    "91": "block_start",
    "11": "question_start",
    "12": "question_start",
    "21": "answer",
    "22": "answer",
    "23": "answer",
    "24": "answer",
    "25": "answer",
    "92": "block_end",
}

SOURCE_LABELS_LEFT = {"F3", "F7", "C3", "T7", "FC5", "CP5", "P7", "P3"}


@dataclass
class LogInfo:
    participant_id: str
    log_path: Path
    expected_event_count: int
    expected_min_count: int
    log_date: str | None


@dataclass
class SessionPaths:
    participant_id: str
    participant_dir_name: str
    session_folder_name: str
    session_id: str
    session_dir: Path
    snirf_file: Path | None
    snirf_backup_file: Path | None
    probe_info_file: Path | None
    config_json_file: Path | None
    description_json_file: Path | None
    calibration_json_file: Path | None
    digpts_file: Path | None
    config_hdr_file: Path | None


def ensure_directories() -> None:
    for path in [INTERMEDIATE_DIR, CLEAN_DIR, PREPROCESSED_DIR, REPORTS_DIR, SESSION_NOTES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def annotate_global_exclusions(df: pd.DataFrame, exclusions: dict[str, dict[str, str]]) -> pd.DataFrame:
    if "participant_id" not in df.columns:
        return df

    excluded_ids = excluded_participant_ids(exclusions)
    annotated_df = df.copy()
    participant_series = annotated_df["participant_id"].astype(str)
    annotated_df["global_analysis_excluded"] = participant_series.isin(excluded_ids)
    annotated_df["global_exclusion_reason"] = participant_series.map(lambda participant_id: participant_exclusion_reason(exclusions, participant_id))
    return annotated_df


def normalize_pid(raw: str | None) -> str | None:
    if not raw:
        return None
    match = re.search(r"pid\s*0*(\d+)", raw, flags=re.IGNORECASE)
    if not match:
        return None
    return f"PID{int(match.group(1)):03d}"


def sanitize_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def parse_date_token(text: str) -> str | None:
    match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", text)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def parse_log_date(file_name: str) -> str | None:
    match = re.search(r"(\d{8})_\d{6}", file_name)
    if not match:
        return None
    token = match.group(1)
    return f"{token[0:4]}-{token[4:6]}-{token[6:8]}"


def parse_timestamp(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return datetime.fromtimestamp(float(value)).isoformat(timespec="seconds")


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


def make_latex_table(rows: list[tuple[str, Any]], column_spec: str = r"p{0.34\linewidth}p{0.58\linewidth}") -> str:
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule", r"Metric & Value\\", r"\midrule"]
    for key, value in rows:
        lines.append(f"{sanitize_for_tex(key)} & {sanitize_for_tex(value)}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def make_latex_longtable(headers: list[str], rows: list[list[Any]], column_spec: str) -> str:
    header_line = " & ".join(sanitize_for_tex(item) for item in headers) + r"\\"
    lines = [rf"\begin{{longtable}}{{{column_spec}}}", r"\toprule", header_line, r"\midrule", r"\endfirsthead"]
    lines.extend([r"\toprule", header_line, r"\midrule", r"\endhead"])
    for row in rows:
        lines.append(" & ".join(sanitize_for_tex(item) for item in row) + r"\\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def build_question_id(row: pd.Series) -> str:
    year = str(row.get("question_year", "")).strip()
    field = str(row.get("question_field", "")).strip().upper()
    number = str(row.get("question_number", "")).strip()
    if not year or not field or not number or year == "nan" or field == "NAN" or number == "nan":
        return ""
    return f"{year}_{field}_{number}"


def list_session_paths() -> list[SessionPaths]:
    rows: list[SessionPaths] = []
    for session_dir in sorted(SESSIONS_DIR.glob("Sessions */PID*/**")):
        if not session_dir.is_dir():
            continue
        snirf_files = sorted(session_dir.glob("*.snirf"))
        if not snirf_files:
            continue

        participant_dir_name = session_dir.parent.name
        participant_id = normalize_pid(participant_dir_name)
        if not participant_id:
            continue

        snirf_file = snirf_files[0]
        snirf_backup_file = next(iter(sorted(session_dir.glob("*.snirf.bak_pre_marker_fix"))), None)
        probe_info_file = next(iter(sorted(session_dir.glob("*_probeInfo.json"))), None)
        config_json_file = next(iter(sorted(session_dir.glob("*_config.json"))), None)
        description_json_file = next(iter(sorted(session_dir.glob("*_description.json"))), None)
        calibration_json_file = next(iter(sorted(session_dir.glob("*_calibration.json"))), None)
        digpts_file = session_dir / "digpts.txt"
        config_hdr_file = next(iter(sorted(session_dir.glob("*_config.hdr"))), None)

        rows.append(
            SessionPaths(
                participant_id=participant_id,
                participant_dir_name=participant_dir_name,
                session_folder_name=session_dir.name,
                session_id=session_dir.name,
                session_dir=session_dir,
                snirf_file=snirf_file,
                snirf_backup_file=snirf_backup_file,
                probe_info_file=probe_info_file,
                config_json_file=config_json_file,
                description_json_file=description_json_file,
                calibration_json_file=calibration_json_file,
                digpts_file=digpts_file if digpts_file.exists() else None,
                config_hdr_file=config_hdr_file,
            )
        )
    return rows


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_snirf_summary(snirf_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "snirf_readable": False,
        "annotation_readable": False,
        "annotation_count": 0,
        "annotation_codes": "",
        "sfreq": None,
        "duration_sec": None,
        "n_raw_channels": None,
        "n_short_channels_raw": None,
        "n_long_channels_raw": None,
        "n_pairs": None,
        "n_short_pairs": None,
        "n_long_pairs": None,
        "mne_channel_types": "",
        "mne_accelerometer_channels": 0,
        "read_error": "",
    }
    try:
        raw = mne.io.read_raw_snirf(snirf_path, preload=False, verbose="ERROR")
        annotations = raw.annotations
        channel_types = raw.get_channel_types()
        short_mask = short_channels(raw.info)
        summary.update(
            {
                "snirf_readable": True,
                "annotation_readable": True,
                "annotation_count": int(len(annotations)),
                "annotation_codes": ";".join(sorted({str(desc) for desc in annotations.description})),
                "sfreq": float(raw.info["sfreq"]),
                "duration_sec": float(raw.n_times / raw.info["sfreq"]),
                "n_raw_channels": int(len(raw.ch_names)),
                "n_short_channels_raw": int(short_mask.sum()),
                "n_long_channels_raw": int(len(raw.ch_names) - short_mask.sum()),
                "n_pairs": int(len(raw.ch_names) / 2),
                "n_short_pairs": int(short_mask.sum() / 2),
                "n_long_pairs": int((len(raw.ch_names) - short_mask.sum()) / 2),
                "mne_channel_types": ";".join(sorted(set(channel_types))),
                "mne_accelerometer_channels": int(sum(ch_type in {"misc", "stim"} for ch_type in channel_types)),
            }
        )
    except Exception as exc:
        summary["read_error"] = str(exc)
    return summary


def read_h5_summary(snirf_path: Path) -> dict[str, Any]:
    summary = {
        "aux_group_count": 0,
        "aux_group_names": "",
        "stim_group_count": 0,
        "stim_names": "",
        "h5_read_error": "",
    }
    try:
        with h5py.File(snirf_path, "r") as handle:
            nirs = handle.get("nirs")
            if nirs is None:
                return summary
            aux_names = sorted([name for name in nirs.keys() if name.startswith("aux")])
            stim_names = sorted([name for name in nirs.keys() if name.startswith("stim")])
            stim_label_values: list[str] = []
            for stim_name in stim_names:
                try:
                    raw_value = nirs[stim_name]["name"][()]
                    if isinstance(raw_value, bytes):
                        stim_label_values.append(raw_value.decode("utf-8", errors="ignore"))
                    else:
                        stim_label_values.append(str(raw_value))
                except Exception:
                    stim_label_values.append(stim_name)
            summary.update(
                {
                    "aux_group_count": len(aux_names),
                    "aux_group_names": ";".join(aux_names),
                    "stim_group_count": len(stim_names),
                    "stim_names": ";".join(stim_label_values),
                }
            )
    except Exception as exc:
        summary["h5_read_error"] = str(exc)
    return summary


def build_log_catalog() -> dict[str, LogInfo]:
    catalog: dict[str, LogInfo] = {}
    for log_path in sorted(SESSIONS_DIR.glob("Sessions */PID*/enem_blocks*.csv")):
        participant_id = normalize_pid(log_path.parent.name)
        if not participant_id:
            continue
        df = pd.read_csv(log_path)
        expected = df[df["marker_name"].map(TRIGGER_MAP).notna()].copy()
        expected["harmonized_event_type"] = expected["marker_name"].map(MIN_EVENT_TYPE_MAP)
        catalog[participant_id] = LogInfo(
            participant_id=participant_id,
            log_path=log_path,
            expected_event_count=int(len(expected)),
            expected_min_count=int(expected["harmonized_event_type"].notna().sum()),
            log_date=parse_log_date(log_path.name),
        )
    return catalog


def choose_best_session_for_log(
    session_rows: list[dict[str, Any]], log_info: LogInfo
) -> dict[str, Any] | None:
    def score(row: dict[str, Any]) -> tuple[int, int, int, int, float]:
        readable = 1 if row["snirf_readable"] else 0
        plausible = 1 if (row.get("duration_sec") or 0) >= MIN_DURATION_SECONDS else 0
        date_match = 1 if row.get("session_date") == log_info.log_date else 0
        annotation_delta = abs((row.get("annotation_count") or 0) - log_info.expected_event_count)
        duration = float(row.get("duration_sec") or 0.0)
        return (readable, plausible, date_match, -annotation_delta, duration)

    ranked = sorted(session_rows, key=score, reverse=True)
    return ranked[0] if ranked else None


def pair_logs_to_sessions(
    session_rows: list[dict[str, Any]], log_catalog: dict[str, LogInfo]
) -> tuple[list[dict[str, Any]], list[str]]:
    sessions_by_pid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        sessions_by_pid[row["participant_id"]].append(row)

    orphan_logs: list[str] = []
    for participant_id, log_info in log_catalog.items():
        candidates = sessions_by_pid.get(participant_id, [])
        if not candidates:
            orphan_logs.append(participant_id)
            continue
        selected = choose_best_session_for_log(candidates, log_info)
        if selected is None:
            continue
        selected["psychopy_file"] = str(log_info.log_path.resolve())
        selected["psychopy_file_name"] = log_info.log_path.name
        selected["expected_event_count"] = log_info.expected_event_count
        selected["expected_min_count"] = log_info.expected_min_count
        selected["log_date"] = log_info.log_date
        for row in candidates:
            if row is selected:
                continue
            row.setdefault("notes", []).append("Additional session for participant; log paired to another session")
    return session_rows, orphan_logs


def build_expected_event_table(row: dict[str, Any]) -> pd.DataFrame:
    psychopy_file = row.get("psychopy_file")
    if not psychopy_file:
        return pd.DataFrame()

    df = pd.read_csv(psychopy_file)
    expected = df[df["marker_name"].map(TRIGGER_MAP).notna()].copy().reset_index(drop=True)
    expected["participant_id"] = row["participant_id"]
    expected["session_id"] = row["session_id"]
    expected["event_order"] = range(1, len(expected) + 1)
    expected["expected_code"] = expected["marker_name"].map(TRIGGER_MAP).astype(int)
    expected["event_type"] = expected["marker_name"]
    expected["harmonized_event_type"] = expected["marker_name"].map(MIN_EVENT_TYPE_MAP)
    expected["event_time_psychopy"] = pd.to_numeric(expected["t_abs"], errors="coerce")
    expected["question_id"] = expected.apply(build_question_id, axis=1)
    expected["source"] = "psychopy"
    expected["block"] = expected["block"].fillna("")
    expected["trial_idx_in_block"] = pd.to_numeric(expected["trial_idx_in_block"], errors="coerce")
    expected["question_number"] = expected["question_number"].fillna("")
    expected["question_year"] = expected["question_year"].fillna("")
    expected["question_field"] = expected["question_field"].fillna("")
    expected["question_type"] = expected["question_type"].fillna("")
    return expected[
        [
            "participant_id",
            "session_id",
            "event_order",
            "event_type",
            "expected_code",
            "harmonized_event_type",
            "event_time_psychopy",
            "block",
            "trial_idx_in_block",
            "question_id",
            "question_number",
            "question_year",
            "question_field",
            "question_type",
            "source",
        ]
    ].copy()


def build_raw_trigger_table(row: dict[str, Any]) -> pd.DataFrame:
    snirf_path = Path(row["snirf_file"])
    raw = mne.io.read_raw_snirf(snirf_path, preload=False, verbose="ERROR")
    annotations = raw.annotations
    records: list[dict[str, Any]] = []
    for index, (onset, duration, desc) in enumerate(
        zip(annotations.onset, annotations.duration, annotations.description), start=1
    ):
        code_int: int | None = None
        try:
            code_int = int(str(desc))
        except ValueError:
            code_int = None
        records.append(
            {
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "event_order": index,
                "event_time_fnirs": float(onset),
                "event_duration_fnirs": float(duration),
                "event_label_raw": str(desc),
                "event_code_raw": code_int,
                "source": "snirf",
            }
        )
    return pd.DataFrame(records)


def infer_raw_pattern(raw_df: pd.DataFrame) -> str:
    if raw_df.empty:
        return "NO_ANNOTATIONS"
    unique_codes = sorted(raw_df["event_label_raw"].astype(str).unique().tolist())
    if unique_codes == ["12"]:
        return "ALL_12_AMBIGUOUS"
    return "DIRECT_CODED"


def align_min_sequences(
    expected_min: pd.DataFrame, raw_min: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    expected_types = expected_min["harmonized_event_type"].tolist()
    raw_types = raw_min["harmonized_event_type"].tolist()
    n_expected = len(expected_types)
    n_raw = len(raw_types)

    large_cost = 10**6
    dp = [[large_cost] * (n_raw + 1) for _ in range(n_expected + 1)]
    back: list[list[tuple[str, int, int] | None]] = [[None] * (n_raw + 1) for _ in range(n_expected + 1)]
    dp[0][0] = 0

    for i in range(n_expected + 1):
        for j in range(n_raw + 1):
            current = dp[i][j]
            if current >= large_cost:
                continue

            if i < n_expected and j < n_raw:
                match_cost = 0 if expected_types[i] == raw_types[j] else 3
                candidate = current + match_cost
                if candidate < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = candidate
                    back[i + 1][j + 1] = ("match", i, j)

            if i < n_expected:
                candidate = current + 1
                if candidate < dp[i + 1][j]:
                    dp[i + 1][j] = candidate
                    back[i + 1][j] = ("missing_raw", i, j)

            if j < n_raw:
                candidate = current + 1
                if candidate < dp[i][j + 1]:
                    dp[i][j + 1] = candidate
                    back[i][j + 1] = ("extra_raw", i, j)

    i = n_expected
    j = n_raw
    matches: list[dict[str, Any]] = []
    missing_expected: list[dict[str, Any]] = []
    extra_raw: list[dict[str, Any]] = []
    substitutions = 0

    while i > 0 or j > 0:
        step = back[i][j]
        if step is None:
            break
        action, prev_i, prev_j = step
        if action == "match":
            expected_row = expected_min.iloc[prev_i].to_dict()
            raw_row = raw_min.iloc[prev_j].to_dict()
            if expected_types[prev_i] == raw_types[prev_j]:
                matches.append({"expected": expected_row, "raw": raw_row})
            else:
                substitutions += 1
                missing_expected.append(expected_row)
                extra_raw.append(raw_row)
        elif action == "missing_raw":
            missing_expected.append(expected_min.iloc[prev_i].to_dict())
        elif action == "extra_raw":
            extra_raw.append(raw_min.iloc[prev_j].to_dict())
        i = prev_i
        j = prev_j

    matches.reverse()
    missing_expected.reverse()
    extra_raw.reverse()
    return matches, missing_expected, extra_raw, substitutions


def best_interval_shift(expected_full: pd.DataFrame, raw_df: pd.DataFrame) -> tuple[int | None, float | None]:
    if expected_full.empty or raw_df.empty or len(raw_df) < 5:
        return None, None

    expected_times = expected_full["event_time_psychopy"].reset_index(drop=True)
    raw_times = raw_df["event_time_fnirs"].reset_index(drop=True)
    best_shift: int | None = None
    best_score: float | None = None

    for shift in range(0, 6):
        expected_slice = expected_times.iloc[shift:].reset_index(drop=True)
        n = min(len(expected_slice), len(raw_times))
        if n < 20:
            continue
        interval_score = (raw_times.iloc[:n].diff() - expected_slice.iloc[:n].diff()).abs().iloc[1:].median()
        if pd.isna(interval_score):
            continue
        score_value = float(interval_score)
        if best_score is None or score_value < best_score:
            best_score = score_value
            best_shift = shift
    return best_shift, best_score


def build_harmonized_for_direct_session(
    row: dict[str, Any],
    expected_full: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    expected_min = expected_full[expected_full["harmonized_event_type"].notna()].copy().reset_index(drop=True)
    raw_min = raw_df[raw_df["event_label_raw"].astype(str).isin(RAW_MIN_CODE_MAP)].copy().reset_index(drop=True)
    raw_min["harmonized_event_type"] = raw_min["event_label_raw"].map(RAW_MIN_CODE_MAP)

    matches, missing_expected, extra_raw, substitutions = align_min_sequences(expected_min, raw_min)

    offset_sec: float | None = None
    if matches:
        deltas = [
            float(match["raw"]["event_time_fnirs"]) - float(match["expected"]["event_time_psychopy"])
            for match in matches
        ]
        offset_sec = float(pd.Series(deltas).median())

    harmonized_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []

    match_map = {int(match["expected"]["event_order"]): match for match in matches}
    missing_orders = {int(item["event_order"]) for item in missing_expected}

    for _, expected_row in expected_min.iterrows():
        event_order = int(expected_row["event_order"])
        if event_order in match_map:
            raw_match = match_map[event_order]["raw"]
            harmonized_rows.append(
                {
                    "participant_id": row["participant_id"],
                    "session_id": row["session_id"],
                    "event_order_final": len(harmonized_rows) + 1,
                    "event_type": expected_row["harmonized_event_type"],
                    "event_time_final": float(raw_match["event_time_fnirs"]),
                    "event_source_final": "snirf_raw",
                    "raw_snirf_present": True,
                    "inserted_from_psychopy": False,
                    "repair_confidence": "high",
                    "block": expected_row["block"],
                    "trial_idx_in_block": expected_row["trial_idx_in_block"],
                    "question_id": expected_row["question_id"],
                    "event_code_raw": raw_match["event_code_raw"],
                    "event_label_raw": raw_match["event_label_raw"],
                    "notes": "",
                }
            )
        elif event_order in missing_orders and offset_sec is not None:
            inserted_time = float(expected_row["event_time_psychopy"]) + offset_sec
            harmonized_rows.append(
                {
                    "participant_id": row["participant_id"],
                    "session_id": row["session_id"],
                    "event_order_final": len(harmonized_rows) + 1,
                    "event_type": expected_row["harmonized_event_type"],
                    "event_time_final": inserted_time,
                    "event_source_final": "psychopy_inserted",
                    "raw_snirf_present": False,
                    "inserted_from_psychopy": True,
                    "repair_confidence": "medium",
                    "block": expected_row["block"],
                    "trial_idx_in_block": expected_row["trial_idx_in_block"],
                    "question_id": expected_row["question_id"],
                    "event_code_raw": None,
                    "event_label_raw": "",
                    "notes": "Inserted from PsychoPy using session offset",
                }
            )
            repair_rows.append(
                {
                    "participant_id": row["participant_id"],
                    "session_id": row["session_id"],
                    "repair_kind": "INSERT_MISSING_MIN_TRIGGER",
                    "event_type": expected_row["harmonized_event_type"],
                    "block": expected_row["block"],
                    "trial_idx_in_block": expected_row["trial_idx_in_block"],
                    "question_id": expected_row["question_id"],
                    "action": "Inserted from PsychoPy",
                    "repair_confidence": "medium",
                    "notes": "Missing raw minimum trigger; inserted at expected time plus offset",
                }
            )
        else:
            problem_rows.append(
                {
                    "participant_id": row["participant_id"],
                    "session_id": row["session_id"],
                    "problem_type": "UNRESOLVED_MIN_TRIGGER_MISMATCH",
                    "severity": "ERROR",
                    "notes": f"Could not resolve expected minimum trigger order={event_order}",
                }
            )

    for extra in extra_raw:
        repair_rows.append(
            {
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "repair_kind": "UNMATCHED_RAW_MIN_TRIGGER",
                "event_type": extra["harmonized_event_type"],
                "block": "",
                "trial_idx_in_block": "",
                "question_id": "",
                "action": "Retained in audit only",
                "repair_confidence": "low",
                "notes": f"Extra raw minimum trigger code={extra['event_label_raw']} at {extra['event_time_fnirs']:.6f}s",
            }
        )

    comparison_result = "MATCH_RAW"
    if len(expected_min) < 80:
        comparison_result = "MATCH_INCOMPLETE_SESSION"
    if substitutions > 0 or len(extra_raw) > 0:
        comparison_result = "MISMATCH_REVIEW_REQUIRED"
    elif len(missing_expected) > 0:
        comparison_result = "MISSING_TRIGGERS_REPAIRABLE"

    comparison = {
        "participant_id": row["participant_id"],
        "session_id": row["session_id"],
        "raw_pattern": infer_raw_pattern(raw_df),
        "alignment_method": "direct_min_event_alignment",
        "expected_full_count": int(len(expected_full)),
        "raw_annotation_count": int(len(raw_df)),
        "expected_min_count": int(len(expected_min)),
        "raw_min_count": int(len(raw_min)),
        "matched_min_count": int(len(matches)),
        "missing_min_count": int(len(missing_expected)),
        "extra_raw_min_count": int(len(extra_raw)),
        "substitution_count": int(substitutions),
        "interval_shift": None,
        "interval_score": None,
        "time_offset_sec": offset_sec,
        "comparison_result": comparison_result,
        "notes": "",
    }
    return pd.DataFrame(harmonized_rows), comparison, repair_rows, problem_rows


def build_harmonized_for_ambiguous_session(
    row: dict[str, Any],
    expected_full: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    expected_min = expected_full[expected_full["harmonized_event_type"].notna()].copy().reset_index(drop=True)
    shift, interval_score = best_interval_shift(expected_full, raw_df)

    repair_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []
    harmonized_rows: list[dict[str, Any]] = []

    if shift is None or interval_score is None or interval_score > 0.5:
        problem_rows.append(
            {
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "problem_type": "AMBIGUOUS_TRIGGER_ALIGNMENT_FAILED",
                "severity": "ERROR",
                "notes": "Could not confidently align raw code-12 annotations to PsychoPy timing",
            }
        )
        comparison = {
            "participant_id": row["participant_id"],
            "session_id": row["session_id"],
            "raw_pattern": infer_raw_pattern(raw_df),
            "alignment_method": "interval_shift_alignment",
            "expected_full_count": int(len(expected_full)),
            "raw_annotation_count": int(len(raw_df)),
            "expected_min_count": int(len(expected_min)),
            "raw_min_count": 0,
            "matched_min_count": 0,
            "missing_min_count": int(len(expected_min)),
            "extra_raw_min_count": 0,
            "substitution_count": 0,
            "interval_shift": shift,
            "interval_score": interval_score,
            "time_offset_sec": None,
            "comparison_result": "MISMATCH_REVIEW_REQUIRED",
            "notes": "Ambiguous raw annotations were not confidently alignable",
        }
        return pd.DataFrame(harmonized_rows), comparison, repair_rows, problem_rows

    expected_slice = expected_full.iloc[shift:].reset_index(drop=True)
    n_matched = min(len(expected_slice), len(raw_df))
    matched_expected = expected_slice.iloc[:n_matched].copy().reset_index(drop=True)
    matched_raw = raw_df.iloc[:n_matched].copy().reset_index(drop=True)
    offset_series = matched_raw["event_time_fnirs"] - matched_expected["event_time_psychopy"]
    offset_sec = float(offset_series.median())

    merged = pd.concat([matched_expected, matched_raw.add_prefix("raw_")], axis=1)
    merged["offset_delta"] = merged["raw_event_time_fnirs"] - merged["event_time_psychopy"]

    merged_min = merged[merged["harmonized_event_type"].notna()].copy().reset_index(drop=True)
    matched_min_orders = set(merged_min["event_order"].astype(int).tolist())

    for _, merged_row in merged_min.iterrows():
        harmonized_rows.append(
            {
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "event_order_final": len(harmonized_rows) + 1,
                "event_type": merged_row["harmonized_event_type"],
                "event_time_final": float(merged_row["raw_event_time_fnirs"]),
                "event_source_final": "snirf_aligned_psychopy",
                "raw_snirf_present": True,
                "inserted_from_psychopy": False,
                "repair_confidence": "high",
                "block": merged_row["block"],
                "trial_idx_in_block": merged_row["trial_idx_in_block"],
                "question_id": merged_row["question_id"],
                "event_code_raw": merged_row["raw_event_code_raw"],
                "event_label_raw": merged_row["raw_event_label_raw"],
                "notes": f"Aligned from ambiguous raw code 12 using expected-sequence shift={shift}",
            }
        )

    for _, expected_row in expected_min.iterrows():
        if int(expected_row["event_order"]) in matched_min_orders:
            continue
        inserted_time = float(expected_row["event_time_psychopy"]) + offset_sec
        harmonized_rows.append(
            {
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "event_order_final": len(harmonized_rows) + 1,
                "event_type": expected_row["harmonized_event_type"],
                "event_time_final": inserted_time,
                "event_source_final": "psychopy_inserted",
                "raw_snirf_present": False,
                "inserted_from_psychopy": True,
                "repair_confidence": "medium",
                "block": expected_row["block"],
                "trial_idx_in_block": expected_row["trial_idx_in_block"],
                "question_id": expected_row["question_id"],
                "event_code_raw": None,
                "event_label_raw": "",
                "notes": f"Inserted from PsychoPy using aligned offset; ambiguous raw sequence shift={shift}",
            }
        )
        repair_rows.append(
            {
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "repair_kind": "INSERT_MISSING_MIN_TRIGGER",
                "event_type": expected_row["harmonized_event_type"],
                "block": expected_row["block"],
                "trial_idx_in_block": expected_row["trial_idx_in_block"],
                "question_id": expected_row["question_id"],
                "action": "Inserted from PsychoPy",
                "repair_confidence": "medium",
                "notes": f"Inserted after ambiguous raw-sequence alignment shift={shift}",
            }
        )

    repair_rows.append(
        {
            "participant_id": row["participant_id"],
            "session_id": row["session_id"],
            "repair_kind": "RELABEL_AMBIGUOUS_RAW_SEQUENCE",
            "event_type": "multiple",
            "block": "",
            "trial_idx_in_block": "",
            "question_id": "",
            "action": "Assigned event labels from PsychoPy to raw code-12 timings",
            "repair_confidence": "high",
            "notes": f"Interval shift={shift}; median interval score={interval_score:.6f}",
        }
    )

    comparison_result = "MISSING_TRIGGERS_REPAIRABLE"
    if len(expected_min) < 80:
        comparison_result = "MATCH_INCOMPLETE_SESSION"
        comparison_result = "MISSING_TRIGGERS_REPAIRABLE"

    comparison = {
        "participant_id": row["participant_id"],
        "session_id": row["session_id"],
        "raw_pattern": infer_raw_pattern(raw_df),
        "alignment_method": "interval_shift_alignment",
        "expected_full_count": int(len(expected_full)),
        "raw_annotation_count": int(len(raw_df)),
        "expected_min_count": int(len(expected_min)),
        "raw_min_count": 0,
        "matched_min_count": int(len(merged_min)),
        "missing_min_count": int(len(expected_min) - len(merged_min)),
        "extra_raw_min_count": 0,
        "substitution_count": 0,
        "interval_shift": int(shift),
        "interval_score": float(interval_score),
        "time_offset_sec": offset_sec,
        "comparison_result": comparison_result,
        "notes": "Ambiguous raw code-12 annotations aligned to PsychoPy by interval structure",
    }
    return pd.DataFrame(harmonized_rows), comparison, repair_rows, problem_rows


def compare_and_harmonize_session(
    row: dict[str, Any], expected_full: pd.DataFrame, raw_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if expected_full.empty:
        comparison = {
            "participant_id": row["participant_id"],
            "session_id": row["session_id"],
            "raw_pattern": infer_raw_pattern(raw_df),
            "alignment_method": "none",
            "expected_full_count": 0,
            "raw_annotation_count": int(len(raw_df)),
            "expected_min_count": 0,
            "raw_min_count": 0,
            "matched_min_count": 0,
            "missing_min_count": 0,
            "extra_raw_min_count": 0,
            "substitution_count": 0,
            "interval_shift": None,
            "interval_score": None,
            "time_offset_sec": None,
            "comparison_result": "MISMATCH_REVIEW_REQUIRED",
            "notes": "No paired PsychoPy file available for trigger comparison",
        }
        problem = [
            {
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "problem_type": "MISSING_PSYCHOPY_FILE",
                "severity": "ERROR",
                "notes": "Session could not be paired to a PsychoPy log",
            }
        ]
        return pd.DataFrame(), comparison, [], problem

    if raw_df.empty:
        comparison = {
            "participant_id": row["participant_id"],
            "session_id": row["session_id"],
            "raw_pattern": "NO_ANNOTATIONS",
            "alignment_method": "none",
            "expected_full_count": int(len(expected_full)),
            "raw_annotation_count": 0,
            "expected_min_count": int(expected_full["harmonized_event_type"].notna().sum()),
            "raw_min_count": 0,
            "matched_min_count": 0,
            "missing_min_count": int(expected_full["harmonized_event_type"].notna().sum()),
            "extra_raw_min_count": 0,
            "substitution_count": 0,
            "interval_shift": None,
            "interval_score": None,
            "time_offset_sec": None,
            "comparison_result": "MISMATCH_REVIEW_REQUIRED",
            "notes": "No native annotations present in the .snirf file",
        }
        problem = [
            {
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "problem_type": "NO_RAW_ANNOTATIONS",
                "severity": "ERROR",
                "notes": "The .snirf file contains zero readable annotations",
            }
        ]
        return pd.DataFrame(), comparison, [], problem

    pattern = infer_raw_pattern(raw_df)
    if pattern == "ALL_12_AMBIGUOUS":
        return build_harmonized_for_ambiguous_session(row, expected_full, raw_df)
    return build_harmonized_for_direct_session(row, expected_full, raw_df)


def session_problem(
    participant_id: str, session_id: str, problem_type: str, severity: str, notes: str
) -> dict[str, Any]:
    return {
        "participant_id": participant_id,
        "session_id": session_id,
        "problem_type": problem_type,
        "severity": severity,
        "notes": notes,
    }


def compute_pair_metrics(od: mne.io.BaseRaw) -> pd.DataFrame:
    sci_values = scalp_coupling_index(od)
    short_mask = short_channels(od.info)
    records: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ch_name, sci_value, is_short in zip(od.ch_names, sci_values, short_mask):
        pair_id = ch_name.rsplit(" ", 1)[0]
        grouped[pair_id].append(
            {
                "raw_channel_name": ch_name,
                "pair_id": pair_id,
                "sci_value": float(sci_value),
                "is_short_channel": bool(is_short),
            }
        )

    for pair_id, items in grouped.items():
        sci_item_values = [item["sci_value"] for item in items]
        is_short_pair = all(item["is_short_channel"] for item in items)
        records.append(
            {
                "pair_id": pair_id,
                "n_wavelength_channels": len(items),
                "is_short_pair": is_short_pair,
                "pair_sci_min": float(min(sci_item_values)),
                "pair_sci_mean": float(sum(sci_item_values) / len(sci_item_values)),
                "raw_channel_names": ";".join(item["raw_channel_name"] for item in items),
            }
        )
    return pd.DataFrame(records).sort_values("pair_id").reset_index(drop=True)


def build_harmonized_annotations(harmonized_df: pd.DataFrame) -> Annotations:
    if harmonized_df.empty:
        return Annotations(onset=[], duration=[], description=[])
    ordered = harmonized_df.sort_values("event_time_final").reset_index(drop=True)
    return Annotations(
        onset=ordered["event_time_final"].astype(float).tolist(),
        duration=[0.0] * len(ordered),
        description=ordered["event_type"].astype(str).tolist(),
    )


def preprocess_session(
    row: dict[str, Any], harmonized_df: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    preproc_manifest: dict[str, Any] = {
        "participant_id": row["participant_id"],
        "session_id": row["session_id"],
        "preprocessed_file": "",
        "preprocessing_success": False,
        "n_bad_pairs": None,
        "n_bad_hb_channels": None,
        "short_channel_regression_applied": False,
        "notes": "",
    }
    channel_rows: list[dict[str, Any]] = []

    if not row["snirf_readable"]:
        problems.append(session_problem(row["participant_id"], row["session_id"], "UNREADABLE_SNIRF", "ERROR", row["read_error"]))
        return preproc_manifest, pd.DataFrame(channel_rows), problems

    if (row.get("duration_sec") or 0.0) < MIN_DURATION_SECONDS:
        problems.append(
            session_problem(
                row["participant_id"],
                row["session_id"],
                "IMPLAUSIBLY_SHORT_RECORDING",
                "ERROR",
                f"Duration {row.get('duration_sec')}s is below {MIN_DURATION_SECONDS}s threshold",
            )
        )
        return preproc_manifest, pd.DataFrame(channel_rows), problems

    if harmonized_df.empty:
        problems.append(
            session_problem(
                row["participant_id"],
                row["session_id"],
                "NO_HARMONIZED_TRIGGERS",
                "ERROR",
                "Preprocessing skipped because no harmonized trigger table was available",
            )
        )
        return preproc_manifest, pd.DataFrame(channel_rows), problems

    try:
        raw = mne.io.read_raw_snirf(row["snirf_file"], preload=True, verbose="ERROR")
        raw.set_annotations(build_harmonized_annotations(harmonized_df))

        od = optical_density(raw)
        pair_qc = compute_pair_metrics(od)
        pair_qc["participant_id"] = row["participant_id"]
        pair_qc["session_id"] = row["session_id"]
        pair_qc["bad_pair"] = (~pair_qc["is_short_pair"]) & (pair_qc["pair_sci_min"] < SCI_THRESHOLD)
        pair_qc["bad_reason"] = pair_qc["bad_pair"].map(lambda value: "LOW_SCI" if value else "")
        channel_rows = pair_qc[
            [
                "participant_id",
                "session_id",
                "pair_id",
                "raw_channel_names",
                "n_wavelength_channels",
                "is_short_pair",
                "pair_sci_min",
                "pair_sci_mean",
                "bad_pair",
                "bad_reason",
            ]
        ].to_dict(orient="records")

        bad_pairs = pair_qc.loc[pair_qc["bad_pair"], "pair_id"].tolist()
        bad_hb_channels: list[str] = []
        for pair_id in bad_pairs:
            bad_hb_channels.extend([f"{pair_id} hbo", f"{pair_id} hbr"])

        scr_applied = bool(pair_qc["is_short_pair"].any())
        if scr_applied:
            od = short_channel_regression(od, max_dist=SHORT_CHANNEL_MAX_DIST)

        od = tddr(od)
        hb = beer_lambert_law(od, ppf=PPF)
        hb.info["bads"] = [name for name in bad_hb_channels if name in hb.ch_names]
        hb = hb.copy().filter(FILTER_L_FREQ, FILTER_H_FREQ, verbose="ERROR")

        output_name = f"{row['participant_id']}_{sanitize_token(row['session_id'])}_preprocessed_raw.fif"
        output_path = PREPROCESSED_DIR / output_name
        hb.save(output_path, overwrite=True, verbose="ERROR")

        preproc_manifest.update(
            {
                "preprocessed_file": str(output_path.resolve()),
                "preprocessing_success": True,
                "n_bad_pairs": int(len(bad_pairs)),
                "n_bad_hb_channels": int(len([name for name in hb.info['bads'] if name in hb.ch_names])),
                "short_channel_regression_applied": scr_applied,
                "notes": "",
            }
        )

        if bad_pairs:
            problems.append(
                session_problem(
                    row["participant_id"],
                    row["session_id"],
                    "LOW_SCI_BAD_PAIRS",
                    "WARNING",
                    f"Marked {len(bad_pairs)} long-channel pairs as bad using SCI < {SCI_THRESHOLD}",
                )
            )
    except Exception as exc:
        problems.append(session_problem(row["participant_id"], row["session_id"], "PREPROCESSING_FAILURE", "ERROR", str(exc)))
    return preproc_manifest, pd.DataFrame(channel_rows), problems


def derive_session_status(
    row: dict[str, Any],
    comparison: dict[str, Any],
    preproc_manifest: dict[str, Any],
    session_problems: list[dict[str, Any]],
) -> dict[str, Any]:
    warning_problem_types = {item["problem_type"] for item in session_problems if item["severity"] == "WARNING"}

    has_error = any(item["severity"] == "ERROR" for item in session_problems)
    if has_error or not preproc_manifest["preprocessing_success"]:
        qc_label = "EXCLUDE"
        final_status = "EXCLUDE_FROM_STEP4"
    elif comparison["comparison_result"] == "MISMATCH_REVIEW_REQUIRED":
        qc_label = "REVIEW_REQUIRED"
        final_status = "REVIEW_BEFORE_STEP4"
    else:
        warning_reasons: list[str] = []
        if comparison["comparison_result"] in {"MATCH_INCOMPLETE_SESSION", "MISSING_TRIGGERS_REPAIRABLE"}:
            warning_reasons.append(comparison["comparison_result"])
        if row.get("backup_differs_from_current"):
            warning_reasons.append("CURRENT_FILE_DIFFERS_FROM_BACKUP")
        if warning_problem_types:
            warning_reasons.extend(sorted(warning_problem_types))
        if warning_reasons:
            qc_label = "OK_WITH_WARNINGS"
            final_status = "READY_FOR_STEP4_WITH_WARNINGS"
        else:
            qc_label = "OK"
            final_status = "READY_FOR_STEP4"

    return {
        "participant_id": row["participant_id"],
        "session_id": row["session_id"],
        "session_folder_path": row["session_folder_path"],
        "psychopy_file": row.get("psychopy_file", ""),
        "snirf_file": row["snirf_file"],
        "snirf_backup_file": row.get("snirf_backup_file", ""),
        "snirf_readable": row["snirf_readable"],
        "duration_sec": row.get("duration_sec"),
        "annotation_count": row.get("annotation_count"),
        "comparison_result": comparison["comparison_result"],
        "expected_min_count": comparison["expected_min_count"],
        "matched_min_count": comparison["matched_min_count"],
        "missing_min_count": comparison["missing_min_count"],
        "extra_raw_min_count": comparison["extra_raw_min_count"],
        "time_offset_sec": comparison["time_offset_sec"],
        "n_raw_channels": row.get("n_raw_channels"),
        "n_short_channels_raw": row.get("n_short_channels_raw"),
        "n_long_channels_raw": row.get("n_long_channels_raw"),
        "n_pairs": row.get("n_pairs"),
        "n_short_pairs": row.get("n_short_pairs"),
        "n_long_pairs": row.get("n_long_pairs"),
        "aux_group_count": row.get("aux_group_count"),
        "accelerometer_configured": row.get("accelerometer_configured"),
        "accelerometer_aux_present": row.get("aux_group_count", 0) > 0,
        "accelerometer_exposed_in_mne_raw": row.get("mne_accelerometer_channels", 0) > 0,
        "montage_left_lateralized": row.get("montage_left_lateralized"),
        "backup_differs_from_current": row.get("backup_differs_from_current"),
        "preprocessing_success": preproc_manifest["preprocessing_success"],
        "preprocessed_file": preproc_manifest["preprocessed_file"],
        "n_bad_pairs": preproc_manifest["n_bad_pairs"],
        "n_bad_hb_channels": preproc_manifest["n_bad_hb_channels"],
        "qc_label": qc_label,
        "final_status": final_status,
    }


def build_file_audit_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    file_specs = [
        ("primary_snirf", row.get("snirf_file")),
        ("backup_snirf", row.get("snirf_backup_file")),
        ("probe_info_json", row.get("probe_info_file")),
        ("config_json", row.get("config_json_file")),
        ("description_json", row.get("description_json_file")),
        ("calibration_json", row.get("calibration_json_file")),
        ("digpts_txt", row.get("digpts_file")),
        ("config_hdr", row.get("config_hdr_file")),
        ("psychopy_log", row.get("psychopy_file")),
    ]

    rows: list[dict[str, Any]] = []
    for role, raw_path in file_specs:
        path_text = str(raw_path) if raw_path else ""
        path_obj = Path(path_text) if path_text else None
        exists = bool(path_obj and path_obj.exists())
        rows.append(
            {
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "file_role": role,
                "file_path": str(path_obj.resolve()) if exists and path_obj else path_text,
                "exists": exists,
                "readable": exists,
                "size_bytes": int(path_obj.stat().st_size) if exists and path_obj else None,
                "mtime_iso": parse_timestamp(path_obj.stat().st_mtime) if exists and path_obj else "",
                "notes": "",
            }
        )
    return rows


def write_session_note(
    status_row: dict[str, Any],
    comparison_row: dict[str, Any],
    session_problem_rows: list[dict[str, Any]],
) -> None:
    note_path = SESSION_NOTES_DIR / f"{sanitize_token(status_row['participant_id'])}_{sanitize_token(status_row['session_id'])}.tex"
    rows = [
        ("Participant", status_row["participant_id"]),
        ("Session", status_row["session_id"]),
        ("Comparison result", comparison_row["comparison_result"]),
        ("QC label", status_row["qc_label"]),
        ("Final status", status_row["final_status"]),
        ("Duration (s)", f"{status_row['duration_sec']:.3f}" if status_row["duration_sec"] else ""),
        ("Annotations in raw .snirf", status_row["annotation_count"]),
        ("Matched minimum triggers", f"{comparison_row['matched_min_count']} / {comparison_row['expected_min_count']}"),
        ("Missing minimum triggers", comparison_row["missing_min_count"]),
        ("Extra raw minimum triggers", comparison_row["extra_raw_min_count"]),
        ("Estimated time offset (s)", "" if comparison_row["time_offset_sec"] is None else f"{comparison_row['time_offset_sec']:.6f}"),
        ("Short raw channels", status_row["n_short_channels_raw"]),
        ("Bad channel pairs", status_row["n_bad_pairs"]),
        ("Preprocessing success", status_row["preprocessing_success"]),
    ]
    lines = [rf"\subsection*{{{sanitize_for_tex(status_row['participant_id'])} / {sanitize_for_tex(status_row['session_id'])}}}", make_latex_table(rows)]
    if session_problem_rows:
        lines.append(r"\paragraph{Logged issues.}")
        lines.append(r"\begin{itemize}")
        for item in session_problem_rows:
            lines.append(
                rf"\item [{sanitize_for_tex(item['severity'])}] {sanitize_for_tex(item['problem_type'])}: {sanitize_for_tex(item['notes'])}"
            )
        lines.append(r"\end{itemize}")
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_master_report(
    manifest_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    status_df: pd.DataFrame,
    orphan_logs: list[str],
    exclusions: dict[str, dict[str, str]],
) -> Path:
    report_path = REPORTS_DIR / "step3_fnirs_preprocessing_report.tex"

    session_note_files = sorted(SESSION_NOTES_DIR.glob("*.tex"))
    status_counts = status_df["final_status"].value_counts().to_dict()
    comparison_counts = comparison_df["comparison_result"].value_counts().to_dict()

    overview_rows = [
        ("Session folders in manifest", len(manifest_df)),
        ("Sessions with readable .snirf", int(manifest_df["snirf_readable"].sum())),
        ("Sessions preprocessed successfully", int(status_df["preprocessing_success"].sum())),
        ("Ready for Step 4", status_counts.get("READY_FOR_STEP4", 0)),
        ("Ready for Step 4 with warnings", status_counts.get("READY_FOR_STEP4_WITH_WARNINGS", 0)),
        ("Review before Step 4", status_counts.get("REVIEW_BEFORE_STEP4", 0)),
        ("Exclude from Step 4", status_counts.get("EXCLUDE_FROM_STEP4", 0)),
        ("Globally excluded participants", len(exclusions)),
    ]

    comparison_rows = [
        ("MATCH_RAW", comparison_counts.get("MATCH_RAW", 0)),
        ("MATCH_INCOMPLETE_SESSION", comparison_counts.get("MATCH_INCOMPLETE_SESSION", 0)),
        ("MISSING_TRIGGERS_REPAIRABLE", comparison_counts.get("MISSING_TRIGGERS_REPAIRABLE", 0)),
        ("MISMATCH_REVIEW_REQUIRED", comparison_counts.get("MISMATCH_REVIEW_REQUIRED", 0)),
    ]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\begin{document}",
        r"\section*{Step 3: fNIRS Trigger Harmonization, QC, and Basic Preprocessing}",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
        r"\subsection*{Trigger Comparison Summary}",
        make_latex_table(comparison_rows),
    ]

    if orphan_logs:
        orphan_text = ", ".join(orphan_logs)
        lines.extend(
            [
                r"\subsection*{Behavioral-Only Logs}",
                rf"The following participant IDs had PsychoPy logs but no fNIRS session folder in the Step 3 manifest: {sanitize_for_tex(orphan_text)}.",
            ]
        )

    if exclusions:
        lines.extend(
            [
                r"\subsection*{Global Downstream Participant Exclusions}",
                r"The following participant IDs are excluded from downstream participant-level summaries and future steps. Step~3 QC and preprocessing outputs are retained as an audit record, but future analysis datasets should continue to honor this exclusion list.",
                make_latex_table(exclusion_table_rows(exclusions)),
            ]
        )

    lines.append(r"\subsection*{Session Notes}")
    for note_file in session_note_files:
        lines.append(rf"\input{{session_notes/{note_file.name}}}")
    lines.append(r"\end{document}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_session_issue_summary(
    status_row: pd.Series, comparison_row: pd.Series | None, session_problem_df: pd.DataFrame
) -> list[str]:
    reasons: list[str] = []

    if comparison_row is not None:
        comparison_result = comparison_row["comparison_result"]
        if comparison_result == "MISSING_TRIGGERS_REPAIRABLE":
            reasons.append("Raw trigger annotations were ambiguous or incomplete and were repaired from PsychoPy alignment.")
        elif comparison_result == "MATCH_INCOMPLETE_SESSION":
            reasons.append("Session was incomplete but trigger sequence matched the completed portion of the task.")
        elif comparison_result == "MISMATCH_REVIEW_REQUIRED":
            extra_raw = int(comparison_row.get("extra_raw_min_count", 0) or 0)
            missing_raw = int(comparison_row.get("missing_min_count", 0) or 0)
            if extra_raw > 0:
                reasons.append(f"Raw minimum trigger sequence contains {extra_raw} extra task trigger(s) relative to PsychoPy.")
            if missing_raw > 0:
                reasons.append(f"{missing_raw} expected minimum trigger(s) were not confirmed in the raw fNIRS sequence.")
            if extra_raw == 0 and missing_raw == 0:
                reasons.append("Trigger sequence requires manual review before Step 4.")

    if bool(status_row.get("backup_differs_from_current", False)):
        reasons.append("Current .snirf file differs from the backup pre-marker-fix file.")

    bad_pairs_value = status_row.get("n_bad_pairs")
    if pd.notna(bad_pairs_value) and int(bad_pairs_value) > 0:
        reasons.append(f"{int(bad_pairs_value)} long-channel pair(s) were marked bad from low scalp-coupling index.")

    for _, problem_row in session_problem_df.sort_values(["severity", "problem_type"]).iterrows():
        problem_type = str(problem_row["problem_type"])
        notes = str(problem_row["notes"])
        if problem_type == "LOW_SCI_BAD_PAIRS":
            continue
        reasons.append(notes if notes and notes != "nan" else problem_type)

    if not reasons:
        reasons.append("No explicit problem rows were logged for this session.")
    return reasons


def write_problem_report(
    status_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    problem_df: pd.DataFrame,
    orphan_logs: list[str],
    exclusions: dict[str, dict[str, str]],
) -> Path:
    report_path = REPORTS_DIR / "step3_fnirs_problems_report.tex"

    comparison_lookup = {
        (row["participant_id"], row["session_id"]): row for _, row in comparison_df.iterrows()
    }

    excluded_df = status_df[status_df["final_status"] == "EXCLUDE_FROM_STEP4"].copy()
    review_df = status_df[status_df["final_status"] == "REVIEW_BEFORE_STEP4"].copy()
    warning_df = status_df[status_df["final_status"] == "READY_FOR_STEP4_WITH_WARNINGS"].copy()
    any_issue_df = status_df[status_df["final_status"] != "READY_FOR_STEP4"].copy()

    overview_rows = [
        ("Sessions with any issue or warning", len(any_issue_df)),
        ("Ready for Step 4 with warnings", len(warning_df)),
        ("Review before Step 4", len(review_df)),
        ("Excluded from Step 4", len(excluded_df)),
        ("Behavioral-only logs without fNIRS session folder", len(orphan_logs)),
        ("Globally excluded participants", len(exclusions)),
    ]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\begin{document}",
        r"\section*{Step 3 Problem and Exclusion Report}",
        r"\subsection*{Overview}",
        make_latex_table(overview_rows),
    ]

    if exclusions:
        lines.extend(
            [
                r"\subsection*{Global Downstream Participant Exclusions}",
                r"The following participant IDs are excluded from downstream participant-level summaries and future steps. Step~3 raw QC outputs are retained for traceability, but future analysis datasets should continue to honor this exclusion list.",
                make_latex_table(exclusion_table_rows(exclusions)),
            ]
        )

    if orphan_logs:
        lines.extend(
            [
                r"\subsection*{Behavioral-Only Participants}",
                rf"The following participant IDs had PsychoPy logs but no fNIRS session folder and therefore do not appear in the Step 3 session-status table: {sanitize_for_tex(', '.join(orphan_logs))}.",
            ]
        )

    if not excluded_df.empty:
        lines.append(r"\subsection*{Excluded Sessions}")
        for _, status_row in excluded_df.iterrows():
            key = (status_row["participant_id"], status_row["session_id"])
            comparison_row = comparison_lookup.get(key)
            session_problem_df = problem_df[
                (problem_df["participant_id"] == status_row["participant_id"])
                & (problem_df["session_id"] == status_row["session_id"])
            ].copy()
            reasons = build_session_issue_summary(status_row, comparison_row, session_problem_df)
            lines.append(
                rf"\paragraph{{{sanitize_for_tex(status_row['participant_id'])} / {sanitize_for_tex(status_row['session_id'])}}}"
            )
            lines.append(r"\begin{itemize}")
            for reason in reasons:
                lines.append(rf"\item {sanitize_for_tex(reason)}")
            lines.append(r"\end{itemize}")

    if not review_df.empty:
        lines.append(r"\subsection*{Sessions Requiring Review Before Step 4}")
        for _, status_row in review_df.iterrows():
            key = (status_row["participant_id"], status_row["session_id"])
            comparison_row = comparison_lookup.get(key)
            session_problem_df = problem_df[
                (problem_df["participant_id"] == status_row["participant_id"])
                & (problem_df["session_id"] == status_row["session_id"])
            ].copy()
            reasons = build_session_issue_summary(status_row, comparison_row, session_problem_df)
            lines.append(
                rf"\paragraph{{{sanitize_for_tex(status_row['participant_id'])} / {sanitize_for_tex(status_row['session_id'])}}}"
            )
            lines.append(r"\begin{itemize}")
            for reason in reasons:
                lines.append(rf"\item {sanitize_for_tex(reason)}")
            lines.append(r"\end{itemize}")

    if not warning_df.empty:
        lines.append(r"\subsection*{Sessions Ready With Warnings}")
        warning_rows: list[list[Any]] = []
        for _, status_row in warning_df.sort_values(["participant_id", "session_id"]).iterrows():
            key = (status_row["participant_id"], status_row["session_id"])
            comparison_row = comparison_lookup.get(key)
            session_problem_df = problem_df[
                (problem_df["participant_id"] == status_row["participant_id"])
                & (problem_df["session_id"] == status_row["session_id"])
            ].copy()
            reasons = build_session_issue_summary(status_row, comparison_row, session_problem_df)
            warning_rows.append(
                [
                    status_row["participant_id"],
                    status_row["session_id"],
                    status_row["comparison_result"],
                    "; ".join(reasons),
                ]
            )
        lines.append(
            make_latex_longtable(
                ["Participant", "Session", "Status driver", "Reason summary"],
                warning_rows,
                r"p{0.10\linewidth}p{0.20\linewidth}p{0.20\linewidth}p{0.40\linewidth}",
            )
        )

    if not problem_df.empty:
        lines.append(r"\subsection*{Raw Problem Log}")
        problem_rows = []
        for _, row in problem_df.sort_values(["severity", "participant_id", "session_id", "problem_type"]).iterrows():
            problem_rows.append(
                [row["participant_id"], row["session_id"], row["severity"], row["problem_type"], row["notes"]]
            )
        lines.append(
            make_latex_longtable(
                ["Participant", "Session", "Severity", "Problem type", "Notes"],
                problem_rows,
                r"p{0.10\linewidth}p{0.18\linewidth}p{0.10\linewidth}p{0.22\linewidth}p{0.28\linewidth}",
            )
        )

    lines.append(r"\end{document}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


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


def main() -> None:
    ensure_directories()
    exclusions = load_participant_exclusions()

    session_paths = list_session_paths()
    log_catalog = build_log_catalog()

    session_rows: list[dict[str, Any]] = []
    file_audit_rows: list[dict[str, Any]] = []

    for session_path in session_paths:
        notes: list[str] = []
        summary = read_snirf_summary(session_path.snirf_file)
        h5_summary = read_h5_summary(session_path.snirf_file)
        description_json = load_json(session_path.description_json_file)
        config_json = load_json(session_path.config_json_file)
        probe_info_json = load_json(session_path.probe_info_file)

        subject_id = normalize_pid(str(description_json.get("subject", "")))
        identity_match = subject_id in {None, session_path.participant_id}
        if not identity_match:
            notes.append(f"Description subject {description_json.get('subject')} does not match folder PID")

        folder_date = parse_date_token(session_path.session_folder_name)
        snirf_date = parse_date_token(session_path.snirf_file.stem if session_path.snirf_file else "")
        session_date = folder_date or snirf_date
        if folder_date and snirf_date and folder_date != snirf_date:
            notes.append("Session folder date and snirf stem date differ")

        backup_annotation_count = None
        backup_differs_from_current = False
        if session_path.snirf_backup_file and session_path.snirf_backup_file.exists():
            backup_summary = read_snirf_summary(session_path.snirf_backup_file)
            backup_annotation_count = backup_summary.get("annotation_count")
            backup_differs_from_current = backup_summary.get("annotation_count") != summary.get("annotation_count")
            if backup_differs_from_current:
                notes.append("Current .snirf differs from backup marker file")

        probe_info_block = probe_info_json.get("probeInfo", {}).get("probes", {})
        source_labels = set(probe_info_block.get("labels_s", []))
        montage_left_lateralized = bool(source_labels == SOURCE_LABELS_LEFT) if source_labels else None

        row = {
            "participant_id": session_path.participant_id,
            "participant_dir_name": session_path.participant_dir_name,
            "session_id": session_path.session_id,
            "session_folder_name": session_path.session_folder_name,
            "session_folder_path": str(session_path.session_dir.resolve()),
            "session_date": session_date,
            "session_folder_readable": session_path.session_dir.exists(),
            "snirf_file": str(session_path.snirf_file.resolve()) if session_path.snirf_file else "",
            "snirf_backup_file": str(session_path.snirf_backup_file.resolve()) if session_path.snirf_backup_file else "",
            "probe_info_file": str(session_path.probe_info_file.resolve()) if session_path.probe_info_file else "",
            "config_json_file": str(session_path.config_json_file.resolve()) if session_path.config_json_file else "",
            "description_json_file": str(session_path.description_json_file.resolve()) if session_path.description_json_file else "",
            "calibration_json_file": str(session_path.calibration_json_file.resolve()) if session_path.calibration_json_file else "",
            "digpts_file": str(session_path.digpts_file.resolve()) if session_path.digpts_file else "",
            "config_hdr_file": str(session_path.config_hdr_file.resolve()) if session_path.config_hdr_file else "",
            "primary_raw_analysis_file": str(session_path.snirf_file.resolve()) if session_path.snirf_file else "",
            "audit_backup_file": str(session_path.snirf_backup_file.resolve()) if session_path.snirf_backup_file else "",
            "auxiliary_metadata_files": ";".join(
                [
                    item
                    for item in [
                        str(session_path.probe_info_file.resolve()) if session_path.probe_info_file else "",
                        str(session_path.config_json_file.resolve()) if session_path.config_json_file else "",
                        str(session_path.description_json_file.resolve()) if session_path.description_json_file else "",
                        str(session_path.calibration_json_file.resolve()) if session_path.calibration_json_file else "",
                        str(session_path.digpts_file.resolve()) if session_path.digpts_file else "",
                        str(session_path.config_hdr_file.resolve()) if session_path.config_hdr_file else "",
                    ]
                    if item
                ]
            ),
            "has_previous_marker_fix_backup": bool(session_path.snirf_backup_file and session_path.snirf_backup_file.exists()),
            "file_timestamp_iso": parse_timestamp(session_path.snirf_file.stat().st_mtime if session_path.snirf_file else None),
            "source_subject_id": description_json.get("subject", ""),
            "session_identity_ok": identity_match,
            "use_accelerometer_config": config_json.get("use_accelerometer"),
            "accelerometer_configured": bool(config_json.get("use_accelerometer", False)),
            "montage_left_lateralized": montage_left_lateralized,
            "backup_annotation_count": backup_annotation_count,
            "backup_differs_from_current": backup_differs_from_current,
            "psychopy_file": "",
            "psychopy_file_name": "",
            "expected_event_count": None,
            "expected_min_count": None,
            "notes": notes,
        }
        row.update(summary)
        row.update(h5_summary)
        session_rows.append(row)
        file_audit_rows.extend(build_file_audit_rows(row))

    session_rows, orphan_logs = pair_logs_to_sessions(session_rows, log_catalog)

    manifest_records: list[dict[str, Any]] = []
    raw_trigger_frames: list[pd.DataFrame] = []
    expected_frames: list[pd.DataFrame] = []
    harmonized_frames: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []
    channel_qc_frames: list[pd.DataFrame] = []
    preproc_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for row in session_rows:
        manifest_records.append(
            {
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "session_folder_path": row["session_folder_path"],
                "session_folder_readable": row["session_folder_readable"],
                "snirf_file": row["snirf_file"],
                "snirf_backup_file": row["snirf_backup_file"],
                "psychopy_file": row["psychopy_file"],
                "file_timestamp_iso": row["file_timestamp_iso"],
                "primary_raw_analysis_file": row["primary_raw_analysis_file"],
                "audit_backup_file": row["audit_backup_file"],
                "auxiliary_metadata_files": row["auxiliary_metadata_files"],
                "has_previous_marker_fix_backup": row["has_previous_marker_fix_backup"],
                "session_identity_ok": row["session_identity_ok"],
                "session_date": row["session_date"],
                "snirf_readable": row["snirf_readable"],
                "annotation_count": row["annotation_count"],
                "duration_sec": row["duration_sec"],
                "notes": " | ".join(row["notes"]),
            }
        )

        expected_full = build_expected_event_table(row)
        if not expected_full.empty:
            expected_frames.append(expected_full)

        raw_df = pd.DataFrame()
        if row["snirf_readable"]:
            raw_df = build_raw_trigger_table(row)
            raw_trigger_frames.append(raw_df)
        elif row["read_error"]:
            problem_rows.append(session_problem(row["participant_id"], row["session_id"], "UNREADABLE_SNIRF", "ERROR", row["read_error"]))

        harmonized_df, comparison_row, session_repair_rows, session_problem_rows = compare_and_harmonize_session(
            row, expected_full, raw_df
        )

        comparison_rows.append(comparison_row)
        repair_rows.extend(session_repair_rows)
        problem_rows.extend(session_problem_rows)
        if not harmonized_df.empty:
            harmonized_frames.append(harmonized_df)

        preproc_manifest, channel_qc_df, preproc_problem_rows = preprocess_session(row, harmonized_df)
        preproc_rows.append(preproc_manifest)
        problem_rows.extend(preproc_problem_rows)
        if not channel_qc_df.empty:
            channel_qc_frames.append(channel_qc_df)

        status_row = derive_session_status(row, comparison_row, preproc_manifest, session_problem_rows + preproc_problem_rows)
        status_rows.append(status_row)
        write_session_note(status_row, comparison_row, session_problem_rows + preproc_problem_rows)

    manifest_df = pd.DataFrame(manifest_records).sort_values(["participant_id", "session_id"]).reset_index(drop=True)
    raw_trigger_df = (
        pd.concat(raw_trigger_frames, ignore_index=True).sort_values(["participant_id", "session_id", "event_order"]).reset_index(drop=True)
        if raw_trigger_frames
        else pd.DataFrame()
    )
    expected_df = (
        pd.concat(expected_frames, ignore_index=True).sort_values(["participant_id", "session_id", "event_order"]).reset_index(drop=True)
        if expected_frames
        else pd.DataFrame()
    )
    comparison_df = pd.DataFrame(comparison_rows).sort_values(["participant_id", "session_id"]).reset_index(drop=True)
    repair_df = pd.DataFrame(repair_rows).sort_values(["participant_id", "session_id"]).reset_index(drop=True)
    harmonized_df = (
        pd.concat(harmonized_frames, ignore_index=True)
        .sort_values(["participant_id", "session_id", "event_order_final"])
        .reset_index(drop=True)
        if harmonized_frames
        else pd.DataFrame()
    )
    status_df = pd.DataFrame(status_rows).sort_values(["participant_id", "session_id"]).reset_index(drop=True)
    channel_qc_df = (
        pd.concat(channel_qc_frames, ignore_index=True).sort_values(["participant_id", "session_id", "pair_id"]).reset_index(drop=True)
        if channel_qc_frames
        else pd.DataFrame()
    )
    problem_df = pd.DataFrame(problem_rows).sort_values(["participant_id", "session_id", "severity", "problem_type"]).reset_index(drop=True)
    file_audit_df = pd.DataFrame(file_audit_rows).sort_values(["participant_id", "session_id", "file_role"]).reset_index(drop=True)
    preproc_manifest_df = pd.DataFrame(preproc_rows).sort_values(["participant_id", "session_id"]).reset_index(drop=True)

    manifest_df = annotate_global_exclusions(manifest_df, exclusions)
    raw_trigger_df = annotate_global_exclusions(raw_trigger_df, exclusions)
    expected_df = annotate_global_exclusions(expected_df, exclusions)
    comparison_df = annotate_global_exclusions(comparison_df, exclusions)
    repair_df = annotate_global_exclusions(repair_df, exclusions)
    harmonized_df = annotate_global_exclusions(harmonized_df, exclusions)
    status_df = annotate_global_exclusions(status_df, exclusions)
    channel_qc_df = annotate_global_exclusions(channel_qc_df, exclusions)
    problem_df = annotate_global_exclusions(problem_df, exclusions)
    file_audit_df = annotate_global_exclusions(file_audit_df, exclusions)
    preproc_manifest_df = annotate_global_exclusions(preproc_manifest_df, exclusions)

    manifest_df.to_csv(INTERMEDIATE_DIR / "01_fnirs_session_manifest.csv", index=False)
    raw_trigger_df.to_csv(INTERMEDIATE_DIR / "02_fnirs_trigger_raw.csv", index=False)
    expected_df.to_csv(INTERMEDIATE_DIR / "03_psychopy_trigger_expected.csv", index=False)
    comparison_df.to_csv(INTERMEDIATE_DIR / "04_trigger_comparison_by_session.csv", index=False)
    repair_df.to_csv(INTERMEDIATE_DIR / "05_trigger_repair_log.csv", index=False)
    harmonized_df.to_csv(CLEAN_DIR / "06_fnirs_trigger_harmonized.csv", index=False)
    status_df.to_csv(CLEAN_DIR / "07_fnirs_session_status.csv", index=False)
    channel_qc_df.to_csv(CLEAN_DIR / "08_fnirs_channel_qc.csv", index=False)
    problem_df.to_csv(CLEAN_DIR / "09_fnirs_problem_log.csv", index=False)
    file_audit_df.to_csv(INTERMEDIATE_DIR / "10_fnirs_file_audit.csv", index=False)

    preprocessing_parameters = {
        "step": "Step 3",
        "snirf_primary_file": "*.snirf",
        "short_channel_max_dist_m": SHORT_CHANNEL_MAX_DIST,
        "sci_threshold": SCI_THRESHOLD,
        "minimum_duration_seconds": MIN_DURATION_SECONDS,
        "preprocessing_order": [
            "read_raw_snirf",
            "replace_annotations_with_harmonized_minimum_set",
            "optical_density",
            "scalp_coupling_index_qc",
            "short_channel_regression_on_od_when_available",
            "tddr",
            "beer_lambert_law",
            "bandpass_filter",
        ],
        "bandpass_l_freq_hz": FILTER_L_FREQ,
        "bandpass_h_freq_hz": FILTER_H_FREQ,
        "beer_lambert_ppf": PPF,
        "software_versions": {
            "mne": mne.__version__,
            "mne_nirs": getattr(__import__("mne_nirs"), "__version__", ""),
            "pandas": pd.__version__,
        },
        "notes": [
            "Ambiguous sessions with raw annotation code 12 were aligned to PsychoPy by interval structure.",
            "Auxiliary streams were audited from the SNIRF HDF5 structure because MNE does not expose them as channels here.",
            "Current .snirf files were never overwritten; all trigger provenance is retained in audit tables.",
            "Global downstream participant exclusions are defined in materials/analysis_participant_exclusions.json and mirrored into Step 3 output tables.",
        ],
    }
    (CLEAN_DIR / "11_preprocessing_parameters.json").write_text(json.dumps(preprocessing_parameters, indent=2), encoding="utf-8")
    preproc_manifest_df.to_csv(CLEAN_DIR / "12_preprocessed_file_manifest.csv", index=False)

    report_path = write_master_report(manifest_df, comparison_df, status_df, orphan_logs, exclusions)
    compile_report(report_path)

    problem_report_path = write_problem_report(status_df, comparison_df, problem_df, orphan_logs, exclusions)
    compile_report(problem_report_path)


if __name__ == "__main__":
    main()
