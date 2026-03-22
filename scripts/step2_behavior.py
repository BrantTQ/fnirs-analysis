#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis_exclusions import exclusion_table_rows, excluded_participant_ids, load_participant_exclusions


ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = ROOT / "sessions"
QUESTIONS_JSON_PATH = ROOT / "materials" / "filtered_questions.json"
STEP1_REGISTRY_PATH = ROOT / "data_intermediate" / "step1" / "02_participant_registry.csv"

INTERMEDIATE_DIR = ROOT / "data_intermediate" / "step2"
CLEAN_DIR = ROOT / "data_clean" / "step2"
FIGURES_DIR = ROOT / "figures" / "step2"
QUESTION_FIGURES_DIR = FIGURES_DIR / "questions"
REPORTS_DIR = ROOT / "reports" / "step2"
QUESTIONS_DIR = REPORTS_DIR / "questions"
GROUPS_DIR = REPORTS_DIR / "groups"

RELEVANT_EVENT_PHASES = {
    "block_start",
    "block_end",
    "iti",
    "q_text_on",
    "q_stem_on",
    "q_options_on",
    "answer",
    "button_click",
}

RESPONSE_STATUS_ORDER = [
    "ANSWERED_VALID",
    "ANSWERED_INVALID",
    "MISSING_ANSWER",
    "PARTIAL_TRIAL",
    "UNMATCHED_QUESTION",
]

ANSWER_OPTIONS = ["A", "B", "C", "D", "E"]


def ensure_directories() -> None:
    for path in [
        INTERMEDIATE_DIR,
        CLEAN_DIR,
        FIGURES_DIR,
        QUESTION_FIGURES_DIR,
        REPORTS_DIR,
        QUESTIONS_DIR,
        GROUPS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_pid(raw: str | None) -> str | None:
    if not raw:
        return None
    match = re.search(r"pid\s*0*(\d+)", raw, flags=re.IGNORECASE)
    if not match:
        return None
    return f"PID{int(match.group(1)):03d}"


def extract_timestamp(value: str) -> tuple[str | None, str | None]:
    match = re.search(r"(\d{8}_\d{6})", value)
    if not match:
        return None, None
    raw = match.group(1)
    iso = f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]} {raw[9:11]}:{raw[11:13]}:{raw[13:15]}"
    return raw, iso


def find_participant_folder(path: Path) -> str | None:
    for part in path.parts:
        if normalize_pid(part):
            return part
    return None


def base_text_cleanup(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s*;\s*", "; ", cleaned)
    return cleaned


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_float(value: float | None) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


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


def make_latex_table(rows: list[tuple[str, Any]], column_spec: str = r"p{0.30\linewidth}p{0.60\linewidth}") -> str:
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule", r"Metric & Value\\", r"\midrule"]
    for key, value in rows:
        lines.append(f"{sanitize_for_tex(key)} & {sanitize_for_tex(value)}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def load_step1_registry() -> dict[str, dict[str, Any]]:
    if not STEP1_REGISTRY_PATH.exists():
        return {}
    df = pd.read_csv(STEP1_REGISTRY_PATH)
    records = df.to_dict(orient="records")
    return {str(record["participant_id"]): record for record in records}


def load_question_metadata_json() -> list[dict[str, Any]]:
    with QUESTIONS_JSON_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_question_id(year: Any, field: Any, question_number: Any) -> str:
    year_text = str(year).strip()
    field_text = str(field).strip().upper()
    number_text = str(question_number).strip()
    return f"{year_text}_{field_text}_{number_text}"


def standardize_question_type(value: Any) -> tuple[str | None, str]:
    cleaned = str(value).strip().lower()
    if cleaned == "abstract":
        return "Abstract", ""
    if cleaned == "concrete":
        return "Concrete", ""
    return None, "invalid_question_type"


def build_file_manifest(step1_registry: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    log_paths = sorted(SESSIONS_DIR.rglob("enem_blocks*.csv"))

    for path in log_paths:
        participant_folder = find_participant_folder(path)
        participant_id = normalize_pid(participant_folder)
        timestamp_raw, timestamp_iso = extract_timestamp(path.name)
        readable = True
        notes: list[str] = []

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                first_row = next(reader, None)
            if first_row is None:
                notes.append("Empty CSV file")
        except Exception as exc:  # pragma: no cover - traceability note
            readable = False
            notes.append(f"Unreadable CSV: {exc}")

        step1_record = step1_registry.get(participant_id or "")
        paired_demographic_exists = bool(step1_record and step1_record.get("socio_json_file_name"))
        paired_demographic_file = step1_record.get("socio_json_file_name", "") if step1_record else ""
        if participant_id and step1_record and step1_record.get("matching_status") != "MATCHED":
            notes.append(f"Step 1 registry status: {step1_record.get('matching_status')}")
        if path.name.startswith("enem_blocks_ "):
            notes.append("Filename contains extra space before PID token")

        rows.append(
            {
                "participant_folder": participant_folder or "",
                "participant_id": participant_id or "",
                "psychopy_log_file_name": path.name,
                "psychopy_log_full_path": str(path.resolve()),
                "timestamp_raw": timestamp_raw or "",
                "timestamp_iso": timestamp_iso or "",
                "log_readable": readable,
                "paired_demographic_file_exists": paired_demographic_exists,
                "paired_demographic_file_name": paired_demographic_file,
                "notes": " | ".join(notes),
            }
        )

    return pd.DataFrame(rows).sort_values(["participant_id", "timestamp_raw"]).reset_index(drop=True)


def build_metadata_tables(raw_metadata: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []

    for item in raw_metadata:
        question_id = build_question_id(item.get("year"), item.get("field"), item.get("question_number"))

        raw_row = {"question_id": question_id}
        raw_row.update(item)
        raw_rows.append(raw_row)

        question_type, type_flag = standardize_question_type(item.get("type"))
        correct_answer = str(item.get("answer", "")).strip().upper()
        enem_correctness = parse_float(item.get("correctness"))
        total_word_count = parse_float(item.get("total_word_count"))
        sentence_count = parse_float(item.get("sentence_count"))
        sentence_length = parse_float(item.get("sentence_length"))

        validation_flags: list[str] = []
        if type_flag:
            validation_flags.append(type_flag)
        if correct_answer not in ANSWER_OPTIONS:
            validation_flags.append("invalid_correct_answer")
            correct_answer = ""
        if enem_correctness is None or not (0 <= enem_correctness <= 1):
            validation_flags.append("invalid_enem_correctness")
            enem_correctness = None
        for field_name, value in {
            "total_word_count": total_word_count,
            "sentence_count": sentence_count,
            "sentence_length": sentence_length,
        }.items():
            if value is None or value <= 0:
                validation_flags.append(f"invalid_{field_name}")

        clean_rows.append(
            {
                "question_id": question_id,
                "question_year": str(item.get("year", "")).strip(),
                "question_field": str(item.get("field", "")).strip().upper(),
                "question_number": str(item.get("question_number", "")).strip(),
                "question_type": question_type,
                "correct_answer": correct_answer,
                "enem_correctness": enem_correctness,
                "total_word_count": total_word_count,
                "sentence_count": sentence_count,
                "sentence_length": sentence_length,
                "question_text_translated": item.get("question_text_translated", ""),
                "question_itself_translated": item.get("question_itself_translated", ""),
                "question_option_A_translated": item.get("question_option_A_translated", ""),
                "question_option_B_translated": item.get("question_option_B_translated", ""),
                "question_option_C_translated": item.get("question_option_C_translated", ""),
                "question_option_D_translated": item.get("question_option_D_translated", ""),
                "question_option_E_translated": item.get("question_option_E_translated", ""),
                "metadata_validation_flags": ";".join(validation_flags),
            }
        )

    raw_df = pd.DataFrame(raw_rows).sort_values("question_id").reset_index(drop=True)
    clean_df = pd.DataFrame(clean_rows).sort_values("question_id").reset_index(drop=True)
    return raw_df, clean_df


def read_log_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_events_raw(file_manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for manifest_row in file_manifest_df.itertuples(index=False):
        if not manifest_row.log_readable:
            continue
        path = Path(manifest_row.psychopy_log_full_path)
        for row in read_log_rows(path):
            if row.get("phase") not in RELEVANT_EVENT_PHASES:
                continue
            rows.append(
                {
                    "participant_id": manifest_row.participant_id,
                    "participant_folder": manifest_row.participant_folder,
                    "timestamp_raw": manifest_row.timestamp_raw,
                    "timestamp_iso": manifest_row.timestamp_iso,
                    "source_log": manifest_row.psychopy_log_file_name,
                    "source_log_full_path": manifest_row.psychopy_log_full_path,
                    "phase": row.get("phase", ""),
                    "block": row.get("block", ""),
                    "trial_idx_in_block": row.get("trial_idx_in_block", ""),
                    "question_number": row.get("question_number", ""),
                    "question_year": row.get("question_year", ""),
                    "question_field": row.get("question_field", ""),
                    "question_type_log": row.get("question_type", ""),
                    "t_abs": parse_float(row.get("t_abs")),
                    "rt_from_phase": parse_float(row.get("rt_from_phase")),
                    "choice": row.get("choice", ""),
                    "correct_raw": row.get("correct", ""),
                    "button_click_time": parse_float(row.get("button_click_time")),
                    "option_view_time": parse_float(row.get("option_view_time")),
                    "marker_name": row.get("marker_name", ""),
                    "marker_code": row.get("marker_code", ""),
                    "note": row.get("note", ""),
                }
            )

    events_df = pd.DataFrame(rows).sort_values(["participant_id", "timestamp_raw", "t_abs", "phase"]).reset_index(drop=True)
    return events_df


def build_block_maps(events_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    block_maps: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    block_start_df = events_df[events_df["phase"] == "block_start"].copy()
    for (participant_id, timestamp_raw), participant_df in block_start_df.groupby(["participant_id", "timestamp_raw"], sort=True):
        participant_df = participant_df.sort_values("t_abs").reset_index(drop=True)
        order_condition = "AbstractFirst" if str(participant_df.iloc[0]["block"]).startswith("A") else "ConcreteFirst"
        block_maps[(participant_id, timestamp_raw)] = {}
        for idx, row in enumerate(participant_df.itertuples(index=False), start=1):
            block_label = str(row.block)
            block_type = "Abstract" if block_label.startswith("A") else "Concrete"
            block_index_match = re.search(r"(\d+)$", block_label)
            block_index_within_type = int(block_index_match.group(1)) if block_index_match else None
            block_maps[(participant_id, timestamp_raw)][block_label] = {
                "block_type": block_type,
                "block_index_within_type": block_index_within_type,
                "global_block_order": idx,
                "order_condition": order_condition,
            }

    return block_maps


def build_trial_raw(events_df: pd.DataFrame, block_maps: dict[tuple[str, str], dict[str, dict[str, Any]]]) -> pd.DataFrame:
    trial_event_df = events_df[events_df["phase"].isin({"iti", "q_text_on", "q_stem_on", "q_options_on", "answer", "button_click"})].copy()
    trial_event_df = trial_event_df[trial_event_df["trial_idx_in_block"].astype(str) != "-1"].copy()

    grouping_columns = [
        "participant_id",
        "participant_folder",
        "timestamp_raw",
        "timestamp_iso",
        "source_log",
        "source_log_full_path",
        "block",
        "trial_idx_in_block",
        "question_year",
        "question_field",
        "question_number",
        "question_type_log",
    ]

    rows: list[dict[str, Any]] = []
    for keys, group_df in trial_event_df.groupby(grouping_columns, dropna=False, sort=True):
        (
            participant_id,
            participant_folder,
            timestamp_raw,
            timestamp_iso,
            source_log,
            source_log_full_path,
            block,
            trial_idx_in_block,
            question_year,
            question_field,
            question_number,
            question_type_log,
        ) = keys
        group_df = group_df.sort_values("t_abs").reset_index(drop=True)
        question_id = build_question_id(question_year, question_field, question_number)

        def first_phase_value(phase: str, column: str = "t_abs") -> Any:
            phase_df = group_df[group_df["phase"] == phase]
            if phase_df.empty:
                return None
            return phase_df.iloc[0][column]

        answer_row = group_df[group_df["phase"] == "answer"]
        answer_row = answer_row.iloc[0] if not answer_row.empty else None
        answer_choice = answer_row["choice"] if answer_row is not None else ""

        block_info = block_maps.get((participant_id, timestamp_raw), {}).get(block, {})
        rows.append(
            {
                "participant_id": participant_id,
                "participant_folder": participant_folder,
                "timestamp_raw": timestamp_raw,
                "timestamp_iso": timestamp_iso,
                "source_log": source_log,
                "source_log_full_path": source_log_full_path,
                "block": block,
                "block_type_from_block": block_info.get("block_type"),
                "block_index_within_type": block_info.get("block_index_within_type"),
                "global_block_order": block_info.get("global_block_order"),
                "order_condition": block_info.get("order_condition"),
                "trial_idx_in_block": int(trial_idx_in_block),
                "question_id": question_id,
                "question_year": str(question_year).strip(),
                "question_field": str(question_field).strip().upper(),
                "question_number": str(question_number).strip(),
                "question_type_log": str(question_type_log).strip().lower(),
                "t_iti": first_phase_value("iti"),
                "t_text_on": first_phase_value("q_text_on"),
                "t_stem_on": first_phase_value("q_stem_on"),
                "t_options_on": first_phase_value("q_options_on"),
                "t_answer": first_phase_value("answer"),
                "rt_from_phase_answer": first_phase_value("answer", "rt_from_phase"),
                "option_view_time": first_phase_value("answer", "option_view_time"),
                "chosen_answer_raw": answer_choice,
                "answer_marker_name": answer_row["marker_name"] if answer_row is not None else "",
                "n_answer_rows": int((group_df["phase"] == "answer").sum()),
                "n_button_click_rows": int((group_df["phase"] == "button_click").sum()),
                "button_click_times_json": json.dumps(
                    [value for value in group_df.loc[group_df["phase"] == "button_click", "button_click_time"].tolist() if value is not None]
                ),
                "trial_notes": " | ".join(sorted({str(note) for note in group_df["note"].tolist() if str(note).strip()})),
            }
        )

    trial_raw_df = pd.DataFrame(rows).sort_values(["participant_id", "timestamp_raw", "global_block_order", "trial_idx_in_block"]).reset_index(drop=True)
    trial_raw_df["trial_idx_global"] = trial_raw_df.groupby("participant_id").cumcount() + 1
    return trial_raw_df


def compute_response_status(row: pd.Series) -> str:
    if not bool(row.get("metadata_matched")):
        return "UNMATCHED_QUESTION"
    if pd.isna(row.get("t_text_on")) or pd.isna(row.get("t_stem_on")) or pd.isna(row.get("t_options_on")):
        return "PARTIAL_TRIAL"
    chosen_answer = str(row.get("chosen_answer", "")).strip()
    if pd.isna(row.get("t_answer")) or chosen_answer == "":
        return "MISSING_ANSWER"
    if chosen_answer not in ANSWER_OPTIONS:
        return "ANSWERED_INVALID"
    return "ANSWERED_VALID"


def build_clean_trial(
    trial_raw_df: pd.DataFrame,
    metadata_clean_df: pd.DataFrame,
) -> pd.DataFrame:
    metadata_subset = metadata_clean_df.copy()
    clean_df = trial_raw_df.merge(metadata_subset, on="question_id", how="left", suffixes=("", "_metadata"))
    clean_df["metadata_matched"] = clean_df["question_type"].notna()

    clean_df["question_type_log_clean"] = clean_df["question_type_log"].map({"abstract": "Abstract", "concrete": "Concrete"})
    clean_df["block_type"] = clean_df["block_type_from_block"]
    clean_df["chosen_answer"] = clean_df["chosen_answer_raw"].astype(str).str.strip().str.upper()
    clean_df.loc[clean_df["chosen_answer"] == "NAN", "chosen_answer"] = ""
    clean_df["answer_available"] = clean_df["chosen_answer"].isin(ANSWER_OPTIONS)

    clean_df["metadata_type_matches_log"] = clean_df.apply(
        lambda row: (
            pd.NA
            if not bool(row["metadata_matched"])
            else bool(row["question_type"] == row["question_type_log_clean"])
        ),
        axis=1,
    )
    clean_df["metadata_year_matches_log"] = clean_df.apply(
        lambda row: (
            pd.NA
            if not bool(row["metadata_matched"])
            else bool(str(row["question_year"]) == str(row["question_year_metadata"]))
        ),
        axis=1,
    )
    clean_df["metadata_field_matches_log"] = clean_df.apply(
        lambda row: (
            pd.NA
            if not bool(row["metadata_matched"])
            else bool(str(row["question_field"]) == str(row["question_field_metadata"]))
        ),
        axis=1,
    )

    clean_df["rt_text_to_stem"] = clean_df["t_stem_on"] - clean_df["t_text_on"]
    clean_df["rt_stem_to_options"] = clean_df["t_options_on"] - clean_df["t_stem_on"]
    clean_df["rt_options_to_answer"] = clean_df["t_answer"] - clean_df["t_options_on"]
    clean_df["rt_text_to_answer"] = clean_df["t_answer"] - clean_df["t_text_on"]
    clean_df["rt_stem_to_answer"] = clean_df["t_answer"] - clean_df["t_stem_on"]

    for column in ["rt_text_to_stem", "rt_stem_to_options", "rt_options_to_answer", "rt_text_to_answer", "rt_stem_to_answer"]:
        clean_df.loc[clean_df[column] < 0, column] = pd.NA

    clean_df["sec_per_word"] = clean_df["rt_text_to_answer"] / clean_df["total_word_count"]
    clean_df["sec_per_sentence"] = clean_df["rt_text_to_answer"] / clean_df["sentence_count"]
    clean_df.loc[(clean_df["total_word_count"].isna()) | (clean_df["total_word_count"] <= 0), "sec_per_word"] = pd.NA
    clean_df.loc[(clean_df["sentence_count"].isna()) | (clean_df["sentence_count"] <= 0), "sec_per_sentence"] = pd.NA

    clean_df["response_status"] = clean_df.apply(compute_response_status, axis=1)
    clean_df["participant_correct"] = clean_df.apply(
        lambda row: (
            int(row["chosen_answer"] == row["correct_answer"])
            if row["response_status"] == "ANSWERED_VALID" and str(row["correct_answer"]).strip() in ANSWER_OPTIONS
            else pd.NA
        ),
        axis=1,
    )

    flags: list[list[str]] = []
    for row in clean_df.itertuples(index=False):
        row_flags: list[str] = []
        if not row.metadata_matched:
            row_flags.append("metadata_unmatched")
        if row.metadata_matched and row.metadata_type_matches_log is False:
            row_flags.append("question_type_mismatch")
        if row.metadata_matched and row.metadata_year_matches_log is False:
            row_flags.append("question_year_mismatch")
        if row.metadata_matched and row.metadata_field_matches_log is False:
            row_flags.append("question_field_mismatch")
        if row.response_status == "PARTIAL_TRIAL":
            row_flags.append("missing_core_event")
        if row.response_status == "ANSWERED_INVALID":
            row_flags.append("invalid_answer_choice")
        flags.append(row_flags)
    clean_df["quality_flags"] = [";".join(values) for values in flags]

    clean_df = clean_df.rename(
        columns={
            "question_year_metadata": "metadata_question_year",
            "question_field_metadata": "metadata_question_field",
            "question_number_metadata": "metadata_question_number",
            "question_type": "metadata_question_type",
        }
    )

    ordered_columns = [
        "participant_id",
        "participant_folder",
        "timestamp_raw",
        "timestamp_iso",
        "source_log",
        "source_log_full_path",
        "block",
        "block_type",
        "block_index_within_type",
        "global_block_order",
        "trial_idx_in_block",
        "trial_idx_global",
        "order_condition",
        "question_id",
        "question_number",
        "question_year",
        "question_field",
        "question_type_log_clean",
        "metadata_question_type",
        "correct_answer",
        "enem_correctness",
        "total_word_count",
        "sentence_count",
        "sentence_length",
        "question_itself_translated",
        "chosen_answer",
        "participant_correct",
        "response_status",
        "answer_available",
        "t_text_on",
        "t_stem_on",
        "t_options_on",
        "t_answer",
        "rt_text_to_stem",
        "rt_stem_to_options",
        "rt_options_to_answer",
        "rt_text_to_answer",
        "rt_stem_to_answer",
        "sec_per_word",
        "sec_per_sentence",
        "metadata_matched",
        "metadata_type_matches_log",
        "metadata_year_matches_log",
        "metadata_field_matches_log",
        "quality_flags",
        "trial_notes",
    ]
    remaining_columns = [col for col in clean_df.columns if col not in ordered_columns]
    return clean_df[ordered_columns + remaining_columns].sort_values(
        ["participant_id", "timestamp_raw", "global_block_order", "trial_idx_in_block"]
    ).reset_index(drop=True)


def summarize_questions(clean_trial_df: pd.DataFrame, metadata_clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    for metadata_row in metadata_clean_df.itertuples(index=False):
        question_id = metadata_row.question_id
        q_df = clean_trial_df[clean_trial_df["question_id"] == question_id].copy()
        valid_df = q_df[q_df["response_status"] == "ANSWERED_VALID"].copy()

        n_exposed = len(q_df)
        n_valid = len(valid_df)
        n_correct = int(valid_df["participant_correct"].fillna(0).sum()) if not valid_df.empty else 0
        sample_accuracy = float(valid_df["participant_correct"].mean()) if not valid_df.empty else math.nan
        mean_rt = float(valid_df["rt_text_to_answer"].mean()) if not valid_df.empty else math.nan
        median_rt = float(valid_df["rt_text_to_answer"].median()) if not valid_df.empty else math.nan
        std_rt = float(valid_df["rt_text_to_answer"].std(ddof=1)) if len(valid_df) > 1 else (0.0 if len(valid_df) == 1 else math.nan)
        mean_sec_per_word = float(valid_df["sec_per_word"].mean()) if not valid_df.empty else math.nan
        mean_sec_per_sentence = float(valid_df["sec_per_sentence"].mean()) if not valid_df.empty else math.nan
        difficulty_gap = sample_accuracy - metadata_row.enem_correctness if not math.isnan(sample_accuracy) and metadata_row.enem_correctness is not None else math.nan

        notes: list[str] = []
        unmatched_count = int((q_df["response_status"] == "UNMATCHED_QUESTION").sum())
        if unmatched_count:
            notes.append(f"{unmatched_count} unmatched trial(s)")
        mismatch_count = int(q_df["quality_flags"].astype(str).str.contains("question_type_mismatch|question_year_mismatch|question_field_mismatch").sum())
        if mismatch_count:
            notes.append(f"{mismatch_count} metadata mismatch flag(s)")

        summary_rows.append(
            {
                "question_id": question_id,
                "question_year": metadata_row.question_year,
                "question_field": metadata_row.question_field,
                "question_number": metadata_row.question_number,
                "question_type": metadata_row.question_type,
                "correct_answer": metadata_row.correct_answer,
                "enem_correctness": metadata_row.enem_correctness,
                "total_word_count": metadata_row.total_word_count,
                "sentence_count": metadata_row.sentence_count,
                "sentence_length": metadata_row.sentence_length,
                "n_exposed": n_exposed,
                "n_valid_answers": n_valid,
                "n_correct": n_correct,
                "sample_accuracy": sample_accuracy,
                "difficulty_gap": difficulty_gap,
                "mean_rt_text_to_answer": mean_rt,
                "median_rt_text_to_answer": median_rt,
                "std_rt_text_to_answer": std_rt,
                "mean_sec_per_word": mean_sec_per_word,
                "mean_sec_per_sentence": mean_sec_per_sentence,
                "notes": " | ".join(notes),
            }
        )

        answer_counts = valid_df["chosen_answer"].value_counts()
        for answer_option in ANSWER_OPTIONS:
            count = int(answer_counts.get(answer_option, 0))
            distribution_rows.append(
                {
                    "question_id": question_id,
                    "question_type": metadata_row.question_type,
                    "answer_option": answer_option,
                    "count": count,
                    "percentage": round((count / n_valid) * 100, 2) if n_valid else 0.0,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("question_id").reset_index(drop=True)
    distribution_df = pd.DataFrame(distribution_rows).sort_values(["question_id", "answer_option"]).reset_index(drop=True)
    return summary_df, distribution_df


def build_type_summary(clean_trial_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for question_type, group_df in clean_trial_df.groupby("metadata_question_type", dropna=False):
        valid_df = group_df[group_df["response_status"] == "ANSWERED_VALID"]
        rows.append(
            {
                "question_type": question_type,
                "n_trials": len(group_df),
                "n_valid_answers": len(valid_df),
                "n_unique_questions": int(group_df["question_id"].nunique()),
                "mean_sample_accuracy": float(valid_df["participant_correct"].mean()) if not valid_df.empty else math.nan,
                "mean_rt_text_to_answer": float(valid_df["rt_text_to_answer"].mean()) if not valid_df.empty else math.nan,
                "median_rt_text_to_answer": float(valid_df["rt_text_to_answer"].median()) if not valid_df.empty else math.nan,
                "mean_enem_correctness": float(group_df["enem_correctness"].dropna().mean()) if not group_df["enem_correctness"].dropna().empty else math.nan,
                "mean_word_count": float(group_df["total_word_count"].dropna().mean()) if not group_df["total_word_count"].dropna().empty else math.nan,
                "mean_sentence_count": float(group_df["sentence_count"].dropna().mean()) if not group_df["sentence_count"].dropna().empty else math.nan,
                "mean_sentence_length": float(group_df["sentence_length"].dropna().mean()) if not group_df["sentence_length"].dropna().empty else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("question_type").reset_index(drop=True)


def build_block_summary(clean_trial_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = ["block", "block_type", "global_block_order", "trial_idx_in_block"]
    for keys, group_df in clean_trial_df.groupby(grouping, dropna=False, sort=True):
        block, block_type, global_block_order, trial_idx_in_block = keys
        valid_df = group_df[group_df["response_status"] == "ANSWERED_VALID"]
        rows.append(
            {
                "block": block,
                "block_type": block_type,
                "global_block_order": global_block_order,
                "trial_idx_in_block": trial_idx_in_block,
                "n_presented_questions": len(group_df),
                "n_answered_questions": int(group_df["answer_available"].sum()),
                "mean_rt_text_to_answer": float(valid_df["rt_text_to_answer"].mean()) if not valid_df.empty else math.nan,
                "mean_accuracy": float(valid_df["participant_correct"].mean()) if not valid_df.empty else math.nan,
                "mean_sec_per_word": float(valid_df["sec_per_word"].mean()) if not valid_df.empty else math.nan,
                "mean_sec_per_sentence": float(valid_df["sec_per_sentence"].mean()) if not valid_df.empty else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["global_block_order", "trial_idx_in_block"]).reset_index(drop=True)


def build_field_summary(clean_trial_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field, group_df in clean_trial_df.groupby("question_field", dropna=False, sort=True):
        valid_df = group_df[group_df["response_status"] == "ANSWERED_VALID"]
        rows.append(
            {
                "question_field": field,
                "n_unique_questions": int(group_df["question_id"].nunique()),
                "n_valid_trials": len(valid_df),
                "mean_accuracy": float(valid_df["participant_correct"].mean()) if not valid_df.empty else math.nan,
                "mean_enem_correctness": float(group_df["enem_correctness"].dropna().mean()) if not group_df["enem_correctness"].dropna().empty else math.nan,
                "mean_rt_text_to_answer": float(valid_df["rt_text_to_answer"].mean()) if not valid_df.empty else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("question_field").reset_index(drop=True)


def build_year_summary(clean_trial_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group_df in clean_trial_df.groupby("question_year", dropna=False, sort=True):
        valid_df = group_df[group_df["response_status"] == "ANSWERED_VALID"]
        rows.append(
            {
                "question_year": year,
                "n_unique_questions": int(group_df["question_id"].nunique()),
                "n_valid_trials": len(valid_df),
                "mean_accuracy": float(valid_df["participant_correct"].mean()) if not valid_df.empty else math.nan,
                "mean_enem_correctness": float(group_df["enem_correctness"].dropna().mean()) if not group_df["enem_correctness"].dropna().empty else math.nan,
                "mean_rt_text_to_answer": float(valid_df["rt_text_to_answer"].mean()) if not valid_df.empty else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("question_year").reset_index(drop=True)


def build_participant_summary(clean_trial_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for participant_id, group_df in clean_trial_df.groupby("participant_id", sort=True):
        valid_df = group_df[group_df["response_status"] == "ANSWERED_VALID"]
        abstract_df = valid_df[valid_df["metadata_question_type"] == "Abstract"]
        concrete_df = valid_df[valid_df["metadata_question_type"] == "Concrete"]
        rows.append(
            {
                "participant_id": participant_id,
                "timestamp_raw": group_df["timestamp_raw"].iloc[0],
                "timestamp_iso": group_df["timestamp_iso"].iloc[0],
                "source_log": group_df["source_log"].iloc[0],
                "order_condition": group_df["order_condition"].iloc[0],
                "n_questions_shown": len(group_df),
                "n_valid_answers": len(valid_df),
                "overall_accuracy": float(valid_df["participant_correct"].mean()) if not valid_df.empty else math.nan,
                "overall_mean_rt_text_to_answer": float(valid_df["rt_text_to_answer"].mean()) if not valid_df.empty else math.nan,
                "abstract_accuracy": float(abstract_df["participant_correct"].mean()) if not abstract_df.empty else math.nan,
                "concrete_accuracy": float(concrete_df["participant_correct"].mean()) if not concrete_df.empty else math.nan,
                "abstract_mean_rt_text_to_answer": float(abstract_df["rt_text_to_answer"].mean()) if not abstract_df.empty else math.nan,
                "concrete_mean_rt_text_to_answer": float(concrete_df["rt_text_to_answer"].mean()) if not concrete_df.empty else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)


def build_exam_vs_sample_comparison(question_summary_df: pd.DataFrame) -> pd.DataFrame:
    return question_summary_df[
        [
            "question_id",
            "question_year",
            "question_field",
            "question_number",
            "question_type",
            "enem_correctness",
            "sample_accuracy",
            "difficulty_gap",
            "n_exposed",
            "n_valid_answers",
        ]
    ].copy()


def build_behavior_missingness(clean_trial_df: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "question_id",
        "block",
        "metadata_question_type",
        "chosen_answer",
        "participant_correct",
        "correct_answer",
        "enem_correctness",
        "t_text_on",
        "t_stem_on",
        "t_options_on",
        "t_answer",
        "rt_text_to_stem",
        "rt_stem_to_options",
        "rt_options_to_answer",
        "rt_text_to_answer",
        "rt_stem_to_answer",
        "sec_per_word",
        "sec_per_sentence",
    ]
    rows: list[dict[str, Any]] = []
    for variable in variables:
        for response_status, group_df in clean_trial_df.groupby("response_status", sort=True):
            series = group_df[variable]
            missing_mask = series.isna()
            if series.dtype == object:
                missing_mask = missing_mask | (series.astype(str).str.strip() == "")
            missing_count = int(missing_mask.sum())
            total_count = len(group_df)
            rows.append(
                {
                    "variable_name": variable,
                    "response_status": response_status,
                    "missing_count": missing_count,
                    "missing_percentage": round((missing_count / total_count) * 100, 2) if total_count else 0.0,
                    "n_trials_in_status": total_count,
                }
            )
    return pd.DataFrame(rows).sort_values(["variable_name", "response_status"]).reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def plot_accuracy_by_type(type_summary_df: pd.DataFrame) -> None:
    path = FIGURES_DIR / "accuracy_by_type.png"
    df = type_summary_df.copy().sort_values("question_type")
    plt.figure(figsize=(7, 5))
    plt.bar(df["question_type"], df["mean_sample_accuracy"], color=["#3C8D5A", "#2E5EAA"])
    plt.ylim(0, 1)
    plt.ylabel("Mean sample accuracy")
    plt.xlabel("Question type")
    plt.title("Sample accuracy by question type")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_rt_by_type(clean_trial_df: pd.DataFrame) -> None:
    path = FIGURES_DIR / "rt_total_by_type.png"
    df = clean_trial_df[clean_trial_df["response_status"] == "ANSWERED_VALID"].copy()
    groups = [df[df["metadata_question_type"] == q]["rt_text_to_answer"].dropna() for q in ["Abstract", "Concrete"]]
    plt.figure(figsize=(7, 5))
    plt.boxplot(groups, labels=["Abstract", "Concrete"])
    plt.ylabel("Response time (s)")
    plt.title("Total response time by question type")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_enem_vs_sample(question_summary_df: pd.DataFrame) -> None:
    path = FIGURES_DIR / "enem_vs_sample_accuracy.png"
    df = question_summary_df.dropna(subset=["enem_correctness", "sample_accuracy"]).copy()
    plt.figure(figsize=(6.5, 6))
    plt.scatter(df["enem_correctness"], df["sample_accuracy"], color="#8A5A44")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("ENEM correctness")
    plt.ylabel("Sample accuracy")
    plt.title("ENEM correctness versus sample accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_word_count_vs_rt(question_summary_df: pd.DataFrame) -> None:
    path = FIGURES_DIR / "word_count_vs_rt.png"
    df = question_summary_df.dropna(subset=["total_word_count", "mean_rt_text_to_answer"]).copy()
    plt.figure(figsize=(7, 5))
    plt.scatter(df["total_word_count"], df["mean_rt_text_to_answer"], color="#2E5EAA")
    plt.xlabel("Word count")
    plt.ylabel("Mean response time (s)")
    plt.title("Word count versus mean response time")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_word_count_vs_accuracy(question_summary_df: pd.DataFrame) -> None:
    path = FIGURES_DIR / "word_count_vs_accuracy.png"
    df = question_summary_df.dropna(subset=["total_word_count", "sample_accuracy"]).copy()
    plt.figure(figsize=(7, 5))
    plt.scatter(df["total_word_count"], df["sample_accuracy"], color="#3C8D5A")
    plt.ylim(0, 1)
    plt.xlabel("Word count")
    plt.ylabel("Sample accuracy")
    plt.title("Word count versus sample accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_question_accuracy_comparison(question_summary_df: pd.DataFrame) -> None:
    path = FIGURES_DIR / "question_accuracy_comparison.png"
    df = question_summary_df.copy().sort_values(["question_type", "question_id"])
    positions = range(len(df))
    width = 0.38
    plt.figure(figsize=(14, 6))
    plt.bar([x - width / 2 for x in positions], df["sample_accuracy"], width=width, label="Sample")
    plt.bar([x + width / 2 for x in positions], df["enem_correctness"], width=width, label="ENEM")
    plt.xticks(list(positions), df["question_id"], rotation=90)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Sample accuracy and ENEM correctness by question")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_question_rt_distribution(question_summary_df: pd.DataFrame) -> None:
    path = FIGURES_DIR / "question_rt_distribution.png"
    df = question_summary_df.copy().sort_values("mean_rt_text_to_answer", na_position="last")
    plt.figure(figsize=(14, 6))
    plt.bar(df["question_id"], df["mean_rt_text_to_answer"], color="#8A5A44")
    plt.xticks(rotation=90)
    plt.ylabel("Mean response time (s)")
    plt.title("Mean response time by question")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_question_answer_distribution(answer_distribution_df: pd.DataFrame) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for question_id, group_df in answer_distribution_df.groupby("question_id", sort=True):
        group_df = group_df.sort_values("answer_option")
        path = QUESTION_FIGURES_DIR / f"{question_id}_answer_distribution.png"
        plt.figure(figsize=(6.5, 4.5))
        plt.bar(group_df["answer_option"], group_df["count"], color="#2E5EAA")
        plt.xlabel("Answer option")
        plt.ylabel("Count")
        plt.title(f"Answer distribution: {question_id}")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths[question_id] = path
    return paths


def generate_plots(clean_trial_df: pd.DataFrame, question_summary_df: pd.DataFrame, answer_distribution_df: pd.DataFrame) -> dict[str, Path]:
    plot_accuracy_by_type(build_type_summary(clean_trial_df))
    plot_rt_by_type(clean_trial_df)
    plot_enem_vs_sample(question_summary_df)
    plot_word_count_vs_rt(question_summary_df)
    plot_word_count_vs_accuracy(question_summary_df)
    plot_question_accuracy_comparison(question_summary_df)
    plot_question_rt_distribution(question_summary_df)
    return plot_question_answer_distribution(answer_distribution_df)


def write_group_reports(
    type_summary_df: pd.DataFrame,
    block_summary_df: pd.DataFrame,
    field_summary_df: pd.DataFrame,
    year_summary_df: pd.DataFrame,
    question_summary_df: pd.DataFrame,
    behavior_missingness_df: pd.DataFrame,
) -> None:
    type_rows = []
    for row in type_summary_df.itertuples(index=False):
        type_rows.append((f"{row.question_type} mean accuracy", round(row.mean_sample_accuracy, 4) if pd.notna(row.mean_sample_accuracy) else ""))
        type_rows.append((f"{row.question_type} mean RT", round(row.mean_rt_text_to_answer, 4) if pd.notna(row.mean_rt_text_to_answer) else ""))

    abstract_content = "\n".join(
        [
            r"\section*{Abstract Versus Concrete}",
            make_latex_table(type_rows),
            "",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=0.48\linewidth]{../../figures/step2/accuracy_by_type.png}",
            r"\includegraphics[width=0.48\linewidth]{../../figures/step2/rt_total_by_type.png}",
            r"\caption{Accuracy and response time by question type.}",
            r"\end{figure}",
            "",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=0.60\linewidth]{../../figures/step2/enem_vs_sample_accuracy.png}",
            r"\caption{Question-level comparison between ENEM correctness and sample accuracy.}",
            r"\end{figure}",
        ]
    )
    (GROUPS_DIR / "group_abstract_vs_concrete.tex").write_text(abstract_content, encoding="utf-8")

    block_rows = []
    for row in block_summary_df.head(12).itertuples(index=False):
        block_rows.append((f"{row.block} trial {row.trial_idx_in_block}", f"accuracy={format_float(row.mean_accuracy)}, rt={format_float(row.mean_rt_text_to_answer)}"))
    block_content = "\n".join(
        [
            r"\section*{Block Summaries}",
            make_latex_table(block_rows),
            "",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=0.90\linewidth]{../../figures/step2/question_rt_distribution.png}",
            r"\caption{Mean response time by question.}",
            r"\end{figure}",
        ]
    )
    (GROUPS_DIR / "group_blocks.tex").write_text(block_content, encoding="utf-8")

    field_rows = []
    for row in field_summary_df.itertuples(index=False):
        field_rows.append((f"Field {row.question_field}", f"questions={row.n_unique_questions}, accuracy={format_float(row.mean_accuracy)}, rt={format_float(row.mean_rt_text_to_answer)}"))
    year_rows = []
    for row in year_summary_df.itertuples(index=False):
        year_rows.append((f"Year {row.question_year}", f"questions={row.n_unique_questions}, accuracy={format_float(row.mean_accuracy)}, rt={format_float(row.mean_rt_text_to_answer)}"))

    missing_rows = []
    for row in behavior_missingness_df[behavior_missingness_df["variable_name"].isin(["chosen_answer", "participant_correct", "rt_text_to_answer"])].head(12).itertuples(index=False):
        missing_rows.append((f"{row.variable_name} / {row.response_status}", f"missing={row.missing_count} ({row.missing_percentage}%)"))

    fields_content = "\n".join(
        [
            r"\section*{Field and Year Summaries}",
            make_latex_table(field_rows),
            "",
            make_latex_table(year_rows),
            "",
            r"\section*{Missingness Highlights}",
            make_latex_table(missing_rows),
            "",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=0.48\linewidth]{../../figures/step2/word_count_vs_rt.png}",
            r"\includegraphics[width=0.48\linewidth]{../../figures/step2/word_count_vs_accuracy.png}",
            r"\caption{Item-length descriptive comparisons.}",
            r"\end{figure}",
        ]
    )
    (GROUPS_DIR / "group_fields.tex").write_text(fields_content, encoding="utf-8")


def write_question_reports(
    metadata_clean_df: pd.DataFrame,
    question_summary_df: pd.DataFrame,
    answer_distribution_df: pd.DataFrame,
    question_plot_paths: dict[str, Path],
) -> None:
    summary_map = question_summary_df.set_index("question_id").to_dict(orient="index")

    for metadata_row in metadata_clean_df.itertuples(index=False):
        question_id = metadata_row.question_id
        summary = summary_map[question_id]
        answer_rows = answer_distribution_df[answer_distribution_df["question_id"] == question_id]

        metadata_table = make_latex_table(
            [
                ("Question ID", question_id),
                ("Year", metadata_row.question_year),
                ("Field", metadata_row.question_field),
                ("Type", metadata_row.question_type),
                ("Correct answer", metadata_row.correct_answer),
                ("ENEM correctness", format_float(metadata_row.enem_correctness)),
                ("Word count", format_float(metadata_row.total_word_count)),
                ("Sentence count", format_float(metadata_row.sentence_count)),
                ("Sentence length", format_float(metadata_row.sentence_length)),
            ]
        )
        performance_table = make_latex_table(
            [
                ("Exposed", summary["n_exposed"]),
                ("Valid answers", summary["n_valid_answers"]),
                ("Correct answers", summary["n_correct"]),
                ("Sample accuracy", format_float(summary["sample_accuracy"])),
                ("ENEM correctness", format_float(summary["enem_correctness"])),
                ("Difficulty gap", format_float(summary["difficulty_gap"])),
                ("Mean RT", format_float(summary["mean_rt_text_to_answer"])),
                ("Median RT", format_float(summary["median_rt_text_to_answer"])),
                ("SD RT", format_float(summary["std_rt_text_to_answer"])),
                ("Mean sec/word", format_float(summary["mean_sec_per_word"])),
                ("Mean sec/sentence", format_float(summary["mean_sec_per_sentence"])),
            ]
        )

        answer_lines = [r"\begin{tabular}{p{0.18\linewidth}rr}", r"\toprule", r"Answer & Count & \% valid\\", r"\midrule"]
        for row in answer_rows.itertuples(index=False):
            answer_lines.append(f"{sanitize_for_tex(row.answer_option)} & {row.count} & {row.percentage}\\\\")
        answer_lines.extend([r"\bottomrule", r"\end{tabular}"])

        question_figure_rel = Path("../../figures/step2/questions") / question_plot_paths[question_id].name
        label = re.sub(r"[^A-Za-z0-9]+", "_", question_id)
        notes = summary["notes"] if summary["notes"] else "No data-quality issues flagged."
        content = "\n".join(
            [
                f"\\subsection*{{Question {sanitize_for_tex(question_id)}}}",
                f"\\label{{sec:q_{label}}}",
                r"\paragraph{Metadata}",
                metadata_table,
                "",
                r"\paragraph{Exposure and validity}",
                performance_table,
                "",
                r"\paragraph{Answer distribution}",
                "\n".join(answer_lines),
                "",
                f"\\paragraph{{Question prompt}} {sanitize_for_tex(metadata_row.question_itself_translated)}",
                "",
                f"\\paragraph{{Notes}} {sanitize_for_tex(notes)}",
                "",
                r"\begin{figure}[htbp]",
                r"\centering",
                f"\\includegraphics[width=0.62\\linewidth]{{{sanitize_for_tex(str(question_figure_rel))}}}",
                f"\\caption{{Answer distribution for {sanitize_for_tex(question_id)}}}",
                r"\end{figure}",
                "",
            ]
        )
        (QUESTIONS_DIR / f"question_{question_id}.tex").write_text(content, encoding="utf-8")


def write_master_report(
    metadata_clean_df: pd.DataFrame,
    clean_trial_df: pd.DataFrame,
    exclusions: dict[str, dict[str, str]],
) -> None:
    overview_rows = [
        ("Participants included in Step 2 summaries", int(clean_trial_df["participant_id"].nunique())),
        ("Trials included in Step 2 summaries", len(clean_trial_df)),
        ("Questions summarized", int(metadata_clean_df["question_id"].nunique())),
        ("Globally excluded participants", len(exclusions)),
    ]
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\title{Step 2 Behavioral Report}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Introduction}",
        r"This report was generated automatically from the Step 2 behavioral harmonization pipeline.",
        r"\section*{Behavioral Overview}",
        make_latex_table(overview_rows),
    ]

    if exclusions:
        lines.extend(
            [
                r"\section*{Global Participant Exclusions}",
                r"The following participant IDs were removed from all Step~2 analysis tables, counts, and plots after Step~3 review. Their files remain on disk for audit only, and future downstream steps should continue to honor this exclusion list.",
                make_latex_table(exclusion_table_rows(exclusions)),
            ]
        )

    lines.extend(
        [
        r"\input{groups/group_abstract_vs_concrete.tex}",
        r"\input{groups/group_blocks.tex}",
        r"\input{groups/group_fields.tex}",
        r"\section*{Question-Level Summaries}",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{../../figures/step2/question_accuracy_comparison.png}",
        r"\caption{Question-level comparison of sample accuracy and ENEM correctness.}",
        r"\end{figure}",
        ]
    )

    for metadata_row in metadata_clean_df.itertuples(index=False):
        lines.append(f"\\input{{questions/question_{metadata_row.question_id}.tex}}")

    lines.append(r"\end{document}")
    (REPORTS_DIR / "step2_behavior_report.tex").write_text("\n".join(lines), encoding="utf-8")


def compile_report() -> None:
    report_path = REPORTS_DIR / "step2_behavior_report.tex"
    try:
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
    excluded_ids = excluded_participant_ids(exclusions)
    step1_registry = load_step1_registry()
    step1_registry = {participant_id: row for participant_id, row in step1_registry.items() if participant_id not in excluded_ids}

    file_manifest_df = build_file_manifest(step1_registry)
    file_manifest_df = file_manifest_df[~file_manifest_df["participant_id"].isin(excluded_ids)].reset_index(drop=True)
    raw_metadata = load_question_metadata_json()
    metadata_raw_df, metadata_clean_df = build_metadata_tables(raw_metadata)
    events_raw_df = build_events_raw(file_manifest_df)
    block_maps = build_block_maps(events_raw_df)
    trial_raw_df = build_trial_raw(events_raw_df, block_maps)
    clean_trial_df = build_clean_trial(trial_raw_df, metadata_clean_df)

    question_summary_df, answer_distribution_df = summarize_questions(clean_trial_df, metadata_clean_df)
    type_summary_df = build_type_summary(clean_trial_df)
    block_summary_df = build_block_summary(clean_trial_df)
    field_summary_df = build_field_summary(clean_trial_df)
    year_summary_df = build_year_summary(clean_trial_df)
    participant_summary_df = build_participant_summary(clean_trial_df)
    exam_vs_sample_df = build_exam_vs_sample_comparison(question_summary_df)
    behavior_missingness_df = build_behavior_missingness(clean_trial_df)

    question_plot_paths = generate_plots(clean_trial_df, question_summary_df, answer_distribution_df)
    write_group_reports(type_summary_df, block_summary_df, field_summary_df, year_summary_df, question_summary_df, behavior_missingness_df)
    write_question_reports(metadata_clean_df, question_summary_df, answer_distribution_df, question_plot_paths)
    write_master_report(metadata_clean_df, clean_trial_df, exclusions)
    compile_report()

    save_dataframe(file_manifest_df, INTERMEDIATE_DIR / "01_question_file_manifest.csv")
    save_dataframe(metadata_raw_df, INTERMEDIATE_DIR / "02_question_metadata_raw.csv")
    save_dataframe(metadata_clean_df, INTERMEDIATE_DIR / "03_question_metadata_clean.csv")
    save_dataframe(events_raw_df, INTERMEDIATE_DIR / "04_behavior_events_raw.csv")
    save_dataframe(trial_raw_df, INTERMEDIATE_DIR / "05_behavior_trial_raw.csv")
    save_dataframe(clean_trial_df, CLEAN_DIR / "06_behavior_trial_clean.csv")
    save_dataframe(question_summary_df, CLEAN_DIR / "07_question_summary.csv")
    save_dataframe(answer_distribution_df, CLEAN_DIR / "08_question_answer_distribution.csv")
    save_dataframe(type_summary_df, CLEAN_DIR / "09_type_summary.csv")
    save_dataframe(block_summary_df, CLEAN_DIR / "10_block_summary.csv")
    save_dataframe(field_summary_df, CLEAN_DIR / "11_field_summary.csv")
    save_dataframe(year_summary_df, CLEAN_DIR / "12_year_summary.csv")
    save_dataframe(participant_summary_df, CLEAN_DIR / "13_participant_behavior_summary.csv")
    save_dataframe(exam_vs_sample_df, CLEAN_DIR / "14_exam_vs_sample_comparison.csv")
    save_dataframe(behavior_missingness_df, CLEAN_DIR / "15_behavior_missingness.csv")

    print("Step 2 outputs generated successfully.")
    print(f"Participants: {file_manifest_df['participant_id'].nunique()}")
    print(f"Excluded participants: {len(exclusions)}")
    print(f"Trials: {len(clean_trial_df)}")
    print(f"Questions in metadata: {len(metadata_clean_df)}")


if __name__ == "__main__":
    main()
