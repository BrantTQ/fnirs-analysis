#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis_exclusions import exclusion_table_rows, excluded_participant_ids, load_participant_exclusions


ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = ROOT / "sessions"
SCHEMA_PATH = ROOT / "materials" / "demographics_questions.json"

INTERMEDIATE_DIR = ROOT / "data_intermediate" / "step1"
CLEAN_DIR = ROOT / "data_clean" / "step1"
FIGURES_DIR = ROOT / "figures" / "step1"
REPORTS_DIR = ROOT / "reports" / "step1"
QUESTIONS_DIR = REPORTS_DIR / "questions"

MISSINGNESS_LABELS = ["MISSING", "NOT_APPLICABLE", "INVALID", "UNKNOWN", "BLANK"]

QUESTION_GROUPS = {
    "age": "Core identification and mobility variables",
    "country_of_birth": "Core identification and mobility variables",
    "nationality": "Core identification and mobility variables",
    "years_in_luxembourg": "Core identification and mobility variables",
    "gender": "Gender and language background",
    "first_language": "Gender and language background",
    "other_languages": "Gender and language background",
    "primary_school_langs": "Gender and language background",
    "affiliation": "Current academic or professional status",
    "department_unit": "Current academic or professional status",
    "programme_name": "Current academic or professional status",
    "study_level": "Current academic or professional status",
    "year_in_programme": "Current academic or professional status",
    "employee_role": "Current academic or professional status",
    "highest_degree": "Educational background",
    "degree_field": "Educational background",
    "country_secondary_ed": "Educational background",
    "parent_a_edu": "Family educational background and living conditions",
    "parent_b_edu": "Family educational background and living conditions",
    "first_gen_he": "Family educational background and living conditions",
    "living_arrangement": "Family educational background and living conditions",
}

NUMERIC_VARIABLES = {"age", "years_in_luxembourg", "year_in_programme"}
ORDERED_EDUCATION_VARIABLES = {
    "study_level",
    "highest_degree",
    "parent_a_edu",
    "parent_b_edu",
}
CATEGORICAL_VARIABLES = {
    "gender",
    "first_gen_he",
    "affiliation",
    "study_level",
    "highest_degree",
    "parent_a_edu",
    "parent_b_edu",
    "employee_role",
}

SPECIAL_WORD_CASE = {
    "bap": "BAP",
    "uk": "UK",
    "usa": "USA",
    "eu": "EU",
    "udm": "UDM",
    "lis": "LIS",
    "liser": "LISER",
    "fhse": "FHSE",
    "c2dh": "C2DH",
    "msc": "MSc",
    "bsc": "BSc",
    "phd": "PhD",
    "na": "NA",
}

TEXT_OVERRIDES = {
    "affiliation": {
        "phd": "PhD candidate",
        "phd candidate": "PhD candidate",
        "phdcandidate": "PhD candidate",
        "phd candidate ": "PhD candidate",
        "phd candidate": "PhD candidate",
        "university employee": "University employee",
        "master graduate": "Master graduate",
        "research associate": "Research associate",
        "employee, lis": "Employee, LIS",
        "liser employee": "LISER employee",
        "lis employee": "LIS employee",
        "postdoctoral student": "Postdoctoral student",
        "working at emile weber": "Working at Emile Weber",
    },
    "employee_role": {
        "admin": "Admin staff",
        "admin staff": "Admin staff",
        "technical": "Technical staff",
        "technical staff": "Technical staff",
        "research staff": "Research staff",
        "postdoc": "Postdoc",
        "data analyst": "Data analyst",
        "research associate and data expert": "Research associate and data expert",
        "team leader": "Team leader",
    },
    "country_of_birth": {
        "uuguay": "Uruguay",
        "uk": "UK",
        "usa": "USA",
        "non-eu": "Non-EU",
    },
    "country_secondary_ed": {
        "gemany": "Germany",
        "uk": "UK",
        "belgium": "Belgium",
        "netherlands": "Netherlands",
    },
    "degree_field": {
        "conomics": "Economics",
        "chinese , french language and cultre": "Chinese, French language and culture",
        "economic-law": "Economic-Law",
        "highschool": "High school",
    },
    "department_unit": {
        "fhse-mediacentre": "FHSE Media Centre",
        "administration media centre": "Administration Media Centre",
        "living conditions, liser": "Living Conditions, LISER",
        "udm liser": "UDM LISER",
        "udm": "UDM",
        "udm liser ": "UDM LISER",
        "liser": "LISER",
        "lis": "LIS",
        "fhse": "FHSE",
        "na": "NA",
    },
    "first_language": {
        "farsi": "Farsi",
    },
    "living_arrangement": {
        "appartment": "Apartment",
        "with faily": "With family",
        "roomates": "Roommates",
        "with one roomate": "With one roommate",
        "student accomodation": "Student accommodation",
    },
    "nationality": {
        "luxembourg": "Luxembourg",
    },
    "other_languages": {
        "english,german,russian": "English, German, Russian",
    },
    "parent_b_edu": {
        "school\\": "School",
        "seconday school diploma": "Secondary school diploma",
    },
    "primary_school_langs": {
        "german english french": "German English French",
    },
}


@dataclass
class HarmonizationResult:
    clean_value: str | None
    numeric_value: float | None
    rule_name: str
    comments: str
    rule_accuracy: str


def ensure_directories() -> None:
    for path in [INTERMEDIATE_DIR, CLEAN_DIR, FIGURES_DIR, REPORTS_DIR, QUESTIONS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_schema() -> list[dict[str, Any]]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_pid(raw: str | None) -> str | None:
    if not raw:
        return None
    match = re.search(r"pid\s*0*(\d+)", raw, flags=re.IGNORECASE)
    if not match:
        return None
    return f"PID{int(match.group(1)):03d}"


def find_participant_folder(path: Path) -> str | None:
    for part in path.parts:
        if normalize_pid(part):
            return part
    return None


def extract_pid_from_filename(file_name: str) -> str | None:
    match = re.search(r"(pid\s*0*\d+)", file_name, flags=re.IGNORECASE)
    return match.group(1) if match else None


def extract_timestamp(value: str) -> tuple[str | None, str | None]:
    match = re.search(r"(\d{8}_\d{6})", value)
    if not match:
        return None, None
    raw = match.group(1)
    iso = f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]} {raw[9:11]}:{raw[11:13]}:{raw[13:15]}"
    return raw, iso


def detect_file_type(path: Path) -> str:
    file_name = path.name.strip()
    lower_name = file_name.lower()
    if path.resolve() == SCHEMA_PATH.resolve():
        return "schema"
    if lower_name.startswith("enem_blocks") and path.suffix.lower() == ".csv":
        return "enem_blocks"
    if lower_name.startswith("socio") and path.suffix.lower() == ".json":
        return "socio_json"
    return "other"


def base_text_cleanup(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s*;\s*", "; ", cleaned)
    cleaned = re.sub(r"\s*/\s*", "/", cleaned)
    return cleaned


def titleize_token(token: str) -> str:
    match = re.match(r"^([^A-Za-zÀ-ÿ]*)([A-Za-zÀ-ÿ]+)([^A-Za-zÀ-ÿ]*)$", token)
    if match:
        prefix, core, suffix = match.groups()
        lower = core.lower()
        if lower in SPECIAL_WORD_CASE:
            return f"{prefix}{SPECIAL_WORD_CASE[lower]}{suffix}"
        return f"{prefix}{core[:1].upper()}{core[1:].lower()}{suffix}"
    lower = token.lower()
    if lower in SPECIAL_WORD_CASE:
        return SPECIAL_WORD_CASE[lower]
    if re.fullmatch(r"[A-Za-zÀ-ÿ]+", token):
        return token[:1].upper() + token[1:].lower()
    return token


def smart_case(text: str) -> str:
    def convert_chunk(chunk: str) -> str:
        if not chunk:
            return chunk
        pieces = re.split(r"([-/])", chunk)
        out: list[str] = []
        for piece in pieces:
            if piece in {"-", "/"}:
                out.append(piece)
            else:
                out.append(titleize_token(piece))
        return "".join(out)

    words = text.split(" ")
    return " ".join(convert_chunk(word) for word in words)


def comparison_key(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def apply_text_override(variable: str, cleaned: str) -> tuple[str, str]:
    override = TEXT_OVERRIDES.get(variable, {}).get(cleaned.lower())
    if override:
        return override, "override"
    return cleaned, "generic_case_cleanup"


def harmonize_text(variable: str, raw_value: str) -> HarmonizationResult:
    cleaned = base_text_cleanup(raw_value)
    cased = smart_case(cleaned)
    clean_value, rule_name = apply_text_override(variable, cased)
    accuracy = "exact"
    if comparison_key(raw_value) != comparison_key(clean_value):
        accuracy = "approximate"
    comments = "Conservative whitespace/case normalization"
    if rule_name == "override":
        comments = "Variable-specific conservative normalization"
    return HarmonizationResult(clean_value=clean_value, numeric_value=None, rule_name=rule_name, comments=comments, rule_accuracy=accuracy)


def harmonize_gender(raw_value: str) -> HarmonizationResult:
    cleaned = base_text_cleanup(raw_value).lower()
    if cleaned == "male":
        return HarmonizationResult("Male", None, "gender_to_male", "Controlled vocabulary for gender", "exact")
    if cleaned == "female":
        return HarmonizationResult("Female", None, "gender_to_female", "Controlled vocabulary for gender", "exact")
    return HarmonizationResult("Unknown", None, "gender_unclear_to_unknown", "Unclear gender response retained as Unknown", "approximate")


def harmonize_first_gen(raw_value: str) -> HarmonizationResult:
    cleaned = base_text_cleanup(raw_value).lower().replace("’", "'")
    if cleaned in {"yes", "y"}:
        return HarmonizationResult("Yes", None, "first_gen_yes", "Controlled vocabulary for first-generation status", "exact")
    if cleaned in {"no", "n"}:
        return HarmonizationResult("No", None, "first_gen_no", "Controlled vocabulary for first-generation status", "exact")
    if cleaned in {"dont know", "don't know", "do not know"}:
        return HarmonizationResult("Unknown", None, "first_gen_uncertain_to_unknown", "Participant uncertainty retained as Unknown", "exact")
    return HarmonizationResult("Unknown", None, "first_gen_ambiguous_to_unknown", "Ambiguous first-generation response retained as Unknown", "approximate")


def harmonize_ordered_education(variable: str, raw_value: str) -> HarmonizationResult:
    cleaned = base_text_cleanup(raw_value).lower()
    if variable == "study_level":
        mapping = {
            "bachelor": "Bachelor",
            "master": "Master",
            "phd": "Doctorate",
        }
        mapped = mapping.get(cleaned)
        if mapped:
            accuracy = "exact" if comparison_key(raw_value) == comparison_key(mapped) else "approximate"
            return HarmonizationResult(mapped, None, "study_level_ordered_harmonization", "Ordered harmonization for study level", accuracy)
        return HarmonizationResult(smart_case(base_text_cleanup(raw_value)), None, "study_level_fallback_text_cleanup", "Unexpected study level retained after conservative cleanup", "approximate")

    if variable == "highest_degree":
        mapping = {
            "upper secondary": "Upper secondary",
            "upper-secondary": "Upper secondary",
            "bachelor": "Bachelor",
            "master": "Master",
            "doctorate": "Doctorate",
        }
        mapped = mapping.get(cleaned)
        if mapped:
            accuracy = "exact" if comparison_key(raw_value) == comparison_key(mapped) else "approximate"
            return HarmonizationResult(mapped, None, "highest_degree_ordered_harmonization", "Ordered harmonization for highest degree", accuracy)
        return HarmonizationResult(smart_case(base_text_cleanup(raw_value)), None, "highest_degree_fallback_text_cleanup", "Unexpected highest degree retained after conservative cleanup", "approximate")

    primary_or_less = {"dropped out after 6th", "primary school", "no"}
    secondary = {
        "hauptschule",
        "high school",
        "highschool",
        "secondary i",
        "secondary education",
        "baccalaureat",
        "high-school level",
        "higher school",
        "secondary school",
        "secondary",
        "school",
        "school\\",
        "abitur",
        "seconday school diploma",
    }
    post_secondary_non_tertiary = {"post-secondary non-tertiary", "technical career"}
    tertiary_unspecified = {"university", "university degree"}
    bachelor = {"bachelor", "bachelors", "bachelor level"}
    master = {"master", "masters", "master degree", "master level"}
    doctorate = {"doctorate", "phd"}
    unknown = {"unknown"}

    if cleaned in primary_or_less:
        target = "Primary or less"
    elif cleaned in secondary:
        target = "Secondary"
    elif cleaned in post_secondary_non_tertiary:
        target = "Post-secondary non-tertiary"
    elif cleaned in tertiary_unspecified:
        target = "Tertiary unspecified"
    elif cleaned in bachelor:
        target = "Bachelor"
    elif cleaned in master:
        target = "Master"
    elif cleaned in doctorate:
        target = "Doctorate"
    elif cleaned in unknown:
        target = "Unknown"
    else:
        target = smart_case(base_text_cleanup(raw_value))

    accuracy = "exact" if comparison_key(raw_value) == comparison_key(target) else "approximate"
    return HarmonizationResult(target, None, f"{variable}_ordered_harmonization", "Ordered harmonization for parental education", accuracy)


def harmonize_affiliation(raw_value: str) -> HarmonizationResult:
    cleaned = base_text_cleanup(raw_value).lower()
    if cleaned in {"phd", "phd candidate", "phdcandidate"}:
        return HarmonizationResult("PhD candidate", None, "affiliation_phd_candidate", "Canonicalized obvious PhD-candidate variants", "exact")
    if cleaned == "university employee":
        return HarmonizationResult("University employee", None, "affiliation_university_employee", "Canonicalized capitalization for university employee", "exact")
    return harmonize_text("affiliation", raw_value)


def harmonize_employee_role(raw_value: str) -> HarmonizationResult:
    return harmonize_text("employee_role", raw_value)


def parse_numeric_value(raw_value: str) -> float | None:
    cleaned = base_text_cleanup(raw_value)
    if cleaned == "":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def format_numeric(value: float | None) -> str | None:
    if value is None:
        return None
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def harmonize_value(variable: str, raw_value: str) -> HarmonizationResult:
    if variable in NUMERIC_VARIABLES:
        numeric_value = parse_numeric_value(raw_value)
        if numeric_value is None:
            return HarmonizationResult(None, None, "numeric_parse_failed", "Could not parse expected numeric value", "exact")
        return HarmonizationResult(format_numeric(numeric_value), numeric_value, "numeric_parse_success", "Parsed numeric value from raw string", "exact")
    if variable == "gender":
        return harmonize_gender(raw_value)
    if variable == "first_gen_he":
        return harmonize_first_gen(raw_value)
    if variable in ORDERED_EDUCATION_VARIABLES:
        return harmonize_ordered_education(variable, raw_value)
    if variable == "affiliation":
        return harmonize_affiliation(raw_value)
    if variable == "employee_role":
        return harmonize_employee_role(raw_value)
    return harmonize_text(variable, raw_value)


def variable_analysis_type(question_id: str) -> str:
    if question_id in NUMERIC_VARIABLES:
        return "numeric"
    if question_id in CATEGORICAL_VARIABLES:
        return "categorical"
    return "free_text"


def collect_manifest_paths() -> list[Path]:
    paths = [path for path in SESSIONS_DIR.rglob("*") if path.is_file()]
    paths.append(SCHEMA_PATH)
    return sorted(paths)


def build_manifest(schema_question_ids: list[str]) -> pd.DataFrame:
    paths = collect_manifest_paths()

    pair_lookup: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in paths:
        file_type = detect_file_type(path)
        if file_type not in {"enem_blocks", "socio_json"}:
            continue
        participant_folder = find_participant_folder(path)
        participant_id = normalize_pid(participant_folder)
        timestamp_raw, _ = extract_timestamp(path.name)
        if participant_id and timestamp_raw:
            pair_lookup[(participant_id, timestamp_raw)].add(file_type)

    rows: list[dict[str, Any]] = []
    for path in paths:
        participant_folder = find_participant_folder(path)
        participant_id = normalize_pid(participant_folder)
        file_type = detect_file_type(path)
        timestamp_raw, timestamp_iso = extract_timestamp(path.name)
        file_pid_raw = extract_pid_from_filename(path.name)
        file_pid_normalized = normalize_pid(file_pid_raw)
        file_pid_matches_folder = None
        if file_pid_normalized and participant_id:
            file_pid_matches_folder = file_pid_normalized == participant_id

        expected_pair_exists: bool | None = None
        if file_type in {"enem_blocks", "socio_json"} and participant_id and timestamp_raw:
            pair_types = pair_lookup.get((participant_id, timestamp_raw), set())
            expected_pair_exists = {"enem_blocks", "socio_json"}.issubset(pair_types)

        notes: list[str] = []
        if file_type == "schema":
            notes.append(f"Schema contains {len(schema_question_ids)} question IDs")
        if file_pid_raw and participant_id and file_pid_normalized and file_pid_normalized != participant_id:
            notes.append("Filename PID does not match participant folder after normalization")
        if file_type in {"enem_blocks", "socio_json"} and expected_pair_exists is False:
            notes.append("Expected socio/enem counterpart missing")
        if " pid" in path.name.lower() or path.name.startswith("socio_ ") or path.name.startswith("enem_blocks_ "):
            notes.append("Filename contains extra space before PID token")

        rows.append(
            {
                "participant_folder_name": participant_folder or "",
                "participant_id": participant_id or "",
                "file_name": path.name,
                "file_type": file_type,
                "timestamp_raw": timestamp_raw or "",
                "timestamp_iso": timestamp_iso or "",
                "full_path": str(path.resolve()),
                "relative_path": str(path.relative_to(ROOT)),
                "file_pid_raw": file_pid_raw or "",
                "file_pid_normalized": file_pid_normalized or "",
                "file_pid_matches_folder": file_pid_matches_folder,
                "expected_pair_exists": expected_pair_exists,
                "notes": " | ".join(notes),
            }
        )

    return pd.DataFrame(rows).sort_values(["participant_id", "file_type", "relative_path"], na_position="last").reset_index(drop=True)


def build_participant_registry(manifest_df: pd.DataFrame) -> pd.DataFrame:
    participant_rows: list[dict[str, Any]] = []
    participant_ids = sorted(x for x in manifest_df["participant_id"].dropna().unique() if x)

    for participant_id in participant_ids:
        participant_manifest = manifest_df[manifest_df["participant_id"] == participant_id]
        folder_name = participant_manifest["participant_folder_name"].replace("", pd.NA).dropna().iloc[0]

        socio_rows = participant_manifest[participant_manifest["file_type"] == "socio_json"].sort_values("relative_path")
        enem_rows = participant_manifest[participant_manifest["file_type"] == "enem_blocks"].sort_values("relative_path")

        socio_file = socio_rows["file_name"].iloc[0] if len(socio_rows) == 1 else ""
        enem_file = enem_rows["file_name"].iloc[0] if len(enem_rows) == 1 else ""
        socio_path = socio_rows["full_path"].iloc[0] if len(socio_rows) == 1 else ""
        enem_path = enem_rows["full_path"].iloc[0] if len(enem_rows) == 1 else ""
        socio_timestamp = socio_rows["timestamp_raw"].iloc[0] if len(socio_rows) == 1 else ""
        enem_timestamp = enem_rows["timestamp_raw"].iloc[0] if len(enem_rows) == 1 else ""

        notes: list[str] = []
        if len(socio_rows) == 0:
            matching_status = "MISSING_SOCIO"
            notes.append("No socio JSON file found")
        elif len(enem_rows) == 0:
            matching_status = "MISSING_ENEM"
            notes.append("No enem_blocks CSV file found")
        elif len(socio_rows) > 1 or len(enem_rows) > 1:
            matching_status = "MULTIPLE_FILES"
            notes.append("More than one socio/enem file found for participant")
        elif socio_timestamp != enem_timestamp:
            matching_status = "TIMESTAMP_MISMATCH"
            notes.append("Socio and enem files do not share the same timestamp")
        else:
            matching_status = "MATCHED"

        if socio_file and normalize_pid(extract_pid_from_filename(socio_file)) == participant_id and extract_pid_from_filename(socio_file) != participant_id:
            notes.append("Socio filename PID matched participant after normalization")
        if enem_file and normalize_pid(extract_pid_from_filename(enem_file)) == participant_id and extract_pid_from_filename(enem_file) != participant_id:
            notes.append("Enem filename PID matched participant after normalization")

        participant_rows.append(
            {
                "participant_id": participant_id,
                "folder_name": folder_name,
                "psychopy_file_name": enem_file,
                "psychopy_full_path": enem_path,
                "socio_json_file_name": socio_file,
                "socio_json_full_path": socio_path,
                "session_timestamp_raw": socio_timestamp or enem_timestamp,
                "session_timestamp_iso": extract_timestamp(socio_timestamp or enem_timestamp)[1] or "",
                "matching_status": matching_status,
                "notes": " | ".join(notes),
            }
        )

    return pd.DataFrame(participant_rows).sort_values("participant_id").reset_index(drop=True)


def build_schema_table(schema: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for question in schema:
        show_if = question.get("show_if") or {}
        rows.append(
            {
                "question_id": question["id"],
                "question_prompt": question.get("prompt", ""),
                "declared_type": question.get("type", ""),
                "required": bool(question.get("required", False)),
                "show_if_dependency": show_if.get("id", ""),
                "show_if_regex": show_if.get("regex", ""),
            }
        )
    return pd.DataFrame(rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_show_if(question: dict[str, Any], raw_payload: dict[str, Any]) -> bool:
    show_if = question.get("show_if")
    if not show_if:
        return True
    dependency_id = show_if.get("id")
    pattern = show_if.get("regex")
    dependency_value = raw_payload.get(dependency_id)
    if dependency_value is None:
        return False
    return bool(re.search(pattern, str(dependency_value), flags=re.IGNORECASE))


def build_raw_tables(
    registry_df: pd.DataFrame, schema: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    schema_order = [question["id"] for question in schema]
    raw_wide_rows: list[dict[str, Any]] = []
    raw_long_rows: list[dict[str, Any]] = []
    participant_payloads: list[dict[str, Any]] = []

    for row in registry_df.itertuples(index=False):
        if not row.socio_json_full_path:
            continue
        raw_payload = read_json(Path(row.socio_json_full_path))
        present_question_ids = sorted(raw_payload.keys())

        wide_row = {
            "participant_id": row.participant_id,
            "folder_name": row.folder_name,
            "timestamp_raw": row.session_timestamp_raw,
            "timestamp_iso": row.session_timestamp_iso,
            "source_json": row.socio_json_file_name,
            "source_json_full_path": row.socio_json_full_path,
            "source_enem_blocks": row.psychopy_file_name,
            "source_enem_blocks_full_path": row.psychopy_full_path,
            "present_question_ids_json": json.dumps(present_question_ids, ensure_ascii=False),
        }
        for question_id in schema_order:
            wide_row[question_id] = raw_payload.get(question_id)
            wide_row[f"{question_id}__key_present"] = question_id in raw_payload
        raw_wide_rows.append(wide_row)

        participant_payloads.append(
            {
                "participant_id": row.participant_id,
                "folder_name": row.folder_name,
                "timestamp_raw": row.session_timestamp_raw,
                "timestamp_iso": row.session_timestamp_iso,
                "source_json": row.socio_json_file_name,
                "source_json_full_path": row.socio_json_full_path,
                "source_enem_blocks": row.psychopy_file_name,
                "source_enem_blocks_full_path": row.psychopy_full_path,
                "raw_payload": raw_payload,
            }
        )

        for question in schema:
            question_id = question["id"]
            raw_long_rows.append(
                {
                    "participant_id": row.participant_id,
                    "folder_name": row.folder_name,
                    "timestamp_raw": row.session_timestamp_raw,
                    "timestamp_iso": row.session_timestamp_iso,
                    "question_id": question_id,
                    "raw_value": raw_payload.get(question_id),
                    "raw_key_present": question_id in raw_payload,
                    "source_json": row.socio_json_file_name,
                    "source_json_full_path": row.socio_json_full_path,
                }
            )

    raw_wide_df = pd.DataFrame(raw_wide_rows).sort_values("participant_id").reset_index(drop=True)
    raw_long_df = pd.DataFrame(raw_long_rows).sort_values(["participant_id", "question_id"]).reset_index(drop=True)
    return raw_wide_df, raw_long_df, participant_payloads


def validate_and_harmonize(
    participant_payloads: list[dict[str, Any]], schema: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schema_map = {question["id"]: question for question in schema}
    harmonization_rows: list[dict[str, Any]] = []
    clean_long_rows: list[dict[str, Any]] = []

    for participant in participant_payloads:
        raw_payload = participant["raw_payload"]
        for question in schema:
            question_id = question["id"]
            raw_key_present = question_id in raw_payload
            raw_value = raw_payload.get(question_id)
            expected_visible = evaluate_show_if(question, raw_payload)
            declared_type = question.get("type", "")
            validation_flags: list[str] = []
            missingness_label = "VALID"
            harmonized_value: str | None = None
            numeric_value: float | None = None
            rule_name = ""
            rule_comments = ""
            rule_accuracy = ""

            if raw_key_present and raw_value not in (None, "") and not expected_visible and question.get("show_if"):
                validation_flags.append("present_despite_show_if")
            if raw_key_present and raw_value == "" and not expected_visible and question.get("show_if"):
                validation_flags.append("blank_despite_show_if")

            if not raw_key_present:
                if question.get("show_if") and not expected_visible:
                    missingness_label = "NOT_APPLICABLE"
                    harmonized_value = "NOT_APPLICABLE"
                    rule_name = "absent_conditional_field_to_not_applicable"
                    rule_comments = "Field absent and not applicable under current questionnaire logic"
                    rule_accuracy = "exact"
                else:
                    missingness_label = "MISSING"
                    harmonized_value = "MISSING"
                    rule_name = "absent_expected_field_to_missing"
                    rule_comments = "Field absent even though expected under current questionnaire logic"
                    rule_accuracy = "exact"
            elif raw_value == "":
                missingness_label = "BLANK"
                harmonized_value = "BLANK"
                rule_name = "empty_string_to_blank"
                rule_comments = "Blank string recorded in the raw JSON"
                rule_accuracy = "exact"
            else:
                result = harmonize_value(question_id, str(raw_value))
                rule_name = result.rule_name
                rule_comments = result.comments
                rule_accuracy = result.rule_accuracy
                harmonized_value = result.clean_value
                numeric_value = result.numeric_value

                if declared_type == "number" and numeric_value is None:
                    missingness_label = "INVALID"
                    harmonized_value = "INVALID"
                    validation_flags.append("numeric_parse_failed")
                elif harmonized_value is None:
                    missingness_label = "UNKNOWN"
                    harmonized_value = "UNKNOWN"
                else:
                    missingness_label = "VALID"

            if question_id in NUMERIC_VARIABLES and missingness_label == "VALID" and numeric_value is None and raw_value not in (None, ""):
                validation_flags.append("numeric_value_missing_after_valid_parse")

            clean_long_rows.append(
                {
                    "participant_id": participant["participant_id"],
                    "folder_name": participant["folder_name"],
                    "timestamp_raw": participant["timestamp_raw"],
                    "timestamp_iso": participant["timestamp_iso"],
                    "source_json": participant["source_json"],
                    "source_json_full_path": participant["source_json_full_path"],
                    "source_enem_blocks": participant["source_enem_blocks"],
                    "source_enem_blocks_full_path": participant["source_enem_blocks_full_path"],
                    "question_id": question_id,
                    "question_prompt": question.get("prompt", ""),
                    "variable_group": QUESTION_GROUPS.get(question_id, ""),
                    "declared_type": declared_type,
                    "analysis_type": variable_analysis_type(question_id),
                    "required": bool(question.get("required", False)),
                    "raw_key_present": raw_key_present,
                    "expected_visible_under_schema": expected_visible,
                    "raw_value": raw_value,
                    "harmonized_value": harmonized_value,
                    "numeric_value": numeric_value,
                    "missingness_label": missingness_label,
                    "validation_flags": ";".join(validation_flags),
                    "rule_name": rule_name,
                    "rule_comments": rule_comments,
                    "rule_accuracy": rule_accuracy,
                }
            )

            harmonization_rows.append(
                {
                    "variable_name": question_id,
                    "raw_value": "<KEY_ABSENT>" if not raw_key_present else raw_value,
                    "harmonized_value": harmonized_value,
                    "transformation_rule": rule_name,
                    "comments": rule_comments,
                    "rule_accuracy": rule_accuracy,
                }
            )

    clean_long_df = pd.DataFrame(clean_long_rows).sort_values(["participant_id", "question_id"]).reset_index(drop=True)
    harmonization_df = pd.DataFrame(harmonization_rows).drop_duplicates().sort_values(["variable_name", "raw_value", "harmonized_value"]).reset_index(drop=True)

    for question_id in schema_map:
        if question_id not in set(harmonization_df["variable_name"]):
            harmonization_df = pd.concat(
                [
                    harmonization_df,
                    pd.DataFrame(
                        [
                            {
                                "variable_name": question_id,
                                "raw_value": "<KEY_ABSENT>",
                                "harmonized_value": "MISSING",
                                "transformation_rule": "absent_expected_field_to_missing",
                                "comments": "Field absent even though expected under current questionnaire logic",
                                "rule_accuracy": "exact",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    harmonization_df = harmonization_df.sort_values(["variable_name", "raw_value", "harmonized_value"]).reset_index(drop=True)
    return clean_long_df, harmonization_df


def build_clean_wide(clean_long_df: pd.DataFrame, schema: list[dict[str, Any]]) -> pd.DataFrame:
    participant_meta_cols = [
        "participant_id",
        "folder_name",
        "timestamp_raw",
        "timestamp_iso",
        "source_json",
        "source_json_full_path",
        "source_enem_blocks",
        "source_enem_blocks_full_path",
    ]

    rows: list[dict[str, Any]] = []
    for participant_id, participant_df in clean_long_df.groupby("participant_id", sort=True):
        row = {column: participant_df.iloc[0][column] for column in participant_meta_cols}
        for question in schema:
            question_id = question["id"]
            q_df = participant_df[participant_df["question_id"] == question_id]
            if q_df.empty:
                continue
            record = q_df.iloc[0]
            row[f"{question_id}_raw"] = record["raw_value"]
            row[f"{question_id}_analysis_value"] = record["harmonized_value"]
            row[f"{question_id}_status"] = record["missingness_label"]
            row[f"{question_id}_expected_visible"] = record["expected_visible_under_schema"]
            row[f"{question_id}_validation_flags"] = record["validation_flags"]
            if question_id in NUMERIC_VARIABLES:
                row[f"{question_id}_numeric"] = record["numeric_value"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)


def summarize_questions(clean_long_df: pd.DataFrame, schema: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    missingness_rows: list[dict[str, Any]] = []

    n_participants = clean_long_df["participant_id"].nunique()

    for question in schema:
        question_id = question["id"]
        q_df = clean_long_df[clean_long_df["question_id"] == question_id].copy()
        valid_df = q_df[q_df["missingness_label"] == "VALID"].copy()
        analysis_type = variable_analysis_type(question_id)
        validation_issue_count = int(q_df["validation_flags"].astype(str).str.contains("present_despite_show_if|blank_despite_show_if").sum())

        summary_row = {
            "question_id": question_id,
            "question_prompt": question.get("prompt", ""),
            "variable_group": QUESTION_GROUPS.get(question_id, ""),
            "variable_type": analysis_type,
            "n_participants": n_participants,
            "n_valid": int((q_df["missingness_label"] == "VALID").sum()),
            "n_missing": int((q_df["missingness_label"] == "MISSING").sum()),
            "n_not_applicable": int((q_df["missingness_label"] == "NOT_APPLICABLE").sum()),
            "n_invalid": int((q_df["missingness_label"] == "INVALID").sum()),
            "n_blank": int((q_df["missingness_label"] == "BLANK").sum()),
            "n_unknown": int((q_df["missingness_label"] == "UNKNOWN").sum()),
            "n_unique_harmonized_categories": int(valid_df["harmonized_value"].nunique(dropna=True)),
            "notes": "",
        }

        notes: list[str] = []
        if validation_issue_count:
            notes.append(f"{validation_issue_count} response(s) present despite current show_if logic")

        if analysis_type == "numeric":
            numeric_series = pd.to_numeric(valid_df["numeric_value"], errors="coerce").dropna()
            if not numeric_series.empty:
                summary_row.update(
                    {
                        "min": float(numeric_series.min()),
                        "max": float(numeric_series.max()),
                        "mean": float(numeric_series.mean()),
                        "std": float(numeric_series.std(ddof=1)) if len(numeric_series) > 1 else 0.0,
                        "median": float(numeric_series.median()),
                        "q1": float(numeric_series.quantile(0.25)),
                        "q3": float(numeric_series.quantile(0.75)),
                    }
                )
                counts = valid_df["harmonized_value"].value_counts().sort_index()
                for category, count in counts.items():
                    count_rows.append(
                        {
                            "question_id": question_id,
                            "harmonized_category": category,
                            "count": int(count),
                            "percentage": round((count / len(valid_df)) * 100, 2),
                            "percentage_of_all_participants": round((count / n_participants) * 100, 2),
                        }
                    )
        else:
            counts = valid_df["harmonized_value"].value_counts(dropna=False)
            for category, count in counts.items():
                count_rows.append(
                    {
                        "question_id": question_id,
                        "harmonized_category": category,
                        "count": int(count),
                        "percentage": round((count / len(valid_df)) * 100, 2) if len(valid_df) else 0.0,
                        "percentage_of_all_participants": round((count / n_participants) * 100, 2),
                    }
                )
            unique_raw_forms = valid_df["raw_value"].astype(str).nunique(dropna=True)
            notes.append(f"{unique_raw_forms} unique raw form(s) among valid responses")

        summary_row["notes"] = " | ".join(notes)
        summary_rows.append(summary_row)

        for label in MISSINGNESS_LABELS:
            count = int((q_df["missingness_label"] == label).sum())
            missingness_rows.append(
                {
                    "question_id": question_id,
                    "missingness_label": label,
                    "count": count,
                    "percentage": round((count / n_participants) * 100, 2),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("question_id").reset_index(drop=True)
    counts_df = pd.DataFrame(count_rows).sort_values(["question_id", "count", "harmonized_category"], ascending=[True, False, True]).reset_index(drop=True)
    missingness_df = pd.DataFrame(missingness_rows).sort_values(["question_id", "missingness_label"]).reset_index(drop=True)
    return summary_df, counts_df, missingness_df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


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


def plot_numeric(question_id: str, question_prompt: str, valid_df: pd.DataFrame) -> Path:
    figure_path = FIGURES_DIR / f"{question_id}_histogram.png"
    numeric_series = pd.to_numeric(valid_df["numeric_value"], errors="coerce").dropna()

    plt.figure(figsize=(8, 5))
    if numeric_series.empty:
        plt.text(0.5, 0.5, "No valid values", ha="center", va="center", fontsize=12)
        plt.axis("off")
    else:
        bins = min(10, max(5, len(numeric_series.unique())))
        plt.hist(numeric_series, bins=bins, color="#2E5EAA", edgecolor="white")
        plt.xlabel(question_id)
        plt.ylabel("Count")
        plt.title(question_prompt)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    return figure_path


def plot_categorical(question_id: str, question_prompt: str, valid_df: pd.DataFrame, analysis_type: str) -> Path:
    figure_path = FIGURES_DIR / f"{question_id}_barplot.png"
    counts = valid_df["harmonized_value"].value_counts().head(10)

    plt.figure(figsize=(9, 5))
    if counts.empty:
        plt.text(0.5, 0.5, "No valid values", ha="center", va="center", fontsize=12)
        plt.axis("off")
    else:
        color = "#3C8D5A" if analysis_type == "categorical" else "#8A5A44"
        counts.iloc[::-1].plot(kind="barh", color=color)
        plt.xlabel("Count")
        plt.ylabel("Response")
        plt.title(question_prompt)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    return figure_path


def generate_plots(clean_long_df: pd.DataFrame, schema: list[dict[str, Any]]) -> dict[str, Path]:
    figure_paths: dict[str, Path] = {}
    for question in schema:
        question_id = question["id"]
        q_df = clean_long_df[clean_long_df["question_id"] == question_id]
        valid_df = q_df[q_df["missingness_label"] == "VALID"]
        analysis_type = variable_analysis_type(question_id)
        if analysis_type == "numeric":
            figure_paths[question_id] = plot_numeric(question_id, question.get("prompt", question_id), valid_df)
        else:
            figure_paths[question_id] = plot_categorical(question_id, question.get("prompt", question_id), valid_df, analysis_type)
    return figure_paths


def make_latex_table(rows: list[tuple[str, Any]], caption: str | None = None) -> str:
    lines = [r"\begin{tabular}{p{0.28\linewidth}p{0.62\linewidth}}", r"\toprule", r"Metric & Value\\", r"\midrule"]
    for key, value in rows:
        lines.append(f"{sanitize_for_tex(key)} & {sanitize_for_tex(value)}\\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        return "\n".join([r"\begin{table}[htbp]", r"\centering", "\n".join(lines), f"\\caption{{{sanitize_for_tex(caption)}}}", r"\end{table}"])
    return "\n".join(lines)


def write_question_reports(
    clean_long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    missingness_df: pd.DataFrame,
    schema: list[dict[str, Any]],
    figure_paths: dict[str, Path],
) -> None:
    summary_map = summary_df.set_index("question_id").to_dict(orient="index")

    for question in schema:
        question_id = question["id"]
        analysis_type = variable_analysis_type(question_id)
        summary = summary_map[question_id]
        valid_df = clean_long_df[(clean_long_df["question_id"] == question_id) & (clean_long_df["missingness_label"] == "VALID")]
        top_counts = counts_df[counts_df["question_id"] == question_id].head(10)
        q_missingness = missingness_df[missingness_df["question_id"] == question_id]
        figure_rel = Path("../../figures/step1") / figure_paths[question_id].name

        summary_rows = [
            ("Group", QUESTION_GROUPS.get(question_id, "")),
            ("Prompt", question.get("prompt", "")),
            ("Type", analysis_type),
            ("Participants", summary["n_participants"]),
            ("Valid responses", summary["n_valid"]),
            ("Missing", summary["n_missing"]),
            ("Not applicable", summary["n_not_applicable"]),
            ("Invalid", summary["n_invalid"]),
            ("Blank", summary.get("n_blank", 0)),
            ("Unknown", summary.get("n_unknown", 0)),
        ]
        if analysis_type == "numeric":
            summary_rows.extend(
                [
                    ("Minimum", summary.get("min", "")),
                    ("Maximum", summary.get("max", "")),
                    ("Mean", summary.get("mean", "")),
                    ("Standard deviation", summary.get("std", "")),
                    ("Median", summary.get("median", "")),
                    ("Q1", summary.get("q1", "")),
                    ("Q3", summary.get("q3", "")),
                ]
            )

        top_lines = []
        if not top_counts.empty:
            top_lines.append(r"\begin{tabular}{p{0.56\linewidth}rr}")
            top_lines.append(r"\toprule")
            top_lines.append(r"Category & Count & \% valid\\")
            top_lines.append(r"\midrule")
            for row in top_counts.itertuples(index=False):
                top_lines.append(
                    f"{sanitize_for_tex(row.harmonized_category)} & {row.count} & {row.percentage}\\\\"
                )
            top_lines.append(r"\bottomrule")
            top_lines.append(r"\end{tabular}")
        else:
            top_lines.append("No valid responses available.")

        missing_lines = [r"\begin{tabular}{p{0.40\linewidth}rr}", r"\toprule", r"Missingness & Count & \% all participants\\", r"\midrule"]
        for row in q_missingness.itertuples(index=False):
            missing_lines.append(f"{sanitize_for_tex(row.missingness_label)} & {row.count} & {row.percentage}\\\\")
        missing_lines.extend([r"\bottomrule", r"\end{tabular}"])

        note_line = summary.get("notes", "")
        if not note_line:
            note_line = "No additional notes."

        content = "\n".join(
            [
                f"\\subsection*{{{sanitize_for_tex(question_id)}}}",
                make_latex_table(summary_rows),
                "",
                r"\paragraph{Top harmonized responses}",
                "\n".join(top_lines),
                "",
                r"\paragraph{Missingness profile}",
                "\n".join(missing_lines),
                "",
                f"\\paragraph{{Notes}} {sanitize_for_tex(note_line)}",
                "",
                r"\begin{figure}[htbp]",
                r"\centering",
                f"\\includegraphics[width=0.82\\linewidth]{{{sanitize_for_tex(str(figure_rel))}}}",
                f"\\caption{{{sanitize_for_tex(question.get('prompt', question_id))}}}",
                r"\end{figure}",
                "",
            ]
        )

        output_path = QUESTIONS_DIR / f"question_{question_id}.tex"
        output_path.write_text(content, encoding="utf-8")


def write_master_report(schema: list[dict[str, Any]], clean_long_df: pd.DataFrame, exclusions: dict[str, dict[str, str]]) -> None:
    group_order = [
        "Core identification and mobility variables",
        "Gender and language background",
        "Current academic or professional status",
        "Educational background",
        "Family educational background and living conditions",
    ]
    overview_rows = [
        ("Participants included in Step 1 summaries", int(clean_long_df["participant_id"].nunique())),
        ("Questions summarized", len(schema)),
        ("Globally excluded participants", len(exclusions)),
    ]

    questions_by_group: dict[str, list[str]] = defaultdict(list)
    for question in schema:
        questions_by_group[QUESTION_GROUPS.get(question["id"], "")].append(question["id"])

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\title{Step 1 Demographics Report}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Overview}",
        r"This report was generated automatically from the Step 1 demographic harmonization pipeline.",
        make_latex_table(overview_rows),
    ]

    if exclusions:
        lines.extend(
            [
                r"\section*{Global Participant Exclusions}",
                r"The following participant IDs were removed from all Step~1 analysis tables, counts, and plots after Step~3 review. Their files remain on disk for audit only, and future downstream steps should continue to honor this exclusion list.",
                make_latex_table(exclusion_table_rows(exclusions)),
            ]
        )

    for group in group_order:
        lines.append(f"\\section*{{{sanitize_for_tex(group)}}}")
        for question_id in questions_by_group.get(group, []):
            lines.append(f"\\input{{questions/question_{question_id}.tex}}")

    lines.append(r"\end{document}")
    (REPORTS_DIR / "step1_demographics_report.tex").write_text("\n".join(lines), encoding="utf-8")


def compile_report() -> None:
    report_path = REPORTS_DIR / "step1_demographics_report.tex"
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
    schema = load_schema()
    schema_question_ids = [question["id"] for question in schema]

    manifest_df = build_manifest(schema_question_ids)
    manifest_df = manifest_df[~manifest_df["participant_id"].isin(excluded_ids)].reset_index(drop=True)
    registry_df = build_participant_registry(manifest_df)
    schema_df = build_schema_table(schema)
    raw_wide_df, raw_long_df, participant_payloads = build_raw_tables(registry_df, schema)
    clean_long_df, harmonization_df = validate_and_harmonize(participant_payloads, schema)
    clean_wide_df = build_clean_wide(clean_long_df, schema)
    summary_df, counts_df, missingness_df = summarize_questions(clean_long_df, schema)
    figure_paths = generate_plots(clean_long_df, schema)
    write_question_reports(clean_long_df, summary_df, counts_df, missingness_df, schema, figure_paths)
    write_master_report(schema, clean_long_df, exclusions)
    compile_report()

    save_dataframe(manifest_df, INTERMEDIATE_DIR / "01_file_manifest.csv")
    save_dataframe(registry_df, INTERMEDIATE_DIR / "02_participant_registry.csv")
    save_dataframe(schema_df, INTERMEDIATE_DIR / "03_demographics_schema.csv")
    save_dataframe(raw_wide_df, INTERMEDIATE_DIR / "04_socio_raw_wide.csv")
    save_dataframe(raw_long_df, INTERMEDIATE_DIR / "05_socio_raw_long.csv")
    save_dataframe(harmonization_df, INTERMEDIATE_DIR / "06_socio_harmonization_dictionary.csv")
    save_dataframe(clean_wide_df, CLEAN_DIR / "07_socio_clean_wide.csv")
    save_dataframe(summary_df, CLEAN_DIR / "08_question_summary.csv")
    save_dataframe(counts_df, CLEAN_DIR / "09_question_level_counts.csv")
    save_dataframe(missingness_df, CLEAN_DIR / "10_question_missingness.csv")

    print("Step 1 outputs generated successfully.")
    print(f"Participants: {registry_df['participant_id'].nunique()}")
    print(f"Excluded participants: {len(exclusions)}")
    print(f"Manifest rows: {len(manifest_df)}")
    print(f"Figures: {len(figure_paths)}")


if __name__ == "__main__":
    main()
