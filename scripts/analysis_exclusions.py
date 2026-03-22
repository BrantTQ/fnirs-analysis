#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXCLUSION_PATH = ROOT / "materials" / "analysis_participant_exclusions.json"


def normalize_pid(raw: str | None) -> str | None:
    if not raw:
        return None
    match = re.search(r"pid\s*0*(\d+)", raw, flags=re.IGNORECASE)
    if not match:
        return None
    return f"PID{int(match.group(1)):03d}"


def load_participant_exclusions() -> dict[str, dict[str, str]]:
    if not EXCLUSION_PATH.exists():
        return {}

    with EXCLUSION_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    exclusions: dict[str, dict[str, str]] = {}
    for item in payload.get("participants", []):
        participant_id = normalize_pid(str(item.get("participant_id", "")))
        if not participant_id:
            continue
        exclusions[participant_id] = {
            "participant_id": participant_id,
            "scope": str(item.get("scope", "")).strip(),
            "effective_date": str(item.get("effective_date", "")).strip(),
            "reason_short": str(item.get("reason_short", "")).strip(),
            "reason_detail": str(item.get("reason_detail", "")).strip(),
        }

    return dict(sorted(exclusions.items()))


def excluded_participant_ids(exclusions: dict[str, dict[str, str]]) -> set[str]:
    return set(exclusions)


def participant_exclusion_reason(exclusions: dict[str, dict[str, str]], participant_id: str) -> str:
    info = exclusions.get(participant_id, {})
    reason_short = info.get("reason_short", "").strip()
    reason_detail = info.get("reason_detail", "").strip()
    if reason_short and reason_detail and reason_short not in reason_detail:
        return f"{reason_short}: {reason_detail}"
    return reason_detail or reason_short


def exclusion_table_rows(exclusions: dict[str, dict[str, str]]) -> list[tuple[str, Any]]:
    return [(participant_id, participant_exclusion_reason(exclusions, participant_id)) for participant_id in sorted(exclusions)]
