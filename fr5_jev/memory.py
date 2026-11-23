"""Small, inspectable case retrieval. Cases are context, never execution permits."""
import json
from pathlib import Path

CASE_DIRECTORY = Path(__file__).resolve().parent / "decisions/cases"


def retrieve(tags, *, robot_id="J1", directory=None, limit=3):
    requested = set(tags)
    matches = []
    for path in sorted(Path(directory or CASE_DIRECTORY).glob("*.json")):
        case = json.loads(path.read_text())
        if case.get("robot_id") != robot_id:
            continue
        overlap = requested & set(case.get("tags", []))
        if not overlap:
            continue
        matches.append((len(overlap), path.name, {
            "case_id": case["case_id"], "matched_tags": sorted(overlap),
            "observed_outcome": case["observed_outcome"],
            "applicability": case["applicability"], "limitations": case["limitations"],
            "evidence": case["evidence"], "grants_execution_permission": False,
        }))
    matches.sort(key=lambda value: (-value[0], value[1]))
    return [case for _, _, case in matches[:limit]]
