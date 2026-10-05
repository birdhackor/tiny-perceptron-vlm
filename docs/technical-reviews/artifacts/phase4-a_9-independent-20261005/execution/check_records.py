"""Bounded independent oracle for A.9's paper records; never loads a model."""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
prerequisite = (BASE / "inputs/A.8.prerequisite.md").read_bytes()
section = (BASE / "inputs/section.md").read_bytes()

def table_records(raw: bytes) -> dict[str, str]:
    return dict(re.findall(r"^\| (T|U\d+) \| (.+) \|$", raw.decode("utf-8"), re.M))

records = table_records(prerequisite)
current = table_records(section)
assert list(records) == ["T", "U1", "U2", "U3", "U4", "U5", "U6", "U7"]
assert len(current) == 4
assert current == {key: records[key] for key in ("T", "U1", "U2", "U3")}

def associations(rows: dict[str, str]):
    clubs, floors = {}, {}
    for label, text in rows.items():
        if m := re.fullmatch(r"(.+)的活動室代碼是 ([A-Z]\d\d)。", text):
            clubs[m[1]] = (m[2], label)
        elif m := re.fullmatch(r"([A-Z]\d\d) 在(.+樓)。", text):
            floors[m[1]] = (m[2], label)
        else:
            raise AssertionError((label, text))
    return clubs, floors

def answers(rows: dict[str, str]):
    clubs, floors = associations(rows)
    requested = ("攝影社", "棋藝社", "合唱社")
    code = clubs.get("攝影社", (None, None))[0]
    return {
        "single": code,
        "three": [clubs.get(club, (None, None))[0] for club in requested],
        "floor": floors.get(code, (None, None))[0],
        "floor_support_labels": [clubs.get("攝影社", (None, None))[1], floors.get(code, (None, None))[1]],
    }

def list_criterion(candidate, expected):
    missing = [value for value in expected if value not in candidate]
    extras = [value for value in candidate if value not in expected]
    duplicates = len(candidate) != len(set(candidate))
    return {
        "expected_item_denominator": len(expected),
        "correct_requested_items": len(set(candidate) & set(expected)),
        "missing": missing,
        "extras": extras,
        "duplicates": duplicates,
        "ordered_complete_answer": candidate == expected,
    }

observed = answers(records)
assert observed == {
    "single": "M17", "three": ["M17", "B04", "C26"],
    "floor": "二樓", "floor_support_labels": ["T", "U3"],
}
clubs, floors = associations(records)
assert clubs["棋藝社"] == ("B04", "U1")
assert floors["B04"] == ("一樓", "U4")
assert clubs["美術社"] == ("E31", "U6")
expected = observed["three"]
candidates = {
    "correct": ["M17", "B04", "C26"],
    "text_omission": ["M17", "B04"],
    "text_extra": ["M17", "B04", "C26", "E31"],
    "swap_associations_or_order": ["M17", "C26", "B04"],
    "duplicate": ["M17", "B04", "C26", "C26"],
    "wrong_code": ["M17", "B04", "E31"],
}
criteria = {name: list_criterion(candidate, expected) for name, candidate in candidates.items()}
assert criteria["correct"]["ordered_complete_answer"]
assert all(not result["ordered_complete_answer"] for name, result in criteria.items() if name != "correct")
assert criteria["text_omission"]["correct_requested_items"] == 2
assert criteria["text_omission"]["missing"] == ["C26"]
assert criteria["text_extra"]["correct_requested_items"] == 3
assert criteria["text_extra"]["extras"] == ["E31"]
assert criteria["swap_associations_or_order"]["correct_requested_items"] == 3

# The short variants retain precisely the information demanded by each task.
short = {
    "single": answers({key: records[key] for key in ("T",)}),
    "three": answers({key: records[key] for key in ("T", "U1", "U2")}),
    "floor": answers({key: records[key] for key in ("T", "U3")}),
}
assert short["single"]["single"] == observed["single"]
assert short["three"]["three"] == observed["three"]
assert short["floor"]["floor"] == observed["floor"]
no_u3 = answers({key: value for key, value in records.items() if key != "U3"})
assert no_u3["single"] == "M17"
assert no_u3["three"] == expected
assert no_u3["floor"] is None
no_t = answers({key: value for key, value in records.items() if key != "T"})
assert no_t["floor"] is None

# Check relevant relation placements without claiming any tokenizer or model test.
others = [key for key in records if key not in ("T", "U3")]
placement_count = 0
for t_pos, u3_pos in itertools.permutations(range(8), 2):
    order = [None] * 8
    order[t_pos] = "T"
    order[u3_pos] = "U3"
    remainder = iter(others)
    order = [next(remainder) if key is None else key for key in order]
    assert answers({key: records[key] for key in order}) == observed
    placement_count += 1
assert placement_count == 8 * 7

result = {
    "purpose": "Verify current manuscript paper-answer logic and counterexamples, not model capability, tokenizer length, or neural computation.",
    "source_sha256": hashlib.sha256(section).hexdigest(),
    "prerequisite_sha256": hashlib.sha256(prerequisite).hexdigest(),
    "records": records,
    "record_counts": {"total": len(records), "club_code_relations": len(clubs), "code_floor_relations": len(floors), "A9_displayed": len(current)},
    "answers": observed,
    "denominators": {"single_expected_output_items": 1, "three_expected_output_items": 3, "floor_expected_output_items": 1, "floor_required_support_records": 2, "material_records": 8, "model_evaluations": 0, "training_steps": 0},
    "candidate_criteria": criteria,
    "short_variants": short,
    "missing_U3_variant": no_u3,
    "missing_T_variant": no_t,
    "relation_placement_cases": placement_count,
    "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "device": "CPU; standard-library-only; no model loaded"},
}
print(json.dumps(result, ensure_ascii=False, indent=2))
