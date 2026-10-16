"""審閱閘門遇到遺漏、改檔與假順序必須拒絕；此處全是虛構測試資料。"""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("incremental_review", ROOT / "docs/review-tools/incremental_revision.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def save(root, name, value, raw=False):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if raw else json.dumps(value).encode()
    path.write_bytes(data)
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}


def session(root, monkeypatch):
    # 單位切法由獨立fixture指定；測試保存完整性，而非複製分段演算法。
    monkeypatch.setattr(review, "module", lambda *_: SimpleNamespace(units=lambda _: ["first", "second"]))
    source = save(root, "source.md", b"fictional source", raw=True)
    opening = save(
        root,
        "opening.json",
        {
            "page_id": "fixture",
            "reviewer_task": "/root/fixture_reader",
            "source_sha256": source["sha256"],
            "figures_sha256": {},
            "started_at": "2026-10-16T01:00:00+00:00",
        },
    )
    notes = []
    for i, unit in enumerate(("first", "second")):
        note = {
            "page_id": "fixture",
            "reviewer_task": "/root/fixture_reader",
            "unit_index": i,
            "unit_sha256": hashlib.sha256(unit.encode()).hexdigest(),
            "saved_at": f"2026-10-16T01:00:0{i + 1}+00:00",
            **{
                name: "fictional fixture"
                for name in (
                    "understanding",
                    "materials_and_labels",
                    "expected_change",
                    "confusion_and_quote",
                    "missing_visuals",
                )
            },
            "concept_checks": [
                {
                    name: "fictional fixture"
                    for name in (
                        "concept",
                        "identity",
                        "mechanism",
                        "need",
                        "usage",
                        "example_scope",
                        "source_quote",
                        "verdict",
                    )
                }
            ],
        }
        notes.append(save(root, f"note-{i}.json", note))
    return {"opening": opening, "source": source, "notes": notes, "figures": {}}


def test_artifact_tamper_and_escape_are_rejected(tmp_path):
    ref = save(tmp_path, "proof.txt", b"original", raw=True)
    (tmp_path / "proof.txt").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        review.artifact(tmp_path, ref)
    with pytest.raises(ValueError, match="escapes"):
        review.local(tmp_path, "../outside.txt")


def test_complete_fixture_preserves_page_and_order(tmp_path, monkeypatch):
    proof = session(tmp_path, monkeypatch)
    page, completed = review.reader_session(tmp_path, proof, "/root/fixture_reader")
    assert page == "fixture"
    assert completed.second == 2


@pytest.mark.parametrize(
    "attack", ["missing_note", "future_time", "wrong_unit", "missing_need", "wrong_owner", "missing_figure"]
)
def test_reader_trace_rejects_incomplete_or_inconsistent_evidence(tmp_path, monkeypatch, attack):
    proof = session(tmp_path, monkeypatch)
    if attack == "missing_note":
        proof["notes"].pop()
    elif attack == "missing_figure":
        opening = json.loads((tmp_path / "opening.json").read_bytes())
        opening["figures_sha256"] = {"figure.svg": "0" * 64}
        proof["opening"] = save(tmp_path, "opening.json", opening)
    else:
        note = json.loads((tmp_path / "note-1.json").read_bytes())
        if attack == "future_time":
            note["saved_at"] = "2026-10-16T00:59:59+00:00"
        elif attack == "wrong_unit":
            note["unit_index"] = 0
        elif attack == "missing_need":
            del note["concept_checks"][0]["need"]
        elif attack == "wrong_owner":
            note["reviewer_task"] = "/root/other_fixture"
        proof["notes"][1] = save(tmp_path, "note-1.json", note)
    with pytest.raises(ValueError):
        review.reader_session(tmp_path, proof, "/root/fixture_reader")
