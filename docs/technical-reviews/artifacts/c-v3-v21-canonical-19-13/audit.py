"""19.13 的小型來源身分核對；不執行模型、訓練或 final。"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("audit.json")
TASK = "/root/c_v3_technical_coord/canonical_19_13"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return (ROOT / path).read_bytes()


def section(path, lesson):
    body = read(path).decode()
    match = re.search(r"^## " + re.escape(lesson) + r"(?:\s|$)", body, re.M)
    assert match, (path, lesson)
    rest = body[match.start() :]
    end = re.search(r"^## ", rest[3:], re.M)
    return rest if end is None else rest[: end.start() + 3]


def main():
    started = datetime.now(UTC).isoformat()
    freeze_path = "docs/course-revision-20261101-c-v3/source-freeze-v21.json"
    old_freeze_path = "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18.json"
    freeze = json.loads(read(freeze_path))
    old_freeze = json.loads(read(old_freeze_path))
    current = next(row for row in freeze["pages"] if row["page_id"] == "19.13")
    previous = next(row for row in old_freeze["pages"] if row["page_id"] == "19.13")
    source = read(current["snapshot"])
    old_source = read(previous["snapshot"])
    addition = "本節先看原來三個成品的用途；現行明確裝置決策已由[19.25](19.md#19.25)的合格C-v3承接，下面的客服與B Auto仍保留自己的有限介面與成績。\n\n".encode()
    assert source.count(addition) == 1
    assert source.replace(addition, b"", 1) == old_source
    assert digest(source) == current["source_sha256"]
    assert digest(old_source) == previous["source_sha256"]
    assert section("course/chapters/19.md", "19.13").encode() == source
    links = []
    for label, href in re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", source.decode()):
        path, _, anchor = href.partition("#")
        target = ROOT / "course/chapters" / path
        text = target.read_text()
        heading = next(line for line in text.splitlines() if re.match(r"^## " + re.escape(anchor) + r"(?:\s|$)", line))
        links.append({"label": label, "href": href, "target": target.relative_to(ROOT).as_posix(), "heading": heading})
    descriptor = json.loads(read("docs/decision-product/v3/public-release.json"))
    acquisition = json.loads(read("docs/decision-product/v3/metadata/acquisition.json"))
    archive = Path("/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip")
    archive_bytes = archive.read_bytes()
    assert len(archive_bytes) == descriptor["bytes"]
    assert digest(archive_bytes) == descriptor["sha256"]
    member_checks = []
    with zipfile.ZipFile(archive) as saved:
        for item in acquisition["files"]:
            data = saved.read(item["archive_member"])
            assert data == read(item["path"])
            assert digest(data) == item["sha256"]
            member_checks.append({"path": item["path"], "archive_member": item["archive_member"], "bytes": len(data), "sha256": digest(data), "equal": True})
        bundle_checks = {}
        for name, expected in descriptor["bundle"].items():
            data = saved.read("tiny-perceptron-c-v3/bundle/" + name)
            assert digest(data) == expected
            bundle_checks[name] = {"bytes": len(data), "sha256": digest(data)}
    opening = json.loads(read("docs/decision-product/v3/metadata/final-opening.json"))
    current_implementation = {}
    for path, expected in descriptor["implementation"].items():
        actual = digest(read(path))
        assert actual == expected == opening["implementation"][Path(path).name]
        current_implementation[path] = actual
    shared_path = "docs/technical-reviews/artifacts/c-v3-v21-numbered-r2/execution.json"
    shared = json.loads(read(shared_path))
    assert digest(read(shared_path)) == "375deeebc7d2107171fd38ed80ffbe127823b7865640ca3118a8fa479c6b1959"
    assert shared["archive"]["sha256"] == digest(archive_bytes)
    example_fingerprints = {}
    for example in shared["new_cpu_examples"]:
        path = "examples/decision_product/" + example["request"]
        example_fingerprints[path] = digest(read(path))
        assert example["returncode"] == 0
        assert example["response"]["product"] == "C-v3"
        assert example["response"]["dispatch"] == "typed-devices"
    # 透明手算的事件語意；這不是神經模型的一次測試。
    trace = [{"A": 2, "B": 6}, {"A": 5, "B": 6}, {"A": 1, "B": 6}]
    result = {
        "schema": "canonical-19.13-small-identity-audit-v1",
        "reviewer_task": TASK,
        "started_at": started,
        "completed_at": datetime.now(UTC).isoformat(),
        "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-19-13/audit.py",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "device": "cpu; identity-only, no model inference"},
        "freeze": {"path": freeze_path, "sha256": digest(read(freeze_path))},
        "source": {"snapshot": current["snapshot"], "sha256": digest(source), "matches_current_section": True, "figure_sha256": {}},
        "baseline": {"freeze": old_freeze_path, "freeze_sha256": digest(read(old_freeze_path)), "snapshot": previous["snapshot"], "sha256": digest(old_source), "delta": addition.decode(), "remaining_bytes_identical": True},
        "links": links,
        "archive": {"path": str(archive), "bytes": len(archive_bytes), "sha256": digest(archive_bytes), "metadata_members": member_checks, "bundle": bundle_checks},
        "implementation": current_implementation,
        "examples": example_fingerprints,
        "shared_raw_execution": {"path": shared_path, "sha256": digest(read(shared_path)), "new_cpu_example_count": len(shared["new_cpu_examples"]), "own_new_model_execution": False},
        "box_event_derivation": {"initial": trace[0], "operations": ["A := 5", "A := 1"], "states": trace, "query": "A", "expected": 1, "observed": trace[-1]["A"], "tolerance": "exact integer equality", "supports": "event semantics only; not learned-model accuracy"},
        "limits": ["No old review, R answer, peer verdict or history issue list read before original.", "Source identity permits a retention request but does not establish retained training, original-source or measurement facts.", "No training, final opening/inference, installation, network download, publication or commit.", "Metadata is inspected as repository source, not relabelled as an execution.", "The shared execution proves only its recorded bounded CPU examples and retained arithmetic; public transport was not retested."]
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": OUT.relative_to(ROOT).as_posix(), "sha256": digest(OUT.read_bytes()), "source_identical_after_delta": True, "metadata_member_count": len(member_checks), "implementation_count": len(current_implementation), "linked_anchors": len(links)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
