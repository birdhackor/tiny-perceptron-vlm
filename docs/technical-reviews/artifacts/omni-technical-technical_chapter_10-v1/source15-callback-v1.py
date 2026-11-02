"""Same-owner source15 callback evidence for the chapter-10 introduction only."""

import argparse
import contextlib
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import platform
import re

import torch


ROOT = Path(__file__).resolve().parents[4]
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json"
ORIGINAL = "docs/course-revision-20261029-omni/reviews/technical/technical_chapter_10/technical-original-v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(raw, identifier):
    if identifier == "chapter-10":
        return raw[: raw.index("## ")]
    start = re.search(r"^## " + re.escape(identifier) + r" .+$", raw, re.M)
    stop = re.search(r"^## ", raw[start.end() :], re.M)
    end = start.end() + stop.start() if stop else len(raw)
    return raw[start.start() : end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert sha(ROOT / ORIGINAL) == "91be5582e60649d4ce4167a42a2c2e7ab4c2b36e3c5326a0f8c24f5e93d02974"
    assert sha(ROOT / FREEZE) == "1094ea8f343a414dc3cbe5401b5753641fdb467d17a4867ef87dc192fe412238"
    original = json.loads((ROOT / ORIGINAL).read_bytes())
    freeze = json.loads((ROOT / FREEZE).read_bytes())
    old_freeze = json.loads((ROOT / original["source_freeze"]["path"]).read_bytes())
    current = (ROOT / "course/chapters/10.md").read_text()
    receipts, slices = [], {}
    identifiers = ["chapter-10", "10.1", "10.2", "10.5", "10.7", "10.8", "10.9", "10.10", "10.11", "10.12"]
    for identifier in identifiers:
        page = next(p for p in freeze["pages"] if p["page_id"] == identifier)
        before = next(p for p in old_freeze["pages"] if p["page_id"] == identifier)
        path = ROOT / page["snapshot"]
        assert sha(path) == page["source_sha256"]
        assert hashlib.sha256(select(current, identifier).encode()).hexdigest() == page["source_sha256"]
        if identifier != "chapter-10":
            assert page["source_sha256"] == before["source_sha256"]
        slices[identifier] = path.read_text()
        receipts.append({"page_id": identifier, "snapshot": page["snapshot"], "sha256": page["source_sha256"], "current_slice_matches": True, "unchanged_since_original": page["source_sha256"] == before["source_sha256"]})
    reused = []
    for source in original["sources"]:
        if source["kind"] == "repository_code":
            assert sha(ROOT / source["path"]) == source["sha256"]
            reused.append({"path": source["path"], "sha256": source["sha256"], "unchanged": True})
    for artifact in original["artifacts"]:
        assert sha(ROOT / artifact["path"]) == artifact["sha256"]
        reused.append({"path": artifact["path"], "sha256": artifact["sha256"], "unchanged": True})

    intro = slices["chapter-10"]
    assert "最後三節改問圖文配對" not in intro
    pairing_quote = "[10.9–10.11](10.md#10.9)改問圖文配對：從現成的候選說明裡選出相配的一段，這和自己生成句子是不同的工作。"
    video_quote = "最後在[10.12](10.md#10.12)為影格加上時間戳，分清「第幾張圖」和「那個時刻看到什麼」。"
    assert pairing_quote in intro and video_quote in intro
    assert "候選" in slices["10.9"] and "候選" in slices["10.10"] and "候選" in slices["10.11"]
    assert "frame_ms" in slices["10.12"] and "sample-and-hold" in slices["10.12"]
    page = next(p for p in freeze["pages"] if p["page_id"] == "chapter-10")
    assert page["figures_sha256"] == {}
    assert not re.findall(r"!\[[^\]]*\]\([^)]+\)", intro)

    blocks = re.findall(r"```python\n(.*?)```", slices["10.12"], re.S)
    assert len(blocks) == 1
    namespace = {"__name__": "__chapter10_source15_callback__"}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(blocks[0], "source15:10.12:python-block-1", "exec"), namespace)
    query = namespace["at"]
    frames = namespace["frames"]
    a, a_pixels = query(frames, namespace["times_a"], 4000.0, 1500.0)
    b, b_pixels = query(frames, namespace["times_b"], 4000.0, 1500.0)
    assert a == 1 and b == 3
    assert torch.equal(a_pixels, namespace["circle"][None])
    assert torch.equal(b_pixels, namespace["square"][None])
    assert not torch.equal(a_pixels, b_pixels)
    boundaries = [query(frames, namespace["times_a"], 4000.0, t)[0] for t in [1999.0, 2000.0]]
    assert boundaries == [1, 2]
    assert len(namespace["records"]) == 17
    try:
        query(frames, namespace["times_a"], 4000.0, 4000.0)
    except ValueError as error:
        end_error = str(error)
    else:
        raise AssertionError("clip end must be rejected")

    output = {
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_task": original["reviewer_task"],
        "callback_started_at": "2026-10-29T18:41:05+00:00",
        "assigned_lesson_id": "chapter-10",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "torch_threads": "1"},
        "source_freeze": {"path": FREEZE, "sha256": sha(ROOT / FREEZE)},
        "original_report": {"path": ORIGINAL, "sha256": sha(ROOT / ORIGINAL)},
        "current_source_sha256": page["source_sha256"],
        "source_receipts": receipts,
        "reused_unchanged_evidence": reused,
        "current_quotes": {"pairing": pairing_quote, "video_time": video_quote},
        "chapter_headings": [{"id": identifier, "title": title} for identifier, title in re.findall(r"^## (10\.\d+) (.+)$", current, re.M)],
        "pairing_section_ids": ["10.9", "10.10", "10.11"],
        "final_video_section_id": "10.12",
        "video_example": {"exact_embedded_block": "10.12 python block 1", "stdout": capture.getvalue(), "query_ms": 1500.0, "a_index": a, "b_index": b, "a_pixels_equal_circle": True, "b_pixels_equal_square": True, "a_boundary_indices": boundaries, "clip_end_rejected": end_error, "records": len(namespace["records"])},
        "derivation": "相同圓、圓、方、方序列。sample-and-hold下A最近且不晚於1500的時間為1000→index1→圓；B為1500→index3→方。序號順序相同而查同一時間所得畫面不同；時間戳不可省略。這是明示顯示規則與確定性查詢，不是模型辨識成績。",
        "training_performed": False,
        "new_reader_review_performed": False,
        "scope": "同owner導讀複核；10.x只作導讀範圍與本次新增時間句的必要前提，不代其完整technical簽名。",
    }
    destination = ROOT / args.output
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": args.output, "sha256": sha(destination), "result": "source15 outline closure and timestamp example verified", "source_sha256": page["source_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
