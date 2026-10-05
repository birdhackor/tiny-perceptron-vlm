"""Apply a coordinator text correction after the original reviewer's real revise.

This is writer/version evidence, not independent factual-review evidence.
"""
import datetime
import hashlib
import json
import platform
import re
import shutil
import sys
from pathlib import Path

root = Path.cwd()
source = root / "course/chapters/14.md"
report_path = root / "docs/technical-reviews/14.1.json"
progress_path = root / "docs/course-revision-20261005/factual-review-progress.json"
report_bytes = report_path.read_bytes()
report = json.loads(report_bytes)
sha = lambda value: hashlib.sha256(value).hexdigest()
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_14_1"
assert report["verdict"] == "revise"
assert sha(report_bytes) == "38beede75b3dbf798d8a1325d7ada6513011dc3de2eb9c7de5a970f71bab51bb"

def sections(raw):
    headings = list(re.finditer(rb"(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+", raw))
    return raw[:headings[0].start()], {
        h[1].decode(): raw[h.start():headings[i + 1].start() if i + 1 < len(headings) else len(raw)]
        for i, h in enumerate(headings)
    }

before = source.read_bytes()
intro_before, sections_before = sections(before)
assert sha(sections_before["14.1"]) == report["source_sha256"]
old = "下圖的虛線圓標出".encode()
new = "下圖的圓標出".encode()
assert sections_before["14.1"].count(old) == before.count(old) == 1
after = before.replace(old, new, 1)
intro_after, sections_after = sections(after)
assert intro_before == intro_after
assert sections_before.keys() == sections_after.keys()
assert all(sections_before[key] == sections_after[key] for key in sections_before if key != "14.1")

history = root / "docs/technical-reviews/history" / (
    "phase4-14_1-own-initial-revise-" + sha(report_bytes) + ".json"
)
assert not history.exists()
shutil.copyfile(report_path, history)
assert history.read_bytes() == report_bytes
source.write_bytes(after)
assert source.read_bytes() == after
now = datetime.datetime.now(datetime.UTC).isoformat()
receipt = {
    "recorded_at": now,
    "role": "coordinator writer/version evidence only",
    "command": [sys.executable, *sys.argv],
    "python": platform.python_version(),
    "source": "course/chapters/14.md#14.1",
    "old_phrase": old.decode(),
    "new_phrase": new.decode(),
    "old_source_sha256": sha(sections_before["14.1"]),
    "new_source_sha256": sha(sections_after["14.1"]),
    "old_whole_frozen_input_sha256": sha(before),
    "new_whole_file_sha256": sha(after),
    "intro_sha256": sha(intro_before),
    "other_sections_byte_unchanged": list(k for k in sections_before if k != "14.1"),
    "initial_report_history_path": history.relative_to(root).as_posix(),
    "initial_report_history_sha256": sha(report_bytes),
    "unchanged": ["introduction", "all other section bytes", "figures", "fences", "implementation", "raw measurements"],
}
artifact_dir = Path(__file__).resolve().parent
receipt_path = artifact_dir / "revision.stdout.json"
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
progress = json.loads(progress_path.read_bytes())
record = progress["records"]["14.1"]
record["status"] = "recheck_queued"
record["events"].append({
    "at": now,
    "event": "coordinator_localized_revision",
    **receipt,
    "writer_artifacts": [
        {"path": path.relative_to(root).as_posix(), "sha256": sha(path.read_bytes())}
        for path in [Path(__file__).resolve(), receipt_path]
    ],
    "note": "Only remove the visually incorrect dashed-line adjective after the original owner's formal revise; original technical and reader actual rechecks still required.",
})
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
main_path = root / "docs/course-revision-20261005/review-progress.json"
main = json.loads(main_path.read_bytes())
main["records"]["14.1"]["status"] = "recheck_queued"
main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False))
