"""Actual coordinator writer/version evidence, not factual-review evidence."""
import datetime
import hashlib
import json
import platform
import re
import shutil
import sys
from pathlib import Path

root = Path.cwd()
source = root / "course/chapters/19.md"
report_path = root / "docs/technical-reviews/19.5.json"
progress_path = root / "docs/course-revision-20261005/factual-review-progress.json"
sha = lambda value: hashlib.sha256(value).hexdigest()
report_bytes = report_path.read_bytes()
report = json.loads(report_bytes)
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_19_5"
assert report["verdict"] == "revise"
assert sha(report_bytes) == "6559424bfc52bea5840a8a026c16ab972358c34e58b8171e8c0f75ffeadf5f5b"


def sections(raw):
    headings = list(re.finditer(rb"(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+", raw))
    return raw[:headings[0].start()], {
        h[1].decode(): raw[h.start():headings[i + 1].start() if i + 1 < len(headings) else len(raw)]
        for i, h in enumerate(headings)
    }


before = source.read_bytes()
intro_before, sections_before = sections(before)
assert sha(sections_before["19.5"]) == report["source_sha256"]
old = "最終文字能力與各項分母統一見[19.12的能力表](19.md#19.12)".encode()
new = "整合成品的能力界定與驗收安排見[19.12](19.md#19.12)".encode()
assert sections_before["19.5"].count(old) == before.count(old) == 1
after = before.replace(old, new, 1)
intro_after, sections_after = sections(after)
assert intro_before == intro_after
assert sections_before.keys() == sections_after.keys()
assert all(sections_before[k] == sections_after[k] for k in sections_before if k != "19.5")
assert re.findall(rb"(?ms)^```.*?^```", sections_before["19.5"]) == re.findall(rb"(?ms)^```.*?^```", sections_after["19.5"])
history = root / "docs/technical-reviews/history" / ("phase4-19_5-own-initial-revise-" + sha(report_bytes) + ".json")
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
    "source": "course/chapters/19.md#19.5",
    "old_phrase": old.decode(),
    "new_phrase": new.decode(),
    "old_source_sha256": sha(sections_before["19.5"]),
    "new_source_sha256": sha(sections_after["19.5"]),
    "old_whole_frozen_input_sha256": sha(before),
    "new_whole_file_sha256": sha(after),
    "intro_sha256": sha(intro_before),
    "other_sections_byte_unchanged": [k for k in sections_before if k != "19.5"],
    "initial_report_history_path": history.relative_to(root).as_posix(),
    "initial_report_history_sha256": sha(report_bytes),
    "unchanged": ["introduction", "all other section bytes including19.12", "figures", "fences", "implementation", "raw measurements", "existing validation links"],
}
receipt_path = Path(__file__).resolve().parent / "revision.stdout.json"
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
progress = json.loads(progress_path.read_bytes())
record = progress["records"]["19.5"]
record["status"] = "recheck_queued"
record["events"].append({
    "at": now,
    "event": "coordinator_localized_revision",
    **receipt,
    "writer_artifacts": [{"path": p.relative_to(root).as_posix(), "sha256": sha(p.read_bytes())} for p in [Path(__file__).resolve(), receipt_path]],
    "note": "Narrow the cross-reference to its actual capability-definition and acceptance-planning scope after original owner formal revise. No unsupported final text scores added to19.12 and no validation records relabelled as final tests. Original technical and full incremental reader rechecks remain required.",
})
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
main_path = root / "docs/course-revision-20261005/review-progress.json"
main = json.loads(main_path.read_bytes())
main["records"]["19.5"]["status"] = "recheck_queued"
main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False))
