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
source = root / "course/chapters/18.md"
report_path = root / "docs/technical-reviews/18.4.json"
progress_path = root / "docs/course-revision-20261005/factual-review-progress.json"
sha = lambda value: hashlib.sha256(value).hexdigest()
report_bytes = report_path.read_bytes()
report = json.loads(report_bytes)
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_18_4"
assert report["verdict"] == "revise"
assert sha(report_bytes) == "68cba459d9632e3667b04dbcab5d7c7445dc21fcb0720c8ebe51b346632c1a35"


def sections(raw):
    headings = list(re.finditer(rb"(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+", raw))
    return raw[:headings[0].start()], {
        h[1].decode(): raw[h.start():headings[i + 1].start() if i + 1 < len(headings) else len(raw)]
        for i, h in enumerate(headings)
    }


before = source.read_bytes()
intro_before, sections_before = sections(before)
assert sha(sections_before["18.4"]) == report["source_sha256"]
old = "若它對3給最高機率，學生會被教著提高錯誤候選".encode()
new = "若教師給3的機率高於學生目前的0.4，沿梯度反方向更新便會提高這個錯誤候選的分數".encode()
assert sections_before["18.4"].count(old) == before.count(old) == 1
after = before.replace(old, new, 1)
intro_after, sections_after = sections(after)
assert intro_before == intro_after
assert sections_before.keys() == sections_after.keys()
assert all(sections_before[k] == sections_after[k] for k in sections_before if k != "18.4")
assert re.findall(rb"(?ms)^```.*?^```", sections_before["18.4"]) == re.findall(rb"(?ms)^```.*?^```", sections_after["18.4"])
history = root / "docs/technical-reviews/history" / ("phase4-18_4-own-initial-revise-" + sha(report_bytes) + ".json")
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
    "source": "course/chapters/18.md#18.4",
    "old_phrase": old.decode(),
    "new_phrase": new.decode(),
    "old_source_sha256": sha(sections_before["18.4"]),
    "new_source_sha256": sha(sections_after["18.4"]),
    "old_whole_frozen_input_sha256": sha(before),
    "new_whole_file_sha256": sha(after),
    "intro_sha256": sha(intro_before),
    "other_sections_byte_unchanged": [k for k in sections_before if k != "18.4"],
    "initial_report_history_path": history.relative_to(root).as_posix(),
    "initial_report_history_sha256": sha(report_bytes),
    "unchanged": ["introduction", "all other section bytes", "figures", "fences", "implementation", "raw measurements"],
}
receipt_path = Path(__file__).resolve().parent / "revision.stdout.json"
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
progress = json.loads(progress_path.read_bytes())
record = progress["records"]["18.4"]
record["status"] = "recheck_queued"
record["events"].append({
    "at": now,
    "event": "coordinator_localized_revision",
    **receipt,
    "writer_artifacts": [{"path": p.relative_to(root).as_posix(), "sha256": sha(p.read_bytes())} for p in [Path(__file__).resolve(), receipt_path]],
    "note": "Qualify the local score-update direction after the original owner's formal revise; its meaningful highest-rank counterexample remains in its own factual evidence. Original technical and full incremental reader rechecks remain required.",
})
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
main_path = root / "docs/course-revision-20261005/review-progress.json"
main = json.loads(main_path.read_bytes())
main["records"]["18.4"]["status"] = "recheck_queued"
main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False))
