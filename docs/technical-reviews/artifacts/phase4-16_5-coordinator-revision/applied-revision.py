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
source = root / "course/chapters/16.md"
report_path = root / "docs/technical-reviews/16.5.json"
progress_path = root / "docs/course-revision-20261005/factual-review-progress.json"
sha = lambda value: hashlib.sha256(value).hexdigest()
report_bytes = report_path.read_bytes()
report = json.loads(report_bytes)
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_16_5"
assert report["verdict"] == "revise"
assert sha(report_bytes) == "9212f0c6e11f1c7a23f4fb799480dabbf2a1dd9ef377ca2b5c9e9308399cb573"


def sections(raw):
    headings = list(re.finditer(rb"(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+", raw))
    return raw[:headings[0].start()], {
        h[1].decode(): raw[h.start():headings[i + 1].start() if i + 1 < len(headings) else len(raw)]
        for i, h in enumerate(headings)
    }


before = source.read_bytes()
intro_before, sections_before = sections(before)
assert sha(sections_before["16.5"]) == report["source_sha256"]
old = "更新後權重最大差5.97909×10⁻⁶；每步中位數PAD為10.799毫秒、packing為11.191毫秒，這輪沒有加速。".encode()
new = "更新後權重最大差5.97909×10⁻⁶。略去各路徑前3次暖身後，餘37次模型更新的計時中位數，PAD為10.799毫秒、packing為11.191毫秒，這輪沒有加速。每次計時包括模型前向計算、代價計算、反向計算、梯度裁剪、參數更新及末尾的GPU同步；抽樣、裝填與清空梯度不計入。".encode()
assert sections_before["16.5"].count(old) == before.count(old) == 1
after = before.replace(old, new, 1)
intro_after, sections_after = sections(after)
assert intro_before == intro_after
assert sections_before.keys() == sections_after.keys()
assert all(sections_before[k] == sections_after[k] for k in sections_before if k != "16.5")
assert re.findall(rb"(?ms)^```.*?^```", sections_before["16.5"]) == re.findall(rb"(?ms)^```.*?^```", sections_after["16.5"])
history = root / "docs/technical-reviews/history" / ("phase4-16_5-own-initial-revise-" + sha(report_bytes) + ".json")
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
    "source": "course/chapters/16.md#16.5",
    "old_phrase": old.decode(),
    "new_phrase": new.decode(),
    "old_source_sha256": sha(sections_before["16.5"]),
    "new_source_sha256": sha(sections_after["16.5"]),
    "old_whole_frozen_input_sha256": sha(before),
    "new_whole_file_sha256": sha(after),
    "intro_sha256": sha(intro_before),
    "other_sections_byte_unchanged": [k for k in sections_before if k != "16.5"],
    "initial_report_history_path": history.relative_to(root).as_posix(),
    "initial_report_history_sha256": sha(report_bytes),
    "unchanged": ["introduction", "all other section bytes", "figures", "fences", "implementation", "raw measurements"],
}
receipt_path = Path(__file__).resolve().parent / "revision.stdout.json"
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
progress = json.loads(progress_path.read_bytes())
record = progress["records"]["16.5"]
record["status"] = "recheck_queued"
record["events"].append({
    "at": now,
    "event": "coordinator_localized_revision",
    **receipt,
    "writer_artifacts": [{"path": p.relative_to(root).as_posix(), "sha256": sha(p.read_bytes())} for p in [Path(__file__).resolve(), receipt_path]],
    "note": "Only the selected-reading timing sentence is rewritten after the original owner's formal revise; original technical and reader actual full rechecks still required.",
})
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
main_path = root / "docs/course-revision-20261005/review-progress.json"
main = json.loads(main_path.read_bytes())
main["records"]["16.5"]["status"] = "recheck_queued"
main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False))
