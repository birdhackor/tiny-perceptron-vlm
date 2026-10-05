"""Apply two navigation corrections after the actual R.2/R.4 revise reports."""
import datetime
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
SOURCE = ROOT / "course/README.md"
REPORT = ROOT / "docs/technical-reviews/R.2.json"
INITIAL_REPORT_SHA = "b16b14bfad7c6826103cfb384b2f14a406f1f3a76168c05eb14cd394d2577b29"
INITIAL_SECTION_SHA = "228bd84cff5dac6ffe6d905c9999879dd43d7270ecc79a6262842cbf0fb05b47"
R4_REPORT_SHA = "7bd62cb306a41c70bd2d702fad20214f11502f777744ed5a2188aa4bffc77186"
R4_SECTION_SHA = "40f3210e4b2f9b9eafab06d5a6ccfde43ce28078304e1093595140985c9eeb1a"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def slices(raw):
    headings = list(re.finditer(rb"(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+", raw))
    return {
        heading[1].decode(): raw[
            heading.start(): headings[index + 1].start() if index + 1 < len(headings) else len(raw)
        ]
        for index, heading in enumerate(headings)
    }


assert digest(REPORT.read_bytes()) == INITIAL_REPORT_SHA
report = json.loads(REPORT.read_bytes())
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_r_2"
assert report["verdict"] == "revise"
archive = ROOT / "docs/technical-reviews/history/phase4-R_2-initial-revise-20261005.opaque.json"
assert digest(archive.read_bytes()) == INITIAL_REPORT_SHA
r4_path = ROOT / "docs/technical-reviews/R.4.json"
assert digest(r4_path.read_bytes()) == R4_REPORT_SHA
r4_report = json.loads(r4_path.read_bytes())
assert r4_report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_r_4_clean"
assert r4_report["verdict"] == "revise"
r4_archive = ROOT / "docs/technical-reviews/history/phase4-R_4-clean-initial-revise-20261005.opaque.json"
assert digest(r4_archive.read_bytes()) == R4_REPORT_SHA
before = SOURCE.read_bytes()
before_slices = slices(before)
assert digest(before_slices["R.2"]) == INITIAL_SECTION_SHA
assert digest(before_slices["R.4"]) == R4_SECTION_SHA
old = "跳讀遇到陌生概念時，使用小節末尾的回讀連結補一個必要步驟，再回到原問題。"
new = "跳讀遇到陌生概念時，使用正文中的回讀連結與小節末尾的補充連結，補一個必要步驟，再回到原問題。"
assert before.count(old.encode()) == 1
after = before.replace(old.encode(), new.encode(), 1)
old_r4 = "成品的能力與限制集中在[19.12](chapters/19.md#19.12)，取得模型與重做的步驟放在[操作頁](training.md)。"
new_r4 = "成品的能力與限制集中在[19.12](chapters/19.md#19.12)；已公開合成示範的下載與重做步驟見[19.11](chapters/19.md#19.11)，各局部實驗的操作見[操作頁](training.md)。"
assert after.count(old_r4.encode()) == 1
after = after.replace(old_r4.encode(), new_r4.encode(), 1)
after_slices = slices(after)
assert set(before_slices) == set(after_slices)
changed = [lesson for lesson in before_slices if before_slices[lesson] != after_slices[lesson]]
assert changed == ["R.2", "R.4"]
for lesson in changed:
    (ARTIFACT / f"before-{lesson}.md").write_bytes(before_slices[lesson])
SOURCE.write_bytes(after)
for lesson in changed:
    (ARTIFACT / f"after-{lesson}.md").write_bytes(after_slices[lesson])
receipt = {
    "executed_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "source": "course/README.md",
    "original_owners": {"R.2": report["reviewer_task"], "R.4": r4_report["reviewer_task"]},
    "original_report_sha256": {"R.2": INITIAL_REPORT_SHA, "R.4": R4_REPORT_SHA},
    "initial_report_archives": {
        "R.2": archive.relative_to(ROOT).as_posix(),
        "R.4": r4_archive.relative_to(ROOT).as_posix(),
    },
    "original_verdict": "revise",
    "changed_sections": changed,
    "before_section_sha256": {lesson: digest(before_slices[lesson]) for lesson in changed},
    "after_section_sha256": {lesson: digest(after_slices[lesson]) for lesson in changed},
    "frozen_before_file_sha256": digest(before),
    "after_file_sha256": digest(after),
    "figure_changes": [],
    "fence_changes": [],
    "implementation_changes": [],
    "measurement_changes": [],
    "scope": "R.2 distinguishes body rereading links from footer supplements. R.4 points the published synthetic product acquisition/reproduction to19.11 and scopes training.md to local experiment operations; its existing previous paragraph distinguishes the old demonstration from the planned new mainline. This writer receipt is not scientific or reader acceptance; genuine original technical and reader callbacks remain required.",
}
print(json.dumps(receipt, ensure_ascii=False, indent=2))
