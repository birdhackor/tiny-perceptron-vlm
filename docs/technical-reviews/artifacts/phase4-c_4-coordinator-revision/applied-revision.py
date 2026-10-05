"""One paragraph scope correction after the actual original C.4 revise receipt."""
import datetime
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
SOURCE = ROOT / "course/chapters/0C.md"
REPORT = ROOT / "docs/technical-reviews/C.4.json"
INITIAL_REPORT_SHA = "81bb0f16f3a422e8e5dc8decaa6e8aa306ebf3edb2829413f5864322ef823025"
INITIAL_SECTION_SHA = "89e972e9cfad8ff829d9365e87281e25eba46a967d5ec2aea69c7ba76f97b030"


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
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_c_4"
assert report["verdict"] == "revise"
before = SOURCE.read_bytes()
before_slices = slices(before)
assert digest(before_slices["C.4"]) == INITIAL_SECTION_SHA
old = "精確驗證器能直接計算這個窄任務的真值，並選第一份全部檢查通過的候選，所以它的選擇成績等於「集合至少有一份正解」的比例（oracle coverage）。"
new = "精確驗證器能直接計算這個窄任務的真值，並選第一份全部檢查通過的候選。候選覆蓋率只檢查集合是否有最後答案正確的候選；中間等式錯、最後答案對的候選仍可計入覆蓋率，卻不會被驗證器接受。本次短答與步驟題的八組紀錄中，含正解的集合也都含一份全部檢查通過的候選，因此驗證器的選擇正確率恰好等於候選覆蓋率（oracle coverage）。"
assert before.count(old.encode()) == 1
after = before.replace(old.encode(), new.encode(), 1)
after_slices = slices(after)
assert set(before_slices) == set(after_slices)
changed = [lesson for lesson in before_slices if before_slices[lesson] != after_slices[lesson]]
assert changed == ["C.4"]
(ARTIFACT / "before-section.md").write_bytes(before_slices["C.4"])
SOURCE.write_bytes(after)
(ARTIFACT / "after-section.md").write_bytes(after_slices["C.4"])
receipt = {
    "executed_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "source": "course/chapters/0C.md#C.4",
    "original_owner": report["reviewer_task"],
    "original_report_sha256": INITIAL_REPORT_SHA,
    "original_verdict": "revise",
    "changed_sections": changed,
    "before_section_sha256": digest(before_slices["C.4"]),
    "after_section_sha256": digest(after_slices["C.4"]),
    "frozen_before_file_sha256": digest(before),
    "after_file_sha256": digest(after),
    "figure_changes": [],
    "implementation_changes": [],
    "measurement_changes": [],
    "scope": "One current C.4 paragraph distinguishes final-answer coverage from full-verification acceptance, limiting their observed equality to the original eight recorded conditions. This writer receipt is not scientific or reader acceptance; genuine original-owner and original-reader callbacks remain required.",
}
print(json.dumps(receipt, ensure_ascii=False, indent=2))
