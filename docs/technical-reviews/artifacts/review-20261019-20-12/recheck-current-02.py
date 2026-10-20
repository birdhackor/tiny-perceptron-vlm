"""核對作者聲明的檔案綁定、既有證據與小節執行紀錄。"""

import ast
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.natural_concepts import text_error_report


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(relative):
    return (ROOT / relative).read_bytes()


previous_path = "docs/course-review-20261019/reviews/rv1019_technical_20-12/technical-20-12-current-01.json"
previous_raw = read(previous_path)
assert digest(previous_raw) == "6583081ee19fd89cdb74cba999a93186c5a1e358d0e27d35088509cb84560560"
previous = json.loads(previous_raw)
original_path = previous["original_report"]["path"]
assert digest(read(original_path)) == previous["original_report"]["sha256"]
source_path = "docs/course-review-20261019/freeze-06/sources/20.12.md"
source_raw = read(source_path)
assert digest(source_raw) == previous["source_sha256"] == "45e1e1d5ec66a7a1d792c7ab76bdbfbbe5749a47cb31ac927d1759b384e52dcc"
freeze_path = previous["source_freeze"]["path"]
assert digest(read(freeze_path)) == previous["source_freeze"]["sha256"]

reuse = []
for group in ("sources", "artifacts"):
    for item in previous[group]:
        if item.get("path"):
            actual = digest(read(item["path"]))
            assert actual == item["sha256"], item["path"]
            reuse.append({"group": group, "id": item["id"], "path": item["path"], "recorded_sha256": item["sha256"], "current_sha256": actual, "matches": True})

authorship_path = "docs/authorship.md"
authorship_raw = read(authorship_path)
assert digest(authorship_raw) == "58092b2cabcac23e39b935b2aaf3716ec2a68eee8ad3d8f937483b8a5fe8fb20"
authorship = authorship_raw.decode()
bindings = re.findall(r"^- `([^`]+)`：`([a-f0-9]{64})`", authorship, re.MULTILINE)
assert {path for path, _ in bindings} == {"scripts/prepare_natural_v4_voice_questions.py", "docs/natural-assistant/v4/data/voice-question-sources.json"}
for path, expected in bindings:
    assert digest(read(path)) == expected

prepare_path = "scripts/prepare_natural_v4_voice_questions.py"
tree = ast.parse(read(prepare_path).decode())
spec_node = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SPEC" for target in node.targets))
spec = ast.literal_eval(spec_node.value)
metadata_path = "docs/natural-assistant/v4/data/voice-question-sources.json"
metadata = json.loads(read(metadata_path))
expected_rubrics = {identifier: (split, rubric) for split, identifier, rubric in spec}
observed_rubrics = {row["source_utterance_id"]: (row["split"], row["semantic_rubric"]) for row in metadata["audio_rows"]}
assert expected_rubrics == observed_rubrics and len(expected_rubrics) == 8
assert Counter(split for split, _ in observed_rubrics.values()) == {"validation": 4, "test": 4}
excerpts = [
    "課文、資料標註與判準的具體措辭由 AI 工具寫成，作者沒有逐條審過，本課也沒有安排其他人逐條審核。",
    "這些判準由 AI 工具撰寫，沒有逐題經人審核。",
]
for excerpt in excerpts:
    assert excerpt in authorship

run = ROOT / "outputs/reviewer-tools/runs/20261020T110305579590Z"
files = {}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run / name).read_bytes()
    files[name] = {"sha256": digest(raw), "text": raw.decode()}
extraction = json.loads(files["extraction.json"]["text"])
execution = json.loads(files["execution.json"]["text"])
assert extraction["source_sha256"] == execution["source_sha256"] == digest(source_raw)
assert execution["exit_code"] == 0
assert files["stdout.txt"]["text"] == "最少編輯次數 1\n正確答案字數 9\nCER百分比 11.1\n兩份問題相同 False\n"
assert files["stderr.txt"]["text"] == ""
old_execution = json.loads((ARTIFACT / "section-execution.json").read_bytes())
assert files["fence-1.py"]["sha256"] == old_execution["files"]["fence-1.py"]["sha256"]

numeric = []
for prediction in ("請推薦辣的晚餐。", "請推薦不辣的晚飯。"):
    result = text_error_report("請推薦不辣的晚餐。", prediction)
    assert result["edits"] == 1 and result["reference_characters"] == 9
    assert result["cer"] == 1 / 9 and round(result["cer"] * 100, 1) == 11.1
    numeric.append(result)
figure_path, expected_figure_sha = next(iter(previous["figure_sha256"].items()))
assert digest(read(figure_path)) == expected_figure_sha
png_path = "outputs/course-review-20261019/figure-renders-04/aaed40e6fc4eebfa6ba59a98b2c08ea42624a481553570313b146d8e1f0085ac.png"
assert digest(read(png_path)) == "f39bac41451c8c50bdc0c0b4192f779e8d1eb190b5d25272d34657a7a577cb67"

value = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-12/recheck-current-02.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
    "previous_report": {"path": previous_path, "sha256": digest(previous_raw)},
    "source": {"path": source_path, "sha256": digest(source_raw), "same_as_previous": True},
    "reused_evidence": reuse,
    "authorship": {"path": authorship_path, "sha256": digest(authorship_raw), "excerpts": excerpts, "bound_files": [{"path": path, "declared_sha256": expected, "actual_sha256": digest(read(path))} for path, expected in bindings], "rubrics_match": True, "utterance_ids": sorted(observed_rubrics), "split_counts": dict(Counter(split for split, _ in observed_rubrics.values())), "evidence_scope": "已核對作者流程聲明的內容、版本及其涵蓋的判準。文字產製者與人審狀態依作者自述，不是由本程式重現歷史事件；本程式不替判準品質背書。"},
    "section_execution": {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/20.md#20.12" --execute', "files": files},
    "numeric": numeric,
    "figure": {"path": figure_path, "sha256": expected_figure_sha, "png": png_path, "png_sha256": digest(read(png_path))},
    "result": "沿用證據雜湊、作者聲明綁定的檔案、全部判準逐字對應、凍結原文與本次執行均符合。作者自述的支持範圍由審閱報告另行說明。",
}
with (ARTIFACT / "recheck-current-02.json").open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
print(value["result"])
