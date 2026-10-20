"""核對凍結差異、舊證據完整性、教材導覽與實驗命令介面。"""

import ast
import contextlib
import difflib
import hashlib
import importlib.util
import io
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
artifact_directory = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise AssertionError(message)

original_path = root / "docs/course-review-20261019/reviews/rv1019_technical_A-10/technical-A-10-original-01.json"
require(sha(original_path) == "c02d75b35d28b019f5798dd802d3730b5600bacfbaa05acd503b291e8770b9b3", "原始報告已改變")
original = json.loads(original_path.read_bytes())
old_snapshot = root / "docs/course-review-20261019/freeze-01/sources/A.10.md"
new_snapshot = root / "docs/course-review-20261019/freeze-07/sources/A.10.md"
require(sha(new_snapshot) == "d27300e6a75bbb2400f4327e9e7442d1377b380e6c405a67ce1196893316b3ab", "目前凍結原文不符")
old_text = old_snapshot.read_text()
new_text = new_snapshot.read_text()
old_lines = old_text.splitlines(keepends=True)
new_lines = new_text.splitlines(keepends=True)
operations = difflib.SequenceMatcher(a=old_lines, b=new_lines).get_opcodes()
changes = [op for op in operations if op[0] != "equal"]
require(len(changes) == 1 and changes[0][0] == "insert", "差異不是單一新增區塊")
change = changes[0]
added = "".join(new_lines[change[3]:change[4]]).strip()
require(added.startswith("本單元到這裡結束。"), "新增段落內容不符")
require(new_text.replace(added + "\n\n", "", 1) == old_text, "移除新增段落後與原始版本不同")
diff_command = ["git", "diff", "--no-index", str(old_snapshot.relative_to(root)), str(new_snapshot.relative_to(root))]
diff = subprocess.run(diff_command, cwd=root, capture_output=True, text=True)
require(diff.returncode == 1, "凍結版本差異指令未如預期回傳差異")

preserved = []
for item in original["artifacts"]:
    actual = sha(root / item["path"])
    require(actual == item["sha256"], "沿用證據的 SHA-256 不符")
    preserved.append({"path": item["path"], "sha256": actual, "unchanged": True})
old_self_check = artifact_directory / "self-check-original-01.txt"
require(sha(old_self_check) == "d70e825b20d4a39d66e8f5fc41f17529e58732bfed9253e584f6bf385407ef7d", "原始自我檢查輸出已改變")

chapter_path = root / "course/chapters/0A.md"
chapter = chapter_path.read_text()
headers = list(re.finditer(r"(?m)^## ([A-Z\d]+\.\d+) (.+)$", chapter))
sections = {}
for position, heading in enumerate(headers):
    end = headers[position + 1].start() if position + 1 < len(headers) else len(chapter)
    sections[heading[2]] = {"lesson_id": heading[1], "text": chapter[heading.start():end]}
selected = next(value for value in sections.values() if value["lesson_id"] == original["lesson_id"])
require(selected["text"].encode() == new_snapshot.read_bytes(), "現行章檔與凍結原文不同")
require(headers[-1][1] == original["lesson_id"], "受審頁不是本單元最後的編號小節")

manual_locators = {
    "給幾個例子能改變回答嗎？": ("examples =", "紙上預期6"),
    "不在訓練資料裡的資訊怎麼使用？": ("source =", "答案應是青街8號"),
    "文件多了怎麼找相關段落？": ("docs =", "所以只交回 d2"),
    "答案能回到來源嗎？": ("answers =", "青街8號得 True、True"),
    "答錯是沒找到還是沒用好？": ("examples =", "有正確來源 2/3、答案正確 1/3"),
    "找不到或文件不可靠時怎麼辦？": ("cases =", "來源未給地址，目前無法確定"),
    "多輪對話和答案怎麼分配長度？": ("parts =", "P＝75、G＝25、P＋G＝100"),
    "同一條線索，放在哪裡會有差嗎？": ("others =", "紙上答案是 `M17`"),
    "找到一條資訊，就代表讀懂長文嗎？": ("code_of =", "紙上答案是 `M17、B04、C26`"),
    "對話很長，最新的條件如何驗收？": ("updates =", "這題的答案是 `15:00`"),
}
require(set(manual_locators) == set(sections), "教材小節與手工定位清單不一致")
inventory = []
for title, (input_marker, answer_marker) in manual_locators.items():
    body = sections[title]["text"]
    for marker in (input_marker, answer_marker, "練習", "先預測", "核對"):
        require(marker in body, "教材的輸入、預期答案或核對定位不存在")
    inventory.append({"lesson_id": sections[title]["lesson_id"], "title": title,
                      "section_sha256": hashlib.sha256(body.encode()).hexdigest(),
                      "input_locator": input_marker, "paper_answer_locator": answer_marker,
                      "verification_locator": "練習；先預測；核對"})

current_run = json.loads((artifact_directory / "section-execution-current-01.json").read_bytes())
old_run = json.loads((artifact_directory / "section-execution.json").read_bytes())
require(current_run["copied_records"]["fence-1.py"] == old_run["copied_records"]["fence-1.py"], "受審程式已改變")
require(current_run["copied_records"]["stdout.txt"] == old_run["copied_records"]["stdout.txt"], "受審程式輸出已改變")
namespace = {}
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    exec(compile(current_run["copied_records"]["fence-1.py"], str(new_snapshot), "exec"), namespace)
require(printed.getvalue() == current_run["copied_records"]["stdout.txt"], "補充執行與原樣執行不同")
base_current = namespace["plan_as_of"](9)
base_past = namespace["plan_as_of"](2)
namespace["updates"].append((5, {"地點": "M18"}))
exercise_current = namespace["plan_as_of"](9)
exercise_past = namespace["plan_as_of"](2)
require(exercise_current == {"時間": "16:00", "地點": "M18", "格式": "兩個條列"}, "新增地點的目前安排不符")
require(exercise_past == {"時間": "15:00", "地點": "M17", "格式": "一小段文字"}, "新增地點改變了舊安排")

runner_path = root / "scripts/course_experiments/run.py"
spec = importlib.util.spec_from_file_location("reviewed_experiment_runner", runner_path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
recipe = runner.experiment_spec("rag")
module_path = root / ("scripts/course_experiments/" + recipe["module"] + ".py")
module_tree = ast.parse(module_path.read_text())
require(any(isinstance(node, ast.FunctionDef) and node.name == recipe["function"] for node in module_tree.body), "實驗配方的函式不存在")
help_command = [str(root / ".venv/bin/python"), "scripts/course_experiments/run.py", "--experiment", "rag", "--device", "cpu", "--help"]
help_result = subprocess.run(help_command, cwd=root, capture_output=True, text=True)
require(help_result.returncode == 0 and "--experiment" in help_result.stdout and "--device {cpu,cuda}" in help_result.stdout, "實驗命令介面不符")

supplements = []
for lesson_id in recipe["lessons"]:
    body = next(value["text"] for value in sections.values() if value["lesson_id"] == lesson_id)
    details = re.findall(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", body, re.S)
    require(any("RAG" in title and "docs/course-experiments/results/rag.json" in content for title, content in details), "導覽小節沒有對應的 RAG 實驗補充與結果連結")
    supplements.append({"lesson_id": lesson_id, "summary": [title for title, content in details if "RAG" in title], "result_path": "docs/course-experiments/results/rag.json"})
command_literal = "python scripts/course_experiments/run.py --experiment rag --device cpu"
require(command_literal in next(value["text"] for value in sections.values() if "RAG 實驗的設定與成對改動" in value["text"]), "重跑命令不在所指補充區")
result_path = root / "docs/course-experiments/results/rag.json"
rag = json.loads(result_path.read_bytes())
rows = rag["results"]["samples"]
require(rag["experiment_id"] == "rag" and rag["evidence_status"] == "complete_run", "實驗結果身分或完成狀態不符")
require(len(rows) == 84 and len(rag["results"]["icl_samples_same_model_weights"]) == 4, "實驗結果紀錄筆數不符")
metric_recounts = {}
for mode, metrics in rag["results"]["metrics"].items():
    selected_rows = [row for row in rows if row["mode"] == mode]
    recount = {}
    for field in ("exact_match", "fact_answer_correct", "citation_valid", "supported_by_cited_source", "unknown"):
        count = sum(bool(row[field]) for row in selected_rows)
        require(metrics[field]["numerator"] == count and metrics[field]["denominator"] == len(selected_rows), "實驗彙總與逐題結果不同")
        recount[field] = {"numerator": count, "denominator": len(selected_rows)}
    require(all(row["generation"]["input_ids"] and row["generation"]["samples"][0]["generated_ids"] for row in selected_rows), "實驗結果缺少原始生成 ID")
    metric_recounts[mode] = recount

inspected_paths = [original_path, old_snapshot, new_snapshot, chapter_path, runner_path, module_path,
                   root / "docs/course-experiments/plan.json", result_path,
                   root / "docs/technical-review-guide.md", root / "docs/course-review-20261019/freeze-07.json",
                   root / "course/figures/rewrite-A-conversation-updates.svg"]
record = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-A-10/recheck-evidence-current-01.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU；字典驗算、文件檢查與 CLI help，未訓練模型"},
    "result": "全部核對符合預期；原有主張與證據仍有效，新增段落的教材導覽、實驗結果連結與命令介面存在。",
    "inputs": [{"path": str(path.relative_to(root)), "sha256": sha(path)} for path in inspected_paths],
    "original_issues": original["issues"], "resolved_issue_ids": [],
    "diff": {"command": diff_command, "exit_code": diff.returncode, "stdout": diff.stdout, "stderr": diff.stderr},
    "added_paragraph": added,
    "preserved_artifacts": preserved,
    "original_self_check_sha256": sha(old_self_check),
    "current_chapter_equals_frozen_section": True,
    "current_fence_equals_original": True, "current_stdout_equals_original": True,
    "current_state": base_current, "historical_state": base_past,
    "exercise_current": exercise_current, "exercise_as_of_day_2": exercise_past,
    "chapter_inventory": inventory,
    "summary_category_mapping": {
        "輸入是否有必要資料": ["給幾個例子能改變回答嗎？", "不在訓練資料裡的資訊怎麼使用？", "文件多了怎麼找相關段落？", "答錯是沒找到還是沒用好？"],
        "回答是否符合來源": ["答案能回到來源嗎？", "找不到或文件不可靠時怎麼辦？"],
        "改變條件後是否仍符合紙上答案": ["多輪對話和答案怎麼分配長度？", "同一條線索，放在哪裡會有差嗎？", "找到一條資訊，就代表讀懂長文嗎？", "對話很長，最新的條件如何驗收？"]},
    "supplement_navigation": supplements,
    "recipe": recipe,
    "cli_help": {"command_argv": help_command, "exit_code": help_result.returncode, "stdout": help_result.stdout, "stderr": help_result.stderr},
    "stored_rag_result": {"path": str(result_path.relative_to(root)), "sha256": sha(result_path),
        "device": rag["device"], "gpu": rag["gpu"], "seed": rag["seed"],
        "evidence_status": rag["evidence_status"], "training_steps": rag["results"]["training"]["steps"],
        "announcement_samples": len(rows), "icl_samples": len(rag["results"]["icl_samples_same_model_weights"]),
        "recomputed_metrics": metric_recounts,
        "applications_code_matches_recorded_hash": rag["code_sha256"].get(str(module_path.relative_to(root))) == sha(module_path)},
    "scope": "核對原有範例、教材所指的既有結果及 CLI 路徑；沒有重跑 RAG 訓練，沒有測量新模型結果，也不以三類紀錄證明互斥的因果歸因。",
}
output = Path(__file__).with_suffix(".json")
with output.open("x", encoding="utf-8") as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"path": str(output.relative_to(root)), "sha256": sha(output), "result": record["result"],
                  "checked_sections": len(inventory), "checked_supplements": len(supplements), "cli_exit_code": help_result.returncode,
                  "applications_code_matches_recorded_hash": record["stored_rag_result"]["applications_code_matches_recorded_hash"]}, ensure_ascii=False))
