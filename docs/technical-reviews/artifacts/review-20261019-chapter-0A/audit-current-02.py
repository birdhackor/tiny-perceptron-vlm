"""複查導言的現行依據與單元回顧；重用未改的驗證程式。"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).with_name("audit.py")
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == "f08fbcda83a5b19107fcfc8fd997fe233f47fe9b7cb8cc14d50dd3a6addb9051"
adapted = BASE.read_text().replace("freeze-04.json", "freeze-07.json")
adapted = adapted.replace("chapter-0A/audit.py", "chapter-0A/audit-current-02.py")
assert adapted.count("print(json.dumps(receipt, ensure_ascii=False, indent=2))") == 1
adapted = adapted.replace("print(json.dumps(receipt, ensure_ascii=False, indent=2))", "")
exec(compile(adapted, str(BASE), "exec"), globals())

previous_report_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_chapter-0A/technical-chapter-0A-current-01.json"
original_report_path = previous_report_path.with_name("technical-chapter-0A-original-01.json")
assert digest(previous_report_path) == "2c9e547ca75df988fc7912868defc684b0275a5e3d8ad884bfa1bd7cc1e69319"
assert digest(original_report_path) == "c13ee440ce738ea196b28924f650f86871608d5a9b621c6e609cc28a4c968af2"
previous_report = json.loads(previous_report_path.read_bytes())
previous_output_path = Path(__file__).with_name("audit-output-current-01.json")
previous_output = json.loads(previous_output_path.read_bytes())
preserved = []
for artifact in previous_report["artifacts"]:
    observed = digest(ROOT / artifact["path"])
    assert observed == artifact["sha256"]
    preserved.append({"path": artifact["path"], "sha256": observed, "unchanged": True})
for source in previous_report["sources"]:
    if source["kind"] == "repository_code" and source["path"] != intro["source"]:
        assert digest(ROOT / source["path"]) == source["sha256"]

baseline, target = "ecf6dec71280", "c6b9339d2595"
diff_command = ["git", "diff", baseline, target, "--", intro["source"]]
diff = subprocess.run(diff_command, cwd=ROOT, capture_output=True, check=True)
old = subprocess.run(["git", "show", f"{baseline}:{intro['source']}"], cwd=ROOT, capture_output=True, check=True)
new = subprocess.run(["git", "show", f"{target}:{intro['source']}"], cwd=ROOT, capture_output=True, check=True)
assert new.stdout == chapter.read_bytes()
assert hashlib.sha256(old.stdout).hexdigest() == next(s["sha256"] for s in previous_report["sources"] if s["id"] == "s1")
assert old.stdout.split(b"\n## ", 1)[0] == new.stdout.split(b"\n## ", 1)[0]
added = [line[1:] for line in diff.stdout.decode().splitlines() if line.startswith("+") and not line.startswith("+++") and line[1:].strip()]
removed = [line[1:] for line in diff.stdout.decode().splitlines() if line.startswith("-") and not line.startswith("---") and line[1:].strip()]
assert len(added) == 1 and not removed
summary = added[0]
sentences = [part + "。" for part in summary.split("。") if part]
assert len(sentences) == 4

program_comparison = []
changed_pages = []
for current, prior in zip(section_records, previous_output["sections"], strict=True):
    assert current["page_id"] == prior["page_id"]
    current_blocks, prior_blocks = current["body_python_blocks"], prior["body_python_blocks"]
    assert len(current_blocks) == len(prior_blocks)
    assert all(c["code_sha256"] == p["code_sha256"] and c["stdout"] == p["stdout"] for c, p in zip(current_blocks, prior_blocks, strict=True))
    program_comparison.append({"page_id": current["page_id"], "code_and_stdout_unchanged": True})
    if current["source_sha256"] != prior["source_sha256"]:
        changed_pages.append(current)
assert len(changed_pages) == 1
changed_page = changed_pages[0]
changed_main, _ = split_details((ROOT / changed_page["snapshot"]).read_text())
assert summary in changed_main and changed_page == section_records[-1]

body_evidence = []
for page in pages:
    main, supplement = split_details((ROOT / page["snapshot"]).read_text())
    body_evidence.append({"page_id": page["page_id"], "title": page["title"],
        "has_body_code": bool(re.search(r"^```python", main, flags=re.M)),
        "paper_and_check_excerpts": [paragraph for paragraph in main.split("\n\n")
            if ("沒有模型" in paragraph or "沒有呼叫模型" in paragraph or "紙上" in paragraph or "核對" in paragraph) and len(paragraph) < 550]})

first_measured = section_records[:7]
assert all(s["model_record_in_supplement"] for s in first_measured)
command = "python scripts/course_experiments/run.py --experiment rag --device cpu"
command_pages = []
for page in pages:
    main, supplement = split_details((ROOT / page["snapshot"]).read_text())
    if command in supplement:
        command_pages.append({"page_id": page["page_id"], "snapshot": page["snapshot"], "sha256": page["source_sha256"]})
assert len(command_pages) == 1
runner = ROOT / "scripts/course_experiments/run.py"
help_command = ["uv", "run", "--no-sync", "python", str(runner.relative_to(ROOT)), "--help"]
help_run = subprocess.run(help_command, cwd=ROOT, capture_output=True, check=True,
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
assert "--experiment" in help_run.stdout.decode() and "--device {cpu,cuda}" in help_run.stdout.decode()
spec = importlib.util.spec_from_file_location("review_experiment_runner", runner)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rag_spec = module.experiment_spec("rag")
assert rag_spec["module"] == "applications" and rag_spec["function"] == "run_rag"

observations = [
    {"case": "資料未進輸入", "has_required_source": False, "answer_correct": False, "baseline_correct": None},
    {"case": "已進輸入但答案不符", "has_required_source": True, "answer_correct": False, "baseline_correct": None},
    {"case": "原條件答對而改動後答錯", "has_required_source": True, "answer_correct": False, "baseline_correct": True},
]
for row in observations:
    row["missing_input"] = not row["has_required_source"]
    row["source_present_but_answer_wrong"] = row["has_required_source"] and not row["answer_correct"]
    row["degraded_after_change"] = row["baseline_correct"] is True and not row["answer_correct"]
assert observations[0]["missing_input"] and not observations[0]["source_present_but_answer_wrong"]
assert observations[1]["source_present_but_answer_wrong"] and not observations[1]["missing_input"]
assert observations[2]["degraded_after_change"] and observations[2]["source_present_but_answer_wrong"]

provenance_path = Path(__file__).with_name("provenance-output.json")
provenance = json.loads(provenance_path.read_bytes())
assert digest(ROOT / previous_output["model_record"]["path"]) == previous_output["model_record"]["sha256"]
assert digest(ROOT / "scripts/course_experiments/applications.py") == next(item["sha256"] for item in previous_output["inspected_inputs"] if item["path"] == "scripts/course_experiments/applications.py")
body_example = provenance["body_generated_examples"][0]
assert digest(ROOT / body_example["snapshot"]) == body_example["snapshot_sha256"]

receipt["base_audit"] = {"path": str(BASE.relative_to(ROOT)), "sha256": digest(BASE),
    "adaptations": ["改讀指定的現行凍結 manifest。", "本次命令改用本檔路徑；原驗證程式的輸出由本檔彙整。"]}
receipt["previous_report"] = {"path": str(previous_report_path.relative_to(ROOT)), "sha256": digest(previous_report_path), "issues": previous_report["issues"]}
receipt["original_report"] = {"path": str(original_report_path.relative_to(ROOT)), "sha256": digest(original_report_path)}
receipt["preserved_artifacts"] = preserved
receipt["source_diff"] = {"command": diff_command, "stdout": diff.stdout.decode(), "stderr": diff.stderr.decode(), "returncode": diff.returncode,
    "baseline_sha256": hashlib.sha256(old.stdout).hexdigest(), "current_sha256": digest(chapter), "introduction_unchanged": True}
receipt["programs_compared_to_previous_run"] = program_comparison
receipt["changed_dependency"] = {"page_id": changed_page["page_id"], "snapshot": changed_page["snapshot"], "sha256": changed_page["source_sha256"],
    "added_paragraph": summary, "sentences": sentences, "last_section": True,
    "body_evidence": body_evidence, "measured_sections": [s["page_id"] for s in first_measured],
    "reproduction_command": command, "command_pages": command_pages,
    "help_check": {"command": help_command, "stdout": help_run.stdout.decode(), "stderr": help_run.stderr.decode(), "returncode": help_run.returncode},
    "rag_plan_spec": rag_spec, "runner_sha256": digest(runner),
    "plan_path": str(module.PLAN_PATH.relative_to(ROOT)), "plan_sha256": digest(module.PLAN_PATH),
    "diagnostic_records": observations,
    "scope": "三類是可分開保存的觀察面向，可能重疊；不能由此推出唯一根因或模型能力。重跑入口僅執行 --help 及讀取計畫，不啟動訓練。"}
receipt["reused_provenance"] = {"path": str(provenance_path.relative_to(ROOT)), "sha256": digest(provenance_path),
    "experiment_record_unchanged": True, "generator_source_unchanged": True, "quoted_body_example_unchanged": True}
print(json.dumps(receipt, ensure_ascii=False, indent=2))
