"""核對單元導言描述的程式邊界、實測紀錄位置與教材連結。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
FREEZE = ROOT / "docs/course-review-20261019/freeze-05.json"
INTRO_ID = "chapter-0A"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_details(text):
    main, supplements = [], []
    depth = 0
    for line in text.splitlines(keepends=True):
        if line.strip() == "<details>":
            depth += 1
        (supplements if depth else main).append(line)
        if line.strip() == "</details>":
            depth -= 1
    assert depth == 0
    return "".join(main), "".join(supplements)


def call_name(node):
    try:
        return ast.unparse(node.func)
    except AttributeError:
        return type(node).__name__


freeze = json.loads(FREEZE.read_bytes())
intro = next(p for p in freeze["pages"] if p["page_id"] == INTRO_ID)
pages = [p for p in freeze["pages"] if p["source"] == intro["source"] and p["kind"] == "lesson"]
chapter = ROOT / intro["source"]
raw_intro = (ROOT / intro["snapshot"]).read_bytes()
assert hashlib.sha256(raw_intro).hexdigest() == intro["source_sha256"]
assert not intro["figures_sha256"]
raw_sections = [(ROOT / p["snapshot"]).read_bytes() for p in pages]
assert raw_intro + b"".join(raw_sections) == chapter.read_bytes()

import torch

model_calls = []
project_calls = set()


def profile_calls(frame, event, arg):
    if event != "call":
        return
    filename = Path(frame.f_code.co_filename)
    name = frame.f_code.co_name
    if filename.is_relative_to(ROOT / "tiny_perceptron"):
        if not name.startswith("<"):
            project_calls.add(f"{filename.relative_to(ROOT)}:{name}")
    if name in {"forward", "_call_impl", "generate", "_sample", "new_lm", "fit_lm"}:
        if "torch" in filename.parts or filename.is_relative_to(ROOT / "tiny_perceptron"):
            model_calls.append(f"{filename}:{name}")


section_records = []
for page, raw in zip(pages, raw_sections, strict=True):
    assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
    main, supplement = split_details(raw.decode("utf-8"))
    blocks = re.findall(r"^```python\n(.*?)^```", main, flags=re.M | re.S)
    namespace = {"__name__": "__review_snippet__"}
    executed = []
    for index, code in enumerate(blocks):
        tree = ast.parse(code)
        output = io.StringIO()
        sys.setprofile(profile_calls)
        try:
            with contextlib.redirect_stdout(output):
                exec(compile(tree, page["snapshot"], "exec"), namespace)
        finally:
            sys.setprofile(None)
        executed.append({
            "block_index": index,
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "calls": sorted({call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)}),
            "stdout": output.getvalue(),
        })
    record_link = "../../docs/course-experiments/results/rag.json"
    assert record_link not in main
    section_records.append({
        "page_id": page["page_id"],
        "title": page["title"],
        "snapshot": page["snapshot"],
        "source_sha256": page["source_sha256"],
        "body_python_blocks": executed,
        "supplement_summaries": re.findall(r"<summary>(.*?)</summary>", supplement),
        "model_record_in_body": record_link in main,
        "model_record_in_supplement": record_link in supplement,
        "no_model_result_notice": [line for line in supplement.splitlines()
            if any(s in line for s in ("沒有提供本課模型", "沒有訓練或模型成績", "沒有執行模型評測"))],
    })

assert not model_calls

links = []
for href in dict.fromkeys(re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", chapter.read_text())):
    if href.startswith("https://"):
        continue
    path, _, anchor = href.partition("#")
    target = (chapter.parent / path).resolve() if path else chapter
    assert target.is_relative_to(ROOT) and target.is_file(), href
    headings = re.findall(r"^## (.+)$", target.read_text(), flags=re.M) if anchor else []
    matching = [h for h in headings if h.split()[0] == anchor]
    assert not anchor or matching, href
    links.append({"href": href, "path": str(target.relative_to(ROOT)), "sha256": digest(target),
                  "matching_heading": matching})

rag_path = ROOT / "docs/course-experiments/results/rag.json"
rag = json.loads(rag_path.read_bytes())
result = rag["results"]
rows = result["samples"]
icl_rows = result["icl_samples_same_model_weights"]
for row in rows + icl_rows:
    generation = row["generation"]
    assert generation["input_tokens"] == len(generation["input_ids"])
    for sample in generation["samples"]:
        output = sample["generated_ids"]
        raw_output = output[:-1] if output and output[-1] == 2 else output
        text = bytes(token - 8 for token in raw_output if token >= 8).decode("utf-8", errors="replace")
        assert text == sample["generated"]

paths = [FREEZE, ROOT / intro["snapshot"], chapter, ROOT / "tiny_perceptron/retrieval.py",
         ROOT / "tiny_perceptron/data.py", ROOT / "scripts/course_experiments/applications.py", rag_path]
receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-0A/audit-current-01.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "platform": platform.platform(), "device": "CPU", "executable": sys.executable},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE)},
    "source_snapshot": {"path": intro["snapshot"], "sha256": intro["source_sha256"]},
    "chapter_equals_frozen_parts": True,
    "figure_count_in_introduction": len(intro["figures_sha256"]),
    "sections": section_records,
    "body_model_calls": model_calls,
    "observed_project_calls": sorted(project_calls),
    "local_links": links,
    "model_record": {"path": str(rag_path.relative_to(ROOT)), "sha256": digest(rag_path),
        "experiment_id": rag["experiment_id"], "status": rag["status"], "revision": rag["revision"],
        "recorded_device": rag["device"], "recorded_gpu": rag["gpu"],
        "announcement_samples": len(rows), "icl_samples": len(icl_rows),
        "conditions": dict(Counter(row["mode"] for row in rows)),
        "all_saved_text_matches_generated_ids": True,
        "icl_expected_and_generated": [{"mode": row["mode"], "expected": row["expected"],
            "generated": row["generation"]["samples"][0]["generated"]} for row in icl_rows]},
    "inspected_inputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths],
    "limits": ["僅執行正文短程式；未啟動補充區的訓練命令。",
               "實驗紀錄只核對存放位置、生成欄位與 ID 解碼；未重訓或重跑模型。",
               "不以本次核對代替各節研究主張、實驗數字或圖解的獨立審閱。",
               "先備安排僅核對技術連結可回查，不推論個別讀者的理解或學習成果。"],
}

import subprocess
from datetime import date

original_report_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_chapter-0A/technical-chapter-0A-original-01.json"
original_report = json.loads(original_report_path.read_bytes())
assert digest(original_report_path) == "c13ee440ce738ea196b28924f650f86871608d5a9b621c6e609cc28a4c968af2"
previous_receipt_path = Path(__file__).parent / "audit-output.json"
previous_receipt = json.loads(previous_receipt_path.read_bytes())
previous_artifacts = []
for artifact in original_report["artifacts"]:
    observed = digest(ROOT / artifact["path"])
    assert observed == artifact["sha256"]
    previous_artifacts.append({"path": artifact["path"], "sha256": observed, "unchanged": True})

baseline_revision, target_revision = "5ef56bc756f3", "3af7f13ef959"
diff_command = ["git", "diff", baseline_revision, target_revision, "--", intro["source"]]
diff_run = subprocess.run(diff_command, cwd=ROOT, capture_output=True, check=True)
old_run = subprocess.run(["git", "show", f"{baseline_revision}:{intro['source']}"], cwd=ROOT, capture_output=True, check=True)
new_run = subprocess.run(["git", "show", f"{target_revision}:{intro['source']}"], cwd=ROOT, capture_output=True, check=True)
assert hashlib.sha256(old_run.stdout).hexdigest() == next(s["sha256"] for s in original_report["sources"] if s["id"] == "s1")
assert new_run.stdout == chapter.read_bytes()
assert old_run.stdout.split(b"\n## ", 1)[0] == new_run.stdout.split(b"\n## ", 1)[0]
unchanged_programs = []
for section, previous in zip(section_records, previous_receipt["sections"], strict=True):
    assert section["page_id"] == previous["page_id"]
    old_blocks = previous["body_python_blocks"]
    new_blocks = section["body_python_blocks"]
    assert len(old_blocks) == len(new_blocks)
    assert all(old["code_sha256"] == new["code_sha256"] and old["stdout"] == new["stdout"]
               for old, new in zip(old_blocks, new_blocks, strict=True))
    unchanged_programs.append({"page_id": section["page_id"], "code_and_stdout_unchanged": True})

announcements = [
    {"id": "current", "shop": "小小書店", "published": "2027-03-01", "effective_from": "2027-03-01", "effective_until": "2027-04-01", "address": "青街8號"},
    {"id": "future", "shop": "小小書店", "published": "2027-03-10", "effective_from": "2027-04-01", "effective_until": None, "address": "紅街9號"},
    {"id": "other", "shop": "另一家書店", "published": "2027-03-12", "effective_from": "2027-03-12", "effective_until": None, "address": "綠街1號"},
]
query_day = date.fromisoformat("2027-03-15")
same_shop = [row for row in announcements if row["shop"] == "小小書店"]
latest_publication = max(same_shop, key=lambda row: row["published"])
applicable = [row for row in same_shop
              if date.fromisoformat(row["published"]) <= query_day
              and date.fromisoformat(row["effective_from"]) <= query_day
              and (row["effective_until"] is None or query_day < date.fromisoformat(row["effective_until"]))]
assert latest_publication["address"] == "紅街9號"
assert [row["address"] for row in applicable] == ["青街8號"]

from tiny_perceptron.retrieval import retrieve
metadata_a = [{"id": "a", "text": "書店地址", "published": "2027-03-01"},
              {"id": "b", "text": "書店地址", "published": "2027-03-10"}]
metadata_b = [{**row, "published": metadata_a[1-index]["published"]} for index, row in enumerate(metadata_a)]
rank_before = [row["id"] for row in retrieve("書店地址", metadata_a)]
rank_after = [row["id"] for row in retrieve("書店地址", metadata_b)]
assert rank_before == rank_after

changed_page = next(page for page in pages if page["source_sha256"] != next(
    row["source_sha256"] for row in previous_receipt["sections"] if row["page_id"] == page["page_id"]))
changed_text = (ROOT / changed_page["snapshot"]).read_text()
changed_main, _ = split_details(changed_text)
exercise_code = re.findall(r"^```python\n(.*?)^```", changed_main, flags=re.M | re.S)[0]
exercise_code = exercise_code.replace('"書店地址尚未公布"', '"書店地址是青街8號"')
exercise_stdout = io.StringIO()
with contextlib.redirect_stdout(exercise_stdout):
    exec(compile(exercise_code, changed_page["snapshot"], "exec"), {"__name__": "__review_exercise__"})
assert exercise_stdout.getvalue() == "[] 沒有取回相關來源，請補公告\n['d2'] 需進一步核對來源與答案\n"
changed_paragraphs = [paragraph for paragraph in changed_main.split("\n\n")
                      if "日期較新是線索" in paragraph or "是不是目前適用的一份" in paragraph]
assert len(changed_paragraphs) == 2

previous_provenance = Path(__file__).parent / "provenance-output.json"
provenance = json.loads(previous_provenance.read_bytes())
assert digest(ROOT / previous_receipt["model_record"]["path"]) == previous_receipt["model_record"]["sha256"]
assert digest(ROOT / "scripts/course_experiments/applications.py") == next(
    row["sha256"] for row in previous_receipt["inspected_inputs"] if row["path"] == "scripts/course_experiments/applications.py")
body_example = provenance["body_generated_examples"][0]
assert digest(ROOT / body_example["snapshot"]) == body_example["snapshot_sha256"]

receipt["original_report"] = {"path": str(original_report_path.relative_to(ROOT)), "sha256": digest(original_report_path), "previous_issues": original_report["issues"]}
receipt["previous_artifacts"] = previous_artifacts
receipt["source_diff"] = {"command": diff_command, "returncode": diff_run.returncode,
    "stdout": diff_run.stdout.decode(), "stderr": diff_run.stderr.decode(),
    "baseline_sha256": hashlib.sha256(old_run.stdout).hexdigest(), "current_sha256": digest(chapter),
    "introduction_unchanged": True}
receipt["programs_compared_to_previous_run"] = unchanged_programs
receipt["changed_dependency"] = {"page_id": changed_page["page_id"], "snapshot": changed_page["snapshot"],
    "sha256": changed_page["source_sha256"], "paragraphs": changed_paragraphs,
    "illustrative_announcements": announcements, "query_day": query_day.isoformat(),
    "latest_publication_address": latest_publication["address"],
    "currently_applicable_addresses": [row["address"] for row in applicable],
    "lexical_rank_with_original_dates": rank_before, "lexical_rank_with_swapped_dates": rank_after,
    "exercise_stdout": exercise_stdout.getvalue(),
    "scope": "人工設定的日期反例與正文練習；只驗證需要區分身分、發布日與生效日，未提出完整文件裁決演算法或宣稱模型做得到。"}
receipt["reused_provenance"] = {"path": str(previous_provenance.relative_to(ROOT)), "sha256": digest(previous_provenance),
    "experiment_record_unchanged": True, "current_generator_source_unchanged": True, "quoted_body_example_unchanged": True}
print(json.dumps(receipt, ensure_ascii=False, indent=2))

