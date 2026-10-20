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
FREEZE = ROOT / "docs/course-review-20261019/freeze-04.json"
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
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-0A/audit.py",
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
print(json.dumps(receipt, ensure_ascii=False, indent=2))
