"""8.5 的有界離線核對；執行原文程式，重算既有七筆輸出，不訓練或生成。"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
ORIGINALS = Path("/workspace/work/reference/omni-technical-technical_8_5-v1")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def definition(raw, name):
    tree = ast.parse(raw)
    node = next(item for item in tree.body if getattr(item, "name", None) == name)
    return ast.Module(body=[node], type_ignores=[])


freeze = json.loads(FREEZE.read_bytes())
checked_pages = {}
for lesson in ("7.13", "8.4", "8.5"):
    page = next(item for item in freeze["pages"] if item["page_id"] == lesson)
    raw = (ROOT / page["snapshot"]).read_bytes()
    chapter = (ROOT / page["source"]).read_bytes()
    headings = list(re.finditer(rb"^## .+$", chapter, re.M))
    index = next(i for i, item in enumerate(headings) if item[0].startswith(f"## {lesson} ".encode()))
    end = headings[index + 1].start() if index + 1 < len(headings) else len(chapter)
    current = chapter[headings[index].start():end]
    assert raw == current and digest(raw) == page["source_sha256"]
    checked_pages[lesson] = {"snapshot": page["snapshot"], "sha256": digest(raw), "matches_current_slice": True}

page = next(item for item in freeze["pages"] if item["page_id"] == "8.5")
raw = (ROOT / page["snapshot"]).read_text()
blocks = re.findall(r"```python\n(.*?)```", raw, re.S)
assert len(blocks) == 1 and page["figures_sha256"] == {}
block = blocks[0]
stdout = io.StringIO()
namespace = {}
with contextlib.redirect_stdout(stdout):
    exec(compile(block, page["snapshot"] + ":python-block", "exec"), namespace)
expected = ['{"answer":4} 格式 True 內容 True', '答案是：{"answer":4} 格式 False 內容 False', '{"answer":5} 格式 True 內容 False']
assert stdout.getvalue().splitlines() == expected

# 保留原 for/try/except 全部行，僅以外部輸入替代答案清單。
tree = ast.parse(block)
loop_tree = ast.Module(body=[item for item in tree.body if isinstance(item, (ast.Import, ast.For))], type_ignores=[])
loop = compile(loop_tree, "8.5-exact-for-try-except", "exec")


def check(text):
    local = {"answers": [text]}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        exec(loop, local)
    return {"text": text, "format": local["valid"], "content": local["correct"]}


cases = [
    ('{"answer":4,"note":"完成"}', False, False),
    ('{}', False, False),
    ('{"result":4}', False, False),
    ('[4]', False, False),
    ('4', False, False),
    ('{"answer":4.0}', True, True),
    ('{"answer":"4"}', True, False),
    ('{"answer":true}', True, False),
    ('{"answer":NaN}', True, False),
    ('{"answer":4,"answer":5}', True, False),
    ('{"answer":5,"answer":4}', True, True),
    ('{"answer":4} trailing', False, False),
]
boundary_results = []
for text, valid, correct in cases:
    item = check(text)
    assert (item["format"], item["content"]) == (valid, correct)
    boundary_results.append(item)
assert math.isnan(json.loads("NaN"))

# Notebook 中的教學 cell 與 Markdown 的 fenced block 相同。
notebook = ROOT / page["notebook"]
assert digest(notebook.read_bytes()) == page["notebook_sha256"]
cells = json.loads(notebook.read_bytes())["cells"]
assert any(c["cell_type"] == "code" and "".join(c["source"]).strip() == block.strip() for c in cells)

result_path = ROOT / "docs/course-experiments/results/style.json"
result = json.loads(result_path.read_bytes())
samples = [item for item in result["results"]["after"]["test"]["samples"] if item.get("style") == "json"]
assert len(samples) == 7
historical_source = subprocess.check_output(["git", "show", result["revision"] + ":scripts/course_experiments/behavior.py"], cwd=ROOT)
assert digest(historical_source) == result["code_sha256"]["scripts/course_experiments/behavior.py"]
historical_cache = ORIGINALS / "behavior-recorded-revision.py"
historical_cache.write_bytes(historical_source)
historical_namespace = {"json": json}
exec(compile(definition(historical_source, "_style_metrics"), "recorded-style-metrics", "exec"), historical_namespace)

rows, reconstructed = [], []
for sample in samples:
    prompt = sample["messages"][0]["content"]
    match = re.fullmatch(r"style=json; (\d+)\+(\d+)=\?", prompt)
    assert match is not None
    a, b = map(int, match.groups())
    ids = sample["generated_ids"]
    assert ids[-1] == 2 and all(8 <= item < 264 for item in ids[:-1])
    decoded = bytes(item - 8 for item in ids[:-1]).decode("utf-8")
    assert decoded == sample["generated"]
    value = json.loads(decoded)
    assert isinstance(value, dict) and set(value) == {"answer"} and type(value["answer"]) is int
    assert json.loads(sample["expected"]) == {"answer": a + b}
    assert value["answer"] != a + b
    rows.append({"a": a, "b": b, "style": "json", "family": sample["family"]})
    reconstructed.append({"prompt": prompt, "generated": decoded, "value": value["answer"], "expected_sum": a + b, "python_type": type(value["answer"]).__name__, "raw_ids_decode_match": True, "illegal_control_ids": [], "eos": True, "format": True, "content": False})
replayed = historical_namespace["_style_metrics"]({"samples": copy.deepcopy(samples)}, rows)["rubric"]["json"]
assert replayed == {"records": 7, "content_correct": 0, "style_correct": 7, "json_valid": 7}
assert replayed == result["results"]["after"]["test"]["rubric"]["json"]

source_receipts = []
for receipt_name in ("fetch-receipts.json", "functions-fetch-receipt.json"):
    receipt = json.loads((ORIGINALS / receipt_name).read_bytes())
    source_receipts.extend(receipt if isinstance(receipt, list) else [receipt])
for receipt in source_receipts:
    assert digest(Path(receipt["path"]).read_bytes()) == receipt["sha256"]

installed_json = {}
for name, official in (("__init__.py", "cpython-v3.13.5-json-init.py"), ("decoder.py", "cpython-v3.13.5-json-decoder.py")):
    path = Path(json.__file__).parent / name
    installed_json[name] = {"path": str(path), "sha256": digest(path.read_bytes()), "matches_official_v3_13_5": path.read_bytes() == (ORIGINALS / official).read_bytes()}

output = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_8_5-v1/verify.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU; standard-library diagnostics only; no model execution"},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE.read_bytes())},
    "checked_pages": checked_pages,
    "embedded_code_sha256": digest(block.encode()),
    "embedded_stdout": stdout.getvalue(),
    "boundary_results": boundary_results,
    "notebook": {"path": page["notebook"], "sha256": digest(notebook.read_bytes()), "teaching_cell_matches_markdown": True},
    "historical_result": {"path": str(result_path.relative_to(ROOT)), "sha256": digest(result_path.read_bytes()), "revision": result["revision"], "seed": result["seed"], "recorded_device": result["device"], "recorded_gpu": result["gpu"], "recorded_python": result["python_version"], "recorded_torch": result["torch_version"], "conditional_training_steps": result["results"]["training"]["steps"], "complete_test_records": result["results"]["after"]["test"]["records"], "json_test_records": len(samples), "rubric_recomputed": replayed, "recorded_metric_source_sha256": digest(historical_source), "recorded_metric_source_cache": str(historical_cache), "rows": reconstructed},
    "official_source_receipts": source_receipts,
    "installed_json_sources": installed_json,
    "scope": "Executed exact 8.5 Markdown code and bounded boundary diagnostics; independently recounted seven saved JSON predictions and replayed their recorded-revision metric. No retraining, checkpoint download, fresh generation, GPU, or paid API.",
    "all_assertions_passed": True,
}
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
