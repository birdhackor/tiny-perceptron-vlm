"""Small offline CPU checks of chapter-10 introduction's concrete navigation promises."""

import ast
import hashlib
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json"
EXPECTED_FREEZE = "1094ea8f343a414dc3cbe5401b5753641fdb467d17a4867ef87dc192fe412238"
assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == EXPECTED_FREEZE
manifest = json.loads(FREEZE.read_bytes())
pages = {page["page_id"]: page for page in manifest["pages"]}
torch.set_num_threads(1)
torch.manual_seed(0)
print("UTC", datetime.now(UTC).isoformat())
print("ENV", json.dumps({"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "threads": str(torch.get_num_threads()), "platform": platform.platform()}))
chapter = (ROOT / "course/chapters/10.md").read_text()
raw_pages = {}
for page_id in ["chapter-10"] + [f"10.{number}" for number in range(1, 13)]:
    page = pages[page_id]
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
    assert raw.decode() in chapter
    raw_pages[page_id] = raw.decode()
    print("FREEZE", page_id, page["source_sha256"], "snapshot=current slice")
assert pages["chapter-10"]["figures_sha256"] == {}
assert "![" not in raw_pages["chapter-10"]
print("INTRO_FIGURES", "none")


def blocks(page_id):
    main = raw_pages[page_id].split("<details>", 1)[0]
    return re.findall(r"```python\n(.*?)```", main, re.S)


def run_block(page_id, index, namespace):
    code = blocks(page_id)[index]
    print("BLOCK", page_id, index, "sha256=" + hashlib.sha256(code.encode()).hexdigest())
    exec(compile(code, str(ROOT / pages[page_id]["snapshot"]) + f":python-block-{index}", "exec"), namespace)


library_path = ROOT / "tiny_perceptron/multimodal.py"
library_ast = ast.parse(library_path.read_text())
for page_id, symbol in [("10.1", "scene"), ("10.2", "patchify"), ("10.5", "VisionEncoder"), ("10.7", "expand_modalities")]:
    reference = next(node for node in library_ast.body if getattr(node, "name", None) == symbol)
    found = [node for block in blocks(page_id) for node in ast.parse(block).body if getattr(node, "name", None) == symbol]
    assert len(found) == 1
    assert ast.dump(reference, include_attributes=False) == ast.dump(found[0], include_attributes=False)
    print("EXACT_LIBRARY_AST", page_id, symbol, True)

scene_ns = {}
run_block("10.1", 0, scene_ns)
assert scene_ns["image"][:, 8, 8].tolist() == [1.0, 0.0, 0.0]
assert scene_ns["image"][:, 0, 0].tolist() == [0.0, 0.0, 0.0]
patch_ns = {}
run_block("10.2", 0, patch_ns)
assert tuple(patch_ns["patches"].shape) == (1, 16, 48)
assert torch.equal(patch_ns["image"], patch_ns["restored"])

expand_ns = {}
run_block("10.7", 0, expand_ns)
assert tuple(expand_ns["x"].shape) == (1, 6, 8)
assert expand_ns["y"].tolist() == [[-100, -100, -100, -100, 73, 2]]

assert "精簡版" in raw_pages["10.8"] and "省略" in raw_pages["10.8"]
wrapper_ns = {}
run_block("10.8", 0, wrapper_ns)
run_block("10.8", 1, wrapper_ns)
wrapped = wrapper_ns["library"]
before = {name: parameter.detach().clone() for name, parameter in wrapped.named_parameters()}
with torch.no_grad():
    output = wrapped(wrapper_ns["sequence"], image=wrapper_ns["scene"]())
assert tuple(output["logits"].shape) == (1, 19, 264)
assert torch.isfinite(output["logits"]).all()
assert all(torch.equal(before[name], parameter) for name, parameter in wrapped.named_parameters())
print("IMAGE_PATH", "logits=(1,19,264); finite=True; parameters_unchanged=True")
print("TRAINING", "no optimizer, no parameter update, no quality evaluation")

retrieval_ns = {}
run_block("10.9", 0, retrieval_ns)
assert retrieval_ns["similarity"].argmax(-1).tolist() == [0, 1]
print("RETRIEVAL", "selects existing text columns; no text is generated")

time_ns = {}
run_block("10.12", 0, time_ns)
at = time_ns["at"]
frames = time_ns["frames"]
duration = time_ns["duration_ms"]
first = at(frames, time_ns["times_a"], duration, 1500.0)
second = at(frames, time_ns["times_b"], duration, 1500.0)
assert first[0] == 1 and second[0] == 3
assert torch.equal(first[1], time_ns["circle"][None])
assert torch.equal(second[1], time_ns["square"][None])
print("TIMESTAMP", "same frames, 1500 ms: A=index1=circle; B=index3=square; no shape recognition model")

glossary = ROOT / "course/glossary.md"
assert glossary.is_file()
assert all(term in glossary.read_text() for term in ["patch", "圖片入口", "影格", "時間戳"])
for path in ["course/chapters/10.md", "course/glossary.md", "tiny_perceptron/multimodal.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py"]:
    print("SOURCE_SHA256", path, hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
print("RESULT", "all introduction-specific checks passed; downstream figures and empirical training results were not reviewed")
