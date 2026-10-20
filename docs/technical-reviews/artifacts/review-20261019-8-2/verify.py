"""核對提示算例、生成時的參數與既有風格實驗紀錄。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.behavior import _style_record
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, generate


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    (ARTIFACTS / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def capture(code):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "prompt-example", "exec"), {})
    return output.getvalue()


torch.set_num_threads(1)
environment = {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"}
run = ROOT / "outputs/reviewer-tools/runs/20261020T061429931621Z"
section_execution = {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/08.md#8.2" --execute',
    "environment": json.loads((run / "environment.json").read_bytes()),
    "execution": json.loads((run / "execution.json").read_bytes()),
    "extraction": json.loads((run / "extraction.json").read_bytes()),
    "stdout": (run / "stdout.txt").read_text(),
    "stderr": (run / "stderr.txt").read_text(),
    "code": (run / "fence-1.py").read_text(),
}
assert section_execution["execution"]["exit_code"] == 0
assert section_execution["extraction"]["source_sha256"] == "cc8bdd22a29d26ddad4fa1e52ce41f0fa9d4d488f8153eb8c3931a84aa5e5277"
assert section_execution["extraction"]["figure_sha256"] == {}
write("section-execution.json", section_execution)
code = section_execution["code"]
assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(ast.parse(code)))
assert {node.func.id for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)} == {"print"}
original = capture(code)
modified = capture(code.replace('"用貼切比喻回答"', '"只回一個數字"'))
assert original == section_execution["stdout"]
assert original.splitlines() == ["用一句話回答：2+2=?", "用貼切比喻回答：2+2=?", "兩次都使用同一組參數與同一個選字規則"]
assert modified.splitlines() == ["用一句話回答：2+2=?", "只回一個數字：2+2=?", "兩次都使用同一組參數與同一個選字規則"]

torch.manual_seed(42)
tok = ByteTokenizer()
model = TinyLM(ModelConfig(vocab_size=tok.vocab_size, width=16, layers=1, heads=2, max_length=128))
before = {name: value.clone() for name, value in model.state_dict().items()}
prompts = ["用一句話回答：2+2=?", "用貼切比喻回答：2+2=?"]
ids = [torch.tensor([[tok.bos_id] + tok.encode(prompt)]) for prompt in prompts]
first = generate(model, ids[0], max_new_tokens=3)
second = generate(model, ids[1], max_new_tokens=3)
repeated = generate(model, ids[0], max_new_tokens=3)
assert torch.equal(first, repeated)
assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
generator = torch.Generator().manual_seed(42)
draws = torch.multinomial(torch.tensor([0.5, 0.5]), 32, replacement=True, generator=generator)
assert len(set(draws.tolist())) == 2

formal_path = ROOT / "docs/course-experiments/results/style.json"
formal = json.loads(formal_path.read_bytes())
results = formal["results"]
styles = ["concise", "vivid", "json"]
expected = {"concise": "3", "vivid": "3，像把兩組積木合在一起再數。", "json": '{"answer": 3}'}
probes = results["prompt_only_comparison_same_weights"]
selected = {}
for style in styles:
    samples = probes[style]["samples"]
    assert len(samples) == probes[style]["records"] == 7
    matching = [s for s in samples if s["messages"] == [{"role": "user", "content": f"style={style}; 2+2=?"}]]
    assert len(matching) == 1
    sample = matching[0]
    assert sample["generated"] == expected[style]
    assert sample["eos"] and sample["generated_ids"][-1] == tok.eos_id
    assert tok.decode(sample["generated_ids"][:-1]) == sample["generated"]
    paired = [s for s in results["after"]["test"]["samples"] if s["messages"] == sample["messages"]]
    assert len(paired) == 1 and paired[0]["generated_ids"] == sample["generated_ids"]
    record = _style_record({"a": 2, "b": 2, "family": "addition"}, style, True)
    assert record["messages"][0] == sample["messages"][0]
    selected[style] = {"sample": sample, "training_record_example": record}

behavior_path = "scripts/course_experiments/behavior.py"
current_behavior = (ROOT / behavior_path).read_bytes()
current_tree = ast.parse(current_behavior)
recorded_hash = formal["code_sha256"][behavior_path]
historical = None
for revision in subprocess.check_output(["git", "log", "--all", "--format=%H", "--", behavior_path], cwd=ROOT, text=True).splitlines():
    raw = subprocess.check_output(["git", "show", revision + ":" + behavior_path], cwd=ROOT)
    if sha(raw) == recorded_hash:
        historical = {"revision": revision, "sha256": sha(raw), "functions": {}}
        historical_tree = ast.parse(raw)
        for name in ["_style_record", "run_style"]:
            node = function(historical_tree, name)
            same = ast.dump(node, include_attributes=False) == ast.dump(function(current_tree, name), include_attributes=False)
            assert same
            historical["functions"][name] = {"matches_current_ast": same, "source": ast.get_source_segment(raw.decode(), node)}
        break
assert historical is not None
inference_sources = {}
for path, name in [("scripts/course_experiments/common.py", "evaluate_lm"), ("tiny_perceptron/model.py", "generate")]:
    raw = (ROOT / path).read_bytes()
    assert sha(raw) == formal["code_sha256"][path]
    inference_sources[path] = {"sha256": sha(raw), "matches_formal_run": True, "source": ast.get_source_segment(raw.decode(), function(ast.parse(raw), name))}
assert results["training"]["steps"] == 1000
assert results["content_training"]["records"] == 49

source_specs = [
    {
        "id": "brown",
        "url": "https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html",
        "version": "NeurIPS 2020, volume 33, published abstract",
        "locator": "Abstract",
        "excerpts": ["GPT-3 is applied without any gradient updates or fine-tuning", "some datasets where GPT-3's few-shot learning still struggles"],
    },
    {
        "id": "generation",
        "url": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/generation_strategies.md",
        "version": "transformers v4.57.1",
        "locator": "Basic decoding methods / Greedy search; Sampling",
        "excerpts": ["It selects the next most likely token at each step.", "randomly selects a token based on the probability distribution"],
    },
    {
        "id": "conditioning",
        "url": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/llm_tutorial.md",
        "version": "transformers v4.57.1",
        "locator": "Text generation, opening paragraph",
        "excerpts": ["given some initial text (prompt) along with its own generated outputs"],
    },
]


def inspect_source(spec):
    argv = ["curl", "--fail", "--location", "--silent", "--show-error", spec["url"]]
    raw = subprocess.check_output(argv)
    normalized = re.sub(r"\s+", " ", raw.decode())
    for excerpt in spec["excerpts"]:
        assert excerpt in normalized
    return {**spec, "command_argv": argv, "accessed_on": datetime.now(UTC).date().isoformat(), "retrieved_bytes": len(raw), "retrieved_sha256": sha(raw), "checked_original": True}


with ThreadPoolExecutor(max_workers=3) as pool:
    source_receipts = list(pool.map(inspect_source, source_specs))
write("source-receipts.json", {"sources": source_receipts, "storage": "只保存支持主張的短摘錄；第三方全文只在讀取時存在記憶體。"})

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-2/verify.py",
    "environment": environment,
    "result": "所有斷言通過",
    "arithmetic": {"expression": "2+2", "observed": 2+2, "expected": 4, "tolerance": "精確相等"},
    "example": {"original_stdout": original, "exercise_stdout": modified, "model_call_count": 0},
    "generation_probe": {"seed": 42, "weights_unchanged": True, "repeat_first_prompt_same_ids": True, "first_ids": first.tolist(), "second_ids": second.tolist(), "sampling_equal_scores_draws": draws.tolist(), "scope": "未訓練小模型只核對推論不更新參數及提示不自動跨呼叫保留，不測風格品質。"},
    "formal_experiment": {
        "path": str(formal_path.relative_to(ROOT)), "sha256": sha(formal_path.read_bytes()),
        "environment": {key: formal[key] for key in ["python_version", "torch_version", "device", "gpu", "seed", "step_scale"]},
        "training_steps": results["training"]["steps"], "training_records": results["training"]["records"],
        "arithmetic_training_records": results["content_training"]["records"],
        "probe_records_per_style": {style: probes[style]["records"] for style in styles},
        "selected": selected, "scope": results["scope"],
        "historical_behavior": historical, "inference_sources": inference_sources,
        "limitation": "本回合執行紀錄核對與程式比較；沒有重新訓練、載入歷史權重或重現 GPU 生成。",
    },
    "freeze": {path: sha((ROOT/path).read_bytes()) for path in ["docs/course-review-20261019/freeze-01/sources/8.2.md", "docs/course-review-20261019/freeze-04.json"]},
}, ensure_ascii=False, indent=2))
