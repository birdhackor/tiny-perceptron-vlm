"""5.13 的獨立短 CPU 核對；不訓練、不改動教材、權重只讀。"""
import ast
import contextlib
import datetime
import hashlib
import io
import json
import platform
import random
import re
from pathlib import Path

import torch
from scripts.course_experiments.common import _nll, records_sha256, split_records, text_examples
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import load_checkpoint

ROOT = Path.cwd()
OUT = ROOT / "docs/technical-reviews/artifacts/omni-technical-technical_5_13-v1"
CACHE = Path("/workspace/work/reference/omni-technical-technical_5_13-v1")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section(path, identifier):
    text = path.read_text()
    return re.search(r"^## " + re.escape(identifier) + r" .*?(?=^## |\Z)", text, re.M | re.S).group()


def run_block(code):
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(code, "5.13-inline", "exec"), {})
    return [ast.literal_eval(line) for line in capture.getvalue().splitlines()]


@torch.no_grad()
def independent_nll(model, examples):
    model.eval()
    total, count = 0.0, 0
    for x, y in examples:
        logits = model(x[None])["logits"][0]
        valid = y != IGNORE
        selected = logits.log_softmax(-1)[valid].gather(1, y[valid, None])
        total += float((-selected).sum())
        count += int(valid.sum())
    return {"nll": total / count, "nll_sum": total, "effective_tokens": count, "examples": len(examples)}


def functions(path):
    tree = ast.parse(path.read_text())
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, ast.FunctionDef)}


torch.set_num_threads(2)
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
snapshot = ROOT / "docs/course-review-20261019/freeze-01/sources/5.13.md"
raw = snapshot.read_text()
current = section(ROOT / "course/chapters/05.md", "5.13")
assert raw == current
assert digest(snapshot) == "b605ca3c39976c1d3b5da3efbf1ee74d7e0416c17ac6071ff65a36c59fdf6aa0"
code = re.search(r"```python\n(.*?)```", raw, re.S).group(1)
rows = run_block(code)
exercise = run_block(code.replace("batch_size, answers_per_record, steps = 4, 8, 100", "batch_size, answers_per_record, steps = 4, 4, 200"))
half = run_block(code.replace("batch_size, answers_per_record, steps = 4, 8, 100", "batch_size, answers_per_record, steps = 4, 4, 100"))
assert len(rows) == len(exercise) == len(half) == 4
assert [row["有效目標預算"] for row in rows] == [3200] * 4
assert [row["有效目標預算"] for row in exercise] == [3200] * 4
assert [row["有效目標預算"] for row in half] == [1600] * 4
parameters = {}
for width in (8, 16, 32):
    model = TinyLM(ModelConfig(width=width))
    count = sum(p.numel() for p in model.parameters())
    assert count == 12 * width * width + 667 * width
    parameters[str(width)] = {"parameters": count, "formula": f"12*{width}^2+667*{width}", "dtype": str(next(model.parameters()).dtype)}

result_path = ROOT / "docs/course-experiments/results/text_foundation.json"
historical = json.loads(result_path.read_text())
public_path = ROOT / "docs/course-experiments/public-releases/text_foundation.json"
public = json.loads(public_path.read_text())["release"]["public_manifest"]
export_path = ROOT / "checkpoints/course/text_foundation/export-manifest.json"
exports = json.loads(export_path.read_text())
assert exports["provenance"]["revision"] == historical["revision"]
pool = [
    {"text": f"object={index};color={color};shape={shape}.", "family": str(index)}
    for index in range(80)
    for color, shape in [("red" if index % 2 else "blue", "circle" if index % 3 else "square")]
]
parts = split_records(pool, seed=42)
split_receipts = {}
for name, records in parts.items():
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode()
    h = hashlib.sha256(encoded).hexdigest()
    expected = historical["results"]["scaling_data"][name]
    assert h == expected["sha256"]
    assert len(records) == expected["records"]
    split_receipts[name] = {"records": len(records), "sha256": h, "families": [r["family"] for r in records]}
assert all(not ({r["family"] for r in parts[a]} & {r["family"] for r in parts[b]}) for a,b in (("train","validation"),("train","test"),("validation","test")))

empirical = {}
for name, original in historical["results"]["scaling"].items():
    file = CACHE / (name + ".pt")
    entry = next(f for f in public["files"] if f["output"] == file.name)
    export = next(f for f in exports["files"] if f["output"] == file.name)
    private = next(f for f in historical["artifacts"] if f["path"] == file.name)
    assert digest(file) == entry["sha256"] == export["sha256"]
    assert file.stat().st_size == entry["bytes"] == export["bytes"]
    assert export["source_sha256"] == private["sha256"]
    model, payload = load_checkpoint(file, device="cpu")
    width = int(name.split("-w")[1].split("-")[0])
    records = int(name.split("-n")[1])
    assert model.config.width == width and model.config.layers == 1
    assert sum(p.numel() for p in model.parameters()) == original["parameters"]
    assert original["training"]["steps"] == 150
    metadata = payload["metadata"]
    assert metadata["revision"] == historical["revision"]
    assert metadata["experiment_id"] == historical["experiment_id"] == "text_foundation"
    assert historical["seed"] == 42
    training = parts["train"][:records]
    assert records_sha256(training) == original["training"]["records_sha256"]
    examples = text_examples(training, "text", model.config.max_length)
    rng = random.Random(42)
    counts = [sum(int((y != IGNORE).sum()) for _, y in rng.choices(examples, k=4)) for _ in range(150)]
    assert sum(counts) == original["training"]["effective_tokens"]
    evaluations = {}
    for split, rows_for_split in (("train", training), ("validation", parts["validation"]), ("test", parts["test"])):
        xs = text_examples(rows_for_split, "text", model.config.max_length)
        observed = _nll(model, xs)
        independent = independent_nll(model, xs)
        expected = original["training"]["final_loss"] if split == "train" else original["evaluation"][split]["nll"]
        assert abs(observed["nll"] - independent["nll"]) < 2e-6
        assert abs(observed["nll"] - expected) < 5e-6
        assert f"{observed['nll']:.5f}" == f"{expected:.5f}"
        if split != "train":
            assert observed["effective_tokens"] == original["evaluation"][split]["effective_tokens"]
        evaluations[split] = {**observed, "independent_nll": independent["nll"], "historical_nll": expected, "absolute_difference": abs(observed["nll"]-expected), "rounded_5dp": f"{observed['nll']:.5f}"}
    empirical[name] = {
        "public_checkpoint": {"path": str(file), "sha256": digest(file), "bytes": file.stat().st_size, "repo": public["repo"], "revision": public["revision"], "source_training_checkpoint_sha256": export["source_sha256"]},
        "payload_format": payload["format_version"], "config": payload["config"], "historical_training_steps": original["training"]["steps"],
        "metadata": metadata, "public_format_limit":"Inference export omits original step, sampler and optimizer state; those historical settings are checked against original result and unchanged generating functions, not inferred from public payload.",
        "records": records, "parameters": original["parameters"], "effective_tokens_recomputed_from_sampler": sum(counts),
        "per_update_effective_target_min_max": [min(counts),max(counts)],
        "evaluation": evaluations,
    }

comparisons = {}
for width in (16,32):
    small = empirical[f"scaling-w{width}-n16"]["evaluation"]
    large = empirical[f"scaling-w{width}-n64"]["evaluation"]
    assert large["train"]["nll"] > small["train"]["nll"]
    assert large["validation"]["nll"] < small["validation"]["nll"]
    comparisons[str(width)] = {"more_data_training_loss_higher": True, "more_data_validation_loss_lower": True}

code_inspection = {}
for path, names in {
    "scripts/course_experiments/common.py": ["new_lm","split_records","records_sha256","text_examples","_nll","fit_lm"],
    "scripts/course_experiments/text.py": ["run_text_foundation","_evaluations","_save_splits"],
}.items():
    original_file = CACHE / ("historical-" + Path(path).name)
    assert digest(original_file) == historical["code_sha256"][path]
    a, b = functions(ROOT/path), functions(original_file)
    matches = {n: a[n] == b[n] for n in names}
    assert all(matches.values())
    code_inspection[path] = {"current_sha256":digest(ROOT/path),"historical_sha256":digest(original_file),"historical_commit":historical["revision"],"relevant_functions_ast_equal":matches}
for path in ["tiny_perceptron/model.py","tiny_perceptron/attention.py","tiny_perceptron/modern.py","tiny_perceptron/data.py","tiny_perceptron/training.py"]:
    assert digest(ROOT/path) == historical["code_sha256"][path]
    code_inspection[path] = {"current_sha256":digest(ROOT/path),"historical_full_file_equal":True}

prerequisites = []
for file, identifier in [("course/chapters/05.md","5.9"),("course/chapters/05.md","5.2"),("course/chapters/01.md","1.8"),("course/chapters/02.md","2.3"),("course/training.md","T.4")]:
    text = section(ROOT/file, identifier)
    prerequisites.append({"source":file+"#"+identifier,"section_sha256":hashlib.sha256(text.encode()).hexdigest(),"bytes":len(text.encode()),"inspection":"Read current precise section; supports parameter counting, effective-target weighting, bias, or linked fixed-recipe command. No unrelated chapter pinned."})

value = {
    "reviewer_task":"/root/omni_technical_coordinator/technical_5_13", "started_at":start,
    "completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command":"uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_5_13-v1/verify.py",
    "environment":{"python":platform.python_version(),"torch":str(torch.__version__),"torch_git":torch.version.git_version,"device":"cpu","dtype":"torch.float32","threads":str(torch.get_num_threads())},
    "source_snapshot_sha256":digest(snapshot), "current_source_equal":True,
    "inline_code_sha256":hashlib.sha256(code.encode()).hexdigest(),"inline_output":rows,"exercise_output":exercise,"half_budget_output":half,
    "parameters":parameters,"linear_flops":{"8x8_approx":2*8*8,"16x16_approx":2*16*16,"convention":"2 FLOPs per multiply-accumulate; ignore bias and lower-order terms, not an exact instruction count."},
    "historical_result":{"path":str(result_path.relative_to(ROOT)),"sha256":digest(result_path),"revision":historical["revision"],"device":historical["device"],"gpu":historical["gpu"],"torch":historical["torch_version"],"seed":historical["seed"]},
    "manifest_sha256":{"public_release":digest(public_path),"export_manifest":digest(export_path)},
    "split_receipts":split_receipts,"empirical":empirical,"comparisons":comparisons,"code_inspection":code_inspection,"current_prerequisites_read":prerequisites,
    "scope":"Read-only CPU re-evaluation of four published tiny checkpoints, deterministic sampler and arithmetic audit. No optimizer steps, no fresh training, no GPU, no claim about large-model scaling laws.",
    "result":"All assertions passed; all 12 table loss cells round to the printed five decimal places; all four exposure totals match."
}
path = OUT / "execution.json"
with path.open("x") as stream:
    json.dump(value,stream,ensure_ascii=False,indent=2)
    stream.write("\n")
print(json.dumps({"result":value["result"],"environment":value["environment"],"empirical":{k:{s:v["rounded_5dp"] for s,v in row["evaluation"].items()} for k,row in empirical.items()},"output":str(path.relative_to(ROOT)),"sha256":digest(path)},ensure_ascii=False,indent=2))
