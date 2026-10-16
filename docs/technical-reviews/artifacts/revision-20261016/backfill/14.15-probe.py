"""Independent, bounded CPU audit of frozen lesson 14.15; never trains/evaluates a test anew."""

import collections
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices
from tiny_perceptron.lesson_products.model import Config, LoopedSolver, Tokenizer, padded
from tiny_perceptron.lesson_products.training import state_digest


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


torch.set_num_threads(1)
result = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU", "platform": platform.platform()}, "scope": "Reaggregates existing primary logits; reads actual checkpoints and receipts; one invented-world CPU forward/backward only, no optimizer step, no retraining, no formal test restart."}
freeze = read("docs/course-revision-20261016/freeze-14.json")
page = next(x for x in freeze["pages"] if x["page_id"] == "14.15")
body = lesson_slices((ROOT / "course/chapters/14.md").read_text())["14.15"]
assert body.encode() == (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(body.encode()).hexdigest() == page["source_sha256"]
result["section"] = {"snapshot": page["snapshot"], "sha256": page["source_sha256"], "live_equals_frozen": True}

report = read("docs/lesson-products/results/final-report-v1.json")
settings = read("docs/lesson-products/results/settings-freeze-v1.json")
pretest = read("docs/lesson-products/results/pretest-freeze-record-v1.json")
assert report["settings_freeze_sha256"] == sha(ROOT / "docs/lesson-products/results/settings-freeze-v1.json") == pretest["settings_freeze_sha256"]
assert pretest["test_outcomes_observed"] is False and settings["test_outcomes_observed"] is False
assert report["data_manifest_sha256"] == sha(ROOT / "outputs/lesson-products/data-v2/manifest.json") == sha(ROOT / "docs/lesson-products/data-v2-manifest.json")
raw = ROOT / "outputs/lesson-products/final-report-v1.predictions.jsonl"
assert sha(raw) == report["predictions"]["sha256"]
assert raw.stat().st_size == report["predictions"]["bytes"]

dataset = [json.loads(line) for line in (ROOT / "outputs/lesson-products/data-v2/b.jsonl").open()]
test = {r["id"]: r for r in dataset if r["split"] == "test"}
train_families = {r["family_id"] for r in dataset if r["split"] == "train"}
boxes = {i: r for i, r in test.items() if r["task"] == "boxes"}
assert len(boxes) == 2956 and len({r["family_id"] for r in boxes.values()}) == 1478
assert not train_families & {r["family_id"] for r in boxes.values()}
assert {len(r["world"]["events"]) for r in boxes.values()} == {1, 2, 3, 4}
for row in boxes.values():
    world = row["world"]
    values = {"A": world["a"], "B": world["b"]}
    for box, value in world["events"]:
        if box == "swap":
            values["A"], values["B"] = values["B"], values["A"]
        else:
            values[box] = value
    assert values[world["query"]] == row["gold"]
example_values = {"A": 2, "B": 6}
example_values["A"], example_values["B"] = example_values["B"], example_values["A"]
assert example_values == {"A": 6, "B": 2}
result["task_label_audit"] = {"swap_example_answers": example_values, "boxes_test_count": len(boxes), "boxes_test_families": 1478, "test_event_length_histogram": dict(collections.Counter(len(r["world"]["events"]) for r in boxes.values())), "all_test_box_gold_labels_match_sequential_rule": True, "train_and_test_box_families_disjoint": True}

grouped = {mode: {d: [] for d in range(1, 7)} for mode in ("fixed3", "odd")}
seen = {mode: set() for mode in grouped}
for line in raw.open():
    row = json.loads(line)
    if row["product"] != "mechanism" or row["task"] != "boxes":
        continue
    mode = row["mode"]
    assert row["id"] not in seen[mode] and row["split"] == "test"
    seen[mode].add(row["id"])
    original = boxes[row["id"]]
    assert row["gold"] == original["gold"] and row["family_id"] == original["family_id"]
    for depth in grouped[mode]:
        scores = row["scores_by_depth"][str(depth)]
        assert len(scores) == 19 and all(math.isfinite(s) for s in scores)
        m = max(scores)
        ce = m + math.log(math.fsum(math.exp(s - m) for s in scores)) - scores[row["gold"]]
        pred = max(range(19), key=lambda k: scores[k])
        grouped[mode][depth].append((ce, pred == row["gold"]))
assert seen["fixed3"] == seen["odd"] == set(boxes)
table = []
for line in body.splitlines():
    match = re.match(r"\| ([1-6]) \| ([\d.]+)／([\d.]+)% \| ([\d.]+)／([\d.]+)% \|", line)
    if match:
        table.append([float(v) for v in match.groups()])
assert len(table) == 6
result["recomputed_table"] = {}
for mode, column in (("fixed3", 1), ("odd", 3)):
    result["recomputed_table"][mode] = {}
    for d, pairs in grouped[mode].items():
        count = len(pairs)
        correct = sum(x[1] for x in pairs)
        ce = math.fsum(x[0] for x in pairs) / count
        accuracy = correct / count
        recorded = report["mechanisms"][mode]["evaluation"][str(d)]["boxes"]
        assert recorded["count"] == count == 2956 and recorded["correct"] == correct and recorded["families"] == 1478
        assert abs(ce - recorded["cross_entropy"]) < 1e-6
        assert abs(accuracy - recorded["accuracy"]) < 1e-7
        assert f"{ce:.4f}" == f"{table[d-1][column]:.4f}"
        assert f"{accuracy * 100:.2f}" == f"{table[d-1][column+1]:.2f}"
        result["recomputed_table"][mode][str(d)] = {"count": count, "families": 1478, "correct": correct, "ce_float64": ce, "accuracy_exact_ratio": accuracy, "display": f"{ce:.4f}／{accuracy * 100:.2f}%"}

result["paired_training"] = {}
last_values = {}
for mode, stem in (("fixed3", "mechanism-fixed3"), ("odd", "mechanism-odd135")):
    directory = ROOT / f"outputs/lesson-products/mechanisms/{mode}"
    runner = read(f"docs/lesson-products/results/{stem}-runner-result.json")
    execution, job = runner["execution"], runner["execution"]["job"]
    receipt = json.loads((directory / "training-receipt.json").read_bytes())
    value = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    last_values[mode] = value
    meta = value["metadata"]
    frozen = settings["mechanisms"][mode]
    assert runner["status"] == "completed" and runner["gpu_execution_status"] == "confirmed"
    assert execution["training"]["last_step"] == meta["step"] == receipt["last_step"] == 1500
    assert job["seed"] == receipt["seed"] == meta["seed"] == 8159
    assert job["batch_size"] == receipt["batch_size"] == meta["training_recipe"]["batch_size"] == 128
    for name in ("last.pt", "training-receipt.json"):
        assert sha(directory / name) == frozen[name]
    assert meta["mode"] == mode and meta["initial_weights_sha256"] == frozen["initial_weights_sha256"]
    assert receipt["initial_weights_sha256"] == execution["training"]["initial_weights_sha256"] == frozen["initial_weights_sha256"]
    assert meta["data_manifest_sha256"] == receipt["data_manifest_sha256"] == report["data_manifest_sha256"]
    for name in ("model.py", "training.py", "train.py", "data.py"):
        path = "tiny_perceptron/lesson_products/" + name
        assert sha(ROOT / path) == execution["source_hashes"][path]
    depths = [3] * 1500 if mode == "fixed3" else [(1, 3, 5)[int(hashlib.sha256(f"8159:{i}".encode()).hexdigest()[:8], 16) % 3] for i in range(1, 1501)]
    work = sum(depths) * 128
    assert work == receipt["work"]["block_sequence_work"] == execution["training"]["work"]["block_sequence_work"]
    result["paired_training"][mode] = {"checkpoint_sha256": sha(directory / "last.pt"), "model_state_sha256": state_digest(value["model"]), "initial_weights_sha256": meta["initial_weights_sha256"], "last_step": meta["step"], "policy_best_step_not_used_for_table": receipt["selected_step"], "seed": receipt["seed"], "batch_size": receipt["batch_size"], "data_manifest_sha256": meta["data_manifest_sha256"], "depth_histogram": dict(collections.Counter(depths)), "block_sequence_work": work, "gpu_primary_receipt": {"run_id": runner["run_id"], "gpu_name": execution["gpu_name"], "code_revision": execution["code_revision"]}}
assert last_values["fixed3"]["config"] == last_values["odd"]["config"]
assert result["paired_training"]["fixed3"]["initial_weights_sha256"] == result["paired_training"]["odd"]["initial_weights_sha256"]
torch.manual_seed(8159)
recreated_initial = LoopedSolver(Config(**last_values["odd"]["config"]))
result["recreated_seed8159_initial_state_sha256"] = state_digest(recreated_initial.state_dict())
assert result["recreated_seed8159_initial_state_sha256"] == result["paired_training"]["odd"]["initial_weights_sha256"]
result["work_increase_percent"] = (580096 - 576000) / 576000 * 100
assert round(result["work_increase_percent"], 2) == 0.71

value = last_values["odd"]
model = LoopedSolver(Config(**value["config"])).eval()
model.load_state_dict(value["model"])
tokenizer = Tokenizer.load(ROOT / "outputs/lesson-products/mechanisms/odd/tokenizer.json")
tokens = padded([tokenizer.encode("兩盒追蹤;A=2;B=6;事件=交換;查詢=B")], model.config.max_length)
with torch.no_grad():
    all_scores = model(tokens, loops=6, all_depths=True)
    diffs = {d: float((model(tokens, loops=d) - all_scores[:, d-1]).abs().max()) for d in range(1, 7)}
assert all(v < 1e-5 for v in diffs.values())
states = []
def retain(module, args, output):
    output.retain_grad()
    states.append(output)
handle = model.shared.register_forward_hook(retain)
loss = F.cross_entropy(model(tokens, loops=5), torch.tensor([2], dtype=torch.long))
loss.backward()
handle.remove()
norms = [float(h.grad.norm()) for h in states]
assert len(states) == 5 and all(n > 0 for n in norms)
result["bounded_cpu_probe"] = {"invented_input": "兩盒追蹤;A=2;B=6;事件=交換;查詢=B", "gold": 2, "separate_depth_vs_all_depths_max_abs_difference": diffs, "five_loop_terminal_ce": float(loss.detach()), "hidden_gradient_norms_rounds_1_to_5": norms, "optimizer_steps": 0, "scope": "Demonstrates executable 1..6 API and nonzero indirect gradient at hidden rounds 2,4 for this invented world; not an accuracy estimate or proof of universally nonzero gradients."}
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    exec(re.search(r"```python\n(.*?)```", body, re.S)[1], {})
assert printed.getvalue() == "1 0.9\n2 0.1\n3 0.9\n4 0.1\n5 0.9\n6 0.1\n"
result["counterexample"] = {"executed_frozen_code_stdout": printed.getvalue(), "formula": "state_r=(-1)^(r+1); odd r -> probability_correct=0.9, even r -> 0.1"}
b = read("outputs/lesson-products/products/b/training-receipt.json")
assert b["seed"] == 7421 and b["mode"] == "multi" and b["depth_weights"] == [1] * 5
result["deployment_b_separate_recipe"] = {"seed": b["seed"], "mode": b["mode"], "depth_weights": b["depth_weights"], "direct_supervision": [1, 2, 3, 4, 5]}

svg_path = freeze["figure_snapshots"]["course/figures/looped-mechanism-measured.svg"]["path"]
assert sha(ROOT / svg_path) == sha(ROOT / "course/figures/looped-mechanism-measured.svg") == page["figures_sha256"]["course/figures/looped-mechanism-measured.svg"]
tree = ET.parse(ROOT / svg_path)
ns = {"s": "http://www.w3.org/2000/svg"}
lines = []
for group in tree.findall(".//s:g", ns):
    if not group.get("id", "").startswith("line2d_"):
        continue
    path = group.find("s:path", ns)
    if path is None:
        continue
    style = path.get("style", "")
    color = next((c for c in ("#175a97", "#b34813") if c in style), None)
    points = re.findall(r"[ML]\s+([\d.]+)\s+([\d.]+)", path.get("d", ""))
    if color and len(points) == 6:
        lines.append((color, [(float(x), float(y)) for x,y in points]))
assert len(lines) == 4
fig_errors = []
for idx, (color, points) in enumerate(lines):
    mode = "fixed3" if color == "#175a97" else "odd"
    metric = "ce_float64" if idx < 2 else "accuracy_exact_ratio"
    values = [result["recomputed_table"][mode][str(d)][metric] for d in range(1,7)]
    target = [math.log10(v) for v in values] if idx < 2 else [100 * v for v in values]
    slope = (points[1][1] - points[0][1]) / (target[1] - target[0])
    intercept = points[0][1] - slope * target[0]
    error = max(abs(intercept + slope * v - p[1]) for v, p in zip(target, points))
    assert error < 3e-5 and slope < 0
    assert max(abs((points[i][0] - points[i-1][0]) - (points[1][0]-points[0][0])) for i in range(1,6)) < 2e-6
    fig_errors.append({"mode": mode, "metric": "log10 CE" if idx < 2 else "accuracy percent", "max_svg_coordinate_error": error})
result["figure_coordinate_check"] = {"sha256": sha(ROOT / svg_path), "series": fig_errors, "scope": "All 24 plotted points match primary-logit aggregates under affine x and log10-CE / linear-percent y transforms."}
paths = [page["snapshot"], "docs/course-revision-20261016/freeze-14.json", "docs/lesson-products/results/final-report-v1.json", "docs/lesson-products/results/settings-freeze-v1.json", "docs/lesson-products/results/pretest-freeze-record-v1.json", "docs/lesson-products/results/mechanism-fixed3-runner-result.json", "docs/lesson-products/results/mechanism-odd135-runner-result.json", "tiny_perceptron/lesson_products/model.py", "tiny_perceptron/lesson_products/training.py", "tiny_perceptron/lesson_products/train.py", "tiny_perceptron/lesson_products/data.py", "tiny_perceptron/lesson_products/evaluation.py", "scripts/lesson_products/plot_results.py", "outputs/lesson-products/products/b/training-receipt.json", "outputs/lesson-products/data-v2/b.jsonl", "outputs/lesson-products/data-v2/manifest.json", "outputs/lesson-products/final-report-v1.predictions.jsonl", svg_path]
result["inspected_input_sha256"] = {p: sha(ROOT / p) for p in paths}
result["status"] = "all assertions passed"
destination = Path(__file__).with_name("14.15-probe-result.json")
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": result["status"], "output": str(destination.relative_to(ROOT)), "table": result["recomputed_table"], "paired_training": result["paired_training"], "bounded_cpu_probe": result["bounded_cpu_probe"], "figure": result["figure_coordinate_check"]}, ensure_ascii=False, indent=2))
