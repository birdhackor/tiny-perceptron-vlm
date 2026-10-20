"""以固定公開權重重算有限題目，並保存本節技術審閱的直接證據。"""

import ast
import hashlib
import io
import itertools
import json
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tiny_perceptron.lesson_products.data import (
    answer,
    arithmetic_worlds,
    box_strata,
    box_worlds,
    canonical,
    digest,
    render,
    split_of,
)
from tiny_perceptron.lesson_products.inference import EFFORT, Product, parse_state
from tiny_perceptron.lesson_products.model import Config, LoopedSolver, Tokenizer, padded


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


def fetch(url):
    result = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    downloads.append({"url": url, "sha256": sha(result.stdout), "bytes": len(result.stdout)})
    return result.stdout


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def official_excerpt(api, snippets):
    url = f"https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.{api}.html"
    raw = fetch(url)
    parser = Text()
    parser.feed(raw.decode())
    body = re.sub(r"\s+", " ", " ".join(parser.parts))
    for snippet in snippets:
        assert snippet in body, (api, snippet)
    return {
        "url": url,
        "version": "PyTorch 2.14 documentation",
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "retrieval": downloads[-1],
        "excerpts": snippets,
    }


def reference(world):
    if world["task"] == "sum":
        return sum(world[k] for k in ("x", "y", "z"))
    state = {"A": world["a"], "B": world["b"]}
    for name, value in world["events"]:
        if name == "swap":
            state = {"A": state["B"], "B": state["A"]}
        else:
            state[name] = value
    return state[world["query"]]


def report_predictions(rows, scores):
    guessed = scores.argmax(-1).tolist()
    groups = {
        "boxes": [i for i, r in enumerate(rows) if r["task"] == "boxes"],
        "sum": [i for i, r in enumerate(rows) if r["task"] == "sum"],
        "repeated_swap": [i for i, r in enumerate(rows) if r.get("strata", {}).get("repeated_swap")],
    }
    return {
        name: {
            "count": len(indices),
            "correct": sum(guessed[i] == rows[i]["gold"] for i in indices),
            "families": len({rows[i]["family_id"] for i in indices}),
        }
        for name, indices in groups.items()
        if indices
    }


@torch.inference_mode()
def score_rows(model, tokenizer, rows, depths):
    scores = []
    for start in range(0, len(rows), 128):
        tokens = padded([tokenizer.encode(r["state"]) for r in rows[start : start + 128]], model.config.max_length)
        scores.append(model(tokens, loops=depths, all_depths=True).cpu())
    return torch.cat(scores)


started = time.perf_counter()
downloads = []
torch.set_num_threads(2)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-15/probe.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "recorded_at": datetime.now(UTC).isoformat(),
    "scope": "僅讀指定教材、程式、原始實驗收據與公開固定權重；不讀其他審閱報告，不訓練或調整模型。",
}
result["official_docs"] = {
    "cross_entropy": official_excerpt("cross_entropy", ["Compute the cross entropy loss between input logits and target.", "no reduction will be applied"]),
    "attention": official_excerpt("scaled_dot_product_attention", ["is_causal=False", "A boolean mask where a value of True indicates that the element should take part in attention."]),
}
print("已讀官方文件並核對 CE 與 attention mask 的原文。", flush=True)

run = ROOT / "outputs/reviewer-tools/runs/20261020T053624670682Z"
execution_copy = {name: (run / name).read_text() if name.endswith(".txt") else json.loads((run / name).read_bytes()) for name in ["execution.json", "extraction.json", "environment.json", "stdout.txt", "stderr.txt"]}
execution_copy["invocation"] = 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/19.md#19.15" --execute'
(ARTIFACT / "section-execution.json").write_text(json.dumps(execution_copy, ensure_ascii=False, indent=2) + "\n")
assert execution_copy["execution.json"]["exit_code"] == 0
assert execution_copy["extraction.json"]["source_sha256"] == "52ee82d0bc4ae64febda40b825d234ac33406bf016aed62665b2464e8ccff738"

torch.manual_seed(0)
tokenizer = Tokenizer()
model = LoopedSolver(Config(len(tokenizer))).eval()
state = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
tokens = torch.tensor([tokenizer.encode(state)])
with torch.no_grad():
    logits = model(tokens, loops=5, all_depths=True)
    ce = F.cross_entropy(logits[0], torch.tensor([2] * 5), reduction="none")
    manual = torch.logsumexp(logits[0], dim=-1) - logits[0, :, 2]
    assert torch.allclose(ce, manual, atol=1e-6, rtol=0)
    padded_logits = model(F.pad(tokens, (0, 7)), loops=5, all_depths=True)
    pad_error = (logits - padded_logits).abs().max().item()
    assert pad_error < 2e-6
    last, norms = model(tokens, loops=5, diagnostics=True)
    assert torch.equal(last, logits[:, -1])
    norm_shapes = len(norms)
    q = torch.zeros(1, 1, 3, 1)
    values = torch.tensor([1.0, 3.0, 99.0]).view(1, 1, 3, 1)
    attended = F.scaled_dot_product_attention(q, q, values, attn_mask=torch.tensor([True, True, False]).view(1, 1, 1, 3))
    assert torch.equal(attended, torch.full_like(attended, 2.0))

source = (ROOT / "docs/course-review-20261019/freeze-01/sources/19.15.md").read_text()
class_fence = re.findall(r"```python\n(.*?)```", source, re.S)[1]
excerpt_class = next(n for n in ast.parse(class_fence).body if isinstance(n, ast.ClassDef))
real_class = next(n for n in ast.parse((ROOT / "tiny_perceptron/lesson_products/model.py").read_text()).body if isinstance(n, ast.ClassDef) and n.name == "LoopedSolver")
assert ast.dump(excerpt_class) == ast.dump(real_class)
result["model_checks"] = {
    "config": vars(model.config), "vocabulary_size": len(tokenizer),
    "parameters": sum(p.numel() for p in model.parameters()),
    "parameter_parts": {name: p.numel() for name, p in model.named_parameters()},
    "shape": list(logits.shape), "ce": ce.tolist(), "ce_mean": ce.mean().item(),
    "manual_ce_max_error": (ce-manual).abs().max().item(),
    "pad_max_error": pad_error, "last_depth_matches": True, "diagnostic_rounds": norm_shapes,
    "attention_mask_test_output": attended.flatten().tolist(), "snippet_ast_matches": True,
    "cls_id": tokenizer.encode(state)[0], "effort": EFFORT,
    "depth_shapes": {str(d): list(model(tokens, loops=d).shape) for d in range(1, 7)},
}
examples = [
    {"task": "boxes", "a": 2, "b": 6, "events": [("swap", 0)], "query": "B"},
    {"task": "boxes", "a": 2, "b": 6, "events": [("swap", 0), ("swap", 0)], "query": "B"},
    {"task": "boxes", "a": 2, "b": 6, "events": [("A", 5), ("A", 1)], "query": "A"},
    {"task": "boxes", "a": 2, "b": 6, "events": [("A", 1), ("A", 5)], "query": "A"},
    {"task": "boxes", "a": 2, "b": 6, "events": [("A", 5)], "query": "A"},
    {"task": "sum", "x": 1, "y": 2, "z": 3},
]
result["examples"] = [{"state": render(w), "gold": reference(w), "parser": parse_state(render(w))} for w in examples]
assert [x["gold"] for x in result["examples"]] == [2, 6, 1, 5, 5, 6]
assert {sum(x) for x in itertools.product(range(7), repeat=3)} == set(range(19))

manifest = read_json("docs/lesson-products/data-v2-manifest.json")
config = manifest["config"]
rows = []
for world in itertools.chain(box_worlds(config["per_length"], config["stress_per_length"], config["generator_seed"]), arithmetic_worlds()):
    split = "long_shift" if world["task"] == "boxes" and len(world["events"]) > 4 else split_of(world["family_id"], config["split_salt"])
    base = {"family_id": world["family_id"], "case_id": digest(world["family_id"])[:20], "split": split, "task": world["task"], "state": render(world), "gold": answer(world), "world": world}
    assert base["gold"] == reference(world)
    if world["task"] == "boxes":
        base["strata"] = box_strata(world)
    rows.append({**base, "id": digest((base["family_id"], base["state"]))})
    if split == "test":
        shifted = {**base, "split": "wording_shift", "state": render(world, shifted=True)}
        rows.append({**shifted, "id": digest((base["family_id"], shifted["state"]))})
        if world["task"] == "boxes":
            renamed = {**base, "split": "entity_shift", "state": base["state"].replace("A", "P").replace("B", "Q")}
            rows.append({**renamed, "id": digest((base["family_id"], renamed["state"]))})
raw_digest = hashlib.sha256()
raw_bytes = 0
for row in rows:
    raw = (canonical(row) + "\n").encode()
    raw_digest.update(raw)
    raw_bytes += len(raw)
assert raw_digest.hexdigest() == manifest["files"]["b.jsonl"]["sha256"]
assert raw_bytes == manifest["files"]["b.jsonl"]["bytes"]
assert len(rows) == manifest["files"]["b.jsonl"]["rows"]
result["data_rebuild"] = {"sha256": raw_digest.hexdigest(), "bytes": raw_bytes, "rows": len(rows), "counts": dict(Counter(r["split"] + "/" + r["task"] for r in rows)), "all_gold_matches_reference": True}
print("有限資料已重建，B 資料逐位元組雜湊與固定 manifest 相符。", flush=True)

release = read_json("tiny_perceptron/lesson_products/release.json")
entry = release["products"]["b"]
base_url = f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/{entry['prefix']}"
public_manifest_bytes = fetch(base_url + "/artifact-manifest.json")
assert sha(public_manifest_bytes) == entry["manifest_sha256"]
public_manifest = json.loads(public_manifest_bytes)
blobs = {name: fetch(base_url + "/" + name) for name in ("model.pt", "config.json", "tokenizer.json")}
for name, blob in blobs.items():
    assert sha(blob) == public_manifest["files"][name]["sha256"]
    assert len(blob) == public_manifest["files"][name]["bytes"]
trained_config = json.loads(blobs["config.json"])
assert trained_config == vars(model.config)
trained_tokenizer = Tokenizer(json.loads(blobs["tokenizer.json"])["characters"])
assert trained_tokenizer.encode(state) == tokenizer.encode(state)
trained = LoopedSolver(Config(**trained_config)).eval()
trained.load_state_dict(torch.load(io.BytesIO(blobs["model.pt"]), map_location="cpu", weights_only=True))
recorded = read_json("docs/lesson-products/results/final-report-v1.json")
core = [r for r in rows if r["split"] == "test"]
core_scores = score_rows(trained, trained_tokenizer, core, 6)
actual_core = {str(d): report_predictions(core, core_scores[:, d-1]) for d in range(1, 7)}
for depth, groups in actual_core.items():
    for group, values in groups.items():
        for key, value in values.items():
            assert recorded["b"][depth][group][key] == value, (depth, group, key, value)
result["replicated_test"] = actual_core
print("六個深度的追蹤、加法與重複交換答對數均與實驗紀錄相符。", flush=True)
result["replicated_shifts"] = {}
for split in ("entity_shift", "long_shift"):
    shifted = [r for r in rows if r["split"] == split]
    scores = score_rows(trained, trained_tokenizer, shifted, 5)[:, -1]
    observed = report_predictions(shifted, scores)
    for group, values in observed.items():
        for key, value in values.items():
            assert recorded["shifts"][split][group][key] == value, (split, group, key, value)
    result["replicated_shifts"][split] = observed
result["percentages"] = {name: {"correct": c, "count": n, "percentage": 100*c/n, "rounded_two_decimals": f"{100*c/n:.2f}%"} for name, c, n in [("high_boxes",2928,2956),("high_sum",2,42),("low_boxes",1598,2956),("repeated_swap",32,42),("entity_shift",1486,2956),("long_shift",1245,3200)]}
result["recorded_protocol"] = read_json("docs/lesson-products/results/settings-freeze-v1.json")["protocol"]
result["recorded_training"] = {}
for name in ("initial-b-execution", "mechanism-fixed3-runner-result", "mechanism-odd135-runner-result"):
    receipt = read_json("docs/lesson-products/results/" + name + ".json")
    receipt = receipt.get("execution", receipt)
    t = receipt["training"]
    fields = ["random_initialization", "resume_source", "steps_this_attempt", "last_step", "selected_step", "seed", "batch_size", "parameters", "data_manifest_sha256", "initial_weights_sha256", "resume_verification", "environment", "depth_weights"]
    result["recorded_training"][name] = {"gpu_name": receipt["gpu_name"], "training": {k:t[k] for k in fields}, "private_checkpoint": receipt["private_checkpoint"], "source_hashes": receipt["source_hashes"]}
fixed = result["recorded_training"]["mechanism-fixed3-runner-result"]["training"]
odd = result["recorded_training"]["mechanism-odd135-runner-result"]["training"]
assert fixed["initial_weights_sha256"] == odd["initial_weights_sha256"]
assert fixed["data_manifest_sha256"] == odd["data_manifest_sha256"]
assert fixed["last_step"] == odd["last_step"] == 1500
assert all(value["training"]["resume_verification"]["exact_next_step"] for value in result["recorded_training"].values())
result["paired_receipts_agree"] = True

product = Product.__new__(Product)
product.product, product.model, product.config, product.tokenizer, product.device = "b", trained, trained.config, trained_tokenizer, torch.device("cpu")
result["cpu_product_solve"] = [product.solve(e["state"], loops=5) for e in result["examples"]]
help_run = subprocess.run([sys.executable, "-m", "tiny_perceptron.lesson_products", "solve", "--help"], cwd=ROOT, capture_output=True, text=True, check=True)
result["cli_solve_help"] = help_run.stdout
paths = [
    "tiny_perceptron/lesson_products/model.py", "tiny_perceptron/lesson_products/data.py",
    "tiny_perceptron/lesson_products/training.py", "tiny_perceptron/lesson_products/train.py",
    "tiny_perceptron/lesson_products/inference.py", "tiny_perceptron/lesson_products/__main__.py",
    "tiny_perceptron/lesson_products/evaluation.py", "tiny_perceptron/lesson_products/release.json",
    "docs/lesson-products/results/final-report-v1.json", "docs/lesson-products/data-v2-manifest.json",
    "docs/lesson-products/results/initial-b-execution.json", "docs/lesson-products/results/b-artifact-manifest.json",
    "docs/lesson-products/results/mechanism-fixed3-runner-result.json", "docs/lesson-products/results/mechanism-odd135-runner-result.json",
    "docs/lesson-products/results/settings-freeze-v1.json", "docs/lesson-products/results/pretest-freeze-record-v1.json",
    "docs/course-review-20261019/freeze-01/sources/19.15.md", "docs/course-review-20261019/freeze-04.json",
]
result["repository_inputs"] = {p: sha((ROOT / p).read_bytes()) for p in paths}
result["receipt_source_matches"] = {
    name: {
        path: receipt["source_hashes"][path] == result["repository_inputs"][path]
        for path in ["tiny_perceptron/lesson_products/" + f for f in ("data.py", "model.py", "training.py", "train.py")]
    }
    for name, receipt in result["recorded_training"].items()
}
assert all(all(matches.values()) for matches in result["receipt_source_matches"].values())
history_command = ["git", "log", "-5", "--format=%H %aI %s", "--", "docs/lesson-products/results/pretest-freeze-record-v1.json", "docs/lesson-products/results/final-report-v1.json"]
history = subprocess.run(history_command, cwd=ROOT, capture_output=True, text=True, check=True)
result["pretest_history"] = {"command_argv": history_command, "stdout": history.stdout, "stderr": history.stderr, "exit_code": history.returncode}
result["downloads"] = downloads
result["elapsed_seconds"] = time.perf_counter()-started
result["result"] = "所有確定性檢查及固定權重答對題數重算通過。L4 訓練與恢復只核對原始收據及程式，未在本機重訓。"
(ARTIFACT / "probe-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(result["result"], flush=True)
print(json.dumps({"artifact": str(ARTIFACT / "probe-results.json"), "sha256": sha((ARTIFACT / "probe-results.json").read_bytes()), "seconds": result["elapsed_seconds"]}, ensure_ascii=False))
