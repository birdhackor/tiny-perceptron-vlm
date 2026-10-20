import gzip
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.lesson_products.evaluation import auto_quality
from tiny_perceptron.lesson_products.inference import auto_solve, parse_state
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, LoopedSolver
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


inputs = {}


def load(path):
    raw = (ROOT / path).read_bytes()
    inputs[path] = digest(raw)
    return json.loads(raw)


def fetch(url):
    return subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "50", url],
        check=True,
        capture_output=True,
    ).stdout


report = load("docs/lesson-products/results/final-report-v1.json")
page_path = "docs/course-review-20261019/freeze-01/sources/19.18.md"
page_raw = (ROOT / page_path).read_bytes()
inputs[page_path] = digest(page_raw)
review_freeze = load("docs/course-review-20261019/freeze-04.json")
page_freeze = next(p for p in review_freeze["pages"] if p["page_id"] == "19.18")
page_receipt = {"sha256": digest(page_raw), "freeze_matches": digest(page_raw) == page_freeze["source_sha256"], "code_fences": page_raw.count(b"```"), "figures": page_freeze["figures_sha256"]}
freeze = load("docs/lesson-products/results/settings-freeze-v1.json")
policy = load("docs/lesson-products/results/b-auto-policy.json")
calibration = load("docs/lesson-products/results/c-calibration.json")
moe = load("docs/selftrained/results/v2-final-public-results.json")
smoke = load("docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-review.json")
moe_receipt = load("docs/selftrained/results/public-raw/moe/test/receipt.json")
release = load("docs/lesson-products/results/publication-v1.json")["execution"]["public"]
base = f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/{release['prefix']}/"
remote = {}
parameter_counts = {}
weight_sizes = {"moe": moe["architectures"]["moe"]["safe_weights"]["bytes"]}
for product, cls in (("b", LoopedSolver), ("c", CandidateScorer)):
    manifest = load(f"docs/lesson-products/results/{product}-artifact-manifest.json")
    config_raw = fetch(base + product + "/config.json")
    model_raw = fetch(base + product + "/model.pt")
    assert digest(config_raw) == manifest["files"]["config.json"]["sha256"]
    assert digest(model_raw) == manifest["files"]["model.pt"]["sha256"]
    assert len(model_raw) == manifest["files"]["model.pt"]["bytes"]
    config = Config(**json.loads(config_raw))
    model = cls(config)
    model.load_state_dict(torch.load(io.BytesIO(model_raw), weights_only=True, map_location="cpu"))
    parameter_counts[product] = sum(p.numel() for p in model.parameters())
    weight_sizes[product] = len(model_raw)
    remote[product] = {"config": json.loads(config_raw), "weight_sha256": digest(model_raw), "weight_bytes": len(model_raw), "loaded_parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()})}
parameter_counts["moe"] = sum(
    p.numel() for p in LimitedAssistant(SelftrainedConfig(**moe["architectures"]["moe"]["capacity"]["config"])).parameters()
)

compressed = fetch(base + "raw/final-report-v1.predictions.jsonl.gz")
raw = gzip.decompress(compressed)
assert digest(raw) == report["predictions"]["sha256"]
assert len(raw) == report["predictions"]["bytes"]
counts = Counter()
quality = defaultdict(Counter)
solver = defaultdict(Counter)
c_main = defaultdict(Counter)
timing_keys = Counter()
shift_examples = []
example_gold = load("docs/lesson-products/results/observed-failure-examples-v1.json")
example_ids = {r["id"] for r in example_gold["examples"]}
b_rows = []
for line in raw.splitlines():
    row = json.loads(line)
    counts[f"{row['product']}:{row.get('split', '')}"] += 1
    timing_keys.update(k for k in row if "second" in k or "time" in k or "budget" in k or "limit" in k)
    if row["product"] == "b" and row["split"] == "test":
        b_rows.append(row)
        accepted = row["c_choice"] in ("boxes", "sum") and row["c_info"] == "yes"
        counts["accepted"] += accepted
        counts["b:" + row["task"]] += 1
        for depth, prediction in row["answers_by_depth"].items():
            solver[depth][row["task"] + ":correct"] += prediction == row["gold"]
        for effort, depth in (("low", 1), ("medium", 3), ("high", 5), ("auto", 3 if row["task"] == "boxes" else policy["sum_depth"])):
            quality[effort]["count"] += 1
            quality[effort]["loops"] += depth if accepted else 0
            quality[effort]["correct"] += accepted and row["answers_by_depth"][str(depth)] == row["gold"]
    if row["product"] == "c" and row["split"] == "test":
        c_main[row["question"]]["count"] += 1
        c_main[row["question"]]["correct"] += row["selected"] == row["gold"]
    if row["product"] == "c" and row["split"] == "wording_shift" and row["id"] in example_ids:
        temperature = calibration["temperatures"][row["question"]]
        scores = [v / temperature for v in row["raw_scores"]]
        top = max(scores)
        weights = [math.exp(v - top) for v in scores]
        probabilities = [w / sum(weights) for w in weights]
        selected = max(range(len(probabilities)), key=probabilities.__getitem__)
        shift_examples.append({"id": row["id"], "gold": row["gold"], "selected": row["candidates"][selected], "probability": probabilities[selected]})

quality_summary = {}
for effort, values in quality.items():
    accuracy = values["correct"] / values["count"]
    loops = values["loops"] / values["count"]
    recorded = report["auto"][effort]
    quality_summary[effort] = dict(values) | {"accuracy_percent": accuracy * 100, "formatted_percent": f"{accuracy * 100:.2f}%", "mean_loops": loops}
    assert values["correct"] == recorded["correct_end_to_end"]
    assert abs(loops - recorded["mean_actual_b_loops_all_complete"]) < 1e-12
    assert abs(accuracy - recorded["accuracy_end_to_end"]) < 1e-12

latency_sample = sorted(b_rows, key=lambda row: digest(("latency-v1:" + row["id"]).encode()))[:24]
assert [row["id"] for row in latency_sample] == report["latency"]["sample_ids"]
times = {effort: {k: v * 1000 if isinstance(v, float) else v for k, v in row.items()} for effort, row in report["latency"]["measurements"].items()}
chat = [r for r in smoke["invocations"] if any(arg.endswith("/chat.py") for arg in r["argv"])]


class BProbe:
    auto_policy = policy

    def synchronize(self):
        pass

    def solve(self, state, loops):
        return {"answer": 0, "loops": loops, "seconds": 0.0}


class CProbe:
    def __init__(self, choice="boxes", info="yes"):
        self.choice, self.info, self.calls = choice, info, []

    def synchronize(self):
        pass

    def judge(self, state, question):
        self.calls.append(question)
        return {"selected": self.choice if question == "choice" else self.info, "seconds": 0.0}


state = "兩盒追蹤;A=1;B=2;事件=交換;查詢=A"
probes = {}
for name, choice, info, effort, cap in (("cap", "boxes", "yes", "high", 2), ("unsupported", "none", "yes", "low", 5), ("clarification", "boxes", "no", "low", 5)):
    c = CProbe(choice, info)
    value = auto_solve(BProbe(), c, state, effort, cap)
    probes[name] = {"status": value["status"], "b_loops": value["b_loops"], "c_calls": c.calls}
denominator_probe = auto_quality(
    [{"state": state, "task": "boxes", "gold": 0}] * 2,
    {3: torch.tensor([0, 0])}, ["none", "yes", "boxes", "yes"], policy, fixed=3,
)
source_matches = {}
source_excerpts = {}
training_receipts = {}
for product in ("b", "c"):
    receipt = load(f"docs/lesson-products/results/initial-{product}-execution.json")
    training_receipts[product] = {"code_revision": receipt["code_revision"], "seed": receipt["training"]["seed"], "data_manifest_sha256": receipt["training"]["data_manifest_sha256"], "parameters": receipt["training"]["parameters"]}
for name in ("data.py", "model.py", "training.py", "inference.py", "evaluation.py"):
    path = "tiny_perceptron/lesson_products/" + name
    content = (ROOT / path).read_bytes()
    inputs[path] = digest(content)
    source_matches[name] = digest(content) == freeze["source_sha256s"][name]
    if name == "evaluation.py":
        text = content.decode()
        source_excerpts["latency_report"] = text[text.index("def latency_report"):text.index("def evaluate_frozen")]
    if name == "inference.py":
        text = content.decode()
        source_excerpts["auto_solve"] = text[text.index("def auto_solve"):]
load("docs/lesson-products/data-v2-manifest.json")
for path in ("tiny_perceptron/selftrained/model.py", "scripts/selftrained/chat.py", "tiny_perceptron/selftrained/inference.py"):
    inputs[path] = digest((ROOT / path).read_bytes())

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-18/verify_evidence.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "input_sha256": inputs,
    "page_receipt": page_receipt,
    "public_url_prefix": base,
    "download_receipt": {"compressed_sha256": digest(compressed), "compressed_bytes": len(compressed), "uncompressed_sha256": digest(raw), "uncompressed_bytes": len(raw)},
    "remote_weights": remote,
    "parameter_counts": parameter_counts,
    "weight_bytes": weight_sizes,
    "raw_counts": dict(counts),
    "recomputed_quality": quality_summary,
    "solver_counts_by_depth": solver,
    "c_main_counts": c_main,
    "shift_examples_from_raw_scores": shift_examples,
    "relative_loop_reduction_percent": (3 - quality_summary["auto"]["mean_loops"]) / 3 * 100,
    "latency_recorded_environment": {k: v for k, v in report["latency"].items() if k not in ("sample_ids", "measurements")},
    "latency_values_milliseconds": times,
    "latency_sampling_ids_match": True,
    "latency_raw_fields": dict(timing_keys),
    "chat_processes": [{"task": r["task"], "seconds": r["elapsed_seconds"], "returncode": r["returncode"]} for r in chat],
    "chat_count": len(chat),
    "chat_seconds_minmax": [min(r["elapsed_seconds"] for r in chat), max(r["elapsed_seconds"] for r in chat)],
    "moe_data_scope": moe["data_scope"],
    "moe_acceptance": moe["architectures"]["moe"]["capability_acceptance_passed"],
    "dates": {"moe_test": moe_receipt["finished_at"], "looped_and_judge": report["recorded_at"]},
    "grammar_probe": parse_state(state),
    "auto_probes": probes,
    "rejection_denominator_probe": denominator_probe,
    "frozen_source_matches": source_matches,
    "training_receipts": training_receipts,
    "source_excerpts": source_excerpts,
    "limits": "重算已公開固定執行紀錄，未重訓。歷史延遲只核對摘要、程式與樣本設定；未重現歷史機器，無逐次時間、最小值或最大值可重算。下載的權重與完整預測僅在記憶體使用，不寫入工作樹。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
