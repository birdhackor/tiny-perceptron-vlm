"""Read-only, bounded CPU verification for frozen lesson 19.16; never trains/tests anew."""

import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.lesson_products.data import CANDIDATES, MEANINGS, QUESTIONS, REQUIRED
from tiny_perceptron.lesson_products.inference import Product
from tiny_perceptron.lesson_products.model import CandidateScorer, Config
from tiny_perceptron.lesson_products.training import state_digest

torch.set_num_threads(1)
OUT = Path(__file__).with_name("19.16-verification-original-01.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text())


report_path = "docs/lesson-products/results/final-report-v1.json"
report = load(report_path)
freeze_path = "docs/lesson-products/results/settings-freeze-v1.json"
freeze = load(freeze_path)
data_path = ROOT / "outputs/lesson-products/data-v2/c.jsonl"
raw_path = ROOT / "outputs/lesson-products/final-report-v1.predictions.jsonl"
manifest = load("docs/lesson-products/data-v2-manifest.json")
assert sha(raw_path) == report["predictions"]["sha256"]
assert sha(data_path) == manifest["files"]["c.jsonl"]["sha256"]
assert sha(ROOT / freeze_path) == report["settings_freeze_sha256"]
assert sha(ROOT / "docs/lesson-products/data-v2-manifest.json") == report["data_manifest_sha256"]
for name, digest in freeze["source_sha256s"].items():
    assert sha(ROOT / "tiny_perceptron/lesson_products" / name) == digest

# Verify labels against the task-specific omission contract; join original
# predictions to exact original data by ID, never use record IDs as model input.
selected_data = {}
observations = defaultdict(lambda: {"splits": set(), "labels": set(), "count": 0})
label_errors = []
counts = Counter()
for line in data_path.open():
    row = json.loads(line)
    if row["question"] == "choice":
        expected = row["task"] if row["task"] in row["candidates"] else "none"
    else:
        total = len(REQUIRED[row["task"]])
        available = total - len(row["omitted"])
        expected = ("yes" if available == total else "no") if row["question"] == "info" else (
            "grade2" if available == total else "grade0" if available == 0 else "grade1"
        )
    if expected != row["gold"]:
        label_errors.append(row["id"])
    counts[(row["split"], row["question"])] += 1
    if row["split"] in ("test", "wording_shift"):
        selected_data[row["id"]] = row
    if row["split"] in ("train", "calibration", "policy", "test"):
        key = (row["state"], row["question"], tuple(sorted(row["candidates"])))
        value = observations[key]
        value["splits"].add(row["split"])
        value["labels"].add(row["gold"])
        value["count"] += 1
assert not label_errors
overlap = {
    "unique_observations": len(observations),
    "cross_split_shared_observations": sum(len(x["splits"]) > 1 for x in observations.values()),
    "rows_in_shared_observations": sum(x["count"] for x in observations.values() if len(x["splits"]) > 1),
    "conflicting_labels": sum(len(x["labels"]) > 1 for x in observations.values()),
}
assert all(overlap[k] == manifest["audit"]["c"][k] for k in overlap)

groups = defaultdict(list)
missing_count = missing_correct = reverse_count = reverse_consistent = 0
for line in raw_path.open():
    row = json.loads(line)
    if row["product"] != "c":
        continue
    original = selected_data[row["id"]]
    assert all(row[k] == original[k] for k in ("split", "question", "candidates", "gold", "family_id"))
    chosen = min(zip(row["candidates"], row["raw_scores"]), key=lambda x: (-x[1], x[0]))[0]
    groups[(row["split"], row["question"])].append(row)
    if row["split"] == "test":
        assert chosen == row["selected"]
        if row["question"] == "choice":
            reverse_count += 1
            reverse_consistent += row["reverse_selected"] == chosen
            if original.get("missing_best"):
                assert row["gold"] == "none" and original["task"] not in row["candidates"]
                missing_count += 1
                missing_correct += chosen == row["gold"]

results = {}
for (split, question), rows in groups.items():
    max_candidates = max(len(row["candidates"]) for row in rows)
    scores = torch.tensor([row["raw_scores"] + [-float("inf")] * (max_candidates - len(row["raw_scores"])) for row in rows])
    gold = torch.tensor([row["candidates"].index(row["gold"]) for row in rows])
    selected = [min(zip(r["candidates"], r["raw_scores"]), key=lambda x: (-x[1], x[0]))[0] for r in rows]
    values = {}
    for label, temperature in (("before", 1.0), ("after", freeze["calibration"]["temperatures"][question])):
        logits = scores / temperature
        probabilities = logits.softmax(-1)
        truth = torch.nn.functional.one_hot(gold, max_candidates).float()
        actual = {
            "count": len(rows),
            "correct": sum(s == r["gold"] for s, r in zip(selected, rows)),
            "cross_entropy": float(torch.nn.functional.cross_entropy(logits, gold)),
            "brier": float((probabilities - truth).square().sum(-1).mean()),
        }
        reference = (report["c"] if split == "test" else report["shifts"]["c_wording"])[question][label]
        assert actual["count"] == reference["count"] and actual["correct"] == reference["correct"]
        assert abs(actual["cross_entropy"] - reference["cross_entropy"]) <= 1e-7
        assert abs(actual["brier"] - reference["brier"]) <= 1e-7
        actual["accuracy"] = actual["correct"] / actual["count"]
        # Independently evaluate stable logsumexp CE using Python doubles.
        losses = []
        for row in rows:
            vals = [s / temperature for s in row["raw_scores"]]
            m = max(vals)
            losses.append(m + math.log(sum(math.exp(x - m) for x in vals)) - vals[row["candidates"].index(row["gold"])] )
        actual["independent_float64_ce"] = sum(losses) / len(losses)
        values[label] = actual
    results[split + "/" + question] = values
assert (reverse_count, reverse_consistent) == (11797, 11797)
assert (missing_count, missing_correct) == (849, 849)

directory = ROOT / "outputs/lesson-products/products/c"
product = Product(directory, "c", "cpu")
receipt = json.loads((directory / "training-receipt.json").read_text())
actual_parameters = sum(x.numel() for x in product.model.parameters())
assert actual_parameters == receipt["parameters"] == 84545
torch.manual_seed(9041)
initial = CandidateScorer(Config(**json.loads((directory / "config.json").read_text())))
assert state_digest(initial.state_dict()) == receipt["initial_weights_sha256"]
assert state_digest(product.model.state_dict()) == receipt["selected_weights_sha256"]
assert receipt["seed"] == 9041 and receipt["steps_this_attempt"] == receipt["last_step"] == receipt["selected_step"] == 800
assert receipt["random_initialization"] and receipt["resume_source"] is None
assert receipt["environment"]["gpu"] == "NVIDIA L4"
gpu_execution = load("docs/lesson-products/results/initial-c-execution.json")
gpu_inspection = load("docs/lesson-products/results/initial-c-inspection.json")
assert sha(ROOT / "docs/lesson-products/results/initial-c-execution.json") == gpu_inspection["execution_sha256"]
assert gpu_execution["gpu_name"] == "NVIDIA L4" and gpu_execution["status"] == "completed"
assert gpu_execution["training"] == receipt
assert gpu_inspection["downloaded_files"]["training-receipt.json"]["sha256"] == sha(directory / "training-receipt.json")
checkpoints = {}
for name in ("best.pt", "last.pt"):
    ckpt = torch.load(directory / name, map_location="cpu", weights_only=True)
    checkpoints[name] = {k: ckpt["metadata"][k] for k in ("seed", "step", "best_step", "initial_weights_sha256", "best_weights_sha256")}
    assert ckpt["metadata"]["step"] == 800
    assert state_digest(ckpt["model"]) == receipt["selected_weights_sha256"]
for name, digest in freeze["products"]["c"].items():
    assert sha(directory / name) == digest

probes = []
for state in ("卡片進度;申請編號=?", "地址填寫規則", "幫我預報明天天氣"):
    for question in CANDIDATES:
        answer = product.judge(state, question)
        answer.pop("seconds")
        probes.append({"state": state, **answer})
normal = product.judge("卡片進度;申請編號=?", "choice")
reverse = product.judge("卡片進度;申請編號=?", "choice", list(reversed(CANDIDATES["choice"])))
assert normal["selected"] == reverse["selected"] == "card"
assert max(abs(normal["probabilities"][x] - reverse["probabilities"][x]) for x in CANDIDATES["choice"]) <= 1e-7
for state, expected in (("卡片進度;申請編號=?", ("no", "grade0")), ("地址填寫規則", ("yes", "grade2"))):
    assert tuple(product.judge(state, q)["selected"] for q in ("info", "rubric")) == expected
weather = product.judge("幫我預報明天天氣", "choice")
assert weather["selected"] == "boxes" and weather["concentration"] > 0.999
saved = load("docs/lesson-products/results/observed-failure-examples-v1.json")["examples"][0]
raw_example = next(r for r in groups[("wording_shift", "choice")] if r["id"] == saved["id"])
assert raw_example["raw_scores"] == saved["raw_scores"] and saved["gold"] == "none"

external = []
fulltexts = Path("/workspace/work/curriculum-20261016/fulltexts")
for name, url, anchors in (
    ("jev-intro.source", "https://docs.typesafe.ai/introduction", ("TypeSafe primitives", "Atomic questions, composed in code")),
    ("jev-confidence.source", "https://docs.typesafe.ai/confidence", ("Noul A Noul", "Choice A Choice")),
):
    path = fulltexts / name
    soup = BeautifulSoup(path.read_text(), "html.parser")
    text = (soup.find("article") or soup.find("main") or soup).get_text(" ", strip=True)
    external.append({"url": url, "version": "live documentation retrieved 2026-10-16; immutable bytes SHA below", "local_fulltext": str(path), "sha256": sha(path), "inspected_2026_10_16": True, "excerpts": [{"locator": a, "text": text[text.index(a):text.index(a)+550]} for a in anchors]})
for name, url, version, locator in (
    ("19.16-nogueira.pdf", "https://arxiv.org/pdf/1901.04085v1", "arXiv:1901.04085v1, 13 Jan 2019", "Section 2 Method, pp.1–2: each query-passage pair receives a score independently; not C quality evidence or a Jev architecture disclosure"),
    ("14.17-guo.pdf", "https://proceedings.mlr.press/v70/guo17a/guo17a.pdf", "ICML 2017 proceedings, PMLR70:1321–1330", "Section2 classification/calibration definition and Section4.2 Eq9: positive scalar temperature chosen on validation NLL leaves argmax unchanged"),
):
    external.append({"url": url, "version": version, "local_fulltext": str(fulltexts / name), "sha256": sha(fulltexts / name), "inspected_2026_10_16": True, "locator": locator})

paths = [report_path, freeze_path, "docs/lesson-products/results/initial-c-execution.json", "docs/lesson-products/results/initial-c-inspection.json", "docs/lesson-products/results/pretest-freeze-record-v1.json", "docs/lesson-products/results/c-calibration.json", "docs/lesson-products/results/observed-failure-examples-v1.json", "docs/lesson-products/data-v2-manifest.json", "docs/lesson-products/data-v2-token-audit.json"]
paths += ["tiny_perceptron/lesson_products/" + n for n in ("model.py", "data.py", "training.py", "train.py", "inference.py", "evaluation.py")]
paths += ["docs/course-revision-20261016/freeze-14/sources/19.16.md"]
payload = {
    "reviewer_task": "/root/backfill_tech_19_16",
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/backfill/19.16-verify-original-01.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    "scope": "Personally executed only a read-only recomputation of original C logits, file/checkpoint/label audits, and nine small public CPU judge probes; did not train, retune, call paid services, or re-enter formal final-test evaluation. Historical GPU facts are existing primary receipts, not my own GPU run.",
    "source_sha256": {p: sha(ROOT / p) for p in paths},
    "large_local_inputs": {str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in (raw_path, data_path)},
    "label_rule_audit": {"rows": sum(counts.values()), "mismatches": len(label_errors), "counts": {"/".join(k): v for k,v in counts.items()}},
    "observation_overlap": overlap,
    "metrics_from_original_logits": results,
    "reverse_choice": {"count": reverse_count, "consistent": reverse_consistent},
    "missing_best_none": {"count": missing_count, "correct": missing_correct},
    "model_checks": {"parameters": actual_parameters, "random_seed_9041_digest_matches": True, "selected_model_digest_matches": True, "existing_training_environment": receipt["environment"], "existing_gpu_run_id": gpu_execution["run_id"], "existing_gpu_receipt_and_inspection_match": True, "checkpoints": checkpoints, "freeze_c_files_all_match": True, "frozen_execution_sources_all_match": True},
    "calibration": freeze["calibration"],
    "cpu_probes": probes,
    "weather_saved_raw_example_matches": True,
    "permutation_probe": {"selected": normal["selected"], "reverse_selected": reverse["selected"], "max_probability_difference": max(abs(normal["probabilities"][x] - reverse["probabilities"][x]) for x in CANDIDATES["choice"])},
    "external_original_receipts": external,
    "result": "All assertions passed; quoted counts and rounded CE values match original C logits and primary receipts; finite grammar/observational duplication and high-confidence wording-shift failures remain explicit limitations.",
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": payload["result"], "output": str(OUT.relative_to(ROOT)), "metrics": results, "sha256": sha(OUT)}, ensure_ascii=False, indent=2))
