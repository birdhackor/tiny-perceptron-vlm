"""Bounded CPU algebra and read-only raw measurement checks for lesson 10.11."""
import hashlib
import json
import math
import platform
from pathlib import Path
import torch
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
scores = torch.tensor([[2.0, 0.1], [0.4, 1.8]], dtype=torch.float64, requires_grad=True)
old = scores.detach().clone()
labels = torch.arange(2)
assert labels.dtype == torch.int64 and labels.tolist() == [0, 1]
assert torch.equal(scores.T, torch.tensor([[2.0, 0.4], [0.1, 1.8]], dtype=torch.float64))
li = F.cross_entropy(scores, labels)
lt = F.cross_entropy(scores.T, labels)
loss = (li + lt) / 2
per_i = [math.log1p(math.exp(-1.9)), math.log1p(math.exp(-1.4))]
per_t = [math.log1p(math.exp(-1.6)), math.log1p(math.exp(-1.7))]
manual_i = sum(per_i) / 2
manual_t = sum(per_t) / 2
assert abs(li.item() - manual_i) < 1e-14
assert abs(lt.item() - manual_t) < 1e-14
pi = scores.detach().softmax(1)
pt = scores.detach().softmax(0)
eye = torch.eye(2, dtype=torch.float64)
gi = (pi - eye) / 2
gt = (pt - eye) / 2
analytic = (gi + gt) / 2
loss.backward()
assert torch.allclose(scores.grad, analytic, atol=1e-14, rtol=0)
assert torch.equal(torch.sign(scores.grad), torch.tensor([[-1., 1.], [1., -1.]], dtype=torch.float64))
assert torch.equal(scores.detach(), old)  # backward did not update scores
assert torch.allclose(pi.sum(1), torch.ones(2, dtype=torch.float64))
assert torch.allclose(pt.sum(0), torch.ones(2, dtype=torch.float64))

def mean_loss(x):
    return (F.cross_entropy(x, labels) + F.cross_entropy(x.T, labels)) / 2

eps = 1e-6
finite = torch.empty_like(old)
for i in range(2):
    for j in range(2):
        delta = torch.zeros_like(old)
        delta[i, j] = eps
        finite[i, j] = (mean_loss(old + delta) - mean_loss(old - delta)) / (2 * eps)
assert torch.allclose(finite, analytic, atol=1e-9, rtol=0)

weighted_scores = old.clone().requires_grad_()
wli = F.cross_entropy(weighted_scores, labels)
wlt = F.cross_entropy(weighted_scores.T, labels)
weighted = (2 * wli + wlt) / 3
weighted.backward()
assert weighted_scores.grad.shape == (2, 2)
assert torch.allclose(weighted_scores.grad, (2 * gi + gt) / 3, atol=1e-14, rtol=0)
assert abs(weighted.item() - li.item()) < abs(weighted.item() - lt.item())
assert torch.equal(weighted_scores.detach(), old)
sym = torch.tensor([[2., 0.], [0., 2.]], dtype=torch.float64)
assert F.cross_entropy(sym, labels) == F.cross_entropy(sym.T, labels)
swapped = old[:, [1, 0]]
correct_swapped_labels = torch.tensor([1, 0])
assert F.cross_entropy(swapped, correct_swapped_labels) == li.detach()
assert F.cross_entropy(swapped.T, correct_swapped_labels) == lt.detach()
wrong_swapped = mean_loss(swapped).item()
assert wrong_swapped > loss.item()
duplicates = torch.zeros((2, 2), dtype=torch.float64, requires_grad=True)
duplicate_loss = F.cross_entropy(duplicates, labels)
duplicate_loss.backward()
assert abs(duplicate_loss.item() - math.log(2)) < 1e-14
assert duplicates.grad[0, 1] > 0 and duplicates.grad[1, 0] > 0

raw_path = ROOT / "docs/course-experiments/results/contrastive.json"
raw = raw_path.read_bytes()
d = json.loads(raw)
# Original bytes are retained without displaying uninspected annotation values.
(OUT / "contrastive-original.json").write_bytes(raw)
pointers = []

def get(pointer):
    pointers.append(pointer)
    value = d
    for key in pointer.split("/")[1:]:
        value = value[key.replace("~1", "/").replace("~0", "~")]
    return value

metadata = {key: get("/" + key) for key in ["revision", "device", "seed", "torch_version", "python_version", "step_scale"]}
config = get("/results/config")
splits, families = {}, {}
for name in ["train", "validation", "test"]:
    base = "/results/data/splits/" + name
    rows = get(base + "/records")
    count = get(base + "/count")
    declared_hash = get(base + "/sha256")
    actual_hash = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert len(rows) == count and actual_hash == declared_hash
    families[name] = {row["family"] for row in rows}
    splits[name] = {"count": count, "sha256_verified": True,
        "unique_answers": sorted({row["answer"] for row in rows}), "offsets": sorted({row["offset"] for row in rows})}
assert not families["train"] & families["validation"]
assert not families["train"] & families["test"]
assert not families["validation"] & families["test"]
labels6 = [f"{c} {s}" for c in ["red", "green", "blue"] for s in ["circle", "square"]]
variants = {}
for variant in ["one_way", "two_way"]:
    base = "/results/variants/" + variant
    training = {k: get(base + "/training/" + k) for k in ["steps", "effective_targets", "weights_changed", "nonzero_gradient_seen", "cpu_smoke"]}
    assert training["steps"] == 250 and training["effective_targets"] == 250 * 6
    evaluations = {}
    for split in ["validation", "test"]:
        b = base + "/" + split
        image_samples = get(b + "/image_samples")
        text_samples = get(b + "/text_samples")
        image_queries = get(b + "/image_queries")
        text_queries = get(b + "/text_queries")
        ia = get(b + "/image_to_text_accuracy")
        ta = get(b + "/text_to_image_accuracy")
        records = get("/results/data/splits/" + split + "/records")
        assert image_queries == len(image_samples) == len(records) == 6
        assert text_queries == len(text_samples) == len(labels6) == 6
        for sample in image_samples:
            assert sample["target"] == records[sample["row"]]["answer"]
            assert sample["predicted"] in labels6
            assert sample["correct"] == (sample["predicted"] == sample["target"])
        for sample in text_samples:
            assert sample["query"] == sample["target"] and sample["query"] in labels6
            assert sample["retrieved"] == records[sample["image_row"]]["answer"]
            assert sample["correct"] == (sample["retrieved"] == sample["target"])
        ic = sum(s["correct"] for s in image_samples)
        tc = sum(s["correct"] for s in text_samples)
        assert ia == ic / image_queries and ta == tc / text_queries
        evaluations[split] = {"image_correct": ic, "image_queries": image_queries,
            "image_to_text_accuracy": ia, "text_correct": tc, "text_queries": text_queries,
            "text_to_image_accuracy": ta, "image_candidates": len(records), "text_candidates": len(labels6),
            "image_samples": image_samples, "text_samples": text_samples}
    variants[variant] = {"training": training, **evaluations}
    assert evaluations["test"]["image_correct"] == evaluations["test"]["text_correct"] == 6
    assert evaluations["validation"]["image_correct"] < 6 or evaluations["validation"]["text_correct"] < 6
code_hash = get("/code_sha256/scripts~1course_experiments~1modalities.py")
assert hashlib.sha256((ROOT / "scripts/course_experiments/modalities.py").read_bytes()).hexdigest() == code_hash
assert hashlib.sha256((OUT / "modalities-ef9263.py").read_bytes()).hexdigest() == code_hash

result = {"environment": {"python": platform.python_version(), "executable": __import__('sys').executable,
    "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "device": "cpu",
    "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads())},
    "algebra": {"scores": old.tolist(), "labels": labels.tolist(), "label_dtype": str(labels.dtype),
        "units": "natural-log loss in nats; scores and probabilities dimensionless",
        "per_image_loss": per_i, "per_text_loss": per_t, "image_to_text": li.item(), "text_to_image": lt.item(),
        "average": loss.item(), "row_probabilities": pi.tolist(), "column_probabilities": pt.tolist(),
        "loss_denominators": {"queries_per_direction": 2, "candidates_per_query": 2, "directions_in_average": 2},
        "gradient": scores.grad.tolist(), "analytic_gradient": analytic.tolist(), "finite_difference_gradient": finite.tolist(),
        "gradient_max_error": (scores.grad - analytic).abs().max().item(), "finite_difference_max_error": (finite - analytic).abs().max().item(),
        "backward_updated_scores": False, "weighted_loss": weighted.item(), "weighted_gradient": weighted_scores.grad.tolist(),
        "weighted_shape": list(weighted_scores.grad.shape), "swap_corrected_loss": loss.item(),
        "swap_wrong_labels_loss": wrong_swapped, "equal_score_duplicate_loss": duplicate_loss.item(), "duplicate_gradient": duplicates.grad.tolist()},
    "raw_measurements": {"path": raw_path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
        "value_inspection_pointers": sorted(set(pointers)), "metadata": metadata, "config": config,
        "splits": splits, "variants": variants, "implementation_sha256": code_hash,
        "scope": "Read-only reconciliation of historical recorded samples; no model load, inference rerun, training, or neural checkpoint writing."}}
(OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
print(json.dumps({"algebra": result["algebra"], "splits": splits,
    "historical_counts": {k: {s: {f: v for f, v in data.items() if not f.endswith('_samples')} for s, data in vals.items() if s in ['validation','test']} for k, vals in variants.items()},
    "raw_sha256": result["raw_measurements"]["sha256"], "inspection_pointer_count": len(set(pointers)), "all_assertions_passed": True}, ensure_ascii=False))
