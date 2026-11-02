"""19.16 reviewer probe: artificial inline CPU only, then preserved-logit arithmetic.

The large original prediction archive and product files remain outside Git. The
JSON output records their public version, exact bytes/digests, and small results.
No trained weights are deserialized and no final-family neural forward is run.
"""

import argparse
import ast
import contextlib
import gzip
import hashlib
import io
import json
import math
import platform
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-gzip", type=Path, required=True)
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    torch.set_num_threads(1)
    freeze_path = root / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v8.json"
    freeze = json.loads(freeze_path.read_bytes())
    page = next(p for p in freeze["pages"] if p["page_id"] == "19.16")
    body = (root / page["snapshot"]).read_text()
    assert sha(body.encode()) == page["source_sha256"]
    chapter = (root / page["source"]).read_text()
    live = re.search(r"^## 19\.16 .*?(?=^## |\Z)", chapter, re.M | re.S).group()
    assert live == body
    blocks = re.findall(r"```python\n(.*?)```", body, re.S)
    assert len(blocks) == 2
    scope = {"__name__": "__review_inline__"}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(blocks[0], page["snapshot"] + ":inline-1", "exec"), scope)
        library_model = scope["model"]
        tokens = scope["tokens"]
        expected = library_model(tokens).detach()
        exec(compile(blocks[1], page["snapshot"] + ":inline-2", "exec"), scope)
        inline = scope["CandidateScorer"](library_model.config).eval()
        inline.load_state_dict(library_model.state_dict())
        with torch.no_grad():
            inline_scores = inline(tokens)
    assert torch.equal(inline_scores, expected)
    model_text = (root / "tiny_perceptron/lesson_products/model.py").read_text()
    library_ast = next(n for n in ast.parse(model_text).body if isinstance(n, ast.ClassDef) and n.name == "CandidateScorer")
    inline_ast = next(n for n in ast.parse(blocks[1]).body if isinstance(n, ast.ClassDef))
    assert ast.dump(library_ast) == ast.dump(inline_ast)
    reordered = {}
    first = {}
    with torch.no_grad():
        for order in (["card", "address", "app", "none"], ["none", "app", "address", "card"]):
            row = {"state": scope["state"], "question": "choice", "candidates": order}
            t, slices, maximum, labels = scope["library_candidate_batch"]([row], scope["tokenizer"], inline.config)
            p = inline(t).softmax(-1)
            current = dict(zip(order, p.tolist(), strict=True))
            if not first:
                first = current
            else:
                reordered = current
    assert first == reordered

    # Additional no-forward boundary checks of the exact public batching code.
    from tiny_perceptron.lesson_products.model import candidate_batch, grouped_scores, validate_candidates

    invalid = []
    for question, candidates in [("choice", ["none"]), ("choice", ["card", "card", "none"]), ("choice", ["card", "app"]), ("choice", ["mystery", "none"]), ("unknown", ["yes", "no"])]:
        try:
            validate_candidates(question, candidates)
        except ValueError as error:
            invalid.append(str(error))
        else:
            raise AssertionError((question, candidates))
    rows = [{"state": scope["state"], "question": "choice", "candidates": ["card", "none"], "gold": "card"}, {"state": scope["state"], "question": "info", "candidates": ["yes", "no"], "gold": "no"}, {"state": scope["state"], "question": "rubric", "candidates": ["grade0", "grade1", "grade2"], "gold": "grade0"}]
    _, slices, maximum, labels = candidate_batch(rows, scope["tokenizer"], inline.config)
    artificial = grouped_scores(torch.arange(7.0), slices, maximum)
    assert slices == [(0, 2), (2, 4), (4, 7)] and maximum == 3 and labels.tolist() == [0, 1, 0]
    assert artificial.softmax(-1)[0, 2].item() == 0.0

    release = json.loads((root / "tiny_perceptron/lesson_products/release.json").read_bytes())
    manifest = json.loads((root / "docs/lesson-products/results/c-artifact-manifest.json").read_bytes())
    originals = {}
    originals_data = {}
    for name in ("config.json", "tokenizer.json", "training-receipt.json", "calibration.json"):
        raw = (args.product / name).read_bytes()
        assert sha(raw) == manifest["files"][name]["sha256"] and len(raw) == manifest["files"][name]["bytes"]
        originals_data[name] = json.loads(raw)
        originals[name] = {"sha256": sha(raw), "bytes": len(raw), "url": f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/{release['products']['c']['prefix']}/{name}"}
    receipt = originals_data["training-receipt.json"]
    assert originals_data["config.json"] == vars(library_model.config)
    assert originals_data["tokenizer.json"]["characters"] == scope["tokenizer"].characters
    assert receipt["parameters"] == 84545 and receipt["seed"] == 9041
    assert receipt["last_step"] == receipt["selected_step"] == receipt["steps_this_attempt"] == 800
    assert receipt["random_initialization"] and receipt["resume_source"] is None and receipt["weights_changed"]
    policy_points = [{"step": h["step"], "choice_accuracy": h["policy"]["choice"]["accuracy"], "choice_ce": h["policy"]["choice"]["cross_entropy"]} for h in receipt["history"] if "policy" in h]
    selected = max(policy_points, key=lambda h: (h["choice_accuracy"], -h["choice_ce"]))
    assert selected["step"] == 800
    calibration = originals_data["calibration.json"]
    assert calibration["temperatures"] == {"choice": 0.25, "info": 0.25, "rubric": 0.25}

    groups = defaultdict(list)
    raw_digest = hashlib.sha256()
    raw_bytes = 0
    row_types = Counter()
    with gzip.open(args.raw_gzip, "rb") as stream:
        for line in stream:
            raw_digest.update(line)
            raw_bytes += len(line)
            row = json.loads(line)
            row_types[row["product"]] += 1
            if row["product"] == "c":
                groups[(row["split"], row["question"])].append(row)
    assert raw_digest.hexdigest() == "8b593421043b2d0ee8b81caea34612b468e8af7eafd5b84c552f8c25b0cf4896"
    assert raw_bytes == 56528235
    report = json.loads((root / "docs/lesson-products/results/final-report-v1.json").read_bytes())
    metrics = {}
    semantic_mismatches = 0
    missing_best = {"count": 0, "correct": 0}
    reordered_count = reordered_consistent = 0
    for (split, question), rows in groups.items():
        width = max(len(row["candidates"]) for row in rows)
        values = torch.full((len(rows), width), -torch.inf)
        labels = []
        correct = 0
        for i, row in enumerate(rows):
            scores = row["raw_scores"]
            values[i, :len(scores)] = torch.tensor(scores)
            labels.append(row["candidates"].index(row["gold"]))
            chosen = min(zip(row["candidates"], scores, strict=True), key=lambda item: (-item[1], item[0]))[0]
            correct += chosen == row["gold"]
            if split == "test":
                semantic_mismatches += chosen != row["selected"]
                if question == "choice":
                    reordered_count += 1
                    reordered_consistent += chosen == row["reverse_selected"]
                    if len(row["candidates"]) == 5:
                        assert row["gold"] == "none"
                        missing_best["count"] += 1
                        missing_best["correct"] += chosen == "none"
        labels_tensor = torch.tensor(labels)
        truth = F.one_hot(labels_tensor, width).float()
        actual = {}
        target = report["c"] if split == "test" else report["shifts"]["c_wording"]
        for phase, temperature in (("before", 1.0), ("after", calibration["temperatures"][question])):
            logits = values / temperature
            p = logits.softmax(-1)
            actual[phase] = {"count": len(rows), "correct": correct, "accuracy": correct / len(rows), "cross_entropy": F.cross_entropy(logits, labels_tensor).item(), "brier": (p - truth).square().sum(-1).mean().item()}
            for key in ("count", "correct", "cross_entropy", "brier"):
                assert math.isclose(actual[phase][key], target[question][phase][key], rel_tol=3e-6, abs_tol=1e-10), (split, question, phase, key, actual[phase][key], target[question][phase][key])
        metrics[f"{split}/{question}"] = actual
    assert semantic_mismatches == 0 and reordered_count == reordered_consistent == 11797
    assert missing_best == {"count": 849, "correct": 849}
    example = json.loads((root / "docs/lesson-products/results/observed-failure-examples-v1.json").read_bytes())["examples"][0]
    original = next(row for row in groups[("wording_shift", "choice")] if row["id"] == example["id"])
    for key in ("raw_scores", "candidates", "gold"):
        assert example[key] == original[key]
    logits = [value / 0.25 for value in original["raw_scores"]]
    weights = [math.exp(value - max(logits)) for value in logits]
    probabilities = [weight / sum(weights) for weight in weights]
    chosen_index = max(range(len(logits)), key=lambda i: logits[i])
    concentration = (len(logits) * probabilities[chosen_index] - 1) / (len(logits) - 1)
    assert original["candidates"][chosen_index] == "boxes" and example["state"] == "幫我預報明天天氣"
    assert math.isclose(concentration, example["concentration"], abs_tol=1e-14)
    parameter_parts = {"embedding": 144 * 64, "position": 128 * 64, "two_blocks": 2 * (128 + 12480 + 4160 + 128 + 8320 + 8256), "final_norm": 128, "score": 65}
    assert sum(parameter_parts.values()) == sum(p.numel() for p in inline.parameters()) == 84545
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu", "threads": "1"},
        "freeze_sha256": sha(freeze_path.read_bytes()), "source_sha256": page["source_sha256"],
        "inline_code_sha256": [sha(block.encode()) for block in blocks],
        "inline_stdout": captured.getvalue(),
        "inline_class_ast_equal": True,
        "inline_library_max_score_difference": (inline_scores - expected).abs().max().item(),
        "candidate_reorder_probabilities": [first, reordered],
        "parameter_derivation": parameter_parts,
        "boundary_checks": {"invalid_cases_rejected": invalid, "multirow_slices": slices, "max_candidates": maximum, "labels": labels_tensor[:0].tolist(), "label_example": [0, 1, 0], "padded_candidate_probability": 0.0},
        "product_original_receipts": originals,
        "training_receipt_excerpt": {key: receipt[key] for key in ("random_initialization", "resume_source", "steps_this_attempt", "last_step", "selected_step", "seed", "parameters", "weights_changed", "initial_weights_sha256", "selected_weights_sha256", "environment")},
        "policy_points_and_selected": {"points": policy_points, "selected": selected},
        "calibration_receipt_excerpt": {key: calibration[key] for key in ("temperatures", "grid", "selected_on", "measurements")},
        "raw_predictions": {"url": f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/lesson-products/v1-20261016/raw/final-report-v1.predictions.jsonl.gz", "gzip_bytes": args.raw_gzip.stat().st_size, "gzip_sha256": sha(args.raw_gzip.read_bytes()), "uncompressed_bytes": raw_bytes, "uncompressed_sha256": raw_digest.hexdigest(), "row_counts_by_product": dict(row_types)},
        "recomputed_metrics": metrics,
        "recorded_selected_vs_raw_score_mismatches": semantic_mismatches,
        "reverse_selected_record_consistency": {"count": reordered_count, "consistent": reordered_consistent, "scope": "Compared saved reverse selections only; no reversed neural forward repeated."},
        "missing_required_candidate_subset": missing_best,
        "weather_failure": {"id": original["id"], "state": example["state"], "candidates": original["candidates"], "gold": original["gold"], "raw_scores": original["raw_scores"], "selected": original["candidates"][chosen_index], "selected_probability": probabilities[chosen_index], "concentration": concentration},
        "limits": ["No training, retuning, formal final-family inference, Modal, workflow, or publication.", "Only artificial random inline model executed, including the displayed class body.", "Training/L4/step and temperature selection are authenticated original receipt/code inspections; calibration logits were not preserved here, so the temperature grid was not independently rerun.", "Metrics are arithmetic recomputation of SHA-matched preserved predictions. Existing reverse_selected values were counted, not neurally regenerated.", "No desktop/mobile layout inspection; assigned lesson contains no figures."]
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha(args.output.read_bytes()), "metrics_groups": len(metrics), "status": "all assertions passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
