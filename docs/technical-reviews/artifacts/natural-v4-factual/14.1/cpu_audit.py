"""Independent bounded CPU audit of lesson 14.1; no training or downloaded weights."""
import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import random
import re
from pathlib import Path

import torch

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from tiny_perceptron.data import ByteTokenizer, IGNORE, load_jsonl, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum
from tiny_perceptron.modern import rope

ROOT = Path(__file__).resolve().parents[5]
torch.set_num_threads(2)
torch.manual_seed(42)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "device": "cpu", "cuda_available": str(torch.cuda.is_available()),
                              "default_dtype": str(torch.get_default_dtype()), "threads": str(torch.get_num_threads())},
              "scope": "Offline deterministic calculations and existing GPU-record audit; no own training, GPU benchmark, or checkpoint replication."}
    source = (ROOT / "course/chapters/14.md").read_text(encoding="utf-8")
    section = source[source.index("## 14.1 "):source.index("## 14.2 ")]
    block = re.search(r"```python\n(.*?)\n```", section, re.S).group(1)
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(block, "lesson-14.1-exact-code", "exec"), namespace)
    result["exact_lesson_code"] = {"stdout": stdout.getvalue(), "code_sha256": hashlib.sha256(block.encode()).hexdigest()}
    assert stdout.getvalue() == "-0.7071\n-0.7071\n0.0\n"
    scores = []
    for a, b in [(2, 5), (12, 15), (3, 5), (22, 25), (22, 26)]:
        observed = namespace["score"](a, b)
        expected = math.cos((b - a) * math.pi / 4)
        assert abs(observed - expected) <= 1e-7
        scores.append({"positions": [a, b], "gap": b-a, "angle_difference_degrees": 45*(b-a),
                       "observed_float32": observed, "expected_real_formula": expected, "rounded4": round(observed, 4)})
    result["scores_and_exercise"] = scores
    result["unit_coordinates"] = {str(deg): [math.cos(math.radians(deg)), math.sin(math.radians(deg))] for deg in [45, 90, 135]}
    # Content vectors remain fixed. The identity is pairwise, not a claim about a re-encoded longer sentence.
    q, k = torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16)
    positions = torch.tensor([2, 5, 8])
    qr, kr = rope(q, positions), rope(k, positions)
    qs, ks = rope(q, positions+10), rope(k, positions+10)
    dot_error = (qr @ kr.transpose(-1, -2) - qs @ ks.transpose(-1, -2)).abs().max().item()
    norm_error = (q.norm(dim=-1)-qr.norm(dim=-1)).abs().max().item()
    assert dot_error < 1e-5 and norm_error < 1e-6
    result["multifrequency_probe"] = {"seed": 42, "shape": list(q.shape), "positions": positions.tolist(),
                                      "common_shift": 10, "base": 10000, "max_dot_error": dot_error,
                                      "max_norm_error": norm_error, "tolerances": [1e-5, 1e-6], "dtype": str(q.dtype)}
    logits = torch.tensor([-math.sqrt(0.5), 0.0, 1.0])
    result["negative_score_softmax"] = {"scores": logits.tolist(), "weights": logits.softmax(0).tolist(),
                                        "sum": logits.softmax(0).sum().item()}
    record_path = ROOT / "docs/course-experiments/results/modern.json"
    record = json.loads(record_path.read_text())
    variants = record["results"]["variants"]
    result["record_receipt"] = {"path": str(record_path.relative_to(ROOT)), "sha256": digest(record_path),
                                "revision": record["revision"], "recorded_environment": {k: record[k] for k in ["device", "seed", "torch_version", "python_version", "gpu", "timing_scope", "step_scale", "evidence_status"]},
                                "recorded_runtime": record["results"]["runtime"]}
    result["parameters_and_run_conditions"] = {}
    for name, variant in variants.items():
        model = TinyLM(ModelConfig(**variant["model"]["config"]))
        count = sum(p.numel() for p in model.parameters())
        assert count == variant["model"]["parameters"]
        training = variant["training"]
        assert training["steps"] == training["optimizer_updates"] == training["requested_steps"] == 240
        assert training["skipped_updates"] == 0 and training["effective_tokens"] == 452102
        result["parameters_and_run_conditions"][name] = {"computed_parameters": count, "config": variant["model"]["config"],
                                                          "training": {k: training[k] for k in ["requested_steps", "steps", "optimizer_updates", "skipped_updates", "effective_tokens", "batch_size", "learning_rate", "gradient_clip_norm", "schedule", "scaler_enabled", "warm_step_median_seconds"]}}
    assert 141568-133376 == 128*64 == 8192
    data_path = ROOT / "data/training/text-initial/tinystories-train-512.jsonl"
    archive_path = ROOT / "assets/training/tinystories-v1.tar.gz"
    assert digest(data_path) == next(f["sha256"] for f in record["assets"][0]["files"] if f["path"].endswith("tinystories-train-512.jsonl"))
    assert digest(archive_path) == record["assets"][0]["archive_sha256"]
    rows = load_jsonl(data_path)
    assert len(rows) == 512
    for row in rows:
        row["family"] = row.get("text_sha256", records_sha256([{"text": row["text"]}]))
    splits = split_records(rows, 42)
    result["input_receipts"] = {"dataset_path": str(data_path.relative_to(ROOT)), "dataset_sha256": digest(data_path),
                               "archive_path": str(archive_path.relative_to(ROOT)), "archive_sha256": digest(archive_path)}
    result["independent_splits"] = {}
    for split, stories in splits.items():
        windows = text_examples(stories, max_length=128)
        targets = sum(int((y != IGNORE).sum()) for _, y in windows)
        byte_plus_eos = sum(len(s["text"].encode("utf-8"))+1 for s in stories)
        fingerprint = records_sha256(stories)
        assert targets == byte_plus_eos
        assert fingerprint == record["results"]["dataset"][split]["sha256"]
        assert len(stories) == record["results"]["dataset"][split]["records"]
        result["independent_splits"][split] = {"records": len(stories), "sha256": fingerprint, "windows": len(windows),
                                               "targets": targets, "sum_bytes_plus_one_eos_per_story": byte_plus_eos}
        expected_targets = variants["baseline"]["training"]["initial"]["effective_tokens"] if split=="train" else variants["baseline"]["heldout"][split]["effective_tokens"]
        assert targets == expected_targets
    families = [{r["family"] for r in splits[name]} for name in ["train", "validation", "test"]]
    assert not any(families[i] & families[j] for i in range(3) for j in range(i+1,3))
    windows = text_examples(splits["train"], max_length=128)
    sampler = random.Random(42)
    sampled_targets = sum(sum(int((y != IGNORE).sum()) for _, y in sampler.choices(windows, k=16)) for _ in range(240))
    assert sampled_targets == 452102
    result["independent_training_sample_count"] = {"seed": 42, "draws_per_update": 16, "updates": 240, "effective_targets": sampled_targets}
    tok = ByteTokenizer()
    result["utf8_units"] = {s: {"bytes": list(s.encode("utf-8")), "byte_token_ids": tok.encode(s)} for s in ["A", "貓", "𠀀"]}
    result["heldout_recalculation"] = {}
    for name in ["baseline", "rope"]:
        result["heldout_recalculation"][name] = {}
        for split in ["validation", "test"]:
            heldout = variants[name]["heldout"][split]
            ratio = heldout["nll_sum"]/heldout["effective_tokens"]
            assert abs(ratio-heldout["nll"]) < 1e-12
            assert all(tok.decode(s["generated_ids"]) == s["generated"] for s in heldout["samples"])
            result["heldout_recalculation"][name][split] = {"numerator": heldout["nll_sum"], "denominator": heldout["effective_tokens"],
                                                          "computed_nll": ratio, "rounded5": round(ratio,5), "recorded_nll": heldout["nll"],
                                                          "sample_count": len(heldout["samples"]), "first_sample": heldout["samples"][0]}
    examples = text_examples([{"text": "A"}, {"text": "貓"}], max_length=128)
    x, y, valid = pad_batch(examples)
    logits = torch.zeros(*y.shape, tok.vocab_size)
    summed, count = loss_sum(logits, y)
    assert int(count) == 6 and valid.sum().item()==6 and (y==IGNORE).sum().item()==2
    result["masked_nll_probe"] = {"input_shape": list(x.shape), "labels": y.tolist(), "dtype": str(x.dtype),
                                  "ignored_padding": int((y==IGNORE).sum()), "effective_targets": int(count),
                                  "observed_mean_nll": (summed/count).item(), "expected_ln264": math.log(264)}
    # Retrieve-time original blobs match the historical result; compare only actually inspected modern functions.
    old = ast.parse((ROOT / "outputs/natural-v4/factual-research/14.1/recorded-architecture.py").read_text())
    now = ast.parse((ROOT / "scripts/course_experiments/architecture.py").read_text())
    names = ["_sync", "_description", "_copy_matching", "_clone_config", "_text_dataset", "_data_report", "_runtime", "_steps", "_amp", "_forward", "_nll", "_train", "_heldout", "run_modern"]
    same = {name: ast.dump(next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name))==ast.dump(next(n for n in now.body if isinstance(n,ast.FunctionDef) and n.name==name)) for name in names}
    assert all(same.values())
    result["historical_modern_function_ast_comparison"] = same
    result["all_assertions_passed"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
