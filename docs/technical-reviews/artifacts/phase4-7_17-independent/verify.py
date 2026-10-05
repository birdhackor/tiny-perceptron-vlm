"""Independent bounded CPU recomputation; no weights, model, or training loaded."""
import ast
import contextlib
import hashlib
import io
import json
import platform
import random
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def selected(path, names, namespace):
    tree = ast.parse(path.read_bytes())
    nodes = []
    found = set()
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            targets = [v.id for v in node.targets if isinstance(v, ast.Name)]
            name = next((v for v in targets if v in names), None)
        if name in names:
            nodes.append(node)
            found.add(name)
    assert found == set(names), (path, found, names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


result = json.loads((HERE / "original/sft-result.json").read_bytes())
provenance = json.loads((HERE / "provenance.json").read_bytes())
for path, entry in provenance["historical_code"].items():
    raw = (HERE / "historical" / path).read_bytes()
    assert sha(raw) == entry["historical_sha256"]
    assert sha(raw) == result["code_sha256"][path]

raw_fence = (HERE / "original/fence-1.py").read_bytes()
fence_outputs = {}
for label, code in [("original", raw_fence), ("cost5", raw_fence.replace(b"1, 20", b"1, 5"))]:
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, label, "exec"), namespace)
    values = [int(line.rsplit(" ", 1)[1]) for line in stdout.getvalue().splitlines()]
    assert values == ([1000, 1000, 20000] if label == "original" else [1000, 250, 5000])
    fence_outputs[label] = {"stdout": stdout.getvalue(), "values": values, "code_sha256": sha(code)}
(HERE / "cost5.py").write_bytes(raw_fence.replace(b"1, 20", b"1, 5"))

ns = {"torch": torch, "json": json, "hashlib": hashlib, "random": random}
selected(HERE / "historical/tiny_perceptron/data.py",
         {"IGNORE", "SPECIALS", "ByteTokenizer", "render_chat", "shifted", "pad_batch"}, ns)
selected(HERE / "historical/scripts/prepare_data.py", {"conversation", "generate_records"}, ns)
selected(HERE / "historical/scripts/course_experiments/common.py",
         {"split_records", "records_sha256", "text_examples"}, ns)
parts = ns["split_records"](ns["generate_records"]("attributes-sft"), seed=42)
d = result["results"]
families = {k: {r["family"] for r in rows} for k, rows in parts.items()}
assert not any(families[a] & families[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
split_checks = {}
(HERE / "reconstructed-inputs").mkdir(exist_ok=True)
for split, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    assert sha(raw) == d["data"][split]["sha256"]
    assert len(rows) == d["data"][split]["records"]
    assert len(families[split]) == d["data"][split]["families"]
    (HERE / "reconstructed-inputs" / (split + ".jsonl")).write_bytes(raw)
    split_checks[split] = {"records": len(rows), "families": len(families[split]), "sha256": sha(raw)}

tok = ns["ByteTokenizer"]()
evaluations = {}
for label, evals in [("direct_sft_after", d["after"]),
                     ("pretrain_before_sft", d["pretrain_then_sft"]["before_sft"]),
                     ("pretrain_after_sft", d["pretrain_then_sft"]["after_sft"])]:
    evaluations[label] = {}
    for split, report in evals.items():
        rows = parts[split]
        assert len(rows) == len(report["samples"]) == report["records"]
        matched = ended = 0
        details = []
        for i, (row, sample) in enumerate(zip(rows, report["samples"], strict=True)):
            assert sample["messages"] == row["messages"][:-1]
            assert sample["expected"] == row["messages"][-1]["content"]
            ids = sample["generated_ids"]
            eos = tok.eos_id in ids
            raw = ids[:ids.index(tok.eos_id)] if eos else ids
            exact = raw == tok.encode(sample["expected"])
            assert exact == sample["exact"] and eos == sample["eos"]
            assert tok.decode(raw) == sample["generated"]
            matched += int(exact)
            ended += int(eos)
            details.append({"index": i, "question": row["messages"][0]["content"],
                            "expected": sample["expected"], "generated_ids": ids,
                            "recomputed_exact": exact, "recomputed_eos": eos})
        examples = ns["text_examples"](rows, "sft", 128)
        effective = sum(int((y != -100).sum()) for _, y in examples)
        assert effective == report["effective_tokens"]
        assert matched == report["matches"]
        assert matched / len(rows) == report["exact_match"]
        assert ended / len(rows) == report["eos_rate"]
        assert abs(report["nll_sum"] / effective - report["nll"]) < 1e-12
        evaluations[label][split] = {"matches": matched, "denominator": len(rows),
                                   "eos_count": ended, "target_tokens": effective,
                                   "recomputed_nll_ratio": report["nll_sum"] / effective,
                                   "sample_checks": details}

text_rows = [{"text": row["messages"][0]["content"] + row["messages"][1]["content"],
              "family": row["family"]} for row in parts["train"]]
recipes = {}
for label, rows, mode, report, expected_steps in [
    ("direct_sft", parts["train"], "sft", d["training"], 900),
    ("pretraining", text_rows, "text", d["pretrain_then_sft"]["pretraining"], 250),
    ("continued_sft", parts["train"], "sft", d["pretrain_then_sft"]["sft"], 900),
]:
    assert report["steps"] == expected_steps and report["records"] == len(rows) == 45
    assert ns["records_sha256"](rows) == report["records_sha256"]
    examples = ns["text_examples"](rows, mode, 128)
    sampler = random.Random(42)
    effective = 0
    for _ in range(expected_steps):
        batch = sampler.choices(examples, k=16)
        effective += sum(int((y != -100).sum()) for _, y in batch)
    assert effective == report["effective_tokens"]
    recipes[label] = {"records": 45, "steps": expected_steps, "batch_size": 16,
                      "sampled_record_occurrences": expected_steps * 16,
                      "target_tokens": effective, "mode": mode,
                      "records_sha256": ns["records_sha256"](rows)}

observed = {
    "scope": "Recomputed fixed original JSON, raw token exactness, EOS, split hashes, and historical sampler target counts. No model weights loaded; no training or generated inference reproduced.",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git": str(torch.version.git_version), "device": "cpu", "cuda": "None"},
    "fence_outputs": fence_outputs, "split_checks": split_checks,
    "evaluations": evaluations, "recipes": recipes,
    "extra_text_updates": recipes["pretraining"]["steps"],
    "units": {"cost": "assumed preparation-cost units per record; not currency or compute",
              "accuracy_denominator": "held-out questions; exact raw content tokens",
              "loss_denominator": "assistant answer bytes plus EOS targets",
              "training_budget": "optimizer updates and sampled effective targets"},
    "tolerance": "Integer counts and raw token lists exact; nll_sum/effective_tokens ratio absolute error < 1e-12.",
}
(HERE / "verification.json").write_text(json.dumps(observed, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: observed[k] for k in ["scope", "environment", "fence_outputs", "split_checks", "recipes", "extra_text_updates", "units", "tolerance"]}, ensure_ascii=False, indent=2))
for label, splits in evaluations.items():
    print(label, {k: {q:v[q] for q in ["matches", "denominator", "eos_count", "target_tokens"]} for k,v in splits.items()})
