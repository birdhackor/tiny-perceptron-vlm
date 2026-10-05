"""Independent bounded CPU review: no weights, training, downloads or external services.

Imports only original historical pure split/token functions, then replays the
original sampler. Runs the two necessary variants of the literal fence.
"""
import ast
import hashlib
import inspect
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, IGNORE, shifted, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum

torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.manual_seed(42)
assert torch.version.cuda is None and not torch.cuda.is_available()

environment = {
    "python": sys.version,
    "python_executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "device": "cpu",
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "threads": str(torch.get_num_threads()),
    "new_training_steps": "0",
    "weight_inputs": "none",
    "network_access": "none in this script",
}
(OUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
(OUT / "sources/installed-linear-source.py").write_text(inspect.getsource(torch.nn.Linear))

j = json.loads((OUT / "inputs/current/docs/course-experiments/results/text_foundation.json").read_text())
hist = OUT / "inputs/historical"
for name in ["scripts/course_experiments/common.py", "scripts/course_experiments/text.py",
             "tiny_perceptron/model.py", "tiny_perceptron/data.py", "tiny_perceptron/attention.py",
             "tiny_perceptron/modern.py"]:
    assert hashlib.sha256((hist / name).read_bytes()).hexdigest() == j["code_sha256"][name]

# Preserve actual historical function source, not a reimplementation.
tree = ast.parse((hist / "scripts/course_experiments/common.py").read_text())
chosen = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
          n.name in {"split_records", "records_sha256", "text_examples", "_nll"}]
assert len(chosen) == 4
namespace = {"random": random, "json": json, "hashlib": hashlib,
             "ByteTokenizer": ByteTokenizer, "shifted": shifted,
             "torch": torch, "pad_batch": pad_batch, "loss_sum": loss_sum}
exec(compile(ast.Module(body=chosen, type_ignores=[]), "historical-common-selected", "exec"), namespace)

# Execute the original evaluation contract on two tiny unequal-length rows.
# This checks its unit and denominator, not any original checkpoint's quality.
probe_examples = namespace["text_examples"]([{"text": "a"}, {"text": "abcd"}])
probe_model = TinyLM(ModelConfig(width=8))
before_state = {k: v.clone() for k, v in probe_model.state_dict().items()}
probe = namespace["_nll"](probe_model, probe_examples, batch_size=2)
probe_sum, probe_count = 0.0, 0
with torch.no_grad():
    for example in probe_examples:
        x, y, valid = pad_batch([example])
        summed, count = loss_sum(probe_model(x, valid=valid)["logits"], y)
        probe_sum += float(summed)
        probe_count += int(count)
assert probe_count == probe["effective_tokens"] == 7  # 5 bytes + 2 EOS
assert abs(probe_sum / probe_count - probe["nll"]) < 2e-6
assert all(torch.equal(before_state[k], v) for k, v in probe_model.state_dict().items())
print("historical _nll unequal-length no-update probe", probe,
      "per-row reaggregation", probe_sum / probe_count, "tolerance", 2e-6)

# Exact deterministic 80-record construction from the historical experiment.
text_tree = ast.parse((hist / "scripts/course_experiments/text.py").read_text())
fn = next(n for n in text_tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_text_foundation")
pool_node = next(n for n in fn.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == "pool" for t in n.targets))
exec(compile(ast.Module(body=[pool_node], type_ignores=[]), "historical-pool-expression", "exec"), namespace)
pool = namespace["pool"]
parts = namespace["split_records"](pool, seed=j["seed"])
assert len(pool) == 80
assert [len(parts[s]) for s in ("train", "validation", "test")] == [64, 8, 8]
family_sets = {s: {r["family"] for r in rows} for s, rows in parts.items()}
assert not (family_sets["train"] & family_sets["validation"])
assert not (family_sets["train"] & family_sets["test"])
assert not (family_sets["validation"] & family_sets["test"])
split_stats = {}
split_dir = OUT / "deterministic-splits"
split_dir.mkdir(exist_ok=True)
for split, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == j["results"]["scaling_data"][split]["sha256"]
    (split_dir / f"{split}.jsonl").write_bytes(raw)
    examples = namespace["text_examples"](rows, mode="text", max_length=128)
    split_stats[split] = {"records": len(rows), "sha256_matches_original": True,
                         "effective_tokens": sum(int((y != IGNORE).sum()) for x, y in examples),
                         "raw_utf8_bytes": sum(len(r["text"].encode()) for r in rows),
                         "families": sorted(family_sets[split])}

table = []
expected = {
    (16, 16): (20765, "0.27166", "0.49288", "0.39779"),
    (16, 64): (20623, "0.33032", "0.31105", "0.31061"),
    (32, 16): (20765, "0.14502", "0.40927", "0.27961"),
    (32, 64): (20623, "0.20107", "0.21914", "0.22258"),
}
for width in [8, 16, 32]:
    model = TinyLM(ModelConfig(width=width))
    counted = model.description()["parameters"]
    # 264w input + 128w positions + 264w untied output; 3 LayerNorms;
    # four attention w*w matrices; FFN w*4w and 4w*w plus 5w bias.
    formula = 12 * width * width + 667 * width
    assert formula == counted
    print("parameter count", width, counted, "= 12*w*w + 667*w")

for width in [16, 32]:
    for count in [16, 64]:
        name = f"scaling-w{width}-n{count}"
        entry = j["results"]["scaling"][name]
        training = entry["training"]
        rows = parts["train"][:count]
        assert namespace["records_sha256"](rows) == training["records_sha256"]
        examples = namespace["text_examples"](rows, mode="text", max_length=128)
        assert len(examples) == count == training["records"] == entry["unique_training_records"]
        assert training["steps"] == 150
        assert training["parameters"] == entry["parameters"] == 12 * width * width + 667 * width
        sampler = random.Random(j["seed"])
        token_count, exposures = 0, Counter()
        for step in range(training["steps"]):
            selected = sampler.choices(range(len(examples)), k=4)
            exposures.update(selected)
            token_count += sum(int((examples[i][1] != IGNORE).sum()) for i in selected)
        assert sum(exposures.values()) == 600
        assert token_count == training["effective_tokens"] == expected[(width, count)][0]
        losses = [training["final_loss"]]
        eval_checks = {}
        for split in ["validation", "test"]:
            e = entry["evaluation"][split]
            assert e["records"] == e["examples"] == 8
            assert e["effective_tokens"] == split_stats[split]["effective_tokens"]
            assert e["raw_utf8_bytes"] == split_stats[split]["raw_utf8_bytes"]
            assert abs(e["nll_sum"] / e["effective_tokens"] - e["nll"]) < 1e-15
            # The original display samples must be the same ordered held-out rows.
            tok = ByteTokenizer()
            assert [v["prompt"] for v in e["samples"]] == [
                tok.decode(tok.encode(row["text"])[:24]) for row in parts[split]]
            losses.append(e["nll"])
            eval_checks[split] = {k: e[k] for k in ["nll", "nll_sum", "effective_tokens", "records"]}
        rounded = [f"{v:.5f}" for v in losses]
        assert rounded == list(expected[(width, count)][1:])
        history = training["history"]
        assert history[0]["step"] == 1 and history[-1]["step"] == 150
        table.append({"name": name, "width": width, "records": count,
                      "steps": 150, "batch_records": 4, "record_exposures": 600,
                      "effective_tokens": token_count, "parameters": entry["parameters"],
                      "mean_record_exposures": 600 / count,
                      "train_evaluation_tokens": sum(int((y != IGNORE).sum()) for x, y in examples),
                      "final_train_loss_from_original_json": training["final_loss"],
                      "rounded_train_validation_test": rounded,
                      "held_out": eval_checks})
        print(name, json.dumps(table[-1], sort_keys=True))
for width in [16, 32]:
    a = j["results"]["scaling"][f"scaling-w{width}-n16"]
    b = j["results"]["scaling"][f"scaling-w{width}-n64"]
    assert b["training"]["final_loss"] > a["training"]["final_loss"]
    assert b["evaluation"]["validation"]["nll"] < a["evaluation"]["validation"]["nll"]

for width in [8, 16]:
    layer = torch.nn.Linear(width, width, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1)
        y = layer(torch.ones(width))
    assert torch.equal(y, torch.full((width,), float(width)))
    print("dense Linear", width, "x", width, "conventional multiply-add FLOPs", 2 * width * width,
          "literal no-bias sum adds", width * (width - 1))

for variant in ["exercise-200.py", "control-100.py"]:
    print("fence variant", variant)
    exec(compile((OUT / variant).read_bytes(), str(OUT / variant), "exec"), {"__name__": "__main__"})
assert 3200 / (4 * 4) == 200 and 4 * 4 * 100 == 1600
summary = {"original_run": {k: j[k] for k in ["revision", "seed", "device", "torch_version", "python_version", "gpu", "step_scale", "evidence_status"]},
           "split_stats": split_stats, "table": table,
           "scope": "Verified existing JSON arithmetic and original code contracts; replayed deterministic data and sampler only. No model weights read, training, optimizer steps or rescoring the original checkpoints.",
           "figures": {"svg_references": 0, "rendering": "not applicable; section has scalar budgets and a text table only"}}
(OUT / "checks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print("PASS: bounded independent checks; 0 training steps; no weight inputs")
