"""Independent bounded CPU checks for 13.9; never load or train an existing checkpoint."""
import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.alignment import dpo_loss, sequence_log_probability
from tiny_perceptron.data import ByteTokenizer, IGNORE, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
read_pointers = []
raw_selected = {}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_digest(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def select(document, name, pointer):
    value = document
    for part in pointer.strip("/").split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    key = name + "#" + pointer
    read_pointers.append(key)
    raw_selected[key] = value
    return value


def isolated_function(path, name, namespace):
    tree = ast.parse(path.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


environment = {
    "python": platform.python_version(), "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version), "cuda_build": str(torch.version.cuda),
    "device": "cpu", "threads": str(torch.get_num_threads()), "cwd": str(Path.cwd()),
}
out = {"environment": environment}

# Execute original fence bytes, then the requested transfer and the meaningful None/0 variant.
fence = (BASE / "inputs/fence-1.py").read_bytes()
ns = {"__name__": "__main__"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(fence, "13.9-original-fence", "exec"), ns)
assert sum(ns["recipe"].values()) == 100
assert len(ns["metrics"]) == 6 and all(v is None for v in ns["metrics"].values())
assert buf.getvalue().count("尚未量測") == 6
changed = dict(ns["recipe"])
changed["應拒絕題"] += changed["正常題完成"]
changed["正常題完成"] = 0
assert changed["應拒絕題"] == 40 and sum(changed.values()) == 100
assert all(v is None for v in ns["metrics"].values())
measured = dict(ns["metrics"])
measured["格式"] = 0
display = {k: "尚未量測" if v is None else v for k, v in measured.items()}
assert display["格式"] == 0 and sum(v == "尚未量測" for v in display.values()) == 5
out["recipe_and_unknown"] = {"original_stdout": buf.getvalue(), "transferred_recipe": changed, "zero_variant": display}

# Effective target count includes assistant answer bytes plus EOS, excludes prompt and padding.
counts = {}
tok = ByteTokenizer()
for positions in (100, 10):
    examples = [render_chat([{"role": "user", "content": "這是很長的問題，仍不計分"},
                             {"role": "assistant", "content": "a" * (positions - 1)}]) for _ in range(30)]
    x, y, valid = pad_batch(examples)
    n = int((y != IGNORE).sum())
    assert n == 30 * positions
    assert int((y == tok.eos_id).sum()) == 30
    counts[str(positions)] = {"pairs": 30, "per_answer_including_eos": positions, "both_sides_positions": 2 * n}
assert counts["100"]["both_sides_positions"] == 6000 and counts["10"]["both_sides_positions"] == 600
mixed = [render_chat([{"role": "user", "content": "p"}, {"role": "assistant", "content": s}]) for s in ["a" * 99, "a" * 9]]
x, y, valid = pad_batch(mixed)
assert int((y != IGNORE).sum()) == 110 and bool((y[~valid] == IGNORE).all())
logits = torch.zeros(y.shape + (tok.vocab_size,))
scores = sequence_log_probability(logits, y)
expected = -torch.tensor([100., 10.]) * math.log(tok.vocab_size)
assert torch.allclose(scores, expected, atol=1e-4, rtol=0)
out["target_positions"] = {"counts": counts, "mixed_mask_count": 110, "uniform_scores": scores.tolist(), "tolerance": "1e-4 absolute float32; counts exact integers"}

# Feasibility example only: one globally shared refusal logit is changed by one DPO update.
# It does not estimate any real language model's refusal rate or safety capability.
theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
logp = torch.stack((theta, torch.zeros_like(theta))).log_softmax(0)
loss = dpo_loss(logp[0:1], logp[1:2], torch.tensor([-math.log(2.)]), torch.tensor([-math.log(2.)]), beta=0.1)
loss.backward()
updated = float(theta.detach() - 0.1 * theta.grad)
assert updated > 0
refusal_probability = 1 / (1 + math.exp(-updated))
out["shared_parameter_feasibility"] = {"model": "all prompts share two-response logits [theta, 0]; candidate 0 is refusal", "single_step_theta": updated, "refusal_probability_all_prompts": refusal_probability, "greedy_refusal_all_prompts": True, "scope": "one scalar CPU update shows a possible failure mechanism; no real-model capability measurement"}

result = json.loads((BASE / "inputs/dpo-original-results.json").read_text())
style = json.loads((BASE / "inputs/style-original-results.json").read_text())
for key in ["revision", "device", "seed", "torch_version", "python_version", "step_scale"]:
    select(result, "dpo", "/" + key)
    select(style, "style", "/" + key)
ns = {"random": random, "json": json, "hashlib": hashlib}
split_records = isolated_function(BASE / "inputs/scripts__course_experiments__common.py", "split_records", ns)
ns = {}
arithmetic_records = isolated_function(BASE / "inputs/text-current.py", "arithmetic_records", ns)
arithmetic = split_records(arithmetic_records(), seed=42)
ns = {"render_chat": render_chat}
pair_parts = isolated_function(BASE / "inputs/behavior-revision-8a757184.py", "_preference_parts", ns)(arithmetic)
format_parts = {s: [{**r, "rejected": r["chosen"] + "; answer complete"} for r in rows] for s, rows in pair_parts.items()}
for name, parts in [("data", pair_parts), ("format_only/data", format_parts)]:
    for split, rows in parts.items():
        for k, v in {"records": len(rows), "families": len({r["family"] for r in rows}), "sha256": digest(jsonl_bytes(rows))}.items():
            assert select(result, "dpo", f"/results/{name}/{split}/{k}") == v
out["arithmetic_splits"] = {s: {"records": len(rows), "families": len({r["family"] for r in rows})} for s, rows in arithmetic.items()}

def arithmetic_eval(document, document_name, pointer):
    samples = select(document, document_name, pointer + "/samples")
    records = select(document, document_name, pointer + "/records")
    assert len(samples) == records == 7
    matches = 0
    eos_count = 0
    for sample in samples:
        ids = sample["generated_ids"]
        eos = tok.eos_id in ids
        raw = ids[:ids.index(tok.eos_id)] if eos else ids
        exact = raw == tok.encode(sample["expected"])
        assert exact == sample["exact"] and eos == sample["eos"]
        assert tok.decode(raw) == sample["generated"]
        matches += int(exact)
        eos_count += int(eos)
    assert matches == select(document, document_name, pointer + "/matches")
    assert matches / records == select(document, document_name, pointer + "/exact_match")
    assert eos_count / records == select(document, document_name, pointer + "/eos_rate")
    assert [(s["messages"][0]["content"], s["expected"]) for s in samples] == [(r["messages"][0]["content"], r["messages"][-1]["content"]) for r in arithmetic["test"]]
    return {"records": records, "matches_recomputed": matches, "eos_recomputed": eos_count}

generation = {"starting_style_content": arithmetic_eval(style, "style", "/results/content_evaluation/test")}
for branch in ["runs/model", "runs/beta1", "format_only"]:
    generation[branch] = arithmetic_eval(result, "dpo", f"/results/{branch}/arithmetic/test")
assert all(x["matches_recomputed"] == 0 for x in generation.values())
out["generation"] = generation

def preference_eval(pointer, expected_records, expected_improved=None, expected_ranked=None):
    samples = select(result, "dpo", pointer + "/samples")
    n = select(result, "dpo", pointer + "/records")
    assert n == len(samples) == expected_records
    improved = ranked = 0
    margins = []
    for s in samples:
        pm = s["policy_chosen_logp"] - s["policy_rejected_logp"]
        rm = pm - s["reference_margin"]
        assert abs(pm - s["policy_margin"]) < 1e-9 and abs(rm - s["relative_margin"]) < 1e-9
        for side in ["chosen", "rejected"]:
            _, y = render_chat([{"role": "user", "content": s["prompt"]}, {"role": "assistant", "content": s[side]}])
            assert int((y != IGNORE).sum()) == s[side + "_answer_tokens"]
        improved += int(rm > 0)
        ranked += int(pm > 0)
        margins.append(rm)
    assert improved == select(result, "dpo", pointer + "/relative_preference_improved")
    assert ranked == select(result, "dpo", pointer + "/chosen_higher_absolute_probability")
    assert abs(sum(margins) / n - select(result, "dpo", pointer + "/mean_relative_margin")) < 1e-9
    if expected_improved is not None: assert improved == expected_improved
    if expected_ranked is not None: assert ranked == expected_ranked
    return {"records": n, "improved_recomputed": improved, "ranked_recomputed": ranked, "tolerance": "1e-9 absolute arithmetic; strict >0 counts"}

prefs = {}
prefs["before_test"] = preference_eval("/results/before/test", 7, 0)
for branch in ["runs/model", "runs/beta1"]:
    prefs[branch] = preference_eval(f"/results/{branch}/preference/test", 7)
prefs["format_only"] = preference_eval("/results/format_only/preference/test", 7, 7, 7)
prefs["natural_validation"] = preference_eval("/results/ultrafeedback_pilot/evaluation/validation", 10, 6, 7)
prefs["natural_test"] = preference_eval("/results/ultrafeedback_pilot/evaluation/test", 10, 8, 4)
out["preference"] = prefs

# Rebuild the natural branch data from original upstream records without downloading data.
raw = [json.loads(line) for line in (BASE / "inputs/ultrafeedback-original-first100.jsonl").read_text().splitlines()]
assert len(raw) == select(result, "dpo", "/results/ultrafeedback_pilot/source_records") == 100
crop = isolated_function(BASE / "inputs/text-current.py", "_utf8_prefix", {})
natural = []
truncated = {"prompt": 0, "chosen": 0, "rejected": 0}
for index, row in enumerate(raw):
    answers = {}
    for side in ["chosen", "rejected"]:
        value = row[side]
        answers[side] = next(m["content"] for m in reversed(value) if m["role"] == "assistant") if isinstance(value, list) else value
    original = {"prompt": row["prompt"], **answers}
    for key, text in original.items(): truncated[key] += int(len(text.encode()) > 120)
    natural.append({"family": row.get("prompt_id", canonical_digest(row["prompt"])),
                    "prompt": crop(row["prompt"], 120), "chosen": crop(answers["chosen"], 120),
                    "rejected": crop(answers["rejected"], 120), "source_row": index,
                    "source_record_sha256": canonical_digest(row)})
parts = split_records(natural, seed=42)
family_sets = [set(r["family"] for r in parts[s]) for s in ["train", "validation", "test"]]
assert not any(a & b for i, a in enumerate(family_sets) for b in family_sets[i+1:])
for split, rows in parts.items():
    for k, v in {"records": len(rows), "families": len({r["family"] for r in rows}), "sha256": digest(jsonl_bytes(rows))}.items():
        assert select(result, "dpo", f"/results/ultrafeedback_pilot/data/{split}/{k}") == v
    assert all(len(r[k].encode()) <= 120 for r in rows for k in ["prompt", "chosen", "rejected"])
    if split != "train":
        samples = select(result, "dpo", f"/results/ultrafeedback_pilot/evaluation/{split}/samples")
        for sample, row in zip(samples, rows, strict=True):
            assert all(sample[k] == row[k] for k in row)
out["natural_data"] = {"records": len(raw), "splits": {k: len(v) for k, v in parts.items()}, "truncated_original_fields": truncated, "max_utf8_bytes_each_field": 120, "source_data_sha256": digest((BASE / "inputs/ultrafeedback-original-first100.jsonl").read_bytes())}

steps = {}
for branch, expected in [("runs/model", 250), ("runs/beta1", 250), ("format_only", 200), ("ultrafeedback_pilot", 80)]:
    pointer = f"/results/{branch}/training"
    n = select(result, "dpo", pointer + "/steps")
    history = select(result, "dpo", pointer + "/history")
    assert n == expected and history[-1]["step"] == n
    steps[branch] = n
n = select(result, "dpo", "/results/ultrafeedback_pilot/sft_training/steps")
assert n == 80
steps["ultrafeedback_sft"] = n
# Only instantiate a fresh architecture; do not run training or load weights.
model = TinyLM(ModelConfig(width=32, layers=1, max_length=256))
params = sum(p.numel() for p in model.parameters())
assert params == select(result, "dpo", "/results/ultrafeedback_pilot/sft_training/parameters") == 37728
assert len(model.blocks) == 1 and model.config.width == 32
out["recipe_contract"] = {"recorded_steps_recomputed_history_endpoint": steps, "fresh_model_width": 32, "fresh_model_layers": 1, "fresh_model_parameter_count": params, "training_executed": False, "existing_weights_loaded": False}

# Preserve precisely selected raw measurement/provenance pointers; do not copy any notes/scope values.
(BASE / "runs/selected-raw-measurements.json").write_text(json.dumps(raw_selected, ensure_ascii=False, indent=2) + "\n")
(BASE / "runs/measurement-pointers.json").write_text(json.dumps(sorted(set(read_pointers)), indent=2) + "\n")
(BASE / "runs/cpu-verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
print("All bounded CPU checks completed; no model checkpoint was loaded or trained.")
