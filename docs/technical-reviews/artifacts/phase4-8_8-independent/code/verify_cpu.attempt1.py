"""Independent bounded CPU checks for section 8.8; no training run or model inference."""
import ast
import copy
import hashlib
import json
import platform
import random
import re
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.data import IGNORE, ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM

torch.set_num_threads(1)
RUN = OUT / "inputs/run-version"
result = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__),
    "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_available": str(torch.cuda.is_available())},
    "scope": "Original single-layer fence was executed separately. This check uses one-layer synthetic gradients and one SGD step, model construction for parameter counts only, and saved JSON accounting. No trained model loading, forward inference, data files, downloads, or training run."}

def digest(b):
    return hashlib.sha256(b).hexdigest()

saved = json.loads((OUT / "inputs/current/docs/course-experiments/results/lora.json").read_bytes())
provenance = {}
for filename in ["scripts/course_experiments/behavior.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py",
                 "tiny_perceptron/alignment.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py",
                 "tiny_perceptron/modern.py", "tiny_perceptron/data.py"]:
    raw = (RUN / filename).read_bytes()
    observed = digest(raw)
    assert observed == saved["code_sha256"][filename]
    provenance[filename] = {"sha256": observed, "recorded_run_hash_matches": True,
                            "current_code_matches": raw == (ROOT / filename).read_bytes()}
result["run_code_provenance"] = {"revision": saved["revision"], "files": provenance}

def original_functions(filename, names):
    """Compile only named original function nodes, unedited, to avoid unrelated entrypoints."""
    path = RUN / filename
    tree = ast.parse(path.read_bytes(), filename=str(path))
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in found} == set(names)
    exec(compile(ast.Module(body=found, type_ignores=[]), str(path), "exec"), globals())
    return {n.name: [n.lineno, n.end_lineno] for n in found}

result["executed_original_functions"] = {
    "behavior.py": original_functions("scripts/course_experiments/behavior.py", ["_conversation", "_style_record", "_style_metrics", "_add_lora"]),
    "common.py": original_functions("scripts/course_experiments/common.py", ["split_records", "text_examples", "records_sha256"]),
    "text.py": original_functions("scripts/course_experiments/text.py", ["arithmetic_records"]),
}

layers = []
for rank in [2, 4]:
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(16, 12), rank=rank, alpha=rank)
    x = torch.randn(3, 16)
    state = {n: p.detach().clone() for n, p in layer.named_parameters()}
    y = layer(x)
    assert torch.equal(y, layer.base(x))
    y.sum().backward()
    assert layer.a.grad.abs().max().item() == 0
    assert layer.b.grad.abs().max().item() > 0
    assert layer.base.weight.grad is None and layer.base.bias.grad is None
    assert all(torch.equal(p, state[n]) for n, p in layer.named_parameters())
    before = {"rank": rank, "alpha": rank, "W_shape": list(layer.base.weight.shape),
              "A_shape": list(layer.a.shape), "B_shape": list(layer.b.shape), "output_shape": list(y.shape),
              "weight_count": layer.base.weight.numel(), "frozen_bias_count": layer.base.bias.numel(),
              "trainable_count": sum(p.numel() for p in layer.parameters() if p.requires_grad),
              "initial_equal": True, "A_gradient_max": layer.a.grad.abs().max().item(),
              "B_gradient_max": layer.b.grad.abs().max().item(), "backward_changes_no_parameters": True}
    optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
    optimizer.step()
    assert torch.equal(layer.base.weight, state["base.weight"])
    assert torch.equal(layer.base.bias, state["base.bias"])
    assert torch.equal(layer.a, state["a"]) and not torch.equal(layer.b, state["b"])
    optimizer.zero_grad(set_to_none=True)
    layer(x).sum().backward()
    assert layer.a.grad.abs().max().item() > 0
    before.update({"one_SGD_step_changes_B_only": True, "base_unchanged_after_step": True,
                   "second_A_gradient_max": layer.a.grad.abs().max().item()})
    layers.append(before)
result["single_layer"] = layers

# A nonzero, deterministic correction checks column convention and alpha/rank scaling.
torch.manual_seed(7)
layer = LoRALinear(nn.Linear(16, 12), rank=4, alpha=2)
with torch.no_grad():
    layer.a.copy_(torch.arange(64, dtype=torch.float32).reshape(4, 16) / 100)
    layer.b.copy_(torch.arange(48, dtype=torch.float32).reshape(12, 4) / 100)
x = torch.arange(16, dtype=torch.float32).reshape(1, 16) / 10
delta = (layer.b @ layer.a) * 0.5
row_output = layer(x)
column_output = (layer.base.weight @ x[0] + layer.base.bias + delta @ x[0]).unsqueeze(0)
torch.testing.assert_close(row_output, column_output, rtol=1e-6, atol=1e-6)
torch.testing.assert_close(row_output, nn.functional.linear(x, layer.merged_weight(), layer.base.bias), rtol=1e-6, atol=1e-6)
result["matrix_axes_and_scaling"] = {"A": [4,16], "B": [12,4], "BA_and_W": [12,16],
    "alpha_over_rank": 0.5, "column_row_max_error": float((row_output-column_output).abs().max()),
    "correction_matrix_rank": int(torch.linalg.matrix_rank(delta)), "tolerance": "rtol=atol=1e-6"}

# Construction only: never call the full model's forward or load neural weights.
model = TinyLM(ModelConfig(width=64, layers=2))
base_count = sum(p.numel() for p in model.parameters())
assert base_count == 141568
assert base_count == 2 * 264 * 64 + 128 * 64 + 2 * (12 * 64**2 + 13 * 64) + 2 * 64
names = _add_lora(model, rank=4, alpha=4)
counts = {n: sum(p.numel() for p in model.get_submodule(n).parameters() if p.requires_grad) for n in names}
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert len(names) == 11 and trainable == 9504 and trainable == 2 * (3*4*(64+64)+2*4*(64+256)) + 4*(64+264)
assert all(n.endswith((".a", ".b")) for n,p in model.named_parameters() if p.requires_grad)
assert names == saved["results"]["runs"]["concise"]["layers"] == saved["results"]["runs"]["vivid"]["layers"]
result["full_model_parameter_accounting"] = {"config": vars(model.config), "base_parameters": base_count,
    "adapter_trainable_parameters": trainable, "module_count": len(names), "module_counts": counts,
    "ratio_to_base_percent": trainable/base_count*100,
    "frozen_base_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    "base_parameters_after_injection": base_count, "total_with_adapter": base_count+trainable,
    "all_trainable_names_A_or_B": True, "full_model_forward_called": False}

arithmetic = split_records(arithmetic_records(), seed=42)
assert [len(arithmetic[s]) for s in ["train","validation","test"]] == [49,8,7]
families = {s: {r["family"] for r in rows} for s,rows in arithmetic.items()}
assert all(not families[a]&families[b] for a,b in [("train","validation"),("train","test"),("validation","test")])
accounting, evals = {}, []
for style in ["concise", "vivid"]:
    parts = {s: [_style_record(r, style, False) for r in rows] for s,rows in arithmetic.items()}
    split_info = {}
    for split, rows in parts.items():
        # In-memory serialization verifies recorded data hashes; it prepares no data files.
        raw = "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode()
        info = saved["results"]["runs"][style]["data"][split]
        assert digest(raw) == info["sha256"] and len(rows) == info["records"]
        split_info[split] = {"records": len(rows), "families": len(families[split]), "sha256": digest(raw)}
    examples = text_examples(parts["train"], mode="sft", max_length=128)
    sampler = random.Random(42)
    effective = sum(sum(int((y != IGNORE).sum()) for x,y in sampler.choices(examples,k=16)) for _ in range(450))
    assert effective == saved["results"]["runs"][style]["training"]["effective_tokens"]
    train_info = {k:v for k,v in saved["results"]["runs"][style]["training"].items() if k not in ["history"]}
    assert train_info["steps"] == 450 and train_info["trainable_parameters"] == 9504
    assert train_info["first_gradients"]["all_a_zero"] and train_info["first_gradients"]["some_b_nonzero"]
    assert saved["results"]["runs"][style]["base_parameters_unchanged"] is True
    accounting[style] = {"splits": split_info, "steps": 450, "batch_size": 16, "draws": 7200,
                         "effective_answer_targets_recomputed": effective, "saved_training": train_info,
                         "base_unchanged_assertion_in_saved_run": True}
    reports = [(style, saved["results"]["runs"][style]["evaluation"])]
    if style == "vivid":
        ft = saved["results"]["full_sft"]["training"]
        assert ft["steps"] == 450 and ft["effective_tokens"] == effective and ft["trainable_parameters"] == 141568
        assert ft["records_sha256"] == records_sha256(parts["train"])
        accounting["full_sft"] = {k:ft[k] for k in ["steps","effective_tokens","parameters","trainable_parameters","records","records_sha256"]}
        reports.append(("full_sft", saved["results"]["full_sft"]["evaluation"]))
    for run, evaluations in reports:
        for split, report in evaluations.items():
            checked = _style_metrics(copy.deepcopy(report), parts[split])
            assert checked["rubric"] == report["rubric"]
            assert len(report["samples"]) == len(parts[split]) == report["records"]
            target_count = sum(int((y!=IGNORE).sum()) for x,y in text_examples(parts[split],mode="sft"))
            assert target_count == report["effective_tokens"]
            records = []
            for row, sample in zip(parts[split],report["samples"],strict=True):
                assert sample["messages"] == row["messages"][:-1]
                assert sample["expected"] == row["messages"][-1]["content"]
                ids=sample["generated_ids"]
                raw_ids=ids[:ids.index(2)] if 2 in ids else ids
                assert ByteTokenizer().decode(raw_ids) == sample["generated"]
                assert (2 in ids) == sample["eos"]
                exact = raw_ids == ByteTokenizer().encode(sample["expected"])
                assert exact == sample["exact"]
                value = row["a"]+row["b"]
                text=sample["generated"]
                content=(text.split("，",1)[0].strip() if style=="vivid" else text.strip())==str(value)
                correct_style="像把兩組積木合在一起再數" in text if style=="vivid" else text.strip().isdigit()
                assert content == sample["content_correct"] and correct_style==sample["style_correct"]
                records.append({"question":sample["messages"][0]["content"],"expected":sample["expected"],
                    "generated":text,"content_correct":content,"style_correct":correct_style,"eos":2 in ids,
                    "hidden_special_ids_before_EOS":[i for i in raw_ids if i<8]})
            rubric = checked["rubric"][style]
            evals.append({"run":run,"split":split,"denominator":len(records),
                "content_correct":rubric["content_correct"],"style_correct":rubric["style_correct"],
                "all_EOS":all(s["eos"] for s in records),"effective_target_tokens":target_count,"samples":records})
result["saved_run_accounting"] = accounting
result["saved_run_recomputed_samples"] = evals

base = json.loads((OUT/"inputs/style-base-evidence.json").read_text())
base_test = base["results"]["content_evaluation"]["test"]
assert len(base_test["samples"]) == 7
correct = 0
for row,s in zip(arithmetic["test"],base_test["samples"],strict=True):
    assert row["messages"][:-1] == s["messages"] and row["messages"][-1]["content"] == s["expected"]
    raw=s["generated_ids"][:s["generated_ids"].index(2)]
    assert ByteTokenizer().decode(raw) == s["generated"]
    correct += int(s["generated"].strip()==str(row["a"]+row["b"]))
assert correct == 0 == base_test["matches"]
for split,rows in arithmetic.items():
    assert digest("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode()) == base["results"]["arithmetic_data"][split]["sha256"]
result["base_saved_test"] = {"revision":base["revision"],"correct":correct,"denominator":7,
                              "arithmetic_data_hashes_verified":True,"scope":"Saved original base JSON; no checkpoint inspected or inference rerun."}
result["status"] = "all_assertions_passed"
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
