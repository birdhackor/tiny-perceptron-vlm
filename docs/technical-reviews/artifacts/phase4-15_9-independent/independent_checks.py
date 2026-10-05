"""Bounded CPU checks for 15.9; no model loading or training reproduction."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import math
import platform
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import IGNORE, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum
from scripts.course_experiments.common import split_records, records_sha256, text_examples
from scripts.course_experiments.architecture import _forward, _routing

torch.set_num_threads(1)
torch.manual_seed(42)
assert torch.version.cuda is None and not torch.cuda.is_available()
OUT = Path(__file__).resolve().parent
result = {"environment": {"python": sys.version, "torch": str(torch.__version__),
          "torch_git_version": str(torch.version.git_version), "device": "cpu",
          "cuda_build": str(torch.version.cuda), "platform": platform.platform()},
          "scope": "Short synthetic CPU checks and arithmetic on existing raw measurements; no checkpoint loading, model evaluation, or experiment retraining."}

# Same original fence, coefficient-only requested exercise.
code = (OUT / "fence-1.py").read_text()
print("coefficient_one_exercise")
exec(compile(code.replace("0.01 * auxiliary", "1 * auxiliary"), "coefficient-one-fence", "exec"))

# The selected fraction is fixed within this differentiable computation.
scores = torch.tensor([[math.log(.8), math.log(.1), math.log(.1)]] * 3,
                      dtype=torch.float64, requires_grad=True)
prob = scores.softmax(-1)
f = torch.bincount(prob.argmax(-1), minlength=3).to(prob.dtype) / len(prob)
aux = 3 * (f.detach() * prob.mean(0)).sum()
aux.backward()
expected_grad = 3 / len(prob) * prob.detach() * (f - (prob.detach() * f).sum(-1, keepdim=True))
assert torch.allclose(scores.grad, expected_grad, atol=1e-14, rtol=0)
with torch.no_grad():
    after = scores - .1 * scores.grad
    after_prob = after.softmax(-1)
    after_f = torch.bincount(after_prob.argmax(-1), minlength=3).to(prob.dtype) / len(prob)
    after_aux = 3 * (after_f * after_prob.mean(0)).sum()
assert after_aux < aux
result["gradient"] = {"fraction": f.tolist(), "aux_before": float(aux.detach()),
 "logit_grad": scores.grad.tolist(), "formula": "dL/dz_ti = E/T * p_ti * (f_i - sum_j f_j*p_tj), treating f fixed",
 "step_size": .1, "aux_after_one_gradient_step": float(after_aux),
 "same_selected_indices": bool(torch.equal(prob.argmax(-1), after_prob.argmax(-1)))}
uniform = torch.full((3, 3), 1/3, dtype=torch.float64)
uf = torch.bincount(uniform.argmax(-1), minlength=3).double() / 3
ua = 3 * (uf * uniform.mean(0)).sum()
assert ua == 1 and uf.tolist() == [1., 0., 0.]
under = torch.tensor([[.51, .49]] * 3 + [[.01, .99]], dtype=torch.float64)
under_f = torch.bincount(under.argmax(-1), minlength=2).double() / 4
under_aux = 2 * (under_f * under.mean(0)).sum()
assert abs(float(under_aux) - .885) < 1e-14
result["counterexamples"] = {"uniform_probability_concentrated_argmax": {"fraction": uf.tolist(), "aux": float(ua)},
 "unique_argmax_aux_below_one": {"probabilities": under.tolist(), "fraction": under_f.tolist(), "aux": float(under_aux)}}

# Validate the actual two-layer valid-mask helper on five inputs / four targets.
model = TinyLM(ModelConfig(vocab_size=16, width=8, layers=2, heads=2, max_length=8, experts=4, top_k=2))
examples = [(torch.tensor([1,2]), torch.tensor([2,IGNORE])),
            (torch.tensor([1,3,4]), torch.tensor([3,4,5]))]
x, y, valid = pad_batch(examples)
captured = []
def hook(module, args, output):
    p = module.router(args[0][valid]).float().softmax(-1)
    choices = output[2][valid.reshape(-1)]
    counts = torch.bincount(choices.flatten(), minlength=4)
    fraction = counts.float() / (int(valid.sum()) * 2)
    aux = 4 * (fraction.detach() * p.mean(0)).sum()
    captured.append({"counts": counts.tolist(), "denominator": int(counts.sum()),
                     "fraction_sum": float(fraction.sum()), "aux": float(aux.detach())})
handles = [b.ffn.register_forward_hook(hook) for b in model.blocks]
try:
    r = _forward(model, x, valid)
finally:
    for h in handles: h.remove()
task_sum, task_count = loss_sum(r["logits"], y)
assert int(valid.sum()) == 5 and int(task_count) == 4
assert all(c["denominator"] == 10 and c["fraction_sum"] == 1 for c in captured)
assert abs(float(r["auxiliary"].detach()) - sum(c["aux"] for c in captured)) < 1e-6
result["valid_masks"] = {"input_count": int(valid.sum()), "task_target_count": int(task_count),
 "layers": captured, "aux_sum": float(r["auxiliary"].detach()),
 "train_objective_example": float((task_sum/task_count + .01*r["auxiliary"]).detach())}

# Existing raw result fields only: no notes, comparison strings, or verdicts.
raw_path = ROOT / "docs/course-experiments/results/moe.json"
raw = raw_path.read_bytes(); original = json.loads(raw)
pointers = ["/schema_version", "/experiment_id", "/revision", "/device", "/seed", "/torch_version", "/python_version", "/step_scale", "/code_sha256", "/results/dataset"]
rows = []
table = {"top1_aux0": (.74231,.64755,2.30115,2.32369),
         "top1_aux0.01": (.99499,.97937,2.29775,2.31963),
         "top2_aux0": (.99800,.75365,2.28011,2.30320),
         "top2_aux0.01": (.99944,.99254,2.27819,2.30170)}
for name, expected in table.items():
    v = original["results"]["variants"][name]
    cfg = v["model"]["config"]
    routing = v["validation_routing"]
    observed = []
    layers = []
    for i, layer in enumerate(routing["layers"]):
        counts = layer["dispatch_counts"]; denominator = sum(counts)
        assert denominator == routing["effective_input_tokens"] * cfg["top_k"]
        fractions = [c/denominator for c in counts]
        entropy = -sum(f*math.log(f) for f in fractions if f > 0)
        normalized = entropy/math.log(cfg["experts"])
        assert abs(normalized - layer["normalized_load_entropy"]) < 1e-12
        assert all(abs(a-b) < 1e-12 for a,b in zip(fractions, layer["load_fraction"]))
        observed.append(normalized)
        layers.append({k: layer[k] for k in ["dispatch_counts", "dispatch_denominator", "mean_router_probability", "mean_batch_auxiliary"]})
        pointers.append(f"/results/variants/{name}/validation_routing/layers/{i}")
    heldout = {}
    for split in ["validation", "test"]:
        h = v["heldout"][split]
        assert abs(h["nll_sum"]/h["effective_tokens"] - h["nll"]) < 1e-12
        heldout[split] = {k: h[k] for k in ["nll", "nll_sum", "effective_tokens", "examples", "records"]}
        observed.append(h["nll"])
        pointers.extend(f"/results/variants/{name}/heldout/{split}/{k}" for k in heldout[split])
    assert tuple(round(x,5) for x in observed) == expected
    t = v["training"]
    keys = ["requested_steps", "steps", "optimizer_updates", "skipped_updates", "auxiliary_weight", "effective_tokens", "all_requested_attempts_completed", "all_parameters_finite"]
    assert t["requested_steps"] == t["steps"] == t["optimizer_updates"] == 180 and t["skipped_updates"] == 0
    pointers.extend(f"/results/variants/{name}/training/{k}" for k in keys)
    pointers.extend([f"/results/variants/{name}/model/config", f"/results/variants/{name}/validation_routing/effective_input_tokens", f"/results/variants/{name}/validation_routing/padding_excluded"])
    rows.append({"variant": name, "config": cfg, "normalized_entropy_recomputed": observed[:2],
                 "heldout": heldout, "routing": layers, "training": {k:t[k] for k in keys}})
result["raw_result"] = {"sha256": hashlib.sha256(raw).hexdigest(), "revision": original["revision"],
                       "seed": original["seed"], "rows": rows, "inspected_pointers": pointers}

# Recreate the documented split/window counts from the already-local source records;
# do not run any model over the validation or test stories.
source = ROOT / "data/training/text-initial/tinystories-train-512.jsonl"
source_bytes = source.read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == "7aa55a657de6499be64a513aa76eb21a9cd52276bf836ba725e1d7f52ea511e0"
records = [json.loads(line) for line in source_bytes.splitlines()]
for record in records:
    record["family"] = record.get("text_sha256", records_sha256([{"text": record["text"]}]))
splits = split_records(records, original["seed"])
families = {k:{r["family"] for r in v} for k,v in splits.items()}
assert not families["train"] & families["validation"] and not families["train"] & families["test"] and not families["validation"] & families["test"]
split_summary = {}
for name, records in splits.items():
    digest = records_sha256(records)
    assert digest == original["results"]["dataset"][name]["sha256"]
    assert len(records) == original["results"]["dataset"][name]["records"]
    windows = text_examples(records, max_length=128)
    targets = sum(int((y != IGNORE).sum()) for _,y in windows)
    inputs = sum(len(x) for x,_ in windows)
    split_summary[name] = {"records":len(records), "sha256":digest, "windows":len(windows), "inputs":inputs, "targets":targets}
    if name != "train":
        assert targets == rows[0]["heldout"][name]["effective_tokens"]
        assert len(windows) == rows[0]["heldout"][name]["examples"]
result["split_recreation"] = {"source_sha256":hashlib.sha256(source_bytes).hexdigest(), "family_sets_disjoint":True, "splits":split_summary}

# Demonstrate the exact _routing weighted-batch aggregation on two tiny batches.
synthetic = [(torch.tensor([1,2,3][:1+(i%3)]), torch.tensor([2,3,4][:1+(i%3)])) for i in range(17)]
batch_terms = []
for start in range(0,len(synthetic),16):
    bx, _, bv = pad_batch(synthetic[start:start+16])
    bs=[]
    def collect(module,args,output):
        p=module.router(args[0][bv]).float().softmax(-1)
        c=output[2][bv.reshape(-1)]
        f=torch.bincount(c.flatten(),minlength=4).float()/c.numel()
        bs.append(float((4*(f*p.mean(0)).sum()).detach()))
    hh=[b.ffn.register_forward_hook(collect) for b in model.blocks]
    try: model(bx,valid=bv)
    finally:
        for h in hh:h.remove()
    batch_terms.append({"input_count":int(bv.sum()), "aux":bs})
summary=_routing(model,synthetic,SimpleNamespace(device="cpu"))
weighted=[sum(b["input_count"]*b["aux"][i] for b in batch_terms)/sum(b["input_count"] for b in batch_terms) for i in range(2)]
assert all(abs(a-summary["layers"][i]["mean_batch_auxiliary"])<1e-7 for i,a in enumerate(weighted))
result["batch_aggregation"]={"batch_terms":batch_terms,"recomputed_weighted_aux":weighted,"actual_routing_summary":summary}
(OUT / "independent-results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
