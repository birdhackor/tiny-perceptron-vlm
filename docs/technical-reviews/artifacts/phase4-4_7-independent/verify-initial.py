"""Bounded independent CPU audit for section 4.7. No training or external datasets."""
from pathlib import Path
import contextlib, hashlib, io, json, os, platform, random, subprocess, sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import torch
from tiny_perceptron import attention
from tiny_perceptron.data import IGNORE, shifted
from tiny_perceptron.model import ModelConfig, TinyLM, generate, loss_sum, masked_loss
from scripts.prepare_data import generate_records
from scripts.course_experiments.common import records_sha256, split_records, text_examples

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
environment = {"python": sys.version, "torch": torch.__version__, "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads()), "platform": platform.platform(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "cwd": str(Path.cwd()), "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-4_7-independent/verify.py"}
out = {"scope": "Original fence and small alignment/causal variants; parse existing run evidence and reconstruct counts without optimizer updates or checkpoint loading."}

# Run the exact saved source, not a rewritten demonstration.
namespace = {"__name__": "__main__"}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile((BASE / "original/fence-1.py").read_bytes(), "original/fence-1.py", "exec"), namespace)
print(captured.getvalue(), end="")
model, x, y, logits, loss = [namespace[k] for k in ("model", "x", "y", "logits", "loss")]
torch.manual_seed(42)
initial = TinyLM(ModelConfig(vocab_size=5, width=8))
assert all(torch.equal(p, initial.state_dict()[name]) for name, p in model.state_dict().items())
grads = {name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None}
assert grads and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert any(value > 0 for value in grads.values())
assert x.tolist() == [1,2,3] and y.tolist() == [2,3,4] and list(logits.shape) == [1,3,5]
per_token = logits.detach().logsumexp(-1) - logits.detach().gather(-1, y[None,:,None]).squeeze(-1)
manual = per_token.mean()
assert torch.allclose(loss.detach(), manual, rtol=0, atol=1e-7)
total, count = loss_sum(logits.detach(), y[None])
assert count.item() == 3 and torch.allclose(total / 3, manual, rtol=0, atol=1e-7)
ignored = y[None].clone(); ignored[0,1] = IGNORE
ignored_loss = masked_loss(logits.detach(), ignored)
assert torch.allclose(ignored_loss, per_token[:,[0,2]].mean(), rtol=0, atol=1e-7)
out["original_fence"] = {"stdout": captured.getvalue(), "pairs": list(zip(x.tolist(),y.tolist())), "input_shape": list(x[None].shape), "score_shape": list(logits.shape), "ce_each_nats": per_token.tolist(), "ce_mean_nats": manual.item(), "count": count.item(), "gradient_parameter_count": len(grads), "gradient_norms": grads, "parameter_max_change_after_backward": 0.0, "ignore_middle_count": 2, "ignore_middle_mean_nats": ignored_loss.item(), "tolerance": "absolute 1e-7 for CE; bitwise parameter equality"}

# All-position pass vs separately computed known prefixes.
model.eval()
with torch.no_grad():
    full = model(x[None])["logits"]
    diffs = [(full[:,i] - model(x[None,:i+1])["logits"][:,-1]).abs().max().item() for i in range(3)]
    mask = attention.attention_mask(torch.arange(3), torch.arange(3))
    assert mask.int().tolist() == [[[[1,0,0],[1,1,0],[1,1,1]]]]
    future_changes = []
    for index in [1,2]:
        changed = x.clone(); changed[index] = 0
        d = (full[:,:index] - model(changed[None])["logits"][:,:index]).abs().max().item()
        assert d == 0.0
        future_changes.append({"changed_input_position": index, "earlier_positions": list(range(index)), "max_abs_difference": d})
    original_mask = attention.attention_mask
    try:
        attention.attention_mask = lambda q,k,valid=None,segments=None: torch.ones(1,1,len(q),len(k),dtype=torch.bool)
        changed = x.clone(); changed[-1] = 0
        unmasked_difference = (model(x[None])["logits"][:,:-1] - model(changed[None])["logits"][:,:-1]).abs().max().item()
        assert unmasked_difference > 1e-3
    finally:
        attention.attention_mask = original_mask
assert max(diffs) < 1e-6
out["causal_and_parallel"] = {"mask_True_means_allowed": mask.int().tolist(), "parallel_vs_prefix_each_max_abs_difference": diffs, "prefix_tolerance": 1e-6, "future_changes": future_changes, "unmasked_positive_control_max_abs_difference": unmasked_difference, "scope": "These specific inputs on fresh seed-42 CPU model; generic causal property is additionally established by original implementation and paper."}

# Shift errors are target-alignment failures even when loss is finite.
double = masked_loss(logits.detach()[:,:-1], y[None,1:])
copy = masked_loss(logits.detach(), x[None].clone())
assert torch.isfinite(double) and torch.isfinite(copy)
out["misalignment"] = {"double_shift_pairs": list(zip(x[:-1].tolist(), y[1:].tolist())), "double_shift_count": 2, "double_shift_ce_nats": double.item(), "copy_pairs": list(zip(x.tolist(), x.tolist())), "copy_ce_nats": copy.item(), "scope": "The conventional extra shift compares logits[:-1] with supplied labels[1:]; unshifted labels set the current visible ID as target. Loss positivity does not identify the intended task."}
ex, ey = shifted([1,2,3,4,0])
with torch.no_grad():
    scores = model(ex[None])["logits"]
assert list(scores.shape) == [1,4,5]
assert list(zip(ex.tolist(), ey.tolist())) == [(1,2),(2,3),(3,4),(4,0)]
out["exercise"] = {"pairs": list(zip(ex.tolist(), ey.tolist())), "input_shape": list(ex[None].shape), "score_shape": list(scores.shape), "valid_ids": [0,1,2,3,4]}
calls = []
hook = model.register_forward_pre_hook(lambda module,args: calls.append(list(args[0].shape)))
generated = generate(model, torch.tensor([[1]]), max_new_tokens=3, eos_id=-1, use_cache=False)
hook.remove()
assert calls == [[1,1],[1,2],[1,3]] and list(generated.shape) == [1,4]
out["generation"] = {"forward_input_shapes": calls, "generated_shape": list(generated.shape), "generated_ids": generated.tolist(), "scope": "No-cache greedy implementation appends one new token per forward; no claim about model learning."}

# Inspect the actual archived result. Reconstruct tiny records/counts only.
report_path = ROOT / "docs/course-experiments/results/text_foundation.json"
raw_report = report_path.read_bytes()
report = json.loads(raw_report)
r = report["results"]
records = generate_records("toy-text")
parts = split_records(records, seed=report["seed"])
counts = {s:len(rows) for s,rows in parts.items()}
assert len(records) == 3*2*2 == 12 and counts == {"train":9,"validation":1,"test":2}
split_hashes = {}
for split, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows).encode()
    digest = hashlib.sha256(raw).hexdigest()
    split_hashes[split] = digest
    assert digest == r["data"][split]["sha256"] and len(rows) == r["data"][split]["records"]
assert records_sha256(parts["train"]) == r["training"]["records_sha256"]
examples = text_examples(parts["train"], mode="text", max_length=128)
sampler = random.Random(report["seed"])
token_count = sum(int((labels != IGNORE).sum()) for _ in range(r["training"]["steps"]) for _,labels in sampler.choices(examples,k=16))
assert token_count == r["training"]["effective_tokens"] == 342462
source_checks = {}
for path in ["tiny_perceptron/attention.py","tiny_perceptron/data.py","tiny_perceptron/model.py","tiny_perceptron/modern.py","scripts/course_experiments/text.py","scripts/course_experiments/common.py"]:
    current = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    recorded = report["code_sha256"][path]
    source_checks[path] = {"current_sha256": current, "recorded_sha256": recorded, "equal": current == recorded}
    assert current == recorded
assert report["status"] == "completed" and report["evidence_status"] == "complete_run"
assert r["training"]["steps"] == 600 and r["training"]["history"][-1]["step"] == 600
assert r["causal_max_difference"] == 0.0
out["archived_measurement"] = {"input_path": str(report_path.relative_to(ROOT)), "input_sha256": hashlib.sha256(raw_report).hexdigest(), "revision": report["revision"], "run_id": report["modal"]["run_id"], "recorded_environment": {k:report[k] for k in ["device","torch_version","python_version","gpu","seed"]}, "records": len(records), "split_counts": counts, "reconstructed_split_sha256": split_hashes, "records_sha256": records_sha256(parts["train"]), "steps": r["training"]["steps"], "history_last_step": r["training"]["history"][-1]["step"], "batch_size_from_source": 16, "effective_tokens_reconstructed_without_training": token_count, "causal_max_difference": r["causal_max_difference"], "probe_shape_from_source": [1,5], "compared_score_shape_from_source": [1,4,264], "compared_scalar_count": 4*264, "source_hash_checks": source_checks, "scope": "Read existing original JSON and implementation, reconstruct record split and sampler token denominator only; no checkpoint download/loading and no retraining. Recorded 0.0 is one specified probe, not exhaustive causal verification."}
(BASE / "verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
(BASE / "environment.json").write_text(json.dumps(environment,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(out,ensure_ascii=False,indent=2))
