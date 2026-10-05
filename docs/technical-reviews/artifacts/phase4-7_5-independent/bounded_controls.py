"""Independent, bounded CPU checks of 7.5; no training or optimizer update."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import warnings

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss, loss_sum
from tiny_perceptron.attention import attention_mask

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
tok = ByteTokenizer()
results = {"environment": {"python": sys.version, "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version), "device": "cpu", "threads": str(torch.get_num_threads()),
    "platform": platform.platform()}, "checks": {}}

def setup(question="Q"):
    torch.manual_seed(42)
    x, y = render_chat([{"role": "user", "content": question}, {"role": "assistant", "content": "A"}])
    return TinyLM(ModelConfig(width=8)), x, y

def run(question="Q", block_question=False, detached=False, retain=True):
    model, x, y = setup(question)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    valid = torch.ones(1, len(x), dtype=torch.bool)
    if block_question:
        valid[0, 2] = False
    if detached:
        logits = model(embeddings=model.embedding(x[None]).detach(), valid=valid)["logits"]
    else:
        logits = model(x[None], valid=valid)["logits"]
    if retain:
        logits.retain_grad()
    loss = masked_loss(logits, y[None])
    loss.backward()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grad = logits.grad
    embedding_grad = model.embedding.weight.grad
    record = {"question": question, "x": x.tolist(), "y": y.tolist(), "logits_shape": list(logits.shape),
        "embedding_shape": list(model.embedding.weight.shape), "question_id": int(x[2]),
        "valid": valid.tolist(), "label_count": int((y != -100).sum()), "loss": loss.item(),
        "logits_is_leaf": logits.is_leaf, "embedding_is_leaf": model.embedding.weight.is_leaf,
        "logits_grad_norms_per_position": None if grad is None else grad.norm(dim=-1).tolist(),
        "embedding_grad_is_none": embedding_grad is None,
        "question_embedding_gradient_norm": None if embedding_grad is None else embedding_grad[x[2]].norm().item(),
        "unused_Q_row_gradient_norm": None if embedding_grad is None else embedding_grad[tok.encode("Q")[0]].norm().item(),
        "output_gradient_norm": model.output.weight.grad.norm().item(),
        "parameters_unchanged": all(torch.equal(before[name], p.detach()) for name, p in model.named_parameters()),
        "expected_nonleaf_warning": [str(w.message) for w in caught]}
    assert record["parameters_unchanged"]
    return record, logits.detach(), grad, embedding_grad, x, y

q, scores, gradients, embedding_grad, x, y = run()
assert x.tolist() == [1, 3, 89, 2, 4, 73] and y.tolist() == [-100, -100, -100, -100, 73, 2]
assert q["logits_shape"] == [1, 6, 264] and q["embedding_shape"] == [264, 8]
assert not q["logits_is_leaf"] and q["embedding_is_leaf"]
assert torch.count_nonzero(gradients[0, :4]) == 0 and q["question_embedding_gradient_norm"] > 0
manual_norm = embedding_grad[x[2]].double().square().sum().sqrt().item()
assert abs(manual_norm - q["question_embedding_gradient_norm"]) < 1e-8
q["norm_manual_sqrt_sum_squares_float64"] = manual_norm
q["norm_absolute_error"] = abs(manual_norm-q["question_embedding_gradient_norm"])
results["checks"]["baseline_Q"] = q

z, *_ = run("Z")
assert z["logits_grad_norms_per_position"][0][0] == 0 and z["question_id"] == 98
assert z["question_embedding_gradient_norm"] > 0 and z["unused_Q_row_gradient_norm"] == 0
results["checks"]["replace_Q_with_Z"] = z

blocked, blocked_scores, *_ = run(block_question=True)
assert blocked["question_embedding_gradient_norm"] == 0 and blocked["output_gradient_norm"] > 0
blocked["supervised_logits_max_change"] = (blocked_scores[:, 4:] - scores[:, 4:]).abs().max().item()
results["checks"]["block_Q_key_in_the_single_attention_layer"] = blocked

detached, detached_scores, *_ = run(detached=True)
assert torch.equal(detached_scores, scores) and detached["embedding_grad_is_none"]
assert detached["output_gradient_norm"] > 0
detached["forward_logits_exactly_unchanged"] = True
results["checks"]["detach_embedding_output"] = detached

unretained, *_ = run(retain=False)
assert unretained["logits_grad_norms_per_position"] is None and unretained["question_embedding_gradient_norm"] > 0
assert len(unretained["expected_nonleaf_warning"]) == 1
results["checks"]["without_retain_grad"] = unretained

model, _, _ = setup()
with torch.no_grad():
    base = model(x[None])["logits"]
    question_changed = x.clone(); question_changed[2] = tok.encode("Z")[0]
    future_changed = x.clone(); future_changed[5] = tok.encode("B")[0]
    q_changed = model(question_changed[None])["logits"]
    future = model(future_changed[None])["logits"]
question_change = (q_changed[0, 4]-base[0, 4]).abs().max().item()
future_change = (future[0, 4]-base[0, 4]).abs().max().item()
assert question_change > 0 and future_change == 0
allowed = attention_mask(torch.arange(6), torch.arange(6))
assert allowed.shape == (1, 1, 6, 6) and allowed[0, 0, 4, 2] and not allowed[0, 0, 4, 5]
results["checks"]["causal_axes"] = {"axis_order": ["batch", "head", "query_position", "key_position"],
    "shape": list(allowed.shape), "allowed": allowed[0,0].tolist(),
    "assistant_query4_can_read_question_key2": bool(allowed[0,0,4,2]),
    "assistant_query4_cannot_read_future_answer_key5": bool(not allowed[0,0,4,5]),
    "change_question_max_delta_at_first_answer_logits": question_change,
    "change_future_answer_max_delta_at_first_answer_logits": future_change}

# Independently derive unweighted class-index CE in float64 at the two effective targets.
raw = scores.double().clone().requires_grad_()
labels = y[None]
selected = labels != -100
manual_sum = (torch.logsumexp(raw[selected], dim=-1) - raw[selected].gather(-1, labels[selected,None]).squeeze(-1)).sum()
manual_mean = manual_sum / 2
observed = masked_loss(raw, labels)
observed.backward()
expected_grad = torch.zeros_like(raw)
expected_grad[selected] = raw.detach()[selected].softmax(-1)
expected_grad[selected] -= torch.nn.functional.one_hot(labels[selected], num_classes=264).double()
expected_grad /= 2
gradient_error = (expected_grad-raw.grad).abs().max().item()
assert torch.allclose(manual_mean, observed, atol=1e-14, rtol=0) and gradient_error < 1e-14
altered = raw.detach().clone(); altered[~selected] = torch.arange(264).double() * 1000
altered_loss = masked_loss(altered, labels)
assert torch.equal(altered_loss, observed.detach())
total, count = loss_sum(raw.detach(), labels)
try:
    masked_loss(raw.detach(), torch.full_like(labels, -100))
except ValueError as error:
    empty_error = str(error)
else:
    raise AssertionError("zero effective denominator must be rejected")
results["checks"]["CE_axes_denominator_and_derivative"] = {
    "original_logits_axes": ["batch=1", "position=6", "candidate_class=264"],
    "flattened_logits_shape": [6, 264], "labels_shape": [1, 6],
    "effective_positions": [4, 5], "effective_target_ids": [73, 2], "effective_denominator": int(count),
    "sum_loss": total.item(), "manual_mean": manual_mean.item(), "observed_mean": observed.item(),
    "mean_absolute_error": abs(manual_mean.item()-observed.item()),
    "analytic_gradient_max_error": gradient_error, "changed_ignored_logits_loss": altered_loss.item(),
    "all_ignored_exception": empty_error,
    "assumptions": "unweighted integer class labels, no label smoothing, finite logits, denominator > 0"}

results["checks"]["limits"] = {"optimizer_steps": 0, "training_runs": 0, "model_downloads": 0,
    "data_downloads": 0, "scope": "one random initialized single-layer untied TinyLM and finite CPU controls; not task accuracy or universal nonzero gradient"}
print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
