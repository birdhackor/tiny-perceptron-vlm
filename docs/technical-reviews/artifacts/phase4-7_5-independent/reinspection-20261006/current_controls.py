"""Actual reinspection of the modified current fence; bounded CPU only."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.model import masked_loss, TinyLM, ModelConfig
from tiny_perceptron.attention import attention_mask

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
fence = (HERE / "current/fence-1.py").read_bytes()
assert hashlib.sha256(fence).hexdigest() == "ce8105e8b2babd7ca071a37a03d8451e2d9fe5316fca7b0830fbeee0583319db"
records = {"environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "threads": "1"}, "current_fence_sha256": hashlib.sha256(fence).hexdigest(), "checks": {}}

def execute_actual_fence(raw, question):
    ns = {"__name__": "__main__"}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(raw, f"current7.5-fence-question-{question}", "exec"), ns)
    x, y, logits, model = (ns[k] for k in ("x", "y", "logits", "model"))
    expected_id = ord(question)+8
    assert int(x[2]) == expected_id and int(y[2]) == -100
    assert logits.shape == (1, 6, 264) and logits.grad[0,2].norm().item() == 0
    assert model.embedding.weight.grad[x[2]].norm().item() > 0
    # Same seeded init: the actual source only computes gradients, so values must be untouched.
    torch.manual_seed(42)
    initialized = TinyLM(ModelConfig(width=8))
    assert all(torch.equal(p.detach(), dict(initialized.named_parameters())[name].detach()) for name,p in model.named_parameters())
    records["checks"][question] = {"executed_code_sha256": hashlib.sha256(raw).hexdigest(), "stdout": capture.getvalue(), "x": x.tolist(), "y": y.tolist(), "question_index": 2, "question_id": int(x[2]), "effective_count": int((y != -100).sum()), "logits_shape": list(logits.shape), "question_logits_gradient_norm": logits.grad[0,2].norm().item(), "question_embedding_gradient_norm": model.embedding.weight.grad[x[2]].norm().item(), "no_parameter_update": True}
    return ns

ns = execute_actual_fence(fence, "Q")
assert fence.count(b'"content": "Q"') == 1
z_code = fence.replace(b'"content": "Q"', b'"content": "Z"')
(HERE / "variant-Z.py").write_bytes(z_code)
execute_actual_fence(z_code, "Z")

scores = ns["logits"].detach().double().requires_grad_()
labels = ns["y"][None]
selected = labels != -100
count = int(selected.sum())
assert count == 2
manual = (scores[selected].logsumexp(-1)-scores[selected].gather(1, labels[selected].unsqueeze(1)).squeeze(1)).sum()/count
observed = masked_loss(scores, labels)
observed.backward()
analytic = torch.zeros_like(scores)
analytic[selected] = (scores.detach()[selected].softmax(-1)-torch.nn.functional.one_hot(labels[selected],264).double())/count
max_error = (scores.grad-analytic).abs().max().item()
assert abs(manual.item()-observed.item()) < 1e-14 and max_error < 1e-14
assert torch.count_nonzero(scores.grad[0,:4]) == 0
new_labels = labels.clone(); new_labels[0,2] = 73
new_scores = scores.detach().clone().requires_grad_()
new_loss = masked_loss(new_scores,new_labels); new_loss.backward()
assert int((new_labels!=-100).sum()) == 3 and new_scores.grad[0,2].norm().item() > 0
try:
    masked_loss(scores.detach(), torch.full_like(labels,-100))
except ValueError as error:
    empty_error = str(error)
else:
    raise AssertionError("empty effective denominator was not rejected")
records["checks"]["math_and_target_axis"] = {"axes": ["batch=1", "position=6", "candidate_class=264"], "effective_positions": [4,5], "effective_denominator": count, "manual_loss": manual.item(), "observed_loss": observed.item(), "absolute_loss_error": abs(manual.item()-observed.item()), "analytic_gradient_max_error": max_error, "question_logit_gradient_norm": scores.grad[0,2].norm().item(), "add_target_at_question_position": {"effective_denominator":3, "question_logit_gradient_norm":new_scores.grad[0,2].norm().item()}, "all_ignored_exception":empty_error, "conditions":"finite unweighted integer-class logits, no label smoothing, nonzero effective denominator"}

model, x, y = ns["model"], ns["x"], ns["y"]
before = {name:p.detach().clone() for name,p in model.named_parameters()}
model.zero_grad(set_to_none=True)
valid = torch.ones(1,6,dtype=torch.bool); valid[0,2]=False
blocked = model(x[None],valid=valid)["logits"]
masked_loss(blocked,y[None]).backward()
assert model.embedding.weight.grad[89].norm().item() == 0
records["checks"]["block_Q_key"] = {"valid":valid.tolist(),"question_embedding_gradient_norm":0.0,"scope":"the unchanged default single-layer untied model; Q blocked as key for every query"}
model.zero_grad(set_to_none=True)
detached = model(embeddings=model.embedding(x[None]).detach())["logits"]
assert torch.equal(detached.detach().double(),scores.detach())
masked_loss(detached,y[None]).backward()
assert model.embedding.weight.grad is None
records["checks"]["detach"] = {"forward_logits_equal":True,"embedding_weight_gradient":None,"output_weight_gradient_nonzero":model.output.weight.grad.norm().item()>0}
allowed = attention_mask(torch.arange(6),torch.arange(6))
assert allowed.shape == (1,1,6,6) and allowed[0,0,4,2] and not allowed[0,0,4,5]
with torch.no_grad():
    original = model(x[None])["logits"]
    z_ids=x.clone(); z_ids[2]=98
    future_ids=x.clone(); future_ids[5]=74
    z_logits=model(z_ids[None])["logits"]
    future_logits=model(future_ids[None])["logits"]
assert torch.equal(original[0,4],future_logits[0,4])
assert not torch.equal(original[0,4],z_logits[0,4])
assert all(torch.equal(before[name],p.detach()) for name,p in model.named_parameters())
records["checks"]["causal_axes"]={"mask_axes":["batch","head","query","key"],"mask_shape":list(allowed.shape),"query4_can_read_Q_key2":True,"query4_cannot_read_future_key5":True,"change_future_answer_max_delta_at_query4":(original[0,4]-future_logits[0,4]).abs().max().item(),"change_Q_to_Z_max_delta_at_query4":(original[0,4]-z_logits[0,4]).abs().max().item()}
records["limits"]={"optimizer_steps":0,"model_or_data_downloads":0,"training":False,"existing_weight_evaluation":False,"rerun_entire_chapter":False,"parameters_unchanged":True}
print(json.dumps(records,ensure_ascii=False,indent=2,allow_nan=False))
