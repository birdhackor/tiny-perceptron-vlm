"""Bounded fresh CPU checks for lesson 11.3; no updates/checkpoints/downloads."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, expand_modalities, scene

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
tok = ByteTokenizer()
answer = tok.encode("red square") + [tok.eos_id]
prefix = [tok.bos_id, tok.user_id, tok.image_id, tok.eos_id, tok.assistant_id]
ids = torch.tensor(prefix + answer)
labels = torch.tensor([-100] * len(prefix) + answer)

def build():
    torch.manual_seed(0)
    m = MultiModalLM(TinyLM(ModelConfig(width=8)))
    m.requires_grad_(False)
    m.image_projector.requires_grad_(True)
    return m

def get_loss(m):
    out = m(ids, labels, image=scene())
    return out, masked_loss(out["logits"], out["labels"])

m = build()
before = {n: p.detach().clone() for n, p in m.named_parameters()}
core_inputs = []
def capture(module, args, kwargs):
    x = kwargs["embeddings"]
    x.retain_grad()
    core_inputs.append(x)
hook = m.language.register_forward_pre_hook(capture, with_kwargs=True)
out, loss = get_loss(m)
valid = out["labels"] != -100
assert len(answer) == 11 and len(ids) == 16
assert tuple(out["logits"].shape) == (1, 30, 264)
assert tuple(out["labels"].shape) == (1, 30)
assert out["labels"][valid].tolist() == answer
assert valid.nonzero()[:, 1].tolist() == list(range(19, 30))
scores = out["logits"][valid]
targets = out["labels"][valid]
independent = (scores.logsumexp(-1) - scores.gather(1, targets[:, None]).squeeze(1)).mean()
assert torch.allclose(loss, independent, atol=1e-6, rtol=1e-6)
masked = F.cross_entropy(out["logits"].reshape(-1,264), out["labels"].reshape(-1), ignore_index=-100, reduction="none").reshape(1,30)
assert torch.equal(masked[~valid], torch.zeros_like(masked[~valid]))
loss.backward()
hook.remove()
assert m.image_projector.weight.grad.norm().item() > 0
assert all(p.grad is None for p in m.language.parameters())
assert all(p.grad is None for p in m.vision.parameters())
assert all(torch.equal(before[n], p) for n,p in m.named_parameters())
record = {"original_shape_check": {"answer_ascii_bytes": len("red square".encode()), "effective_targets": int(valid.sum()), "input_ids": len(ids), "vision_features": [16,16], "projected_features": [16,8], "logits_shape": list(out["logits"].shape), "labels_shape": list(out["labels"].shape), "supervised_logit_positions_zero_based": valid.nonzero()[:,1].tolist(), "mean_cross_entropy_nats_per_target": loss.item(), "independent_mean_nll": independent.item(), "absolute_difference": abs(loss.item()-independent.item()), "ignored_positions": int((~valid).sum()), "projector_weight_gradient_frobenius_norm": m.image_projector.weight.grad.norm().item(), "fixed_core_input_gradient_norm": core_inputs[0].grad.norm().item(), "all_language_and_vision_grads_none": True, "all_parameter_values_unchanged_after_backward": True, "trainable_parameter_count": sum(p.numel() for p in m.parameters() if p.requires_grad)}}

# Freeze the projector before rebuilding the forward graph.
early = build()
early.image_projector.requires_grad_(False)
_, early_loss = get_loss(early)
early_count = sum(p.numel() for p in early.parameters() if p.requires_grad)
assert early_count == 0 and not early_loss.requires_grad
try:
    early_loss.backward()
except RuntimeError as e:
    early_error = str(e)
else:
    raise AssertionError("early-freeze backward should fail")
record["freeze_before_forward"] = {"trainable_parameters": early_count,"loss_requires_grad": early_loss.requires_grad,"exception": early_error}

# Freeze only after the forward: the graph already exists, so backward need not fail.
late = build()
_, late_loss = get_loss(late)
late.image_projector.requires_grad_(False)
late_count = sum(p.numel() for p in late.parameters() if p.requires_grad)
assert late_count == 0 and late_loss.requires_grad
late_loss.backward()
assert all(p.grad is None for p in late.parameters())
record["freeze_after_forward"] = {"trainable_parameters": late_count,"loss_requires_grad":late_loss.requires_grad,"backward_raised":False,"parameter_grads_all_none":True}

# No-grad around fixed core severs the route, whereas eval keeps autograd enabled.
cut = build()
feats = cut.image_projector(cut.vision(scene().unsqueeze(0)))[0]
x,y = expand_modalities(ids,labels,cut.language.embedding,{tok.image_id:feats},{tok.image_id,tok.audio_id})
with torch.no_grad():
    cut_out = cut.language(embeddings=x)
cut_loss = masked_loss(cut_out["logits"],y)
assert not cut_loss.requires_grad
evaluated = build().eval()
_, eval_loss = get_loss(evaluated)
eval_loss.backward()
assert evaluated.image_projector.weight.grad.norm().item() > 0
record["grad_mode_variants"] = {"no_grad_core_loss_requires_grad": cut_loss.requires_grad,"eval_loss_requires_grad":eval_loss.requires_grad,"eval_projector_grad_norm": evaluated.image_projector.weight.grad.norm().item()}

# Swap one ASCII token and EOS-only answer to verify target masking is structural.
for name,tail in [("blue_square",tok.encode("blue square")+[tok.eos_id]),("EOS_only",[tok.eos_id])]:
    mm = build()
    oo = mm(torch.tensor(prefix+tail),torch.tensor([-100]*len(prefix)+tail),image=scene("blue"))
    yy=oo["labels"]; assert yy[yy != -100].tolist()==tail
    ll=masked_loss(oo["logits"],yy);ll.backward()
    record[name] = {"effective_targets":int((yy!=-100).sum()),"logits_shape":list(oo["logits"].shape),"projector_grad_norm":mm.image_projector.weight.grad.norm().item()}

# If a deterministic image entry maps two images to identical features, the same
# projector cannot recreate a distinction for the same prompt.
with torch.no_grad():
    same = torch.zeros(16,16)
    projected_a = m.image_projector(same)
    projected_b = m.image_projector(same.clone())
assert torch.equal(projected_a,projected_b)
record["identical_entry_features"] = {"same_projected_features":True,"scope":"identical deterministic entry representation and same prompt; not a universal claim that projectors cannot infer correlations"}
record["environment"] = {"python":sys.version,"torch":str(torch.__version__),"torch_git":torch.version.git_version,"device":"cpu","cuda_build":str(torch.version.cuda),"threads":str(torch.get_num_threads())}
record["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
print(json.dumps(record,ensure_ascii=False,indent=2))
