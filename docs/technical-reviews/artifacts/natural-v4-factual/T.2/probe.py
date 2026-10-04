"""Bounded factual review probe: no optimizer step, checkpoint, data download, or training."""
import contextlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import TinyLM
from tiny_perceptron.multimodal import expand_modalities, tone
from tiny_perceptron.training import choose_device

spec = importlib.util.spec_from_file_location("review_train", ROOT / "scripts/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
result = {"environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu"}, "runs": []}

for task in ("text", "sft", "vision"):
    stats = {"task": task, "forward_calls": 0, "backward_calls": 0,
             "optimizer_step_calls": 0, "checkpoint_calls": 0, "torch_save_calls": 0,
             "targets": [], "preclip_norms": []}
    saved_params = []
    original_forward = TinyLM.forward
    original_backward = torch.Tensor.backward
    original_init = torch.optim.AdamW.__init__
    original_loss = train.masked_loss
    original_clip = torch.nn.utils.clip_grad_norm_

    def forward(self, *args, **kwargs):
        stats["forward_calls"] += 1
        return original_forward(self, *args, **kwargs)

    def backward(self, *args, **kwargs):
        stats["backward_calls"] += 1
        return original_backward(self, *args, **kwargs)

    def optimizer_init(self, params, *args, **kwargs):
        params = list(params)
        saved_params.extend((p, p.detach().clone()) for p in params)
        original_init(self, params, *args, **kwargs)

    def forbidden(kind):
        def call(*args, **kwargs):
            stats[kind] += 1
            raise AssertionError("Dry run attempted " + kind)
        return call

    def loss(logits, labels):
        value = original_loss(logits, labels)
        stats["targets"].append({"logits_shape": list(logits.shape),
            "effective_by_row": (labels != -100).sum(-1).tolist(),
            "effective_total": int((labels != -100).sum()),
            "active_label_ids": labels[labels != -100].tolist(), "loss": float(value.detach())})
        return value

    def clip(params, *args, **kwargs):
        params = list(params)
        grads = [p.grad for p in params if p.grad is not None]
        independent = torch.cat([g.detach().reshape(-1).double() for g in grads]).norm().item()
        returned = original_clip(params, *args, **kwargs)
        stats["preclip_norms"].append({"independent_float64_l2": independent,
            "returned": float(returned), "postclip_float64_l2": torch.cat([g.detach().reshape(-1).double() for g in grads]).norm().item(),
            "parameters_with_grad": len(grads), "parameters_without_grad": sum(p.grad is None for p in params)})
        assert math.isclose(independent, float(returned), rel_tol=1e-6)
        return returned

    buf = io.StringIO()
    with patch.object(sys, "argv", ["scripts/train.py", "--task", task, "--device", "cpu"]), \
         patch.object(TinyLM, "forward", forward), patch.object(torch.Tensor, "backward", backward), \
         patch.object(torch.optim.AdamW, "__init__", optimizer_init), \
         patch.object(torch.optim.AdamW, "step", forbidden("optimizer_step_calls")), \
         patch.object(train, "save_checkpoint", forbidden("checkpoint_calls")), \
         patch.object(torch, "save", forbidden("torch_save_calls")), \
         patch.object(train, "masked_loss", loss), patch.object(torch.nn.utils, "clip_grad_norm_", clip), \
         contextlib.redirect_stdout(buf):
        train.main()
    report = json.loads(buf.getvalue().splitlines()[-1])
    stats["report"] = report
    stats["all_parameters_bitwise_unchanged"] = all(torch.equal(p, before) for p, before in saved_params)
    assert stats["all_parameters_bitwise_unchanged"]
    assert stats["forward_calls"] == (4 if task == "vision" else 1)
    assert stats["backward_calls"] == 1
    assert all(stats[k] == 0 for k in ("optimizer_step_calls", "checkpoint_calls", "torch_save_calls"))
    assert len(report["history"]) == 1 and report["mode"] == "dry-run-no-weight-update"
    e = report["history"][0]
    assert math.isfinite(e["loss"]) and math.isfinite(e["grad_norm"]) and e["grad_norm"] > 0
    assert e["effective_tokens"] == {"text": 111, "sft": 8, "vision": None}[task]
    if task == "vision":
        stats["actual_effective_total"] = sum(t["effective_total"] for t in stats["targets"])
        assert math.isclose(e["loss"], sum(t["loss"] for t in stats["targets"])/4, rel_tol=1e-6)
    result["runs"].append(stats)

tok = ByteTokenizer()
x, y = render_chat([{"role": "user", "content": "只回答yes"}, {"role": "assistant", "content": "yes"}])
active = y[y != -100].tolist()
assert active == tok.encode("yes") + [tok.eos_id]
result["yes_mask"] = {"input_ids": x.tolist(), "target_ids": y.tolist(), "active_target_ids": active,
    "active_text": tok.decode(active), "active_count": len(active), "note": "First y answer byte is aligned with preceding assistant role input; not with current input role."}

embedding = torch.nn.Embedding(264, 8)
ids = torch.tensor([1, 5, 4, 73, 2]); labels = torch.tensor([-100, -100, -100, 73, 2])
xx, yy = expand_modalities(ids, labels, embedding, {5: torch.randn(3, 8)}, {5})
assert tuple(xx.shape) == (1, 6, 8) and yy.tolist() == [[-100, -100, -100, -100, 73, 2]]
result["image_example"] = {"shape": list(xx.shape), "targets": yy.tolist(), "effective_count": int((yy != -100).sum())}

wave = tone(440.0, seconds=0.2)
spectrum = torch.stft(wave, 400, 160, window=torch.hann_window(400), return_complex=True)
power = spectrum.abs().square(); peak = int(power.mean(-1).argmax())
assert tuple(power.shape) == (201, 21) and peak == 11
result["audio_example"] = {"samples": len(wave), "power_shape": list(power.shape), "peak_bin": peak, "peak_hz": peak*16000/400}

result["device_selection"] = []
for cuda, mps, expected in [(True,True,"cuda"),(False,True,"mps"),(False,False,"cpu")]:
    with patch.object(torch.cuda,"is_available",return_value=cuda), patch.object(torch.backends.mps,"is_available",return_value=mps):
        actual = str(choose_device()); assert actual == expected
        result["device_selection"].append({"cuda_available":cuda,"mps_available":mps,"selected":actual,"simulated_capability_only":True})
print(json.dumps(result,indent=2,ensure_ascii=False))
