"""Bounded CPU forward/gradient checks; no optimizer, training, downloads or weights saved."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
ART = Path(__file__).resolve().parents[1]

def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().contiguous().numpy().tobytes())
    return h.hexdigest()

checks = []
for seed in (0, 1):
    torch.manual_seed(seed)
    lm = TinyLM(ModelConfig(width=8)).eval()
    before = state_hash(lm)
    wrapped = MultiModalLM(lm).eval()
    assert wrapped.language is lm
    assert state_hash(lm) == before
    assert all(not module.training for module in wrapped.modules())
    assert all(a is b for a, b in zip(lm.parameters(), wrapped.language.parameters(), strict=True))
    active_stochastic_modules = [type(m).__name__ for m in lm.modules()
                                 if isinstance(m, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm))]
    assert not active_stochastic_modules
    for values in ([1, 20, 30], [1, 40, 50]):
        ids = torch.tensor(values)
        assert ids.dtype == torch.int64
        assert torch.equal(ids[None], ids.unsqueeze(0))
        with torch.no_grad():
            original = lm(ids[None])
            result = wrapped(ids)
        a, b = original["logits"], result["logits"]
        assert tuple(a.shape) == tuple(b.shape) == (1, 3, 264)
        assert a.dtype == b.dtype == torch.float32
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert torch.equal(a, b)
        assert torch.equal(original["auxiliary"], result["auxiliary"])
        for ac, bc in zip(original["cache"], result["cache"], strict=True):
            for av, bv in zip(ac, bc, strict=True):
                assert torch.equal(av, bv)
        checks.append({"seed": seed, "ids": values, "input_shape": list(ids.shape),
                       "batched_input_shape": list(ids[None].shape),
                       "logits_shape": list(a.shape), "dtype": str(a.dtype),
                       "element_count_each": a.numel(), "finite": True, "torch_equal": True,
                       "maximum_absolute_difference": (a-b).abs().max().item(),
                       "same_language_object": True, "same_parameter_objects": True,
                       "language_state_unchanged": True, "all_modules_eval": True,
                       "default_language_dropout_or_batchnorm_modules": active_stochastic_modules})
    assert state_hash(lm) == before

tok = ByteTokenizer()
assert (tok.image_id, tok.audio_id, tok.vocab_size) == (5, 6, 264)
errors = []
for values in ([1, 5, 4], [1, 6, 4], [1, 5, 6, 4]):
    try:
        wrapped(torch.tensor(values))
    except ValueError as ex:
        assert str(ex) == "placeholder 缺少配對圖片／聲音"
        errors.append({"ids": values, "exception": "ValueError", "message": str(ex)})
    else:
        raise AssertionError("Missing modality did not raise")

# Compare derivative paths while leaving all parameter values unchanged.
torch.manual_seed(0)
lm = TinyLM(ModelConfig(width=8)).eval()
wrapped = MultiModalLM(lm).eval()
before = state_hash(lm)
ids = torch.tensor([1, 20, 30])
params = tuple(lm.parameters())
direct = lm(ids[None])["logits"]
wrap = wrapped(ids)["logits"]
direct_grad = torch.autograd.grad(direct[0, -1, 20], params)
wrap_grad = torch.autograd.grad(wrap[0, -1, 20], params)
assert all(torch.equal(a, b) for a, b in zip(direct_grad, wrap_grad, strict=True))
assert state_hash(lm) == before
assert all(p.grad is None for p in params)
gradient = {"scalar": "logits[0,-1,20]", "parameter_tensors": len(params),
            "all_gradients_exactly_equal": True,
            "maximum_gradient_absolute_difference": max((a-b).abs().max().item()
                                                         for a,b in zip(direct_grad, wrap_grad, strict=True)),
            "output_weight_gradient_nonzero": bool(wrap_grad[-1].abs().max() > 0),
            "optimizer_steps": 0, "parameter_values_unchanged": True,
            "scope": "Local derivative equality only; no fine-tuning or skill-retention score measured."}
assert gradient["output_weight_gradient_nonzero"]

result = {"environment": {"python": sys.version, "executable": sys.executable,
                          "torch": torch.__version__, "torch_git_version": torch.version.git_version,
                          "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
                          "device": "cpu", "platform": platform.platform(), "torch_threads": str(torch.get_num_threads())},
          "forward_cases": checks, "marker_errors": errors, "gradient": gradient,
          "tolerance": "Exact same-size element equality in this unchanged CPU float32 path; atol=rtol=0.",
          "denominators": {"forward_pairs": len(checks), "positions_each": 3, "candidates_each": 264,
                           "score_elements_each": 792, "missing_modality_cases": len(errors),
                           "gradient_pairs": len(params), "optimizer_steps": 0},
          "nonclaims": "No training, generation quality, skill retention, backend equivalence, batched wrapper input, or GPU result established."}
(ART / "variants.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
