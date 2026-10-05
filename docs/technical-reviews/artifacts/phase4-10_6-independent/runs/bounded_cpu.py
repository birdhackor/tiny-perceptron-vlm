"""Independent CPU checks for 10.6; no optimizer, downloads, or model persistence."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, VisionEncoder, scene

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.manual_seed(0)
vision = torch.randn(1, 16, 8)
projector = nn.Linear(8, 12)
projected = projector(vision)
manual = vision @ projector.weight.T + projector.bias
max_error = (manual - projected).abs().max().item()
assert torch.allclose(manual, projected, atol=1e-6, rtol=1e-6)
assert projected.shape == (1, 16, 12)
permutation = torch.arange(15, -1, -1)
assert torch.equal(projector(vision[:, permutation]), projected[:, permutation])
duplicated = vision[:, :1].expand(1, 16, 8).contiguous()
duplicate_output = projector(duplicated)
assert torch.equal(duplicate_output[:, :1].expand_as(duplicate_output), duplicate_output)

toy = nn.Linear(2, 3, bias=False)
with torch.no_grad():
    toy.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 1.]]))
toy_output = toy(torch.tensor([1., 2.]))
assert toy_output.tolist() == [1., 2., 3.]

collapsed_inputs = vision.expand(2, -1, -1).clone()
collapsed_outputs = projector(collapsed_inputs)
assert torch.equal(collapsed_outputs[0], collapsed_outputs[1])
rank = torch.linalg.matrix_rank(projector.weight).item()
assert rank <= 8

language = TinyLM(ModelConfig(width=12, max_length=32))
good = language(embeddings=projected)
assert good["logits"].shape == (1, 16, 264)
assert torch.isfinite(good["logits"]).all()
projector10 = nn.Linear(8, 10)
mismatches = []
for positions in [4, 7, 16]:
    wrong = projector10(vision[:, :positions])
    assert wrong.shape == (1, positions, 10)
    try:
        language(embeddings=wrong)
    except RuntimeError as error:
        mismatches.append({"positions": positions, "shape": list(wrong.shape), "exception": str(error)})
    else:
        raise AssertionError("12-wide TinyLM accepted a 10-wide input")

torch.manual_seed(0)
encoder = VisionEncoder(width=8)
encoded = encoder(scene()[None])
assert encoded.shape == (1, 16, 8)
assert projector(encoded).shape == (1, 16, 12)

# Only show that an answer-token loss can differentiate projector parameters.
# All model parameters remain unchanged; this is not an alignment evaluation.
torch.manual_seed(0)
multimodal = MultiModalLM(TinyLM(ModelConfig(width=12, max_length=32)), vision_width=8)
multimodal.language.requires_grad_(False)
multimodal.vision.requires_grad_(False)
tok = ByteTokenizer()
answer_id = tok.encode("r")[0]
ids = torch.tensor([tok.bos_id, tok.image_id, tok.assistant_id, answer_id, tok.eos_id])
labels = torch.tensor([IGNORE, IGNORE, IGNORE, answer_id, tok.eos_id])
before = {name: p.detach().clone() for name, p in multimodal.named_parameters()}
output = multimodal(ids, labels, image=scene("red", "square"))
loss = masked_loss(output["logits"], output["labels"])
loss.backward()
grad_norm = multimodal.image_projector.weight.grad.norm().item()
assert grad_norm > 0 and torch.isfinite(loss)
assert output["logits"].shape == (1, 19, 264)
assert output["labels"].shape == (1, 19)
assert (output["labels"] != IGNORE).sum().item() == 2
assert all(p.grad is None for p in multimodal.language.parameters())
assert all(p.grad is None for p in multimodal.vision.parameters())
assert all(torch.equal(before[name], p.detach()) for name, p in multimodal.named_parameters())

module_hashes = {}
for name, module in sorted(sys.modules.items()):
    filename = getattr(module, "__file__", None)
    if filename and name.startswith("tiny_perceptron"):
        path = Path(filename).resolve()
        if path.is_file():
            module_hashes[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps({
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": str(vision.device), "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads())},
    "linear": {"input_shape": list(vision.shape), "output_shape": list(projected.shape), "weight_shape": list(projector.weight.shape), "bias_shape": list(projector.bias.shape), "manual_affine_max_abs_error": max_error, "atol": 1e-6, "rtol": 1e-6, "position_permutation_exact": True, "duplicated_positions_exact": True, "weight_rank": rank},
    "toy": {"input": [1., 2.], "weight": toy.weight.tolist(), "bias": None, "output": toy_output.tolist()},
    "information_collision": {"identical_inputs": True, "identical_projected_outputs": True, "scope": "Deterministic projector only; no side information is provided."},
    "core_interface": {"language_width": 12, "logits_shape": list(good["logits"].shape), "wrong_weight_shape": list(projector10.weight.shape), "width10_exceptions": mismatches},
    "real_encoder": {"image_shape": [1, 3, 16, 16], "feature_shape": list(encoded.shape), "projected_shape": list(projector(encoded).shape)},
    "answer_gradient": {"input_token_count": len(ids), "image_positions": 16, "expanded_positions_before_shift": 20, "positions_after_shift": 19, "logits_shape": list(output["logits"].shape), "label_shape": list(output["labels"].shape), "answer_tokens_denominator": 2, "loss_nats_per_answer_token": loss.item(), "image_projector_weight_gradient_norm": grad_norm, "language_and_vision_frozen": True, "all_parameters_unchanged": True, "optimizer_steps": 0, "scope": "One untrained CPU forward/backward; not training, generated-answer assessment, or evidence of alignment."},
    "repository_modules": module_hashes,
}, ensure_ascii=False, indent=2))
