"""Bounded, CPU-only independent checks of section 4.6; no learning or input files."""
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from tiny_perceptron.attention import attention_mask
from tiny_perceptron.model import ModelConfig, TinyLM, generate

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
torch.set_num_threads(1)
torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=20, width=8))
params_before = {name: value.detach().clone() for name, value in model.named_parameters()}
stages = {}
hooks = []
for name, module in [("embedding", model.embedding), ("position", model.position),
                     ("block", model.blocks[0]), ("final_norm", model.final_norm),
                     ("output", model.output)]:
    def record(module, args, output, name=name):
        tensor = output[0] if isinstance(output, tuple) else output
        stages[name] = list(tensor.shape)
    hooks.append(module.register_forward_hook(record))
ids = torch.tensor([[1, 2, 3]])
target = torch.tensor([[2, 3, 4]])
logits = model(ids)["logits"]
for hook in hooks:
    hook.remove()
assert stages == {"embedding": [1, 3, 8], "position": [3, 8], "block": [1, 3, 8],
                  "final_norm": [1, 3, 8], "output": [1, 3, 20]}
x = model.embedding(ids) + model.position(torch.arange(3))
x = model.blocks[0](x)[0]
hidden = model.final_norm(x)
dot_product_logits = hidden @ model.output.weight.T
linear_error = float((logits - dot_product_logits).detach().abs().max())
assert linear_error == 0.0
assert model.output.bias is None
assert model.embedding.weight is not model.output.weight
assert model.embedding.weight.data_ptr() != model.output.weight.data_ptr()
assert list(model.embedding.weight.shape) == [20, 8]
assert list(model.output.weight.shape) == [20, 8]
assert model.embedding.weight.requires_grad and model.output.weight.requires_grad
assert logits.shape == (1, 3, 20)
assert logits.argmax(-1).shape == (1, 3)
assert logits.min().item() < 0

scores = logits.reshape(-1, logits.shape[-1])
answers = target.reshape(-1)
assert scores.shape == (3, 20) and answers.shape == (3,)
assert torch.equal(scores[0], logits[0, 0])
assert torch.equal(scores[1], logits[0, 1])
assert torch.equal(scores[2], logits[0, 2])
assert answers.tolist() == [2, 3, 4]
loss = F.cross_entropy(scores, answers)
per_question = F.cross_entropy(scores, answers, reduction="none")
# Independent float64 calculation from the saved raw class scores.
scores64 = scores.detach().double()
negative_log_prob = scores64.logsumexp(-1) - scores64[torch.arange(3), answers]
manual_mean = negative_log_prob.sum() / 3
mean_error = abs(loss.item() - manual_mean.item())
assert mean_error < 1e-6
assert round(loss.item(), 3) == 3.336
assert abs(loss.item() - per_question.mean().item()) < 1e-6
changed_target = torch.tensor([4, 3, 2])
changed_loss = F.cross_entropy(scores, changed_target)
assert abs(loss.item() - changed_loss.item()) > 1e-3

allowed = attention_mask(torch.arange(3), torch.arange(3))[0, 0]
assert allowed.tolist() == [[True, False, False], [True, True, False], [True, True, True]]
perturbed = model(torch.tensor([[1, 2, 19]]))["logits"]
prefix_perturb_error = (logits[:, :2] - perturbed[:, :2]).detach().abs().max().item()
assert prefix_perturb_error == 0.0
prefix_errors = []
for length in (1, 2, 3):
    prefix_logits = model(ids[:, :length])["logits"]
    error = (prefix_logits[:, -1] - logits[:, length - 1]).detach().abs().max().item()
    prefix_errors.append(error)
    assert error < 1e-6

generation_inputs = []
def record_generation(module, args):
    generation_inputs.append(args[0].tolist())
hook = model.register_forward_pre_hook(record_generation)
generated = generate(model, ids, max_new_tokens=2, eos_id=-1, use_cache=False)
hook.remove()
first = logits[:, -1].argmax(-1, keepdim=True)
second_input = torch.cat((ids, first), dim=1)
second = model(second_input)["logits"][:, -1].argmax(-1, keepdim=True)
assert torch.equal(generated, torch.cat((second_input, second), dim=1))
assert generation_inputs == [ids.tolist(), second_input.tolist()]
assert generated.shape == (1, 5)

torch.manual_seed(42)
larger = TinyLM(ModelConfig(vocab_size=25, width=8))
logits25 = larger(ids)["logits"]
assert logits25.shape == (1, 3, 25)
assert model.output.weight.numel() == 160 and larger.output.weight.numel() == 200
assert logits25.numel() - logits.numel() == 3 * 5
boundary = {}
for value in (0, 19, 20, -1):
    try:
        output = model(torch.tensor([[value]]))["logits"]
        boundary[str(value)] = {"success": True, "shape": list(output.shape)}
    except Exception as error:
        boundary[str(value)] = {"success": False, "type": type(error).__name__, "message": str(error)}
assert boundary["0"]["success"] and boundary["19"]["success"]
assert not boundary["20"]["success"] and not boundary["-1"]["success"]
assert all(torch.equal(params_before[name], p.detach()) for name, p in model.named_parameters())
assert all(p.grad is None for p in model.parameters())

result = {
    "stage_shapes": stages,
    "output_mapping": {"weight_shape": list(model.output.weight.shape), "bias": None,
                       "max_dot_product_error": linear_error, "embedding_tied": False,
                       "trainable_embedding_and_output": True},
    "original_logits": logits.detach().tolist(),
    "argmax": logits.argmax(-1).tolist(),
    "raw_score_range": [logits.min().item(), logits.max().item()],
    "candidate_axis_sums_before_softmax": logits.sum(-1).detach().tolist(),
    "loss": {"scores_shape": list(scores.shape), "answers_shape": list(answers.shape),
             "answers": answers.tolist(), "per_question": per_question.detach().tolist(),
             "manual_float64_per_question": negative_log_prob.tolist(),
             "mean": loss.item(), "manual_float64_mean": manual_mean.item(),
             "mean_error": mean_error, "rounded_to_3_decimals": round(loss.item(), 3),
             "denominator_questions": 3, "log_base": "natural", "unit": "nats/question",
             "changed_labels": changed_target.tolist(), "changed_label_mean": changed_loss.item()},
    "causal_check": {"allowed_mask": allowed.tolist(), "future_id_change": [3, 19],
                     "prior_positions_max_error": prefix_perturb_error,
                     "separate_prefix_max_errors": prefix_errors},
    "generation": {"actual_forward_inputs": generation_inputs, "generated_ids": generated.tolist(),
                   "max_new_tokens": 2, "eos_id": -1, "use_cache": False,
                   "manual_last_row_generation_matches": True},
    "vocab_25": {"shape": list(logits25.shape), "argmax": logits25.argmax(-1).tolist(),
                 "additional_candidates_per_position": 5,
                 "output_weight_counts": [model.output.weight.numel(), larger.output.weight.numel()]},
    "boundary_ids": boundary,
    "execution_kind": "Forward and loss demonstrations only; no backward, optimizer, training, checkpoints or datasets.",
    "all_original_model_parameters_unchanged": True,
    "all_original_model_parameter_gradients_none": True,
}
environment = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
               "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
               "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
               "threads": str(torch.get_num_threads()), "seed": "42", "input_provenance": "Literal synthetic IDs only"}
for name, value in [("probe-result.json", result), ("probe-environment.json", environment)]:
    (ART / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
print("PASS: all independent CPU assertions completed")
