"""Small CPU calculations for 1.8, with independent scalar math expectations."""
import ast
import contextlib
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import platform
import shutil
import sys

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert not torch.cuda.is_available() and torch.version.cuda is None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def close(actual, expected, tolerance=1e-12):
    assert abs(actual - expected) <= tolerance, (actual, expected, tolerance)


original = Path("/tmp/phase4-1_8-fresh")
for name in ["section.md", "fence-1.py", "bootstrap.py", "extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"]:
    shutil.copyfile(original / name, HERE / ("original-" + name))

source = (original / "fence-1.py").read_text()
namespace = {"__name__": "__main__"}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(source, "original-fence-1.py", "exec"), namespace)
(HERE / "original-rerun-stdout.txt").write_text(captured.getvalue())
logits = namespace["logits"]
target = namespace["target"]
probabilities = namespace["probabilities"]
manual = namespace["manual"]
loss = namespace["loss"]
expected_denominator = 2 + math.exp(2)
expected_probabilities = [1 / expected_denominator, math.exp(2) / expected_denominator, 1 / expected_denominator]
expected_loss = math.log(expected_denominator) - 2
for actual, expected, printed in zip(probabilities[0].tolist(), expected_probabilities, [0.1065, 0.7870, 0.1065]):
    close(actual, expected, 1e-7)
    close(actual, printed, 5e-5)
close(loss.item(), expected_loss, 1e-7)
close(loss.item(), 0.2395, 5e-5)
assert tuple(logits.shape) == (1, 3) and tuple(target.shape) == (1,)
assert target.dtype == torch.int64 and logits.dtype == torch.float32
assert F is torch.nn.functional
assert isinstance(loss.item(), float)
assert not logits.requires_grad and not loss.requires_grad

log_scalars = []
for p, rounded in [(0.1, 2.303), (0.5, 0.693), (0.9, 0.105)]:
    expected = -math.log(p)
    actual = -torch.tensor(p, dtype=torch.float64).log().item()
    close(actual, expected)
    close(actual, rounded, 0.0005)
    log_scalars.append({"p": p, "negative_ln_p": actual, "rounded_text": rounded})

accuracy_example = torch.tensor([[0.3, 0.5, 0.2], [0.05, 0.9, 0.05]], dtype=torch.float64)
assert accuracy_example.argmax(dim=-1).tolist() == [1, 1]
confidence_losses = -accuracy_example[:, 1].log()
assert confidence_losses[0] > confidence_losses[1]

changed_source = source.replace("target = torch.tensor([1])", "target = torch.tensor([0])").replace("probabilities[0, 1]", "probabilities[0, 0]")
assert changed_source != source
(HERE / "exercise-target-0.py").write_text(changed_source)
exercise_namespace = {"__name__": "__main__"}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(changed_source, "exercise-target-0.py", "exec"), exercise_namespace)
(HERE / "exercise-target-0-stdout.txt").write_text(captured.getvalue())
exercise_loss = exercise_namespace["loss"].item()
expected_exercise_loss = math.log(expected_denominator)
close(exercise_loss, expected_exercise_loss, 1e-7)
close(exercise_loss, 2.2395, 5e-5)
close(exercise_loss - loss.item(), 2, 1e-7)

x = torch.tensor([[0., 2., 0.]], dtype=torch.float64)
prob = x.softmax(dim=-1)
wrong_loss = F.cross_entropy(prob, torch.tensor([1])).item()
wrong_expected = math.log(sum(math.exp(p) for p in expected_probabilities)) - expected_probabilities[1]
close(wrong_loss, wrong_expected)
assert abs(wrong_loss - expected_loss) > 0.1

batch = x.repeat(3, 1)
batch_targets = torch.tensor([1, 0, 2])
per_question = F.cross_entropy(batch, batch_targets, reduction="none")
batch_mean = F.cross_entropy(batch, batch_targets).item()
expected_each = [expected_loss, expected_exercise_loss, expected_exercise_loss]
for actual, expected in zip(per_question.tolist(), expected_each):
    close(actual, expected)
close(batch_mean, sum(expected_each) / 3)
close(F.cross_entropy(batch, batch_targets, reduction="sum").item(), sum(expected_each))
assert torch.allclose(batch.softmax(dim=-1).sum(dim=-1), torch.ones(3, dtype=torch.float64))
ignored_targets = torch.tensor([1, 0, -100])
ignored_mean = F.cross_entropy(batch, ignored_targets).item()
close(ignored_mean, sum(expected_each[:2]) / 2)

extreme = torch.tensor([[1000., 0., -1000.]], dtype=torch.float64)
extreme_target = torch.tensor([2])
stable_loss = F.cross_entropy(extreme, extreme_target).item()
naive_loss = -extreme.softmax(dim=-1)[0, 2].log().item()
close(stable_loss, 2000.)
assert math.isinf(naive_loss)

bits = expected_loss / math.log(2)
close(-math.log(0.5) / math.log(2), 1)
close(bits, -math.log2(expected_probabilities[1]))

installed_pieces = []
source_matches = {}
remote_tree = ast.parse((HERE / "torch-functional-installed-commit.py").read_text())
remote_functions = {node.name: node for node in remote_tree.body if isinstance(node, ast.FunctionDef)}
for name in ["softmax", "log_softmax", "cross_entropy"]:
    installed_source = inspect.getsource(getattr(F, name))
    installed_pieces.append(installed_source)
    installed_ast = ast.parse(installed_source).body[0]
    # AST equality includes the API contract and body, but ignores line offsets.
    match = ast.dump(installed_ast, include_attributes=False) == ast.dump(remote_functions[name], include_attributes=False)
    assert match, name
    source_matches[name] = {"installed_source_sha256": sha(installed_source.encode()), "official_commit_ast_matches": match}
(HERE / "installed-functional-inspected.py").write_text("\n\n".join(installed_pieces))

result = {
    "environment": {"python": platform.python_version(), "executable": sys.executable, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads())},
    "original": {"code_sha256": sha(source.encode()), "logits_shape": list(logits.shape), "target_shape": list(target.shape), "logits_dtype": str(logits.dtype), "target_dtype": str(target.dtype), "class_axis": -1, "class_normalizer": expected_denominator, "mean_denominator_questions": 1, "probabilities": probabilities.tolist(), "expected_probabilities_scalar_math": expected_probabilities, "manual_loss": manual.item(), "cross_entropy_loss": loss.item(), "expected_loss_scalar_math": expected_loss, "manual_ce_difference": abs(manual.item() - loss.item()), "allclose_default_rtol": 1e-5, "allclose_default_atol": 1e-8, "allclose_threshold_for_manual": 1e-8 + 1e-5 * abs(manual.item()), "requires_grad": bool(loss.requires_grad), "parameters_updated": False},
    "log_scalar_values": log_scalars,
    "accuracy_comparison": {"predicted_ids": [1, 1], "true_id": 1, "correct_questions": 2, "question_denominator": 2, "nll_nats": confidence_losses.tolist()},
    "exercise": {"true_id": 0, "selected_probability": exercise_namespace["probabilities"][0, 0].item(), "manual_loss": exercise_namespace["manual"].item(), "ce_loss": exercise_loss, "expected_scalar_math": expected_exercise_loss, "delta": exercise_loss - loss.item()},
    "incorrect_probability_input": {"actual_loss": wrong_loss, "expected_re_softmax_scalar_math": wrong_expected, "correct_logits_loss": expected_loss},
    "batch_mean": {"shape": [3, 3], "true_ids": [1, 0, 2], "class_count": 3, "question_count": 3, "per_question_nats": per_question.tolist(), "mean_nats": batch_mean, "sum_nats": sum(expected_each), "class_normalizer_per_question": expected_denominator, "mean_denominator": 3, "ignored_targets": [1, 0, -100], "nonignored_denominator": 2, "mean_with_one_ignored": ignored_mean},
    "stability": {"logits": extreme.tolist(), "target": 2, "cross_entropy_nats": stable_loss, "direct_softmax_then_log": "infinity (underflowed target probability)"},
    "unit_conversion": {"original_loss_nats": expected_loss, "original_loss_bits": bits, "ln_two": math.log(2), "p_one_half_bits": 1.0},
    "source_matches": source_matches,
    "tolerances": {"float64_vs_independent_math": 1e-12, "float32_vs_independent_math": 1e-7, "four_decimal_text": 5e-5, "three_decimal_text": 0.0005},
    "scope": "Original fence and bounded arithmetic/API variations only. No gradients, optimizer steps, model/data downloads, training, GPU, or empirical model evaluation.",
    "status": "passed"
}
(HERE / "validation-result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, indent=2, allow_nan=False))
