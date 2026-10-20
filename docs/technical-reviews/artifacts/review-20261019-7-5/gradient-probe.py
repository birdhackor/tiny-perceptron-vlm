"""核對被忽略位置的 logits 與輸入 embedding 的不同梯度路徑。"""

import json
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.data import render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss


def inspect_case(question, mode="normal", only_first_answer=False):
    torch.manual_seed(42)
    x, y = render_chat([
        {"role": "user", "content": question},
        {"role": "assistant", "content": "A"},
    ])
    if only_first_answer:
        y = y.clone()
        y[5] = -100
    model = TinyLM(ModelConfig(width=8))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    embedded = model.embedding(x[None])
    embedded.retain_grad()
    if mode == "detach_question":
        keep = torch.ones_like(embedded)
        keep[:, 2] = 0
        model_input = embedded * keep + embedded.detach() * (1 - keep)
    else:
        model_input = embedded
    arguments = {}
    if mode == "isolate_question":
        arguments["segments"] = torch.tensor([[0, 0, 1, 0, 0, 0]])
    logits = model(embeddings=model_input, **arguments)["logits"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        grad_before_retain = logits.grad
    logits.retain_grad()
    loss = masked_loss(logits, y[None])
    loss.backward()
    row = int(x[2])
    weight_grad = model.embedding.weight.grad
    result = {
        "question": question,
        "mode": mode,
        "only_first_answer": only_first_answer,
        "x": x.tolist(),
        "y": y.tolist(),
        "x_dtype": str(x.dtype),
        "y_dtype": str(y.dtype),
        "batch_shape": list(x[None].shape),
        "logits_shape": list(logits.shape),
        "logits_is_leaf": logits.is_leaf,
        "logits_grad_before_retain_is_none": grad_before_retain is None,
        "logits_grad_retained": logits.grad is not None,
        "question_row": row,
        "question_occurrences": int((x == row).sum()),
        "tied_output": model.output.weight is model.embedding.weight,
        "effective_targets": int((y != -100).sum()),
        "loss": float(loss.detach()),
        "question_logits_gradient_norm": float(logits.grad[0, 2].norm()),
        "ignored_logits_gradient_max_abs": float(logits.grad[0, y == -100].abs().max()),
        "question_embedding_gradient_norm": float(weight_grad[row].norm()),
        "question_input_gradient_norm": float(embedded.grad[0, 2].norm()),
        "row_gradient_equals_input_gradient": bool(torch.equal(weight_grad[row], embedded.grad[0, 2])),
        "embedding_layer_grad_is_none": weight_grad is None,
        "unused_row_gradient_norm": float(weight_grad[0].norm()),
        "manual_norm_matches": bool(torch.allclose(weight_grad[row].norm(), weight_grad[row].square().sum().sqrt())),
        "parameters_unchanged_after_backward": all(
            torch.equal(before[name], value) for name, value in model.named_parameters()
        ),
    }
    assert result["question_logits_gradient_norm"] == 0
    assert result["ignored_logits_gradient_max_abs"] == 0
    assert result["row_gradient_equals_input_gradient"]
    assert result["manual_norm_matches"]
    assert result["parameters_unchanged_after_backward"]
    if mode == "normal":
        assert result["question_embedding_gradient_norm"] > 0
    else:
        assert result["question_embedding_gradient_norm"] == 0
    return result


torch.set_num_threads(1)
torch.set_default_device("cpu")
result = {
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "dtype": str(torch.get_default_dtype()),
        "threads": str(torch.get_num_threads()),
    },
    "seed": 42,
    "optimizer_steps": 0,
    "cases": [
        inspect_case("Q"),
        inspect_case("Z"),
        inspect_case("Q", only_first_answer=True),
        inspect_case("Q", mode="detach_question"),
        inspect_case("Q", mode="isolate_question"),
    ],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
