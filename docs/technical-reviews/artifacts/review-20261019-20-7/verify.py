"""核對 byte 算例、忽略代價、因果前文及訓練資料長度邊界。"""

import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tiny_perceptron.attention import attention_mask
from tiny_perceptron.data import ByteTokenizer, IGNORE, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.natural_assistant import encode_training_row


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def example(answer):
    return [{"role": "user", "content": "Q"}, {"role": "assistant", "content": answer}]


def inspect_example(answer, expected_x, expected_y):
    x, y = render_chat(example(answer))
    assert x.tolist() == expected_x
    assert y.tolist() == expected_y
    active = y != IGNORE
    first = active.nonzero()[0].item()
    assert first == 4
    assert x[first].item() == ByteTokenizer.assistant_id
    return {
        "answer": answer,
        "x": x.tolist(),
        "y": y.tolist(),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "dtype": str(x.dtype),
        "active": active.tolist(),
        "active_dtype": str(active.dtype),
        "nonzero": active.nonzero().tolist(),
        "first": first,
        "count": active.sum().item(),
        "active_targets": y[active].tolist(),
    }


class FixtureProcessor:
    """替身只隔離資料包裝邏輯；不模擬 Qwen 圖片或 tokenizer。"""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        tok = ByteTokenizer()
        ids = [tok.bos_id]
        roles = {"user": tok.user_id, "assistant": tok.assistant_id, "system": tok.system_id}
        for message in messages:
            ids += [roles[message["role"]]]
            ids += tok.encode("".join(part["text"] for part in message["content"]))
            ids += [tok.eos_id]
        if add_generation_prompt:
            ids += [tok.assistant_id]
        return ids

    def __call__(self, text, return_tensors):
        assert return_tensors == "pt"
        inputs = torch.tensor(text, dtype=torch.long)
        return {"input_ids": inputs, "attention_mask": torch.ones_like(inputs)}


def main():
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-7/verify.py",
        "environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "device": "cpu",
            "platform": platform.platform(),
            "transformers": "not installed" if importlib.util.find_spec("transformers") is None else importlib.metadata.version("transformers"),
        },
        "repository_sha256": {
            path: digest(path) for path in [
                "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py",
                "tiny_perceptron/natural_assistant.py", "scripts/natural_assistant.py",
                "requirements-natural.txt", "docs/natural-assistant/v4/experiment-plan.json",
            ]
        },
        "examples": [
            inspect_example("A", [1, 3, 89, 2, 4, 73], [-100, -100, -100, -100, 73, 2]),
            inspect_example("AB", [1, 3, 89, 2, 4, 73, 74], [-100, -100, -100, -100, 73, 74, 2]),
        ],
    }
    x, y = render_chat(example("A"))
    logits = torch.zeros((1, len(x), ByteTokenizer.vocab_size), requires_grad=True)
    loss = masked_loss(logits, y.unsqueeze(0))
    unreduced = F.cross_entropy(logits.reshape(-1, ByteTokenizer.vocab_size), y, ignore_index=IGNORE, reduction="none")
    loss.backward()
    assert torch.all(logits.grad[0, :4] == 0)
    assert torch.all(unreduced[:4] == 0)
    assert math.isclose(loss.item(), math.log(264), abs_tol=1e-6)
    result["masked_loss"] = {
        "logits_shape": list(logits.shape), "labels_dtype": str(y.dtype),
        "effective_denominator": int((y != IGNORE).sum()),
        "loss": loss.item(), "expected_log_vocab": math.log(264), "tolerance": 1e-6,
        "per_position_loss": unreduced.tolist(),
        "ignored_logit_gradient_max": logits.grad[0, :4].abs().max().item(),
        "active_logit_gradient_max": logits.grad[0, 4:].abs().max().item(),
    }

    torch.manual_seed(42)
    model = TinyLM(ModelConfig(width=16, layers=2, heads=2, max_length=16)).eval()
    ids = x.unsqueeze(0)
    future_changed, question_changed = ids.clone(), ids.clone()
    future_changed[0, 5] = 74
    question_changed[0, 2] = 90
    with torch.no_grad():
        baseline = model(ids)["logits"]
        future = model(future_changed)["logits"]
        question = model(question_changed)["logits"]
    right_difference = (baseline[:, 4] - future[:, 4]).abs().max().item()
    question_difference = (baseline[:, 4] - question[:, 4]).abs().max().item()
    assert right_difference == 0.0
    assert question_difference > 0.0
    model.zero_grad(set_to_none=True)
    masked_loss(model(ids)["logits"], y.unsqueeze(0)).backward()
    question_gradient = model.embedding.weight.grad[89].abs().max().item()
    assert question_gradient > 0.0
    allowed = attention_mask(torch.arange(6), torch.arange(6))[0, 0]
    assert allowed[4].tolist() == [True, True, True, True, True, False]
    result["causal_conditioning"] = {
        "seed": 42, "dtype": "torch.float32", "model_config": model.description()["config"],
        "assistant_allowed_keys": allowed[4].tolist(),
        "assistant_logits_max_difference_when_right_answer_changes": right_difference,
        "assistant_logits_max_difference_when_question_changes": question_difference,
        "question_embedding_gradient_max_from_answer_loss": question_gradient,
        "scope": "只核對此固定 CPU 算例；不衡量模型品質或 Qwen 數值。",
    }

    truncation = []
    for length in [4, 5, 6]:
        try:
            bx, by, valid = pad_batch([(x, y)], max_length=length)
            truncation.append({"max_length": length, "input": bx.tolist(), "active_targets": by[by != IGNORE].tolist()})
        except ValueError as error:
            truncation.append({"max_length": length, "exception": type(error).__name__, "message": str(error)})
    assert truncation[0]["exception"] == "ValueError"
    assert truncation[1]["active_targets"] == [73]
    assert truncation[2]["active_targets"] == [73, 2]
    result["truncation"] = truncation

    processor = FixtureProcessor()
    ordinary = encode_training_row(processor, {"id": "fixture", "user": "Q", "answer": "A"}, ROOT)
    assert ordinary["input_ids"].tolist() == [[1, 3, 89, 2, 4, 73, 2]]
    assert ordinary["labels"].tolist() == [[-100, -100, -100, -100, -100, 73, 2]]
    boundary = encode_training_row(processor, {"id": "boundary", "user": "Q", "answer": "A" * 2042}, ROOT)
    assert boundary["input_ids"].shape[-1] == 2048
    try:
        encode_training_row(processor, {"id": "over", "user": "Q", "answer": "A" * 2043}, ROOT)
    except ValueError as error:
        rejected = {"exception": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError("超過預設上限的資料應被拒絕")
    spec = importlib.util.spec_from_file_location("natural_cli", ROOT / "scripts/natural_assistant.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    options = cli.parser().parse_args(["prepare", "--output", "unused-fixture-directory"])
    assert options.max_tokens == 2048
    plan = json.loads((ROOT / "docs/natural-assistant/v4/experiment-plan.json").read_bytes())
    assert plan["fresh_lora"]["max_tokens"] == 2048
    result["training_wrapper"] = {
        "fixture_input_ids": ordinary["input_ids"].tolist(), "fixture_labels": ordinary["labels"].tolist(),
        "fixture_supervised_after_internal_shift": ordinary["labels"][:, 1:][ordinary["labels"][:, 1:] != IGNORE].tolist(),
        "boundary_full_tokens": boundary["input_ids"].shape[-1], "over_limit": rejected,
        "function_default": 2048, "cli_default": options.max_tokens,
        "experiment_plan_max_tokens": plan["fresh_lora"]["max_tokens"],
        "scope": "執行真實資料包裝函式，processor 為明示替身；未載入 Qwen tokenizer、圖片 processor 或模型。",
    }
    result["result"] = "全部斷言通過"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
