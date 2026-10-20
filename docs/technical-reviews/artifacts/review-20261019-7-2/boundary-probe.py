"""核對對話前綴、移位目標與角色標記練習；不訓練模型。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.data import ByteTokenizer, render_chat, toy_conversations
from tiny_perceptron.model import ModelConfig, TinyLM, generate


def execute_example(code):
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    error = None
    with contextlib.redirect_stdout(output):
        try:
            exec(compile(code, "frozen-example", "exec"), namespace)
        except AssertionError as exc:
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            error = {"type": type(exc).__name__, "statement": code.splitlines()[frame.lineno - 1]}
    return namespace, {"stdout": output.getvalue(), "error": error}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    source_path = ROOT / "docs/course-review-20261019/freeze-04/sources/7.2.md"
    raw = source_path.read_bytes()
    code = re.search(r"```python\n(.*?)```", raw.decode(), re.S)[1]
    original, original_result = execute_example(code)
    changed = code.replace("+ [tok.eos_id, tok.assistant_id]", "+ [tok.eos_id]")
    assert changed != code
    removed, removed_result = execute_example(changed)
    first_assertion = next(node for node in ast.parse(changed).body if isinstance(node, ast.Assert))
    assert removed_result["error"]["statement"] == changed.splitlines()[first_assertion.lineno - 1]
    assert removed_result["error"]["type"] == "AssertionError"
    assert original_result["error"] is None

    tok = ByteTokenizer()
    expected = [1, 3, 57, 51, 57, 69, 71, 2, 4]
    assert original["prompt"] == expected
    assert removed["prompt"] == expected[:-1]
    x, y = render_chat([{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "2"}])
    assert original["_"].tolist() == y.tolist()
    assert x[len(expected) - 1].item() == tok.assistant_id
    assert y[len(expected) - 1].item() == tok.encode("2")[0]

    boundaries = []
    for messages in toy_conversations():
        inputs, labels = render_chat(messages)
        prefix = [tok.bos_id, tok.user_id] + tok.encode(messages[0]["content"]) + [tok.eos_id, tok.assistant_id]
        assert inputs[: len(prefix)].tolist() == prefix
        assert labels[len(prefix) - 1].item() == tok.encode(messages[1]["content"])[0]
        boundaries.append({"question": messages[0]["content"], "answer": messages[1]["content"], "prefix": prefix})

    history = [
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "2+1=?"},
    ]
    history_prefix = [tok.bos_id]
    role_ids = {"user": tok.user_id, "assistant": tok.assistant_id}
    for message in history:
        history_prefix += [role_ids[message["role"]]] + tok.encode(message["content"]) + [tok.eos_id]
    history_prefix += [tok.assistant_id]
    multi_x, multi_y = render_chat(history + [{"role": "assistant", "content": "3"}])
    assert multi_x[: len(history_prefix)].tolist() == history_prefix
    assert multi_y[len(history_prefix) - 1].item() == tok.encode("3")[0]

    model = TinyLM(ModelConfig(width=8, heads=1, layers=1, max_length=32)).eval()
    prefix_tensor = torch.tensor([expected], dtype=torch.long)
    with torch.no_grad():
        prefix_logits = model(prefix_tensor)["logits"]
        full_logits = model(x.unsqueeze(0))["logits"]
        difference = (prefix_logits[0, -1] - full_logits[0, len(expected) - 1]).abs().max().item()
        predicted = prefix_logits[:, -1].argmax(-1).item()
        generated = generate(model, prefix_tensor, max_new_tokens=1, temperature=0.0)
    assert prefix_logits.shape == (1, len(expected), tok.vocab_size)
    assert difference < 1e-6
    assert generated[0, -1].item() == predicted

    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-2/boundary-probe.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "seed": "23", "debug": str(__debug__)},
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "original": original_result,
        "removed_assistant": {**removed_result, "prompt": removed["prompt"], "last_id": removed["prompt"][-1]},
        "utf8_bytes": list("1+1=?".encode()),
        "encoded_question": tok.encode("1+1=?"),
        "prompt": expected,
        "prompt_length": len(expected),
        "training_x": x.tolist(),
        "training_y": y.tolist(),
        "answer_content_id": tok.encode("2")[0],
        "underscore_contains_y": original["_"].tolist() == y.tolist(),
        "toy_records_checked": len(boundaries),
        "toy_boundaries": boundaries,
        "history_prefix": history_prefix,
        "history_training_prefix_matches": True,
        "multi_turn_current_answer_target": multi_y[len(history_prefix) - 1].item(),
        "logits_shape": list(prefix_logits.shape),
        "prefix_vs_training_position_max_abs_difference": difference,
        "last_logits_argmax_matches_generation": True,
        "generation_token": generated[0, -1].item(),
        "scope": "只核對格式、標籤與未訓練模型的生成位置；不衡量算術能力或訓練後品質。",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
