"""獨立 technical reviewer 的 7.24 短 CPU 診斷；不建立或更新模型。"""

import argparse
import ast
import contextlib
import hashlib
import inspect
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from tiny_perceptron.data import IGNORE, ByteTokenizer, render_chat
from tiny_perceptron.model import masked_loss


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    snapshot = Path("docs/course-review-20261019/freeze-01/sources/7.24.md")
    raw = snapshot.read_bytes()
    assert sha(raw) == "2bbcdaabd6cb0c0ae49afd4ae8e6beccb05b33a455414e7bbd4f3d15a61d747a"
    blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
    assert len(blocks) == 1
    tree = ast.parse(blocks[0])
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any("backward" in call or "step" in call or "optimizer" in call for call in calls)
    namespace = {}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(blocks[0], str(snapshot) + "#python-1", "exec"), namespace)

    tok = ByteTokenizer()
    diagnostics = []
    conditioned_prompts = []
    unconditioned_prompts = []
    for row, expected_count in zip(namespace["records"], (2, 27), strict=True):
        marker = f"<{row['effort']}>"
        prompt = marker + " " + row["question"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row["answer"]},
        ]
        x, y = render_chat(messages)
        # 以 Python UTF-8 bytes 獨立組出移位序列與遮罩，不再呼叫 tokenizer 編碼。
        prefix_ids = [1, 3] + [b + 8 for b in prompt.encode("utf-8")] + [2, 4]
        answer_ids = [b + 8 for b in row["answer"].encode("utf-8")]
        full_ids = prefix_ids + answer_ids + [2]
        full_targets = [IGNORE] * len(prefix_ids) + answer_ids + [2]
        assert x.tolist() == full_ids[:-1]
        assert y.tolist() == full_targets[1:]
        assert x.dtype == y.dtype == torch.int64
        assert x.device.type == y.device.type == "cpu"
        assert x.shape == y.shape
        selected = y[y != IGNORE]
        assert selected.tolist() == answer_ids + [2]
        assert len(selected) == expected_count
        assert tok.decode(selected.tolist()) == row["answer"]
        marker_ids = tok.encode(marker)
        assert all(i >= 8 for i in marker_ids)
        assert tok.decode(marker_ids) == marker
        first_target = int((y != IGNORE).nonzero()[0].item())
        assert x[first_target].item() == tok.assistant_id
        assert x[: first_target + 1].tolist() == prefix_ids
        conditioned_prompts.append(prefix_ids)
        no_marker_x, no_marker_y = render_chat([
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ])
        start = int((no_marker_y != IGNORE).nonzero()[0].item())
        unconditioned_prompts.append(no_marker_x[: start + 1].tolist())

        # 另外核對現行 masked_loss 的 ignore 行為；只有合成 logits 求導，沒有模型或更新。
        logits = torch.zeros((len(y), tok.vocab_size), requires_grad=True)
        loss = masked_loss(logits, y)
        loss.backward()
        assert torch.count_nonzero(logits.grad[y == IGNORE]).item() == 0
        assert (logits.grad[y != IGNORE].abs().sum(dim=-1) > 0).all().item()
        diagnostics.append({
            "effort": row["effort"],
            "input_length": len(x),
            "dtype": str(x.dtype),
            "marker_ids": marker_ids,
            "marker_is_ordinary_bytes": True,
            "answer_ascii_count": sum(ord(c) < 128 for c in row["answer"]),
            "answer_nonascii_count": sum(ord(c) >= 128 for c in row["answer"]),
            "answer_utf8_bytes": len(answer_ids),
            "effective_targets": len(selected),
            "first_supervised_position": first_target,
            "first_supervised_target": selected[0].item(),
            "last_supervised_target": selected[-1].item(),
            "decoded_supervised_answer": tok.decode(selected.tolist()),
            "mask_and_shift_match_independent_construction": True,
            "synthetic_logits_loss": loss.detach().item(),
            "ignored_logits_gradient_zero": True,
            "answer_and_eos_logits_have_gradient": True,
        })

    assert conditioned_prompts[0] != conditioned_prompts[1]
    assert unconditioned_prompts[0] == unconditioned_prompts[1]
    invalid_answer = "2+3=6；6-1=4"
    exercise_messages = [
        {"role": "user", "content": "<high> " + namespace["records"][1]["question"]},
        {"role": "assistant", "content": invalid_answer},
    ]
    _, exercise_y = render_chat(exercise_messages)
    exercise_text = tok.decode(exercise_y[exercise_y != IGNORE].tolist())
    assert exercise_text == invalid_answer
    exercise = {
        "question_truth": 2 + 3 - 1,
        "claimed_final_answer": 4,
        "first_equation_correct": 2 + 3 == 6,
        "second_equation_correct": 6 - 1 == 4,
        "next_step_uses_previous_claim": True,
        "wrong_equations_are_in_supervised_text": exercise_text == invalid_answer,
    }
    assert exercise["question_truth"] == exercise["claimed_final_answer"] == 4
    assert not exercise["first_equation_correct"] and not exercise["second_equation_correct"]

    prerequisites = []
    freeze_path = Path("docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json")
    freeze = json.loads(freeze_path.read_bytes())
    for lesson in ("7.24", "7.11", "7.1", "6.6", "14.13", "14.14", "C.1", "C.2"):
        entry = next(p for p in freeze["pages"] if p["page_id"] == lesson)
        current = Path(entry["source"]).read_text()
        section = re.search(rf"^## {re.escape(lesson)} .*?(?=^## |\Z)", current, re.M | re.S).group()
        current_digest = sha(section.encode())
        assert current_digest == entry["source_sha256"]
        assert sha(Path(entry["snapshot"]).read_bytes()) == current_digest
        prerequisites.append({
            "lesson_id": lesson,
            "current_source": entry["source"],
            "selector": "section:" + lesson,
            "sha256": current_digest,
            "matching_existing_snapshot": entry["snapshot"],
            "figures_sha256": entry["figures_sha256"],
        })

    notebook = Path("notebooks/07/7.24.ipynb")
    assert sha(notebook.read_bytes()) == "fb3916cab7a6e14ea751942b093f1d62f73663de370ec6927bb3b869f0d6e2c0"
    notebook_code = "\n".join("".join(c["source"]) for c in json.loads(notebook.read_bytes())["cells"] if c["cell_type"] == "code")
    assert blocks[0].strip() in notebook_code
    functional_path = Path(inspect.getsourcefile(torch.nn.functional.cross_entropy))
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_task": "/root/omni_technical_coordinator/technical_7_24",
        "command": "uv run --extra cpu python " + str(Path(__file__).relative_to(root)) + " --output " + str(args.output),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "source_snapshot": {"path": str(snapshot), "sha256": sha(raw), "python_block_sha256": sha(blocks[0].encode())},
        "notebook": {"path": str(notebook), "sha256": sha(notebook.read_bytes()), "contains_exact_lesson_python_block": True},
        "inline_code_stdout": captured.getvalue(),
        "inline_code_updates_parameters": False,
        "diagnostics": diagnostics,
        "effective_target_ratio_high_over_low": diagnostics[1]["effective_targets"] / diagnostics[0]["effective_targets"],
        "conditioned_prefixes_distinct": True,
        "prefixes_identical_if_effort_column_is_omitted_from_input": True,
        "exercise": exercise,
        "source_freeze": {"path": str(freeze_path), "sha256": sha(freeze_path.read_bytes())},
        "current_prerequisite_slices": prerequisites,
        "inspected_installed_torch_source": {"path": str(functional_path), "sha256": sha(functional_path.read_bytes()), "cross_entropy_function_sha256": sha(inspect.getsource(torch.nn.functional.cross_entropy).encode())},
        "result": "PASS: 精確執行教材唯一 Python 區塊；byte/移位/遮罩/EOS/2對27/條件輸入/錯誤步驟均符合正文，未建立或更新模型。",
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
