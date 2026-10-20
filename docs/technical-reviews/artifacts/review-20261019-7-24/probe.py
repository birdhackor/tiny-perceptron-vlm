"""核對條件文字、助手目標、代價遮罩與算術示範。"""

import ast
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tiny_perceptron.data import SPECIALS, ByteTokenizer, render_chat


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


tok = ByteTokenizer()
question = "先有2件，新增3件，再移除1件，剩幾件？"
answers = {"low": "4", "high": "2+3=5；5-1=4。答案4。"}
rows = []
prefixes = {}
for effort, answer in answers.items():
    prompt = f"<{effort}> {question}"
    x, y = render_chat([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ])
    target = y[y != -100].tolist()
    expected_target = tok.encode(answer) + [tok.eos_id]
    assert target == expected_target
    prompt_ids = tok.encode(prompt)
    assert x[2:2 + len(prompt_ids)].tolist() == prompt_ids
    marker = tok.encode(f"<{effort}>")
    assert all(i >= len(SPECIALS) for i in marker)
    assert tok.decode(marker) == f"<{effort}>"
    logits = torch.zeros((len(y), tok.vocab_size), dtype=torch.float64, requires_grad=True)
    loss = F.cross_entropy(logits, y, ignore_index=-100)
    loss.backward()
    magnitude = logits.grad.abs().sum(dim=-1)
    assert bool((magnitude[y == -100] == 0).all())
    assert bool((magnitude[y != -100] > 0).all())
    prefixes[effort] = x[:2 + len(prompt_ids)].tolist()
    rows.append({
        "effort": effort, "input_length": len(x),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "answer_utf8_bytes": len(answer.encode("utf-8")),
        "valid_targets": len(target), "answer_target_ids": target,
        "answer_roundtrip": tok.decode(target), "eos_is_supervised": target[-1] == tok.eos_id,
        "marker_ids": marker, "marker_is_ordinary_bytes": True,
        "ignored_positions_have_zero_logit_gradient": True,
        "answer_and_eos_positions_have_nonzero_logit_gradient": True,
    })

wrong = "2+3=6；6-1=4"
_, wrong_y = render_chat([
    {"role": "user", "content": f"<high> {question}"},
    {"role": "assistant", "content": wrong},
])
wrong_target = wrong_y[wrong_y != -100].tolist()
assert wrong_target == tok.encode(wrong) + [tok.eos_id]
snapshot = ROOT / "docs/course-review-20261019/freeze-01/sources/7.24.md"
source = snapshot.read_text()
code = re.search(r"```python\n(.*?)```", source, re.S).group(1)
calls = sorted({ast.unparse(n.func) for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call)})
assert not any(name in calls for name in ("backward", "optimizer.step", "loss.backward"))
assert prefixes["low"] != prefixes["high"]
without_effort = [{"role": "user", "content": question}, {"role": "assistant", "content": "4"}]
a, _ = render_chat(without_effort)
b, _ = render_chat(without_effort)
assert torch.equal(a, b)

print(json.dumps({
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "source_sha256": sha(snapshot),
    "repository_sha256": {p: sha(ROOT / p) for p in ("tiny_perceptron/data.py", "tiny_perceptron/model.py")},
    "rows": rows,
    "arithmetic": {"initial": 2, "added": 3, "removed": 1, "result": 2 + 3 - 1},
    "target_ratio_high_over_low": rows[1]["valid_targets"] / rows[0]["valid_targets"],
    "high_byte_breakdown": {s: len(s.encode("utf-8")) for s in ("2+3=5", "；", "5-1=4", "。答案4。")},
    "distinct_effort_input_prefixes": True,
    "omitted_effort_leaves_identical_input": True,
    "original_fence_calls": calls,
    "original_fence_creates_no_model_or_optimizer": not any(s in code for s in ("TinyLM", "optim.", ".step(", ".backward(")),
    "wrong_demonstration": {
        "text": wrong, "first_equation_correct": 2 + 3 == 6,
        "first_equation_correct_value": 2 + 3,
        "second_equation_correct": 6 - 1 == 4, "second_equation_correct_value": 6 - 1,
        "final_text_answer_matches_question": 4 == 2 + 3 - 1,
        "all_wrong_process_bytes_and_eos_are_targets": True,
        "target_ids": wrong_target,
    },
    "question_overlap_if_versions_split": sorted({question} & {question}),
    "scope": "確定性資料與代價核對；沒有訓練模型，沒有測量解題品質、速度或跨平台行為。",
}, ensure_ascii=False, indent=2))
