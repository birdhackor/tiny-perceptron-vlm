"""Independent, bounded CPU checks of the exact 7.18 data-card fence.

No model, sampling, optimization or externally downloaded dataset is used.
The changed examples check material contracts, not learning effectiveness.
"""
import ast
import contextlib
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
raw = (base / "fence-1.py").read_bytes()
assert hashlib.sha256(raw).hexdigest() == "a7875a3c109380647be58d346a0f09e870e8dc503cc5f431ca8da6fe68fd821b"
capture = io.StringIO()
namespace = {"__name__": "__main__"}
with contextlib.redirect_stdout(capture):
    exec(compile(raw, "original-course-7.18-fence-1", "exec"), namespace)
expected = "示範給什麼 3\n偏好比較什麼 3 勝過 答案是3喔！\n作答後得到什麼 4 0\n"
assert capture.getvalue() == expected
demonstration = namespace["demonstration"]
preference = namespace["preference"]
feedback = namespace["feedback"]
assert demonstration["question"] == preference["question"] == feedback["question"]
assert int(demonstration["answer"]) == 1 + 2 == 3
assert int(feedback["sampled_answer"]) != 1 + 2
assert feedback["reward"] == 0
tree = ast.parse(raw)
calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
assert len(calls) == 3
assert all(isinstance(node.func, ast.Name) and node.func.id == "print" for node in calls)

# Meaningful change: sampled_answer is a field, not an implemented grader.
changed_feedback = dict(feedback, sampled_answer="3")
assert changed_feedback["reward"] == 0
assert feedback["sampled_answer"] == "4"
rescored_feedback = dict(changed_feedback, reward=1)
assert rescored_feedback["reward"] == 1

# Meaningful change: the prompt asks for an explanation, so the ideal data changes.
changed_question = "用一句話解釋1+2"
changed_demo = {"question": changed_question, "answer": "1+2表示把1個和2個合起來，共有3個。"}
assert changed_demo["answer"] != demonstration["answer"]
swapped = {"question": changed_question, "chosen": preference["rejected"], "rejected": preference["chosen"]}
assert swapped["chosen"] == "答案是3喔！"
# This exact string still supplies only the answer, not an explanation of addition;
# the assertion checks bytes only. The semantic judgment is recorded in notes.md.
assert "合起來" not in swapped["chosen"]

# Boundary: dictionary lookup is exact, and no missing field is automatically scored.
try:
    feedback["computed_reward"]
except KeyError:
    missing_key = "KeyError"
else:
    raise AssertionError("dictionary missing-key behavior changed")

result = {
    "original_stdout": capture.getvalue(),
    "original_fence_sha256": hashlib.sha256(raw).hexdigest(),
    "original_calls": ["print", "print", "print"],
    "original_cards": {"demonstration": demonstration, "preference": preference, "feedback": feedback},
    "arithmetic": {"operands": [1, 2], "sum": 1 + 2, "wrong_answer": 4, "stored_reward": 0},
    "changed_answer_without_regrading": changed_feedback,
    "explicit_manual_rescoring": rescored_feedback,
    "changed_prompt_demonstration": changed_demo,
    "swapped_preference_under_changed_prompt": swapped,
    "missing_key_boundary": missing_key,
    "tolerance": "Exact Unicode text equality; integer arithmetic equality; no rounding",
    "denominators": "3 manually authored cards and 3 print calls; no samples, tokens, steps or empirical score",
    "environment": {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(), "device": "CPU; Python dictionaries only"},
    "scope": "Only data-card representation and observed Python behavior; no sampling, training, quality evaluator, or method efficacy is demonstrated",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
