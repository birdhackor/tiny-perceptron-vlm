"""Bounded independent checks of the raw 8.11 fence; no model or training."""

import contextlib
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raw = (ROOT / "inputs/fence-1.py").read_bytes()
expected = (
    "原順序選 correct 2+2=4\n"
    "交換後選 wrong 2+2=5\n"
    "內容身分一致 False\n"
)


def run(code, name):
    namespace = {"__name__": "__main__"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, name, "exec"), namespace)
    return namespace, stdout.getvalue()


namespace, original_stdout = run(raw, "original-raw-fence-1.py")
assert original_stdout == expected
answers = namespace["answers"]
original_order = namespace["original_order"]
swapped_order = namespace["swapped_order"]
assert original_order == ["correct", "wrong"]
assert swapped_order == ["wrong", "correct"]
assert original_order is not swapped_order
assert 2 + 2 == 4 and 2 + 2 != 5

ideal = raw.replace(b"return order[0]", b'return "correct"')
assert raw.count(b"return order[0]") == 1
(ROOT / "ideal-variant.py").write_bytes(ideal)
ideal_namespace, ideal_stdout = run(ideal, "exercise-return-correct.py")
assert ideal_stdout == expected.replace(
    "交換後選 wrong 2+2=5", "交換後選 correct 2+2=4"
).replace("一致 False", "一致 True")

labels = ["A", "B"]


def stable_digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


controls = {
    "question": "2+2=?",
    "rubric": "Prefer the answer containing the correct arithmetic result.",
    "answers": answers,
}
before_controls_sha256 = stable_digest(controls)
records = []
for name, choose in [
    ("first_position", namespace["choose_first"]),
    ("correct_identity_handwritten", ideal_namespace["choose_first"]),
    ("wrong_identity_handwritten", lambda order: "wrong"),
    ("tie_handwritten", lambda order: None),
]:
    judgments = []
    for order in [original_order, swapped_order]:
        display = dict(zip(labels, order, strict=True))
        selected_identity = choose(order)
        selected_label = (
            None if selected_identity is None else labels[order.index(selected_identity)]
        )
        normalized_identity = (
            None if selected_label is None else display[selected_label]
        )
        assert normalized_identity == selected_identity
        judgments.append(
            {
                "order": order,
                "display_labels_to_identity": display,
                "raw_selected_label": selected_label,
                "normalized_identity": normalized_identity,
                "selected_text": (
                    None if selected_identity is None else answers[selected_identity]
                ),
                "correct": selected_identity == "correct",
            }
        )
    records.append(
        {
            "judge_rule": name,
            "judgments": judgments,
            "same_display_label": judgments[0]["raw_selected_label"]
            == judgments[1]["raw_selected_label"],
            "same_content_identity": judgments[0]["normalized_identity"]
            == judgments[1]["normalized_identity"],
        }
    )

after_controls_sha256 = stable_digest(controls)
assert before_controls_sha256 == after_controls_sha256
assert records[0]["same_display_label"] is True
assert records[0]["same_content_identity"] is False
assert records[1]["same_display_label"] is False
assert records[1]["same_content_identity"] is True
assert records[2]["same_content_identity"] is True
assert all(not item["correct"] for item in records[2]["judgments"])
assert records[3]["same_content_identity"] is True
assert all(item["normalized_identity"] is None for item in records[3]["judgments"])

reversed_start = raw.replace(
    b'original_order = ["correct", "wrong"]',
    b'original_order = ["wrong", "correct"]',
)
(ROOT / "reversed-start-variant.py").write_bytes(reversed_start)
_, reversed_stdout = run(reversed_start, "reversed-start-variant.py")
assert reversed_stdout == (
    "原順序選 wrong 2+2=5\n"
    "交換後選 correct 2+2=4\n"
    "內容身分一致 False\n"
)

result = {
    "status": "passed",
    "environment": {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "device": "CPU; pure Python, no tensors or model inference",
        "scope": "two fixed strings; deterministic handwritten functions only",
    },
    "raw_fence_sha256": hashlib.sha256(raw).hexdigest(),
    "original_stdout": original_stdout,
    "ideal_variant_sha256": hashlib.sha256(ideal).hexdigest(),
    "ideal_stdout": ideal_stdout,
    "reversed_start_stdout": reversed_stdout,
    "controls": controls,
    "before_controls_sha256": before_controls_sha256,
    "after_controls_sha256": after_controls_sha256,
    "records": records,
    "checks": {
        "raw_fence_matches_expected_stdout": True,
        "reverse_preserves_original_list": True,
        "dictionary_content_unchanged": True,
        "arithmetic_2_plus_2_checked": True,
        "exercise_replaces_only_function_return": True,
        "same_display_label_can_mean_different_content": True,
        "different_display_labels_can_mean_same_content": True,
        "consistent_choice_can_still_be_wrong": True,
        "explicit_tie_maps_to_no_winner": True,
        "reversed_start_is_still_position_sensitive": True,
    },
    "limitations": [
        "The added controls record a testing contract; the original choose_first function does not receive question or rubric.",
        "A fixed correct identity is a handwritten ideal rule and does not evaluate arithmetic semantically.",
        "The tie convention None is a bounded example, not a universal evaluator output format.",
        "No external judge, empirical model evaluation, parameter update, training, or weight inference ran.",
    ],
}
(ROOT / "cpu-result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("Original fence:\n" + original_stdout, end="")
print("Specified exercise:\n" + ideal_stdout, end="")
print("Reversed starting order:\n" + reversed_stdout, end="")
print("All 10 bounded checks passed; model measurements: none.")
