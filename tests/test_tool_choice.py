"""Check split integrity and routing evidence, not neural-network convergence."""

import pytest

from scripts.course_experiments.tool_choice import build_records, parse_action, split_records, summarize


def test_unordered_operands_and_all_context_variants_stay_in_one_split():
    splits = split_records(build_records())
    owners = {}
    prompts = {}
    for name, rows in splits.items():
        for row in rows:
            pair = tuple(sorted(row["operands"]))
            owners.setdefault(pair, set()).add(name)
            prompt = tuple(message["content"] for message in row["messages"][:-1])
            prompts.setdefault(prompt, set()).add(name)
    assert all(len(names) == 1 for names in owners.values())
    assert all(len(names) == 1 for names in prompts.values())
    assert owners[(1, 2)] == {"test"}
    assert [len(rows) for rows in splits.values()] == [704, 80, 96]


def test_heldout_paraphrases_never_appear_in_training_templates():
    primary = build_records()
    diagnostic = build_records(paraphrase=True)
    primary_questions = {row["messages"][1]["content"] for row in primary}
    assert not primary_questions.intersection(row["messages"][1]["content"] for row in diagnostic)
    assert all(
        not any(marker in row["messages"][1]["content"] for marker in ("CALC", "COPY", "DIRECT", "TOOL", "ASK"))
        for row in primary + diagnostic
    )


def test_tool_unavailable_changes_arithmetic_choice_but_not_copy_or_explanation():
    rows = [row for row in build_records() if row["family"] == "pair:1:2"]
    by_prompt = {}
    for row in rows:
        by_prompt.setdefault(row["messages"][1]["content"], {})[row["calculator_available"]] = row
    arithmetic = by_prompt["1 加 2 等於多少？"]
    assert arithmetic[True]["expected_action"] == "TOOL"
    assert arithmetic[False]["expected_action"] == "ASK"
    for variants in by_prompt.values():
        if variants[True]["intent"] in ("copy", "explanation"):
            assert variants[True]["expected_action"] == variants[False]["expected_action"] == "DIRECT"
        if variants[True]["intent"] == "missing_quantity":
            assert variants[True]["expected_action"] == variants[False]["expected_action"] == "ASK"


@pytest.mark.parametrize(
    "raw,eos,specials", [("TOOL ", True, []), ("TOOL", False, []), ("TOOL", True, [4]), ("TOOL\n", True, [])]
)
def test_decoded_label_alone_does_not_hide_protocol_errors(raw, eos, specials):
    assert parse_action({"generated": raw, "eos": eos, "invalid_special_tokens": specials}) is None


def test_accuracy_does_not_hide_missed_unnecessary_or_unavailable_calls():
    rows = [
        {"expected_action": "TOOL", "intent": "numerical_addition", "calculator_available": True},
        {"expected_action": "DIRECT", "intent": "copy", "calculator_available": True},
        {"expected_action": "ASK", "intent": "numerical_addition", "calculator_available": False},
    ]
    score = summarize(rows, ["DIRECT", "TOOL", "TOOL"])
    assert score["accuracy"]["numerator"] == 0
    assert score["needed_tool_missed"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert score["unnecessary_tool_selected"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert score["unavailable_tool_selected"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert score["confusion"]["ASK"] == {"TOOL": 1}
