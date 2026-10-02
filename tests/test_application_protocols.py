"""只測協議拒絕、真實工具執行與受控公告介入；不測模型收斂。"""

import copy
import json
from types import SimpleNamespace

import pytest

from scripts.course_experiments import applications as app
from tiny_perceptron.data import ByteTokenizer


@pytest.mark.parametrize(
    "raw",
    [
        '{"name":"add","arguments":{"a":NaN,"b":3}}',
        '{"name":"add","arguments":{"a":Infinity,"b":3}}',
        '{"name":"add","arguments":{"a":1e999,"b":3}}',
        '{"name":"add","arguments":{"a":-1e999,"b":3}}',
        '{"done":true,"answer":1e999}',
        '{"name":"add","name":"multiply","arguments":{"a":2,"b":3}}',
        '{"name":"add","arguments":{"a":2,"a":3,"b":3}}',
        r'{"name":"add","arguments":{"a":2,"\u0061":3,"b":3}}',
    ],
)
def test_parser_rejects_nonfinite_numbers_and_all_duplicate_object_keys(raw):
    with pytest.raises(ValueError):
        app._parse_json_action(raw)


def _inject_protocol_output(monkeypatch, outputs):
    """故障注入供 controller 單元測試；這不是模型能力或生成成績。"""
    pending = iter(outputs)
    tokenizer = ByteTokenizer()

    def injected(_model, messages, _ctx, **_kwargs):
        text, invalid = next(pending)
        return {
            "unit_test_injected_output": True,
            "messages": copy.deepcopy(messages),
            "generated_tokens": len(tokenizer.encode(text)) + len(invalid) + 1,
            "samples": [
                {
                    "generated": text,
                    "generated_ids": tokenizer.encode(text) + invalid + [tokenizer.eos_id],
                    "invalid_special_tokens": invalid,
                    "eos": True,
                }
            ],
        }

    monkeypatch.setattr(app, "_sample", injected)


def _record():
    return {
        "family": "pair:2:3",
        "operation": "multiply",
        "a": 2,
        "b": 3,
        "answer": 6,
        "messages": [
            {"role": "system", "content": app._TOOLS_SYSTEM},
            {"role": "user", "content": "CALC:multiply(2,3)"},
        ],
    }


def test_exponent_overflow_argument_never_executes_tool(monkeypatch):
    _inject_protocol_output(monkeypatch, [('{"name":"multiply","arguments":{"a":1e999,"b":3}}', [])])
    episode = app._tool_episode(None, _record(), SimpleNamespace())
    assert episode["status"] == "invalid_request" and episode["actual_tool_calls"] == 0
    assert episode["trace"][0]["executed"] is False
    json.dumps(episode, allow_nan=False)


def test_finite_arguments_overflow_records_actual_execution_and_serializable_evidence(monkeypatch):
    _inject_protocol_output(monkeypatch, [('{"name":"multiply","arguments":{"a":1e308,"b":1e308}}', [])])
    episode = app._tool_episode(None, _record(), SimpleNamespace())
    assert episode["status"] == "invalid_tool_result" and episode["actual_tool_calls"] == 1
    event = episode["trace"][0]
    assert event["executed"] is True and event["finite_result"] is False
    assert event["tool_result"] == "inf" and "overflow" in event["error"]
    assert not episode["correct"] and not episode["required_tool_behavior"]
    json.dumps(episode, allow_nan=False)


def test_valid_request_executes_and_final_uses_the_actual_result(monkeypatch):
    _inject_protocol_output(
        monkeypatch, [('{"name":"multiply","arguments":{"a":2,"b":3}}', []), ('{"done":true,"answer":6}', [])]
    )
    episode = app._tool_episode(None, _record(), SimpleNamespace())
    assert episode["status"] == "done" and episode["correct"]
    assert episode["actual_tool_calls"] == 1
    assert episode["trace"][0]["finite_result"] is True and episode["trace"][0]["tool_result"] == 6
    assert episode["trace"][1]["generation"]["messages"][-1] == {"role": "user", "content": "TOOL_RESULT:6"}


def test_decoded_done_without_tool_call_is_not_arithmetic_tool_success(monkeypatch):
    _inject_protocol_output(monkeypatch, [('{"done":true,"answer":6}', [])])
    episode = app._tool_episode(None, _record(), SimpleNamespace())
    assert episode["answer_correct"] is True and episode["correct"] is False
    assert episode["actual_tool_calls"] == 0


def test_hidden_special_tokens_cannot_become_valid_tool_success(monkeypatch):
    _inject_protocol_output(monkeypatch, [('{"name":"multiply","arguments":{"a":2,"b":3}}', [4])])
    episode = app._tool_episode(None, _record(), SimpleNamespace())
    assert episode["status"] == "invalid_request" and episode["actual_tool_calls"] == 0


def test_decoded_answers_with_hidden_control_ids_fail_rag_and_reasoning():
    sample = {"generated": "A7[D2]", "invalid_special_tokens": [4], "eos": True}
    score = app._grounding(sample, [{"id": "D2", "text": "K001 address=A7"}], "A7[D2]")
    assert not score["exact_match"] and not score["citation_valid"] and not score["supported_by_cited_source"]
    sample["generated"] = "2+3=5;5+1=6;answer=6"
    reasoning = app._verify_reasoning(sample, {"a": 2, "b": 3, "c": 1, "truth": 6}, "steps")
    assert not reasoning["final_correct"] and not reasoning["fully_verified"]


def test_paired_counterfactuals_change_only_one_context_field_and_preserve_fact():
    fact = {"family": "K001", "key": "K001", "address": "A9", "source": "D9"}
    before = copy.deepcopy(fact)
    cases = app._rag_counterfactual_contexts(fact)
    assert fact == before
    address = cases["changed_address_context"]
    source = cases["changed_source_context"]
    assert address["fact"] == before | {"address": "A0"}
    assert source["fact"] == before | {"source": "D0"}
    assert address["documents"] == [{"id": "D9", "text": "K001 address=A0"}]
    assert source["documents"] == [{"id": "D0", "text": "K001 address=A9"}]
    for case in cases.values():
        expected = f"{case['fact']['address']}[{case['fact']['source']}]"
        score = app._grounding(
            {"generated": expected, "invalid_special_tokens": [], "eos": True}, case["documents"], expected
        )
        assert score["exact_match"] and score["supported_by_cited_source"]
