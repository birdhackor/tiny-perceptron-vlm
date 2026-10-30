"""推論隔離、錯路由不改派，以及串流介面不污染呼叫者的 grad 模式。"""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tiny_perceptron.omni_products.av import AVModel
from tiny_perceptron.omni_products.choice import ChoiceModel
from tiny_perceptron.omni_products.controlled_diagnostics import controlled_tool_reply
from tiny_perceptron.omni_products.data import Event
from tiny_perceptron.omni_products.inference import CJudge, OmniAssistant
from tiny_perceptron.omni_products.materialize import public_payload
from tiny_perceptron.omni_products.media_bindings import (
    counterfactual_tool_panel,
    public_input,
    retention_panel,
    retention_result,
)
from tiny_perceptron.omni_products.questions import panel_c
from tiny_perceptron.omni_products.routing import Request
from tiny_perceptron.omni_products.sound import ProgramTalker, SoundCodec


def test_public_prediction_ignores_research_gold_and_uses_same_float64_calibration_as_policy():
    judge = CJudge(ChoiceModel().eval(), temperature=2)
    question = panel_c("dev")[0]["questions"]["choice"]
    first = judge.predict(question)
    alternative = next(name for name in ("none", "stop", "wait", "record", "play", "save") if name != question.gold)
    second = judge.predict(replace(question, gold=alternative))
    assert first["logits"] == second["logits"]
    assert first["probabilities"] == second["probabilities"]
    assert abs(sum(first["probabilities"]) - 1) < 1e-12
    with pytest.raises(ValueError):
        CJudge(ChoiceModel().eval(), temperature=float("nan"))


def make_assistant():
    return OmniAssistant(AVModel(), ChoiceModel(), SoundCodec(), ProgramTalker())


def test_wrong_auto_route_stays_a_failure_without_falling_back_to_gold_task():
    assistant = make_assistant()
    assistant.c.route = lambda request: {"answer": "A", "p_max": 0.99}
    result = assistant.auto(Request("計算三個整數的和：1、2、3。"), {"numbers": [1, 2, 3]})
    assert result["route"] == "A" and result["contract_error"] and result["result"] is None


def test_old_reply_is_the_real_fixed_old_a_entry_and_no_current_gold_is_accepted():
    assistant = make_assistant()
    calls = []
    marker = {"answer": "actual old reply"}
    assistant.old_a = SimpleNamespace(reply=lambda *args, **kwargs: calls.append((args, kwargs)) or marker)
    public = {
        "task": "voice_qa",
        "messages": [{"role": "user", "content": "請回答這段語音問題。"}],
        "metadata": {"audio": "audio/fixed.wav"},
    }
    assert assistant.old_reply(public) is marker
    assert calls[0][1]["task"] == "voice_qa" and calls[0][1]["audio"] == "audio/fixed.wav"


def test_chunks_are_actual_causal_decoder_blocks_and_leave_caller_grad_mode_unchanged():
    assistant = make_assistant()
    codes = torch.tensor([0, 2, 1])
    assistant.talker.generate = lambda _: {"codes": codes, "eos": True, "capped": False}
    program = (Event(220, 100),)
    stream = assistant.sound_chunks(program)
    with torch.enable_grad():
        first = next(stream)
        assert torch.is_grad_enabled() and not first["samples"].requires_grad
        remaining = list(stream)
        assert torch.is_grad_enabled()
    assert first["ready_seconds"] >= first["whole_code_generation_seconds"]
    actual = torch.cat([row["samples"] for row in (first, *remaining)])
    with torch.no_grad():
        assert torch.equal(actual, assistant.codec.decode_codes(codes))


def test_same_state_b_information_cannot_authorize_a_choice():
    row = panel_c("policy")[-1]
    judge = CJudge(ChoiceModel().eval(), threshold=0.5)
    with pytest.raises(ValueError, match="同entity"):
        judge.judge(row["questions"]["choice"], row["questions"]["info"])
    judge.judge(row["questions"]["choice"], row["questions"]["choice_info"])


def test_tool_retention_starts_with_user_and_scores_initial_call_and_live_reply_separately():
    gold_call = {"tool": "calculator", "operation": "add", "a": 2, "b": 3}
    record = {
        "task": "tool_reply",
        "messages": [
            {"role": "user", "content": "請把2和3相加。"},
            {"role": "assistant", "content": '{"tool":"calculator","operation":"add","a":2,"b":3}'},
            {"role": "tool", "content": '{"tool":"calculator","ok":true,"result":5}'},
            {"role": "assistant", "content": "答案是5。"},
        ],
    }
    public = public_input(record)
    assert public["task"] == "tool_call" and public["tools"] is True
    assert public["messages"] == record["messages"][:1]
    response = {
        "answer": "答案是5。",
        "tool_trace": {"initial_output": record["messages"][1]["content"], "tool_call": gold_call, "executed": True},
    }
    row = {"task": "tool_call", "gold_tool_call": gold_call, "scoring_field": "initial"}
    assert retention_result(row, response)["correct"]
    row = {"task": "tool_reply", "gold": "答案是5。", "gold_tool_call": gold_call, "scoring_field": "answer"}
    assert retention_result(row, response)["correct"]
    response["tool_trace"]["executed"] = False
    assert not retention_result(row, response)["correct"]


def test_public_materialization_converts_media_and_discards_current_gold():
    payload = {
        "video_audio_definition": {"pattern": 13, "variant": 73, "panel": 1, "which": 0, "gold": "untrusted target"}
    }
    result = public_payload(payload)
    assert set(result) == {"video_audio"}
    result["video_audio"].validate()
    question = panel_c("dev")[0]["questions"]["choice"]
    from dataclasses import asdict

    source = {"questions": {kind: asdict(value) for kind, value in panel_c("dev")[0]["questions"].items()}}
    converted = public_payload(source)
    assert all(value.gold is None for value in converted["questions"].values())
    assert converted["questions"]["choice"].state == question.state


def test_live_retention_selects_actual_execution_even_when_counterfactual_record_sorts_first(tmp_path):
    call = '{"tool":"calculator","operation":"add","a":2,"b":3}'
    user = {"role": "user", "content": "請把2和3相加。"}

    def record(identifier, replay, value):
        return {
            "id": identifier,
            "group_id": "same-operands-parent",
            "task": "tool_reply",
            "messages": [
                user,
                {"role": "assistant", "content": call},
                {"role": "tool", "content": json.dumps({"tool": "calculator", "ok": True, "result": value})},
                {"role": "assistant", "content": f"答案是{value}。"},
            ],
            "supervision": {"replay_kind": replay, "actual_result": 5, "expected_result": value},
        }

    rows = [record("00-counterfactual", "counterfactual_test_only", 6), record("01-live", "actual_executor", 5)]
    manifest = {"records": []}
    for name in ("text-tools-test.jsonl", "vision-test.jsonl", "ocr-test.jsonl", "voice-test.jsonl"):
        raw = (
            ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
            if name.startswith("text-")
            else b""
        )
        (tmp_path / name).write_bytes(raw)
        manifest["records"].append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    live = retention_panel(tmp_path, manifest)
    controlled = counterfactual_tool_panel(tmp_path, manifest)
    assert len(live) == len(controlled) == 1
    assert live[0]["record"] == "01-live" and live[0]["gold"] == "答案是5。"
    assert live[0]["input"]["messages"] == [user]
    assert controlled[0]["record"] == "00-counterfactual" and controlled[0]["gold"] == "答案是6。"
    assert controlled[0]["provided_tool_result"] != controlled[0]["actual_executor_result"]


def test_controlled_tool_diagnostic_conditions_on_provided_prefix_without_claiming_real_execution():
    seen = []
    prefix = [
        {"role": "user", "content": "請相加。"},
        {"role": "assistant", "content": "earlier model call"},
        {"role": "tool", "content": '{"tool":"calculator","ok":true,"result":6}'},
    ]
    tokenizer = SimpleNamespace(eos_id=1, decode=lambda ids, **kwargs: "controlled continuation")

    def encode(record, **kwargs):
        seen.append((record, kwargs))
        return {"input_ids": [3, 4], "modalities": []}

    old = SimpleNamespace(
        encoder=SimpleNamespace(
            context=512, device="cpu", tokenizer=tokenizer, encode=encode, to_device=lambda value: value
        ),
        model=SimpleNamespace(generate=lambda *args, **kwargs: torch.tensor([[7, 1]])),
    )
    result = controlled_tool_reply(old, prefix)
    assert result["actual_executor_used"] is False
    assert seen[0][0] == {"id": "controlled-tool-result-diagnostic", "task": "tool_reply", "messages": prefix}
    assert set(seen[0][0]) == {"id", "task", "messages"}
    with pytest.raises(ValueError):
        controlled_tool_reply(old, prefix, max_new_tokens=1000)
