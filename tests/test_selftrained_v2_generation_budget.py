"""V2 generation ceiling and exhausted-history outcomes, with synthetic CPU data."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.selftrained import evaluate, train
from tiny_perceptron.selftrained.dataset import RecordEncoder, read_records
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer
from tiny_perceptron.selftrained.tools import run_tool_loop, serialize_tool_call, serialize_tool_result


@pytest.fixture
def tokenizer():
    return CharacterTokenizer.build(["好。秘密答案"], required_chars="".join(chr(i) for i in range(32, 127)))


def example(messages):
    return {
        "id": "independent-synthetic-source",
        "group_id": "independent-synthetic-source",
        "split": "validation",
        "task": "text",
        "messages": messages + [{"role": "assistant", "content": "好。"}],
        "supervision": {"format": "one_sentence", "semantic_all": ["好"]},
    }


def tiny_model(tokenizer):
    torch.manual_seed(31)
    torch.set_num_threads(2)
    return LimitedAssistant(
        SelftrainedConfig(
            vocab_size=tokenizer.vocab_size,
            width=16,
            layers=1,
            heads=2,
            kv_heads=1,
            ffn_hidden=32,
            experts=2,
            max_length=512,
        )
    )


@pytest.mark.parametrize(
    "history",
    [
        [{"role": "user", "content": "q"}],
        [{"role": "system", "content": "q" * 120}, {"role": "user", "content": "u" * 120}],
        [
            {"role": "user", "content": "q" * 180},
            {"role": "assistant", "content": "v" * 90},
            {"role": "user", "content": "w" * 106},
        ],
    ],
)
def test_normal_history_preserves_actual_neural_inputs_kwargs_and_tokens(tmp_path, tokenizer, history):
    encoder = RecordEncoder(tokenizer, tmp_path, 512)
    record = example(history)
    old_encoded = encoder.encode(record, generation=True, messages=history)
    assert len(old_encoded["input_ids"]) <= 384
    model = tiny_model(tokenizer)
    # The exact former call, on the same actual neural model and cached generation.
    expected = model.generate(
        torch.tensor([old_encoded["input_ids"]]),
        modalities=[encoder.to_device(old_encoded["modalities"])],
        max_new_tokens=128,
        eos_id=tokenizer.eos_id,
    )[0].tolist()
    generator = evaluate.Generator(model, encoder, 128)
    output = generator.generate(record, history)
    call = generator.calls[-1]
    assert call["prompt_ids"] == old_encoded["input_ids"]
    assert call["generated_ids"] == expected
    assert output == tokenizer.decode(expected)
    assert call["requested_max_new_tokens"] == call["effective_max_new_tokens"] == 128
    assert encoder.context == model.config.max_length == 512
    assert call["neural_generation"] is True
    changed = copy.deepcopy(record)
    changed["messages"][-1]["content"] = "秘密答案" * 300
    changed["supervision"] = {"intent_id": 9, "ocr_text": "秘密答案"}
    generator.generate(changed, history)
    assert generator.calls[-1]["prompt_ids"] == call["prompt_ids"]
    assert generator.calls[-1]["generated_ids"] == call["generated_ids"]


def test_malformed_tool_first_output_leaves_exact106_second_budget(tmp_path, tokenizer):
    history = [
        {"role": "system", "content": "q" * 175},
        {"role": "user", "content": "u" * 10},
        {"role": "assistant", "content": "v" * 11},
        {"role": "user", "content": "w" * 13},
    ]
    record = example(history)
    record["task"] = "tool_call"
    encoder = RecordEncoder(tokenizer, tmp_path, 512)

    class ExplicitBoundaryFixture:
        def __init__(self):
            self.calls = []

        def generate(self, input_ids, **kwargs):
            self.calls.append((input_ids.clone(), kwargs))
            output = "{" + "x" * 127 if len(self.calls) == 1 else "好。"
            tokens = tokenizer.encode(output)
            if len(self.calls) == 2:
                tokens.append(tokenizer.eos_id)
            assert len(tokens) <= kwargs["max_new_tokens"]
            return torch.tensor([tokens])

    fixture = ExplicitBoundaryFixture()
    generator = evaluate.Generator(fixture, encoder, 128)
    trace = run_tool_loop(lambda messages: generator.generate(record, messages), history)
    assert trace["status"] == "invalid_call" and not trace["executed"]
    assert trace["initial_output"] == "{" + "x" * 127
    assert trace["tool_message"]["content"] == '{"tool":"calculator","ok":false,"error":"invalid_call"}'
    first, second = generator.calls
    assert first["prompt_token_count"] == 219
    assert second["prompt_token_count"] == 406
    assert second["remaining_context_tokens"] == second["effective_max_new_tokens"] == 106
    assert fixture.calls[1][1]["max_new_tokens"] == 106
    assert second["prompt_messages"] == history + [
        {"role": "assistant", "content": trace["initial_output"]},
        trace["tool_message"],
    ]
    assert trace["final_output"] == "好。"
    assert encoder.context == 512


@pytest.mark.parametrize("content_length", [508, 509])
def test_exhausted_history_retains_all_ids_without_calling_model(tmp_path, tokenizer, content_length):
    history = [{"role": "user", "content": "q" * content_length}]
    record = example(history)
    encoder = RecordEncoder(tokenizer, tmp_path, 512)

    class NeverGenerate:
        def generate(self, *args, **kwargs):
            pytest.fail("Exhausted history cannot invoke model.generate")

    generator = evaluate.Generator(NeverGenerate(), encoder, 128)
    assert generator.generate(record, history) == ""
    call = generator.calls[-1]
    assert call["prompt_token_count"] == content_length + 4
    assert call["prompt_ids"] == [tokenizer.bos_id, tokenizer.user_id] + tokenizer.encode(history[0]["content"]) + [
        tokenizer.eos_id,
        tokenizer.assistant_id,
    ]
    assert call["prompt_messages"] == history
    assert call["effective_max_new_tokens"] == call["remaining_context_tokens"] == 0
    assert call["generation_status"] == call["stop_reason"] == "context_budget_exhausted"
    assert call["neural_generation"] is False and call["generated_ids"] == []
    trace = {"generation_failure": "context_budget_exhausted"}
    score = evaluate.score_reply(record, "", trace)
    assert score["exact"] is score["semantic"] is score["format"] is False
    assert encoder.context == 512


@pytest.fixture
def genuine_checkpoint(tmp_path):
    call = serialize_tool_call("add", 2, 3)
    records = [
        {
            "id": "original-train-source",
            "group_id": "original-train-source",
            "split": "train",
            "task": "text",
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "好。"}],
            "supervision": {},
        },
        {
            "id": "independent-validation-source",
            "group_id": "independent-validation-source",
            "split": "validation",
            "task": "tool_call",
            "messages": [{"role": "user", "content": "q" * 390}, {"role": "assistant", "content": call}],
            "supervision": {"format": "one_sentence", "expected_call": json.loads(call), "expected_result": 5},
        },
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"width": 16, "layers": 1, "heads": 2, "kv_heads": 1, "ffn_hidden": 32, "experts": 2}))
    output = tmp_path / "actual-one-step-training"
    train.main(
        [
            "--records",
            str(path),
            "--asset-dir",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--stage",
            "sft",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--context",
            "512",
            "--config",
            str(config),
        ]
    )
    return tmp_path, path, output / "best.pt"


def test_protocol_binds_ceiling_policy_and_rejects_old_or_missing_policy(genuine_checkpoint):
    root, path, checkpoint_path = genuine_checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    args = SimpleNamespace(
        checkpoint=checkpoint_path, model_dir=None, records=[path], asset_dir=root, max_new_tokens=128, controls="none"
    )
    current = evaluate.protocol_contents(args, checkpoint, read_records([path]))
    evaluate.validate_protocol(current, current)
    for key, change in (
        ("generation_budget_policy", "old-fixed-reservation"),
        ("generation_budget_policy", None),
        ("version", "selftrained-generation-v1"),
    ):
        altered = copy.deepcopy(current)
        if change is None:
            del altered[key]
        else:
            altered[key] = change
        with pytest.raises(ValueError, match=key):
            evaluate.validate_protocol(altered, current)


def test_main_commits_exhausted_second_hop_as_failure_without_second_neural_call(
    genuine_checkpoint, monkeypatch, capsys
):
    root, path, checkpoint = genuine_checkpoint
    calls = []

    def explicit_fixture_generation(model, input_ids, **kwargs):
        # Marked artificial boundary fixture, not a model-capability claim.
        calls.append({"length": input_ids.shape[1], "maximum": kwargs["max_new_tokens"]})
        tokenizer = CharacterTokenizer.from_dict(torch.load(checkpoint, weights_only=False)["tokenizer"])
        return torch.tensor([tokenizer.encode("{" + "q" * (kwargs["max_new_tokens"] - 1))])

    monkeypatch.setattr(LimitedAssistant, "generate", explicit_fixture_generation)
    output = root / "budget-validation"
    metrics = evaluate.main(
        [
            "--records",
            str(path),
            "--asset-dir",
            str(root),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--controls",
            "none",
            "--max-new-tokens",
            "128",
        ]
    )
    row = json.loads((output / "outputs.jsonl").read_text())
    receipt = json.loads((output / "evaluation-receipt.json").read_text())
    assert calls == [{"length": 394, "maximum": 118}]
    assert row["trace"]["initial_output"] == "{" + "q" * 117
    assert row["trace"]["tool_result"]["error"] == "invalid_call"
    assert row["trace"]["final_output"] == ""
    assert row["trace"]["generation_failure"] == "context_budget_exhausted"
    assert row["model_generations"][1]["prompt_token_count"] == 571
    assert row["model_generations"][1]["neural_generation"] is False
    assert row["score"]["semantic"] is row["score"]["format"] is row["score"]["tool_roundtrip"] is False
    assert metrics["evaluation_complete"] is True
    assert metrics["per_task_final_reply"]["tool_call"]["semantic"] == {"numerator": 0, "denominator": 1, "rate": 0.0}
    assert metrics["generation_budget"]["exhausted_attempts"] == 1
    assert receipt["status"] == "complete" and receipt["completed_count"] == 1
    capsys.readouterr()


def test_zero_control_does_not_claim_change_when_either_reply_had_no_budget():
    record = example([{"role": "user", "content": "q"}])
    trace = {"final_output": "", "generation_failure": "context_budget_exhausted"}
    row = {
        "record": record,
        "trace": trace,
        "score": evaluate.score_reply(record, "", trace),
        "perception": [],
        "routing": [],
        "generation_budget_failure": True,
        "zero_output": "好。",
        "zero_generation_budget_failure": False,
    }
    assert evaluate.summarize([row])["controls"]["zero_modality"] == {"count": 1, "changed": 0}
    row["generation_budget_failure"] = False
    row["zero_generation_budget_failure"] = True
    assert evaluate.summarize([row])["controls"]["zero_modality"] == {"count": 1, "changed": 0}
    row["zero_generation_budget_failure"] = False
    assert evaluate.summarize([row])["controls"]["zero_modality"] == {"count": 1, "changed": 1}


@pytest.mark.parametrize("user_length, expected_main_prompt, dependency_correct", [(1, 140, 1), (373, 512, 0)])
def test_valid_executor_minus1_to0_budget_boundary_retains_raw_without_false_dependency(
    tmp_path, tokenizer, user_length, expected_main_prompt, dependency_correct
):
    history = [{"role": "user", "content": "q" * user_length}]
    record = example(history)
    record["task"] = "tool_call"
    call = serialize_tool_call("subtract", 1, 2)
    record["messages"][-1]["content"] = call
    initial = call + " " * (88 - len(call))
    record["supervision"]["expected_call"] = json.loads(call)
    record["supervision"]["expected_result"] = -1
    encoder = RecordEncoder(tokenizer, tmp_path, 512)

    class ExplicitBoundaryFixture:
        def generate(self, input_ids, **kwargs):
            prompt = tokenizer.decode(input_ids[0])
            # Explicit fixture responses to public tool return, not capability evidence.
            output = initial if tokenizer.tool_id not in input_ids[0] else "0" if '"result":0' in prompt else "-1"
            ids = tokenizer.encode(output)
            assert len(ids) <= kwargs["max_new_tokens"]
            return torch.tensor([ids])

    generator = evaluate.Generator(ExplicitBoundaryFixture(), encoder, 128)
    trace = run_tool_loop(lambda messages: generator.generate(record, messages), history)
    assert trace["executed"] and trace["tool_result"]["result"] == -1
    main_failure = generator.calls[-1]["generation_status"] == "context_budget_exhausted"
    assert generator.calls[-1]["prompt_token_count"] == expected_main_prompt
    if main_failure:
        trace["generation_failure"] = "context_budget_exhausted"
        assert generator.calls[-1]["neural_generation"] is False
    replay_message = {"role": "tool", "content": serialize_tool_result({"tool": "calculator", "ok": True, "result": 0})}
    replay_output = generator.generate(record, history + [{"role": "assistant", "content": initial}, replay_message])
    assert replay_output == "0"
    assert generator.calls[-1]["prompt_token_count"] == expected_main_prompt - 1
    assert generator.calls[-1]["neural_generation"] is True
    if main_failure:
        assert generator.calls[-1]["effective_max_new_tokens"] == 1
    row = {
        "record": record,
        "trace": trace,
        "score": evaluate.score_reply(record, trace["final_output"], trace),
        "perception": [],
        "routing": [],
        "generation_budget_failure": main_failure,
        "replay_tool_message": replay_message,
        "replay_output": replay_output,
        "replay_score": evaluate.numeric_reply_value(replay_output) == 0,
        "replay_generation_budget_failure": False,
    }
    assert evaluate.summarize([row])["controls"]["tool_return_dependency"] == {
        "count": 1,
        "correct": dependency_correct,
    }
    if not main_failure:
        assert row["score"]["tool_roundtrip"] is True
        row["replay_generation_budget_failure"] = True
        assert evaluate.summarize([row])["controls"]["tool_return_dependency"] == {"count": 1, "correct": 0}
