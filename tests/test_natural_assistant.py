"""Trust boundaries for the practical pretrained branch, without remote weights."""

import contextlib
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tiny_perceptron import natural_assistant as assistant


class CharacterProcessor:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize
        text = ""
        for message in messages:
            text += f"<{message['role']}>"
            text += "".join(part.get("text", "<image>") for part in message["content"])
            text += "</turn>"
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, *, text, return_tensors, images=None):
        assert return_tensors == "pt"
        ids = torch.tensor([[ord(character) for character in text[0]]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def manifest(tmp_path, rows=None, audio_rows=None):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_version": "test",
                "rows": rows or [],
                "audio_rows": audio_rows or [],
                "sources": [],
            }
        )
    )
    return path


def test_masking_keeps_only_final_assistant_answer_and_end_marker(tmp_path):
    Image.new("RGB", (8, 8), "red").save(tmp_path / "sample.png")
    row = {
        "id": "one",
        "user": "question",
        "answer": "final",
        "image": "sample.png",
        "history": [{"role": "user", "content": "previous"}, {"role": "assistant", "content": "old answer"}],
    }
    inputs = assistant.encode_training_row(CharacterProcessor(), row, tmp_path)
    labels = inputs["labels"][0]
    supervised = "".join(chr(value) for value in labels[labels != -100].tolist())
    assert supervised == "final</turn>"
    assert (labels == -100).sum() > 0
    with pytest.raises(ValueError, match="refusing silent multimodal truncation"):
        assistant.encode_training_row(CharacterProcessor(), row, tmp_path, max_tokens=8)


def test_template_mismatch_fails_instead_of_training_prompt_tokens(tmp_path):
    class BrokenProcessor(CharacterProcessor):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            text = super().apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
            return "DIFFERENT" + text if add_generation_prompt else text

    with pytest.raises(ValueError, match="not an exact prefix"):
        assistant.encode_training_row(BrokenProcessor(), {"id": "x", "user": "q", "answer": "a"}, tmp_path)


def test_manifest_rejects_family_leakage_and_duplicate_ids(tmp_path):
    rows = [
        {"id": "a", "user": "q", "answer": "a", "family": "original", "split": "train"},
        {"id": "b", "user": "q2", "answer": "a2", "family": "original", "split": "test"},
    ]
    with pytest.raises(ValueError, match="crosses splits"):
        assistant.load_manifest(manifest(tmp_path, rows))
    rows[1].update(family="other", id="a")
    with pytest.raises(ValueError, match="unique id"):
        assistant.load_manifest(manifest(tmp_path, rows))


def test_asset_escape_and_changed_bytes_are_detectable(tmp_path):
    outside = tmp_path.parent / "outside-natural-test.txt"
    outside.write_text("outside")
    with pytest.raises(ValueError, match="inside data_root"):
        assistant.asset_path("../outside-natural-test.txt", tmp_path)
    image = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "red").save(image)
    row = {"id": "a", "user": "q", "answer": "a", "split": "train", "image": "sample.png"}
    path = manifest(tmp_path, [row])
    before, _ = assistant.load_manifest(path)
    Image.new("RGB", (8, 8), "blue").save(image)
    after, _ = assistant.load_manifest(path)
    assert before["manifest_sha256"] == after["manifest_sha256"]
    assert before["asset_sha256"] != after["asset_sha256"]


def test_frozen_expected_asset_hash_is_verified_not_just_reported(tmp_path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "red").save(image)
    row = {"id": "a", "user": "q", "answer": "a", "split": "train", "image": "sample.png"}
    path = manifest(tmp_path, [row])
    content = json.loads(path.read_text())
    content["files"] = [{"path": "sample.png", "bytes": image.stat().st_size, "sha256": assistant.sha256(image)}]
    path.write_text(json.dumps(content))
    assistant.load_manifest(path)
    Image.new("RGB", (8, 8), "blue").save(image)
    with pytest.raises(ValueError, match="frozen manifest SHA"):
        assistant.load_manifest(path)
    content["files"][0]["sha256"] = assistant.sha256(image)
    content["files"][0]["bytes"] = 1
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="byte count"):
        assistant.load_manifest(path)


def test_value_hashes_detect_adapter_updates_and_frozen_sample_changes():
    class Small(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.arange(12, dtype=torch.bfloat16).reshape(3, 4), requires_grad=False)
            self.lora_A = torch.nn.Parameter(torch.ones(2, 3))
            self.lora_B = torch.nn.Parameter(torch.zeros(3, 2))

    model = Small()
    initial = assistant.adapter_tensor_hashes(model)
    frozen = assistant.frozen_parameter_samples(model)
    with torch.no_grad():
        model.lora_B[0, 0] = 0.5
    after = assistant.adapter_tensor_hashes(model)
    assert initial["lora_A"] == after["lora_A"]
    assert initial["lora_B"] != after["lora_B"]
    assert frozen == assistant.frozen_parameter_samples(model)
    with torch.no_grad():
        model.base.reshape(-1)[0] += 1
    assert frozen != assistant.frozen_parameter_samples(model)


def test_ocr_preserves_traditional_characters_and_reading_order():
    row = {"answer": "臺灣\n學生", "references": {"kind": "ocr_order", "text": "臺灣\n學生"}}
    assert assistant.score_output(row, "臺灣\n學生")["passed"]
    assert not assistant.score_output(row, "学生\n台湾")["passed"]
    assert not assistant.score_output(row, "臺灣學生")["passed"]
    row["references"] = {"kind": "ocr", "text": "臺灣\n學生"}
    assert assistant.score_output(row, "臺灣學生")["passed"]
    score = assistant.score_output({"answer": "a", "references": {"kind": "ocr"}}, "aaaa")
    assert score["errors"] == 3 and score["cer"] == 3
    empty = assistant.score_output({"answer": "", "references": {"kind": "ocr"}}, "幻覺")
    assert empty["cer"] is None and empty["empty_reference_false_positive"]


def test_asr_reports_raw_and_normalized_cer_without_script_correction():
    score = assistant.asr_cer_metrics("Ａ \n臺。", "A臺。")
    assert score["raw_errors"] == 3
    assert score["raw_reference_characters"] == 5
    assert score["raw_cer"] == 3 / 5
    assert score["errors"] == 0 and score["normalized_cer"] == 0
    assert score["reference_characters"] == 3
    assert score["normalized_reference"] == "A臺。"
    assert assistant.asr_cer_metrics("臺灣", "台湾")["errors"] == 2
    assert assistant.asr_cer_metrics("A。", "a")["errors"] == 2
    assert "no Traditional/Simplified conversion" in score["normalization"]


@pytest.mark.parametrize(
    ("suffix", "eos", "stop_reason", "truncated"),
    [([5, 42], True, "eos", False), ([5] * 127 + [42], True, "eos", False), ([5] * 128, False, "max_new_tokens", True)],
)
def test_asr_preserves_raw_generation_ids_eos_and_truncation(tmp_path, suffix, eos, stop_reason, truncated):
    import soundfile as sf

    path = tmp_path / "clip.wav"
    sf.write(path, [0.0] * 160, 16000)
    prefix = [1, 2, 3, 4]

    class Processor:
        def __call__(self, waveform, *, sampling_rate, return_tensors, return_attention_mask):
            assert sampling_rate == 16000 and return_tensors == "pt" and return_attention_mask
            return SimpleNamespace(input_features=torch.zeros(1, 80, 3000), attention_mask=torch.ones(1, 3000))

        def get_decoder_prompt_ids(self, *, language, task):
            assert (language, task) == ("chinese", "transcribe")
            return [(1, 2), (2, 3), (3, 4)]

        def batch_decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            assert skip_special_tokens and not clean_up_tokenization_spaces
            return ["真實生成的測試字串"]

    class Model:
        generation_config = SimpleNamespace(decoder_start_token_id=1, eos_token_id=42)

        def generate(self, features, **kwargs):
            assert kwargs["return_dict_in_generate"] is True
            assert kwargs["return_timestamps"] is False
            assert kwargs["max_new_tokens"] == 128
            assert kwargs["attention_mask"].shape == (1, 3000)
            return SimpleNamespace(sequences=torch.tensor([prefix + suffix]))

    record = assistant.transcribe(Model(), Processor(), path)
    assert record["raw_token_ids"] == prefix + suffix
    assert record["raw_token_count"] == 4 + len(suffix)
    assert record["generated_token_ids"] == suffix
    assert record["generated_token_count"] == len(suffix)
    assert record["ended_with_eos"] == eos
    assert record["stop_reason"] == stop_reason
    assert record["truncated"] == truncated
    assert record["transcript"] == "真實生成的測試字串"


def test_scene_rubric_does_not_pass_vacuously_or_match_inside_words():
    row = {
        "answer": "A cat sits on a bench",
        "references": {"kind": "facts", "fact_groups": [["cat", "貓"], ["bench", "椅子"]], "forbidden": ["dog"]},
    }
    assert assistant.score_output(row, "A CAT sits on the bench.")["passed"]
    assert not assistant.score_output(row, "catch a bench")["passed"]
    assert not assistant.score_output(row, "A cat and dog on a bench")["passed"]
    with pytest.raises(ValueError, match="nonempty groups"):
        assistant.score_output({"answer": "", "references": {"kind": "facts"}}, "anything")


def test_shuffled_training_order_is_reproducible_across_resume():
    rows = list(range(7))
    whole = [assistant.training_row_at(rows, step, 42) for step in range(21)]
    resumed = [assistant.training_row_at(rows, step, 42) for step in range(8, 21)]
    assert whole[8:] == resumed
    for offset in (0, 7, 14):
        assert sorted(whole[offset : offset + 7]) == rows


def test_speech_chat_uses_actual_asr_hypothesis_and_separate_typed_control(tmp_path, monkeypatch):
    (tmp_path / "clip.wav").write_bytes(b"not-decoded-in-this-unit-test")
    row = {
        "id": "voice",
        "split": "test",
        "family": "voice-family",
        "task": "asr",
        "audio": "clip.wav",
        "user": "reference sentence",
        "answer": None,
    }
    path = manifest(tmp_path, audio_rows=[row])
    options = SimpleNamespace(
        manifest=path,
        data_root=None,
        max_seconds=1000,
        model=assistant.MODEL_ID,
        model_revision=assistant.MODEL_REVISION,
        asr_model=assistant.ASR_ID,
        asr_revision=assistant.ASR_REVISION,
        device="cpu",
        dtype="float32",
        min_pixels=65536,
        max_pixels=524288,
        max_tokens=2048,
        seed=42,
        split="test",
        adapter=None,
        output=tmp_path / "result",
    )
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(assistant, "load_core", lambda *args, **kwargs: (model, object()))
    monkeypatch.setattr(assistant, "load_asr", lambda *args, **kwargs: (model, object()))
    monkeypatch.setattr(
        assistant,
        "transcribe",
        lambda *args, **kwargs: {
            "transcript": "mistaken hypothesis",
            "truncated": True,
            "completion_unknown": False,
            "stop_reason": "max_new_tokens",
            "generated_token_count": 128,
        },
    )
    observed = []

    def generation(*args):
        request = args[2]
        observed.append(request["user"])
        return {"id": request["id"], "task": request["task"], "score": None, "prediction": "actual answer"}

    monkeypatch.setattr(assistant, "generate", generation)
    result = assistant.run_evaluate(options, baseline=True)
    assert observed == ["mistaken hypothesis", "reference sentence"]
    assert result["status"] == "completed"
    assert result["variants"]["base"]["generation_count"] == 2
    transcripts = json.loads((options.output / "transcripts.json").read_text())
    assert transcripts[0]["errors"] > 0
    assert result["asr"]["raw_errors"] == transcripts[0]["raw_errors"]
    assert result["asr"]["micro_cer"] == result["asr"]["normalized_micro_cer"]
    assert result["asr"]["truncated_count"] == 1
    speech_record = json.loads((options.output / "generations.json").read_text())[0]
    assert speech_record["asr_truncated"] is True and speech_record["asr_stop_reason"] == "max_new_tokens"
    assert result["variants"]["base"]["tasks"]["speech_chat"]["pass_rate"] is None


def test_inference_retains_history_image_instead_of_only_latest_text(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "sample.png")
    history = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "sample.png"}, {"type": "text", "text": "remember this"}],
        },
        {"role": "assistant", "content": "I see it"},
    ]
    row = {"id": "followup", "user": "what color was it?", "history": history}
    messages = assistant.messages_for(row, tmp_path)
    assert messages[0]["content"][0]["image"] == "sample.png"
    assert messages[1]["content"][0]["text"] == "I see it"
    assert messages[-1]["content"][0]["text"] == "what color was it?"
    assert len(messages) == 3


def test_partial_timeout_is_not_reported_as_full_evaluation(tmp_path, monkeypatch):
    rows = [{"id": "a", "split": "test", "family": "a", "task": "ocr", "user": "q", "answer": "a"}]
    path = manifest(tmp_path, rows)
    options = SimpleNamespace(
        manifest=path,
        data_root=None,
        max_seconds=1,
        model=assistant.MODEL_ID,
        model_revision=assistant.MODEL_REVISION,
        asr_model=assistant.ASR_ID,
        asr_revision=assistant.ASR_REVISION,
        device="cpu",
        dtype="float32",
        min_pixels=65536,
        max_pixels=524288,
        max_tokens=2048,
        seed=42,
        split="test",
        adapter=None,
        output=tmp_path / "result",
    )
    monkeypatch.setattr(assistant, "load_core", lambda *args, **kwargs: (torch.nn.Linear(1, 1), object()))
    clock = iter([0, 2, 2, 2])
    monkeypatch.setattr(assistant.time, "monotonic", lambda: next(clock))
    result = assistant.run_evaluate(options, baseline=True)
    assert result["status"] == "time_limit_partial"
    assert not result["variants"]["base"]["completed"]
    assert result["requested_visual_text_rows"] == 1


def test_adapter_and_base_are_separate_variants(tmp_path, monkeypatch):
    rows = [{"id": "a", "split": "test", "family": "a", "task": "ocr", "user": "q", "answer": "a"}]
    path = manifest(tmp_path, rows)
    options = SimpleNamespace(
        manifest=path,
        data_root=None,
        max_seconds=1000,
        model=assistant.MODEL_ID,
        model_revision=assistant.MODEL_REVISION,
        asr_model=assistant.ASR_ID,
        asr_revision=assistant.ASR_REVISION,
        device="cpu",
        dtype="float32",
        min_pixels=65536,
        max_pixels=524288,
        max_tokens=2048,
        seed=42,
        split="test",
        adapter=tmp_path / "adapter",
        output=tmp_path / "result",
    )

    class Model(torch.nn.Linear):
        active = True

        @contextlib.contextmanager
        def disable_adapter(self):
            self.active = False
            try:
                yield
            finally:
                self.active = True

    model = Model(1, 1)
    monkeypatch.setattr(assistant, "load_core", lambda *args, **kwargs: (model, object()))
    seen = []

    def generation(*args):
        seen.append(model.active)
        return {"id": "a", "task": "ocr", "prediction": str(model.active), "score": {"passed": model.active}}

    monkeypatch.setattr(assistant, "generate", generation)
    result = assistant.run_evaluate(options)
    assert seen == [False, True]
    assert result["variants"]["base"]["tasks"]["ocr"]["passed"] == 0
    assert result["variants"]["adapter"]["tasks"]["ocr"]["passed"] == 1
