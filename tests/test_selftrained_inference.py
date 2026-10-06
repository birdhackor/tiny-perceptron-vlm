"""Inference integrity and history/tool boundaries; synthetic outputs are engineering fixtures."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from scripts.selftrained.chat import main as chat_main
from tiny_perceptron.selftrained.dataset import PREPROCESS_VERSION, file_sha256
from tiny_perceptron.selftrained.inference import (
    PAYLOAD_FILES,
    ChatSession,
    InferenceAssistant,
    attach_public_metadata,
    fetch_public_export,
    verify_export,
)
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer
from tiny_perceptron.selftrained.tools import serialize_tool_call


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(18)


def refresh_manifest(root):
    path = root / "inference-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {name: file_sha256(root / name) for name in PAYLOAD_FILES}
    path.write_text(json.dumps(manifest) + "\n")


@pytest.fixture
def export(tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    tokenizer = CharacterTokenizer.build(
        ["你好。結果是請回答語音問題改成兩點。"], required_chars="".join(chr(i) for i in range(32, 127))
    )
    config = SelftrainedConfig(vocab_size=tokenizer.vocab_size, width=16, layers=1, heads=2, kv_heads=1, ffn_hidden=32)
    model = LimitedAssistant(config)
    state = {name: tensor.detach().contiguous().clone() for name, tensor in model.state_dict().items()}
    save_file(
        state,
        root / "model.safetensors",
        metadata={"origin": "all-neural-weights-random", "schema": "selftrained-random-v1"},
    )
    (root / "model-config.json").write_text(json.dumps(dataclasses.asdict(config)))
    tokenizer.save(root / "tokenizer.json")
    manifest = {
        "schema": "selftrained-random-v1",
        "files": {},
        "selected_checkpoint_sha256": "a" * 64,
        "stage": "joint",
        "selected_step": 0,
        "origin": {"kind": "all-neural-weights-random", "seed": 18},
        "preprocess_version": PREPROCESS_VERSION,
        "selection": "validation_loss",
    }
    (root / "inference-manifest.json").write_text(json.dumps(manifest))
    refresh_manifest(root)
    return root


def test_loader_never_deserializes_training_pickle_and_pins_manifest(export, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must never call torch.load")

    monkeypatch.setattr(torch, "load", forbidden)
    pinned = file_sha256(export / "inference-manifest.json")
    assistant = InferenceAssistant(export, tmp_path, manifest_sha256=pinned)
    assert assistant.receipt["manifest_hash_pinned"]
    assert not assistant.model.training
    assert assistant.model.lm.embedding.weight is assistant.model.lm.output.weight
    with pytest.raises(ValueError, match="manifest SHA"):
        InferenceAssistant(export, tmp_path, manifest_sha256="b" * 64)


def test_payload_tampering_rejected_before_any_model_allocation(export, monkeypatch):
    import tiny_perceptron.selftrained.inference as inference

    def forbidden(*args, **kwargs):
        raise AssertionError("Hash verification must precede allocation")

    monkeypatch.setattr(inference, "LimitedAssistant", forbidden)
    (export / "model-config.json").write_text("{}")
    with pytest.raises(ValueError, match="payload SHA"):
        InferenceAssistant(export, export)


@pytest.mark.parametrize("violation", ["missing_key", "tied_disagreement", "dtype"])
def test_strict_full_model_state_rejects_incompatible_safe_tensors(export, violation):
    state = load_file(export / "model.safetensors")
    if violation == "missing_key":
        del state["audio_encoder.head.bias"]
        reason = "keys differ"
    elif violation == "tied_disagreement":
        state["lm.output.weight"] = state["lm.output.weight"] + 1
        reason = "Tied embedding"
    else:
        state["audio_encoder.head.bias"] = state["audio_encoder.head.bias"].double()
        reason = "dtype mismatch"
    save_file(
        state,
        export / "model.safetensors",
        metadata={"origin": "all-neural-weights-random", "schema": "selftrained-random-v1"},
    )
    refresh_manifest(export)
    with pytest.raises(ValueError, match=reason):
        InferenceAssistant(export, export)


def test_public_history_has_no_target_or_supervision_and_requires_explicit_multiturn_asset_index():
    history = [
        {"role": "user", "content": "請回答語音問題。"},
        {"role": "assistant", "content": "回覆"},
        {"role": "user", "content": "改成兩點。"},
    ]
    with pytest.raises(ValueError, match="multi-turn assets"):
        attach_public_metadata(history, audio="audio/opaque.wav")
    attached = attach_public_metadata(history, audio="audio/opaque.wav", modality_message_index=0)
    assert attached[0]["audio"] == "audio/opaque.wav" and "audio" not in attached[2]
    assert "audio" not in history[0]
    with pytest.raises(ValueError, match="public image/audio"):
        attach_public_metadata([{"role": "user", "content": "問句", "supervision": {"intent_id": 0}}])
    with pytest.raises(ValueError, match="current user"):
        attach_public_metadata(history[:-1])


def test_same_loaded_model_tool_executor_tool_role_and_second_generation(export, tmp_path, monkeypatch):
    assistant = InferenceAssistant(export, tmp_path)
    tokenizer = assistant.encoder.tokenizer
    generated_prompts = []

    def fixture_generation(ids, **kwargs):
        # These explicit synthetic outputs test the runtime, not learned model capability.
        prompt = tokenizer.decode(ids[0], skip_special_tokens=False)
        generated_prompts.append(prompt)
        output = serialize_tool_call("multiply", 17, 23) if len(generated_prompts) == 1 else "結果是391。"
        if len(generated_prompts) == 2:
            assert '<tool>{"tool":"calculator","ok":true,"result":391}' in prompt
        return torch.tensor([tokenizer.encode(output) + [tokenizer.eos_id]])

    monkeypatch.setattr(assistant.model, "generate", fixture_generation)
    result = assistant.reply([{"role": "user", "content": "17*23"}], task="tool_call", tools=True)
    assert result["answer"] == "結果是391。"
    assert result["tool_trace"]["executed"]
    assert result["tool_trace"]["tool_result"]["result"] == 391
    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert len(result["generations"]) == 2
    assert all(call["eos"] for call in result["generations"])
    assert len({call["model_sha256"] for call in result["generations"]}) == 1


def test_voice_actual_first_reply_and_original_audio_survive_typed_rewrite(export, tmp_path):
    sf.write(tmp_path / "opaque.wav", np.sin(np.arange(800, dtype=np.float32) / 10), 8000)
    assistant = InferenceAssistant(export, tmp_path)
    session = ChatSession(assistant)
    first = session.ask("請回答語音問題。", task="voice_qa", audio="opaque.wav", max_new_tokens=2)
    second = session.ask("改成兩點。", task="voice_topic_continuation", max_new_tokens=2)
    call = second["generations"][0]
    assert call["prompt_messages"][0]["audio"] == "opaque.wav"
    assert "audio" not in call["prompt_messages"][2]
    assert call["prompt_messages"][1]["content"] == first["answer"]
    assert call["modality_kinds"] == ["audio"]
    assert "opaque.wav" not in assistant.encoder.tokenizer.decode(call["prompt_ids"])


def test_image_and_ocr_history_share_the_same_model_and_encoder(export, tmp_path):
    Image.new("L", (112, 112), 64).save(tmp_path / "scene.png")
    Image.new("L", (160, 40), 192).save(tmp_path / "card.png")
    history = [
        {
            "role": "user",
            "content": "你好。",
            "image": "scene.png",
            "image_layout": {"axis": "horizontal", "slots": [[0, 0, 56, 112], [56, 0, 112, 112]]},
        },
        {"role": "assistant", "content": "你好。"},
        {"role": "user", "content": "你好。", "image": "card.png", "roi": [0, 0, 160, 40]},
    ]
    result = InferenceAssistant(export, tmp_path).reply(history, task="ocr", max_new_tokens=2)
    assert result["generations"][0]["modality_kinds"] == ["image", "ocr"]


def test_public_fetch_pins_revision_disables_auth_and_verifies_all_payloads(export, tmp_path, monkeypatch):
    import huggingface_hub

    requests = []

    def cached_public_file(**kwargs):
        requests.append(kwargs)
        return str(export / Path(kwargs["filename"]).name)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", cached_public_file)
    destination = tmp_path / "public-copy"
    source = fetch_public_export("owner/repository", "c" * 40, destination, prefix="releases/moe")
    assert len(requests) == 4 and all(request["token"] is False for request in requests)
    assert all(request["revision"] == "c" * 40 for request in requests)
    assert source["authentication"] == "disabled"
    verify_export(destination)
    with pytest.raises(ValueError, match="immutable"):
        fetch_public_export("owner/repository", "main", destination)
    with pytest.raises(ValueError, match="relative"):
        fetch_public_export("owner/repository", "c" * 40, destination, prefix="../outside")


def test_cli_saves_raw_generation_and_continued_history(export, tmp_path, capsys):
    messages = tmp_path / "messages.json"
    messages.write_text(json.dumps([{"role": "user", "content": "你好。"}]))
    output = tmp_path / "result.json"
    history = tmp_path / "history.json"
    result = chat_main(
        [
            "--model-dir",
            str(export),
            "--messages",
            str(messages),
            "--asset-dir",
            str(tmp_path),
            "--max-new-tokens",
            "2",
            "--threads",
            "1",
            "--output",
            str(output),
            "--history-output",
            str(history),
        ]
    )
    assert json.loads(capsys.readouterr().out) == result
    assert json.loads(output.read_text()) == result
    assert json.loads(history.read_text()) == result["messages"]
    assert result["generations"][0]["generated_ids"]
    assert isinstance(result["generations"][0]["eos"], bool)
