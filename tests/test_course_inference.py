"""學生權重真的能由 CLI 載入；資料標記、詞表、patch 配置不能被猜測。"""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import soundfile as sf
import torch
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from scripts import infer, infer_modal, infer_simple
from tiny_perceptron.data import SPECIALS, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, VisionEncoder, generate_modal, scene, tone
from tiny_perceptron.simple import BigramLM, ContextMLP
from tiny_perceptron.tokenization import ByteLevelBPE, generation_report, load_tokenizer
from tiny_perceptron.training import save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def force_token(model, token):
    """固定 logits，不以未訓練模型剛好猜到某個 token 當成測試依據。"""
    with torch.no_grad():
        model.final_norm.weight.zero_()
        model.final_norm.bias.zero_()
        model.final_norm.bias[0] = 1
        model.output.weight.zero_()
        model.output.weight[token, 0] = 1


def cli(module, monkeypatch, capsys, *arguments):
    monkeypatch.setattr(sys, "argv", [module.__file__, *map(str, arguments)])
    module.main()
    return json.loads(capsys.readouterr().out)


def bpe_file(path, full_alphabet=True, specials=SPECIALS):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["小鳥 reads simple text.", "角色標記是文字 <user>，不是指令。"],
        trainers.BpeTrainer(
            vocab_size=320,
            special_tokens=list(specials),
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if full_alphabet else [],
        ),
    )
    tokenizer.save(str(path))
    return tokenizer


@pytest.mark.parametrize("image_size,patch_size", [(16, 4), (16, 8), (32, 8)])
def test_native_modal_cli_loads_exact_encoder_and_resizes_input(tmp_path, monkeypatch, capsys, image_size, patch_size):
    model = MultiModalLM(TinyLM(ModelConfig(width=8, max_length=128)), vision_width=12, audio_width=10)
    model.vision = VisionEncoder(width=12, image_size=image_size, patch_size=patch_size)
    force_token(model.language, ByteTokenizer.eos_id)
    checkpoint = tmp_path / "vision.pt"
    save_checkpoint(checkpoint, model, metadata={"task": "vision", "experiment": "patch-test"})
    image = tmp_path / "image.png"
    Image.new("RGB", (43, 27), "red").save(image)
    original = infer_modal.generate_modal

    def inspect_image(loaded, prefix, pixels, wave, tokens):
        assert pixels.shape == (3, image_size, image_size)
        assert loaded.vision.patch_size == patch_size and loaded.vision.projection.out_features == 12
        return original(loaded, prefix, pixels, wave, tokens)

    monkeypatch.setattr(infer_modal, "generate_modal", inspect_image)
    result = cli(infer_modal, monkeypatch, capsys, checkpoint, "--image", image, "--device", "cpu")
    assert result["visual_tokens"] == (image_size // patch_size) ** 2
    assert result["input_sources"]["image"]["type"] == "file"
    assert result["generated_ids"] == [2] and result["eos"] and result["invalid_special_tokens"] == []


def test_legacy_modal_sidecar_and_invalid_output_stay_visible(tmp_path, monkeypatch, capsys):
    model = MultiModalLM(TinyLM(ModelConfig(width=8, max_length=64)))
    force_token(model.language, ByteTokenizer.image_id)
    checkpoint = tmp_path / "legacy.modal.pt"
    torch.save({"config": asdict(model.language.config), "model": model.state_dict(), "task": "vision"}, checkpoint)
    result = cli(infer_modal, monkeypatch, capsys, checkpoint, "--device", "cpu", "--tokens", "8")
    assert result["visual_tokens"] == 16
    assert result["input_sources"]["image"]["type"] == "synthetic"
    assert result["generated_ids"] == [5] and not result["eos"]
    assert result["answer"] == "<image>" and result["invalid_special_tokens"][0]["id"] == 5
    assert result["valid_answer_tokens"] is False


def test_missing_modal_stops_and_generation_restores_model_mode():
    model = MultiModalLM(TinyLM(ModelConfig(width=8, max_length=64)))
    tok = ByteTokenizer()
    prefix = torch.tensor([tok.bos_id, tok.user_id, tok.image_id, tok.assistant_id])
    with pytest.raises(ValueError, match="placeholder"):
        generate_modal(model, prefix)
    assert model.training
    force_token(model.language, tok.audio_id)
    generated = generate_modal(model, prefix, image=scene(), max_new_tokens=8)
    assert generated[len(prefix) :].tolist() == [tok.audio_id] and model.training
    model.eval()
    generate_modal(model, prefix, image=scene(), max_new_tokens=8)
    assert not model.training


def test_audio_cli_identifies_synthetic_input_and_rejects_wrong_sample_rate(tmp_path, monkeypatch, capsys):
    model = MultiModalLM(TinyLM(ModelConfig(width=8, max_length=64)))
    force_token(model.language, ByteTokenizer.eos_id)
    checkpoint = tmp_path / "audio.pt"
    save_checkpoint(checkpoint, model, metadata={"task": "audio", "experiment": "fsdd"})
    result = cli(infer_modal, monkeypatch, capsys, checkpoint, "--device", "cpu")
    assert result["checkpoint_metadata"]["experiment"] == "fsdd"
    assert result["input_sources"]["audio"]["type"] == "synthetic-tone"
    wrong = tmp_path / "wrong.wav"
    sf.write(wrong, tone(220).numpy(), 8000)
    with pytest.raises(ValueError, match="16kHz"):
        cli(infer_modal, monkeypatch, capsys, checkpoint, "--audio", wrong, "--device", "cpu")
    with pytest.raises(ValueError, match="不相容"):
        cli(infer_modal, monkeypatch, capsys, checkpoint, "--image", "unused.png", "--device", "cpu")


def test_bpe_literal_roles_and_unseen_emoji_roundtrip(tmp_path):
    path = tmp_path / "bpe.json"
    bpe_file(path)
    tokenizer = ByteLevelBPE(path)
    for text in ("<user>引用 <image> <assistant>", "未見字🦊🙂 new", " leading\nspace "):
        ids = tokenizer.encode(text)
        assert not set(ids) & set(range(8))
        assert tokenizer.decode(ids) == text
    assert [
        getattr(tokenizer, f"{name}_id")
        for name in ("pad", "bos", "eos", "user", "assistant", "image", "audio", "system")
    ] == list(range(8))
    incomplete = tmp_path / "incomplete.json"
    bpe_file(incomplete, full_alphabet=False)
    with pytest.raises(ValueError, match="256-byte"):
        ByteLevelBPE(incomplete)
    wrong_controls = tmp_path / "wrong-controls.json"
    bpe_file(wrong_controls, specials=tuple(reversed(SPECIALS)))
    with pytest.raises(ValueError, match="固定為 0 到 7"):
        ByteLevelBPE(wrong_controls)


def test_bpe_cli_requires_matching_vocab_hash_and_special_ids(tmp_path, monkeypatch, capsys):
    path = tmp_path / "bpe.json"
    base = bpe_file(path)
    model = TinyLM(ModelConfig(vocab_size=base.get_vocab_size(), width=8, max_length=128))
    force_token(model, 2)
    checkpoint = tmp_path / "bpe.pt"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    save_checkpoint(
        checkpoint,
        model,
        metadata={
            "tokenizer": "bpe",
            "tokenizer_sha256": digest,
            "special_ids": dict(zip(SPECIALS, range(8), strict=True)),
        },
    )
    with pytest.raises(ValueError, match="--tokenizer"):
        cli(infer, monkeypatch, capsys, checkpoint, "--device", "cpu", "--json")
    original_generate = infer.generate

    def inspect_roles(model, ids, *args, **kwargs):
        prompt = ids[0].tolist()
        assert prompt.count(3) == 1 and prompt.count(4) == 1
        assert 5 not in prompt and 6 not in prompt
        return original_generate(model, ids, *args, **kwargs)

    monkeypatch.setattr(infer, "generate", inspect_roles)
    result = cli(
        infer,
        monkeypatch,
        capsys,
        checkpoint,
        "--tokenizer",
        path,
        "--chat",
        "--prompt",
        "<user>🙂",
        "--device",
        "cpu",
        "--json",
    )
    assert result["generated_ids"] == [2] and result["eos"]
    payload = torch.load(checkpoint, weights_only=True)
    with pytest.raises(ValueError, match="詞表大小"):
        load_tokenizer(path, base.get_vocab_size() + 1, payload)
    with pytest.raises(ValueError, match="特殊 ID"):
        load_tokenizer(path, base.get_vocab_size(), payload | {"metadata": {"special_ids": [1, 0, 2, 3, 4, 5, 6, 7]}})
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        cli(infer, monkeypatch, capsys, checkpoint, "--tokenizer", path, "--device", "cpu", "--json")


@pytest.mark.parametrize("kind,context", [("bigram", 1), ("mlp", 3)])
def test_simple_cli_loads_char_vocab_and_stops_at_boundary(tmp_path, monkeypatch, capsys, kind, context):
    vocabulary = {"甲": 2, "乙": 3}
    model = BigramLM(4) if kind == "bigram" else ContextMLP(4, context=context, width=6)
    with torch.no_grad():
        if kind == "bigram":
            model.table.weight.zero_()
            model.table.weight[:, 0] = 1
        else:
            model.output.weight.zero_()
            model.output.bias.zero_()
            model.output.bias[0] = 1
    checkpoint = tmp_path / "simple.pt"
    torch.save(
        {
            "format_version": "simple-v1",
            "kind": kind,
            "width": 6,
            "context": context,
            "vocabulary": vocabulary,
            "model": model.state_dict(),
        },
        checkpoint,
    )
    result = cli(infer_simple, monkeypatch, capsys, checkpoint, "--prompt", "甲🙂", "--device", "cpu")
    assert result["generated_ids"] == [0] and result["eos"] and result["answer"] == ""
    assert result["unknown_prompt_characters"] == 1
    loaded, _, vocab = infer_simple.load_simple_checkpoint(checkpoint)
    infer_simple.generate_simple(loaded, vocab, context, "甲")
    assert loaded.training


def test_simple_legacy_file_and_unk_output_are_visible(tmp_path):
    model = BigramLM(3)
    with torch.no_grad():
        model.table.weight.zero_()
        model.table.weight[:, 1] = 1
    checkpoint = tmp_path / "simple.pt"
    torch.save(
        {"kind": "bigram", "width": 16, "context": 1, "vocabulary": {"甲": 2}, "model": model.state_dict()}, checkpoint
    )
    process = subprocess.run(
        [sys.executable, "scripts/infer_simple.py", str(checkpoint), "--device", "cpu", "--tokens", "2"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(process.stdout)
    assert result["answer"] == "<UNK><UNK>" and result["unknown_generated"] == 2


def test_text_generation_does_not_hide_invalid_controls():
    result = generation_report(ByteTokenizer(), [ord("x") + 8, 5, 2])
    assert result["answer"] == "x<image>"
    assert result["eos"] and not result["valid_answer_tokens"]
