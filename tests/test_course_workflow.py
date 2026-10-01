"""驗證跨零件的契約，不靠模型已訓練能力或外部下載。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf
import torch

from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.efficiency import online_attention
from tiny_perceptron.modal_data import modal_example
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, generate_modal, scene, tone
from tiny_perceptron.quantization import replace_linear_layers
from tiny_perceptron.simple import BigramLM, ContextMLP
from tiny_perceptron.training import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_online_attention_stable_and_block_independent():
    q, k, v = torch.randn(7, 8) * 20, torch.randn(9, 8) * 20, torch.randn(9, 5)
    expected = (q @ k.T / 8**0.5).softmax(-1) @ v
    for block in (1, 4, 20):
        actual = online_attention(q, k, v, block)
        assert torch.isfinite(actual).all()
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_modal_data_roundtrip_and_first_answer(tmp_path):
    from PIL import Image

    Image.new("RGB", (30, 20), "red").save(tmp_path / "image.png")
    sf.write(tmp_path / "audio.wav", tone(220).numpy(), 16000)
    record = {"image": "image.png", "audio": "audio.wav", "question": "joint?", "answer": "red,low"}
    ids, labels, image, wave = modal_example(record, tmp_path, "joint")
    tok = ByteTokenizer()
    first = (labels != IGNORE).nonzero()[0].item()
    assert ids[first - 1] == tok.assistant_id
    assert labels[first] == tok.encode("red,low")[0]
    assert image.shape == (3, 16, 16) and wave.shape == (1600,)
    model = MultiModalLM(TinyLM(ModelConfig(width=8)))
    output = model(ids, labels, image=image, waveform=wave)
    assert (output["labels"] != IGNORE).sum() == len(tok.encode("red,low")) + 1
    sf.write(tmp_path / "audio.wav", tone(220).numpy(), 8000)
    with pytest.raises(ValueError, match="16 kHz"):
        modal_example(record, tmp_path, "joint")


def test_missing_modal_and_context_limit():
    tok = ByteTokenizer()
    model = MultiModalLM(TinyLM(ModelConfig(width=8, max_length=22)))
    prefix = torch.tensor([tok.bos_id, tok.user_id, tok.image_id, 20, tok.eos_id, tok.assistant_id])
    with pytest.raises(ValueError, match="placeholder"):
        model(prefix)
    # 不因随机 EOS 提前停止，使上限检查确定可重复。
    with torch.no_grad():
        model.language.output.weight.zero_()
    result = generate_modal(model, prefix, scene(), max_new_tokens=10)
    assert len(result) == len(prefix) + 2
    assert model.training
    model.language.config.max_length = 20
    with pytest.raises(ValueError, match="prompt"):
        generate_modal(model, prefix, scene())


@pytest.mark.parametrize("tied", [False, True])
def test_packed_checkpoint_reload_logits(tmp_path, tied):
    model = TinyLM(ModelConfig(width=8, tied=tied))
    original = tmp_path / "float.pt"
    target = tmp_path / "packed.pt"
    save_checkpoint(original, model)
    result = subprocess.run(
        [sys.executable, "scripts/quantize.py", str(original), "--output", str(target)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["input_output_sharing_removed"] == tied
    expected = replace_linear_layers(model)
    actual, payload = load_checkpoint(target)
    ids = torch.tensor([[1, 14, 17]])
    assert torch.equal(actual(ids)["logits"], expected(ids)["logits"])
    assert payload["config"]["tied"] is False
    with pytest.raises(ValueError, match="resume"):
        load_checkpoint(target, restore_rng=True)


def test_simple_models_context_dependency():
    contexts = torch.tensor([[2, 3, 4], [1, 3, 4]])
    table = BigramLM(10)
    mlp = ContextMLP(10)
    assert torch.equal(table(contexts)[0], table(contexts)[1])
    assert not torch.equal(mlp(contexts)[0], mlp(contexts)[1])
    with pytest.raises(ValueError, match="窗口"):
        mlp(contexts[:, 1:])
