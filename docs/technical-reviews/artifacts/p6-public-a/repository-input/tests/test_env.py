"""環境冒煙測試：只確認之後會用到的基本零件都能動，不測模型本身。

全部離線執行，不會下載任何資料：
    uv run pytest
"""

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image


def accelerator():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def test_torch_backward():
    layer = torch.nn.Linear(8, 4)
    layer(torch.randn(2, 8)).pow(2).mean().backward()
    assert layer.weight.grad is not None


@pytest.mark.skipif(accelerator() is None, reason="沒有 CUDA / MPS")
def test_accelerator_matches_cpu():
    x = torch.randn(32, 32)
    y = (x.to(accelerator()) @ x.to(accelerator())).cpu()
    assert torch.allclose(y, x @ x, rtol=1e-3, atol=1e-2)


def test_image_roundtrip(tmp_path):
    pixels = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "x.png")
    assert (np.asarray(Image.open(tmp_path / "x.png").convert("RGB")) == pixels).all()


def test_audio_roundtrip(tmp_path):
    sr = 16_000
    wave = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    sf.write(tmp_path / "x.flac", wave, sr)
    loaded, loaded_sr = sf.read(tmp_path / "x.flac", dtype="float32")
    assert loaded_sr == sr
    assert np.abs(loaded - wave).max() < 1e-3


def test_stft():
    # log-mel 頻譜的第一步：1 秒 16 kHz 音訊，25 ms 窗、10 ms 步長
    spec = torch.stft(
        torch.randn(16_000), n_fft=400, hop_length=160, window=torch.hann_window(400), return_complex=True
    )
    assert spec.shape == (201, 101)


def test_bpe_tokenizer_roundtrip():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = ["小小感知機會看圖、聽聲音、讀文字。", "A tiny perceptron that sees, hears and reads."]
    tok.train_from_iterator(corpus, trainer)
    text = "小小感知機 tiny perceptron 🙂"
    assert tok.decode(tok.encode(text).ids) == text


def test_datasets_from_dict():
    from datasets import Dataset

    ds = Dataset.from_dict({"text": ["你好", "hello"]})
    assert ds[0]["text"] == "你好"
