"""驗證學生的音訊輸入保持真正時間軸，而非改取樣率後悄悄變調。"""

import json
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts import infer_modal
from scripts.audio_utils import load_mono_audio, resample_waveform
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM
from tiny_perceptron.training import save_checkpoint


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_resampling_retains_frequency_duration_and_training_preprocessing(tmp_path):
    from scripts.course_experiments.modalities import _resample_8_to_16

    source = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(2000) / 8000)
    path = tmp_path / "tone.wav"
    sf.write(path, source, 8000, subtype="PCM_16")
    with pytest.raises(ValueError, match="--resample-audio"):
        load_mono_audio(path)
    decoded, _ = sf.read(path, dtype="float32")
    result, metadata = load_mono_audio(path, resample=True)
    assert len(result) == 4000
    assert metadata["source_duration_seconds"] == metadata["duration_seconds"] == 0.25
    assert metadata["source_sample_rate"] == 8000 and metadata["sample_rate"] == 16000
    frequencies = np.fft.rfftfreq(len(result), d=1 / 16000)
    assert frequencies[np.abs(np.fft.rfft(result)).argmax()] == 1000
    assert np.array_equal(result, _resample_8_to_16(decoded))


def test_pcm_extrema_decode_to_normalized_range_without_gain_change(tmp_path):
    path = tmp_path / "pcm.wav"
    integers = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
    sf.write(path, integers, 16000, subtype="PCM_16")
    values, metadata = load_mono_audio(path)
    assert np.array_equal(values, integers.astype("float32") / 32768)
    assert values.dtype == np.float32 and metadata["resampled"] is False
    assert metadata["subtype"] == "PCM_16"


@pytest.mark.parametrize(
    "values,subtype,pattern",
    [
        (np.zeros((10, 2)), "PCM_16", "單聲道"),
        (np.array([], dtype=np.float32), "FLOAT", "非空"),
        (np.array([0, np.nan]), "FLOAT", "有限振幅"),
        (np.array([0, 1.1]), "FLOAT", "正規化"),
    ],
)
def test_invalid_audio_is_rejected_before_inference(tmp_path, values, subtype, pattern):
    path = tmp_path / "invalid.wav"
    sf.write(path, values, 16000, subtype=subtype)
    with pytest.raises(ValueError, match=pattern):
        load_mono_audio(path, resample=True)


def test_downsampling_suppresses_above_new_nyquist():
    # 6 kHz 不可直接折回 2 kHz；重採樣 kernel 必須先減少這段高頻。
    times = np.arange(8000) / 16000
    source = 0.5 * np.sin(2 * np.pi * 6000 * times)
    result = resample_waveform(source, 16000, 8000)
    assert len(result) == 4000
    assert np.sqrt(np.mean(result[64:-64] ** 2)) < 0.005


def test_modal_cli_explicitly_resamples_and_reports_real_source(tmp_path, monkeypatch, capsys):
    model = MultiModalLM(TinyLM(ModelConfig(width=8, max_length=64)))
    with torch.no_grad():
        model.language.final_norm.weight.zero_()
        model.language.final_norm.bias.zero_()
        model.language.final_norm.bias[0] = 1
        model.language.output.weight.zero_()
        model.language.output.weight[ByteTokenizer.eos_id, 0] = 1
    checkpoint = tmp_path / "fsdd.pt"
    save_checkpoint(checkpoint, model, metadata={"task": "audio", "experiment": "fsdd"})
    audio = tmp_path / "digit.wav"
    sf.write(audio, 0.5 * np.sin(2 * np.pi * 1000 * np.arange(800) / 8000), 8000, subtype="PCM_16")
    original = infer_modal.generate_modal

    def inspect_waveform(loaded, prefix, image, waveform, tokens):
        assert image is None and len(waveform) == 1600
        return original(loaded, prefix, image, waveform, tokens)

    monkeypatch.setattr(infer_modal, "generate_modal", inspect_waveform)
    monkeypatch.setattr(
        sys,
        "argv",
        [infer_modal.__file__, str(checkpoint), "--audio", str(audio), "--resample-audio", "--device", "cpu"],
    )
    infer_modal.main()
    report = json.loads(capsys.readouterr().out)
    assert report["prompt"] == "digit?"
    assert report["generated_ids"] == [ByteTokenizer.eos_id]
    assert report["input_sources"]["audio"]["resampled"] is True
    assert report["input_sources"]["audio"]["duration_seconds"] == 0.1
