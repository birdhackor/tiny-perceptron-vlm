"""Independent CPU review probe: test real routing, with no pretrained models."""

import contextlib
import io
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from scripts import natural_assistant as cli
from tiny_perceptron import natural_assistant as assistant
from tiny_perceptron.natural_concepts import speech_stages

torch.set_num_threads(1)
report = speech_stages("我不吃辣", "我不吃拉", "選清淡的湯麵。", "選清淡的湯麵。")
assert report["transcription"]["edits"] == 1
assert report["transcription"]["reference_characters"] == 4
assert report["transcription"]["cer"] == 0.25
assert report["answers_identical"] is True
assert report["answers_correct"] is None
wrong_agreement = speech_stages("2+2?", "2+2?", "5", "5")
assert wrong_agreement["answers_identical"] is True
assert wrong_agreement["answers_correct"] is None

prior = [
    {"role": "user", "content": "我不吃辣"},
    {"role": "assistant", "content": "記住了。"},
]
reference = "給我晚餐建議"
hypothesis = "給我晚參建議"
core_marker = object()
asr_calls = []
core_calls = []
generation_calls = []


class ProcessorMarker:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.messages = messages
        return "interface probe only"

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": torch.tensor([[1, 2]], device="cpu")}


processor_marker = ProcessorMarker()
try:
    assistant.encode_messages(
        processor_marker,
        [{"role": "user", "content": [{"type": "audio", "audio": "unused"}]}],
        ROOT,
        generation_prompt=True,
    )
except ValueError as error:
    raw_audio_rejection = str(error)
else:
    raise AssertionError("The core interface unexpectedly accepted raw audio")


def fake_load_core(options, *, adapter=None, train=False):
    assert options.device == "cpu"
    core_calls.append({"model": options.model, "revision": options.model_revision, "adapter": str(adapter)})
    return core_marker, processor_marker


def fake_transcribe(model, processor, path):
    asr_calls.append(Path(path).name)
    return {"transcript": hypothesis, "scope": "test double; no ASR model executed"}


def fake_generate(model, processor, row, data_root, options):
    assert model is core_marker and processor is processor_marker
    assert row["history"] == prior
    inputs = assistant.encode_messages(
        processor, assistant.messages_for(row, data_root), data_root, generation_prompt=True
    )
    assert inputs["input_ids"].device.type == "cpu"
    image = processor.kwargs["images"][0]
    assert image.mode == "RGB" and image.size == (2, 2)
    assert image.getpixel((0, 0)) == (17, 43, 89)
    generation_calls.append(
        {
            "user": row["user"],
            "messages": processor.messages,
            "same_core_marker": model is core_marker,
            "pixel": list(image.getpixel((0, 0))),
            "prediction_scope": "test double; no chat model executed",
        }
    )
    return {"prediction": "介面探針輸出（非模型回答）", "task": "interface_probe"}


with tempfile.TemporaryDirectory(prefix="natural-12.14-interface-") as temporary:
    root = Path(temporary)
    (root / "history.json").write_text(json.dumps(prior, ensure_ascii=False), encoding="utf-8")
    (root / "audio.wav").write_bytes(b"test marker, not an ASR audio test")
    Image.new("RGB", (2, 2), (17, 43, 89)).save(root / "picture.png")
    results = []
    with (
        patch.object(assistant, "load_core", fake_load_core),
        patch.object(assistant, "load_asr", lambda options: (object(), object())),
        patch.object(assistant, "transcribe", fake_transcribe),
        patch.object(assistant, "generate", fake_generate),
    ):
        for mode, args in (
            ("typed", ["--user", reference]),
            ("speech", ["--audio", "audio.wav", "--user", "CORRECT REFERENCE MUST NOT REPLACE HYPOTHESIS"]),
        ):
            argv = [
                "natural_assistant.py", "chat", "--device", "cpu", "--dtype", "float32",
                "--data-root", str(root), "--history", str(root / "history.json"),
                "--image", "picture.png", "--output", str(root / mode), *args,
            ]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            result = json.loads((root / mode / "result.json").read_text(encoding="utf-8"))
            assert len(result["history"]) == 4
            assert result["history"][0]["content"][0]["text"] == "我不吃辣"
            assert result["history"][1]["content"][0]["text"] == "記住了。"
            results.append({"mode": mode, "history": result["history"], "asr": result["asr"]})
    assert len(asr_calls) == 1
    assert generation_calls[0]["user"] == reference
    assert generation_calls[1]["user"] == hypothesis
    assert all(call["model"] == assistant.MODEL_ID for call in core_calls)
    assert all(call["revision"] == assistant.MODEL_REVISION for call in core_calls)

output = {
    "scope": "CPU-only real helper/messages_for/encode_messages/CLI routing. ASR and chat loaders, transcribe, and generate are test doubles; this does not measure recognition, answer quality, or model ability.",
    "environment": {
        "python": platform.python_version(), "python_executable": sys.executable,
        "torch": str(torch.__version__), "torch_cuda_build": str(torch.version.cuda),
        "device": "cpu", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    },
    "short_example": report,
    "equal_wrong_answer_witness": wrong_agreement,
    "transcribe_call_count": len(asr_calls),
    "raw_audio_core_rejection": raw_audio_rejection,
    "same_model_configuration": core_calls,
    "generation_calls": generation_calls,
    "two_turn_histories": results,
    "assertions": "All passed; typed bypassed ASR, speech used distinct hypothesis, prior user and assistant messages remained present, image pixels were actually opened, saved history included the new user and written response.",
}
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
