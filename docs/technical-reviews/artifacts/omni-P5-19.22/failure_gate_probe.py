"""Artificial doubles only: inspect decoder gating without any neural inference."""

import json
import platform

import torch

from scripts.omni_revision.core_evaluation import generated_sound
from tiny_perceptron.omni_products.data import Event
from tiny_perceptron.omni_products.inference import OmniAssistant


class FailedTalker:
    def generate(self, program):
        return {"codes": torch.tensor([0]), "eos": False, "capped": True}


class RecordingCodec:
    def __init__(self):
        self.decode_calls = 0
        self.chunk_calls = 0

    def decode_codes(self, codes):
        self.decode_calls += 1
        return torch.zeros(80)

    def chunks(self, codes):
        self.chunk_calls += 1
        yield torch.zeros(80)


program = (Event(220, 100),)
codec = RecordingCodec()
record = generated_sound(codec, FailedTalker(), program)
assert codec.decode_calls == 1
assert record["legal"] is False and record["program_success"] is False
assistant = OmniAssistant.__new__(OmniAssistant)
assistant.codec, assistant.talker = RecordingCodec(), FailedTalker()
try:
    list(assistant.sound_chunks(program))
except ValueError as exc:
    streaming = str(exc)
else:
    raise AssertionError("streaming entry should reject failed EOS")
assert assistant.codec.decode_calls == assistant.codec.chunk_calls == 0
print(json.dumps({
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "scope": "One nonempty failed-generation fixture with Python doubles, no neural model or formal weights.",
    "fixture": {"codes": [0], "eos": False, "capped": True},
    "evaluation_generated_sound": {"decoder_calls": codec.decode_calls, "legal": record["legal"], "program_success": record["program_success"], "waveform_samples": record["waveform"]["samples"]},
    "streaming_sound_chunks": {"decoder_calls": assistant.codec.decode_calls, "chunk_calls": assistant.codec.chunk_calls, "error": streaming},
    "finding": "The evaluation route decodes a nonempty failed prefix, then rejects it for capability scoring. The streaming playback route rejects it before decoding. Frozen prose L452 states the stronger common no-decoder behavior without that scope.",
}, ensure_ascii=False, indent=2))
