"""Bounded CPU check of existing message/UI contracts with explicit test doubles.

No pretrained models, acoustic recognition, image recognition, training, or
model-quality scoring are performed. The doubles isolate message routing.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import base64
import copy
import importlib.metadata
import io
import json
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from tiny_perceptron import natural_assistant as assistant
from tiny_perceptron import natural_ui
from scripts.natural_assistant import parser

BASE = Path(__file__).resolve().parents[1]
assert torch.version.cuda is None
torch.set_num_threads(1)

class ProcessorDouble:
    tokenizer = SimpleNamespace(pad_token_id=0)

    def __init__(self):
        self.messages = []
        self.image_counts = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "explicit processor test double"

    def __call__(self, **kwargs):
        self.image_counts.append(len(kwargs.get("images", [])))
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    def batch_decode(self, ids, **kwargs):
        assert ids.tolist() == [[7, 9]]
        return ["測試回答"]

class ModelDouble:
    generation_config = SimpleNamespace(eos_token_id=9)

    def generate(self, **kwargs):
        assert kwargs["do_sample"] is False
        return torch.tensor([[1, 2, 7, 9]], dtype=torch.long)

def image_upload():
    payload = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(payload, format="PNG")
    return {"filename": "fixture.png", "kind": "image", "base64": base64.b64encode(payload.getvalue()).decode()}

def audio_upload():
    payload = io.BytesIO()
    with wave.open(payload, "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(16000)
        sound.writeframes(b"\0\0" * 160)
    return {"filename": "fixture.wav", "kind": "audio", "base64": base64.b64encode(payload.getvalue()).decode()}

processor = ProcessorDouble()
options = SimpleNamespace(device="cpu", max_tokens=32, max_new_tokens=8)
with tempfile.TemporaryDirectory(prefix="20_1-routing-", dir=BASE / "execution") as data_root:
    server = natural_ui.create_server(ModelDouble(), processor, options, data_root, port=0, asr=(object(), object()))
    try:
        sid = server.operation("/api/session", {})["session"]
        server.operation("/api/chat", {"session": sid, "prompt": "接下來請用繁體中文回答。"})
        photo = server.operation("/api/upload", {"session": sid, **image_upload()})["asset"]
        server.operation("/api/chat", {"session": sid, "prompt": "請讀出照片裡的招牌。", "image": photo})
        earlier = copy.deepcopy(server.session(sid)["history"])
        recording = server.operation("/api/upload", {"session": sid, **audio_upload()})["asset"]
        raw_transcript = "招排上的字是什麼？"
        with patch.object(assistant, "transcribe", return_value={"transcript": raw_transcript, "fixture": "explicit ASR double; not recognition"}):
            transcribed = server.operation("/api/transcribe", {"session": sid, "audio": recording})
        corrected_text = "招牌上的字是什麼？"
        reply = server.operation("/api/chat", {"session": sid, "prompt": corrected_text, "speech": transcribed["speech"]})
        final_messages = processor.messages[-1]
        assert final_messages[:-1] == earlier
        assert final_messages[0]["content"][0]["text"] == "接下來請用繁體中文回答。"
        assert final_messages[2]["content"][0]["type"] == "image"
        assert final_messages[-1]["content"] == [{"type": "text", "text": corrected_text}]
        assert processor.image_counts == [0, 1, 1]
        assert reply["asr"]["transcript"] == raw_transcript
        assert reply["asr"]["submitted_text"] == corrected_text
        assert reply["asr"]["corrected"] is True
        assert server.session(sid)["transcriptions"][recording]["transcript"] == raw_transcript
        assert reply["prediction"] == "測試回答"
        assert "audio" not in reply and "waveform" not in reply
        print(json.dumps({"scope": "actual existing helper/UI methods with model, processor and ASR test doubles", "preserved_prior_turns": len(earlier), "image_counts_by_turn": processor.image_counts, "raw_transcript": reply["asr"]["transcript"], "submitted_text": reply["asr"]["submitted_text"], "corrected": reply["asr"]["corrected"], "response_keys": sorted(reply), "assertions": "passed"}, ensure_ascii=False))
    finally:
        server.server_close()

namespace = {}
exec(compile((BASE / "original-fence/fence-1.py").read_bytes(), "raw-fence-1", "exec"), namespace)
assert namespace["typed_chat"] == namespace["spoken_chat"]
assert len(namespace["history"]) == 1
changed = namespace["history"] + [{"role": "user", "content": "請讀出照片裡的日期。"}]
assert namespace["typed_chat"] != changed
print("Original fence equality: True; changed recognized condition equality: False; input history unchanged: True")
defaults = parser().parse_args(["serve", "--output", str(BASE / "execution/unused-output")])
assert defaults.adapter is None
assert (defaults.model, defaults.model_revision) == (assistant.MODEL_ID, assistant.MODEL_REVISION)
print(json.dumps({"default_adapter": defaults.adapter, "model": defaults.model, "revision": defaults.model_revision, "cpu_device": "cpu", "python": sys.version, "torch": torch.__version__, "numpy": importlib.metadata.version("numpy"), "Pillow": importlib.metadata.version("pillow"), "soundfile": importlib.metadata.version("soundfile")}, ensure_ascii=False))
