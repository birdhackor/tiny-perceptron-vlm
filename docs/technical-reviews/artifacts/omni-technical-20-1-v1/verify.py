"""20.1 審閱：原碼與練習真跑；成品只作 CPU mock 路由驗證。"""

import base64
import contextlib
import copy
import hashlib
import io
import json
import platform
import re
import sys
import types
import wave
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from tiny_perceptron import natural_assistant as assistant
from tiny_perceptron import natural_ui as ui

OUT = Path(__file__).with_name("execution.json")
assert not OUT.exists(), "保留已執行收據，重跑請另用新版本"
freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
freeze = json.loads(freeze_path.read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "20.1")
raw = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
assert len(blocks) == 1
code = blocks[0]


def execute(text):
    scope, stream = {}, io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(text, page["snapshot"] + "#20.1", "exec"), scope)
    return scope, stream.getvalue()


baseline, baseline_stdout = execute(code)
assert baseline_stdout == "Qwen收到相同對話 True\n這輪問題 請讀出照片裡的招牌。\n"
assert baseline["history"] == [{"role": "user", "content": "接下來請用繁體中文回答。"}]
assert baseline["typed_chat"] is not baseline["history"]
assert baseline["spoken_chat"] is not baseline["typed_chat"]
assert baseline["typed_chat"][0] is baseline["history"][0]  # 清單串接是淺層的。
exercise_code = code.replace(
    'recognized_question = "請讀出照片裡的招牌。"',
    'recognized_question = "請讀出照片裡的找牌。"',
)
assert exercise_code != code
exercise, exercise_stdout = execute(exercise_code)
assert exercise_stdout == "Qwen收到相同對話 False\n這輪問題 請讀出照片裡的找牌。\n"
notebook = json.loads((ROOT / page["notebook"]).read_bytes())
notebook_code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
assert any(c.strip() == code.strip() for c in notebook_code)


class MockProcessor:
    tokenizer = types.SimpleNamespace(pad_token_id=0)

    def __init__(self):
        self.messages = []
        self.image_counts = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        return json.dumps(messages, ensure_ascii=False, sort_keys=True)

    def __call__(self, *, text, return_tensors, images=()):
        self.image_counts.append(len(images))
        return {"input_ids": torch.tensor([list(text[0].encode("utf-8"))])}

    def batch_decode(self, *args, **kwargs):
        return ["示意回答（mock，非模型能力）"]


class MockModel:
    generation_config = types.SimpleNamespace(eos_token_id=2)

    def generate(self, *, input_ids, **kwargs):
        assert kwargs["do_sample"] is False
        return torch.cat((input_ids, torch.tensor([[77, 2]])), dim=1)

    def to(self, device):
        assert device == "cpu"
        return self

    def requires_grad_(self, value):
        assert value is False

    def eval(self):
        return self


class MockASRProcessor:
    def __call__(self, waveform, **kwargs):
        assert kwargs["sampling_rate"] == 16000
        return types.SimpleNamespace(input_features=torch.zeros(1, 2), attention_mask=torch.ones(1, 2))

    def get_decoder_prompt_ids(self, **kwargs):
        assert kwargs == {"language": "chinese", "task": "transcribe"}
        return [(1, 20), (2, 30)]

    def batch_decode(self, *args, **kwargs):
        return ["請讀出照片裡的找牌。"]


class MockASRModel:
    generation_config = types.SimpleNamespace(decoder_start_token_id=10, eos_token_id=2)

    def generate(self, features, **kwargs):
        assert kwargs["language"] == "chinese" and kwargs["task"] == "transcribe"
        assert kwargs["do_sample"] is False
        return types.SimpleNamespace(sequences=torch.tensor([[10, 20, 30, 77, 2]]))


cache = Path("/workspace/work/omni-technical-20-1-v1/ui-data")
processor, model = MockProcessor(), MockModel()
options = types.SimpleNamespace(device="cpu", max_tokens=50000, max_new_tokens=20)
server = ui.create_server(model, processor, options, cache, port=0, asr=(MockASRModel(), MockASRProcessor()))
try:
    session = server.operation("/api/session", {})["session"]
    server.operation("/api/chat", {"session": session, "prompt": "接下來請用繁體中文回答。"})
    image_bytes = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(image_bytes, format="PNG")
    picture = server.operation("/api/upload", {
        "session": session, "filename": "mock.png", "kind": "image",
        "base64": base64.b64encode(image_bytes.getvalue()).decode(),
    })["asset"]
    server.operation("/api/chat", {"session": session, "prompt": baseline["typed_question"], "image": picture})
    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(bytes(3200))
    audio = server.operation("/api/upload", {
        "session": session, "filename": "mock.wav", "kind": "audio",
        "base64": base64.b64encode(wav_bytes.getvalue()).decode(),
    })["asset"]
    before_transcribe = copy.deepcopy(server.sessions[session]["history"])
    transcription = server.operation("/api/transcribe", {"session": session, "audio": audio})
    assert server.sessions[session]["history"] == before_transcribe
    uncorrected = server.operation("/api/chat", {
        "session": session, "prompt": transcription["asr"]["transcript"], "speech": audio,
    })
    raw_messages = copy.deepcopy(processor.messages[-1])
    assert raw_messages[0]["content"][0]["text"] == "接下來請用繁體中文回答。"
    assert any(p["type"] == "image" for t in raw_messages for p in t["content"])
    assert raw_messages[-1]["content"] == [{"type": "text", "text": "請讀出照片裡的找牌。"}]
    assert uncorrected["asr"]["corrected"] is False
    corrected = server.operation("/api/chat", {
        "session": session, "prompt": baseline["typed_question"], "speech": audio,
    })
    assert corrected["asr"]["transcript"] == "請讀出照片裡的找牌。"
    assert corrected["asr"]["submitted_text"] == "請讀出照片裡的招牌。"
    assert corrected["asr"]["corrected"] is True
    assert processor.messages[-1][-1]["content"] == [{"type": "text", "text": baseline["typed_question"]}]
    assert all(set(t) == {"role", "content"} for t in processor.messages[-1])
    assert all(set(p) <= {"type", "text", "image"} for t in processor.messages[-1] for p in t["content"])
    assert isinstance(corrected["prediction"], str)
    identical_row = {"user": baseline["typed_question"], "history": before_transcribe}
    typed_inputs = assistant.encode_messages(processor, assistant.messages_for(identical_row, cache), cache, generation_prompt=True)
    spoken_inputs = assistant.encode_messages(processor, assistant.messages_for(dict(identical_row), cache), cache, generation_prompt=True)
    assert torch.equal(typed_inputs["input_ids"], spoken_inputs["input_ids"])
    assert "辨識原稿：" in ui.PAGE and "$('prompt').value=result.asr.transcript" in ui.PAGE
    assert "speechSynthesis" not in ui.PAGE
    ui_result = {
        "chat_turns": 4, "transcribe_does_not_append_chat": True,
        "prior_language_and_photo_retained": True,
        "image_counts_at_generation": processor.image_counts[:4],
        "uncorrected_question": uncorrected["user"],
        "raw_transcript_retained_after_correction": corrected["asr"]["transcript"],
        "corrected_submitted_question": corrected["asr"]["submitted_text"],
        "corrected_flag": corrected["asr"]["corrected"],
        "same_text_history_photo_produce_same_mock_processor_inputs": True,
        "no_speech_provenance_in_model_messages": True,
        "prediction_type": type(corrected["prediction"]).__name__,
        "browser_handler_inspected_not_browser_executed": True,
    }
finally:
    server.server_close()

release = json.loads((ROOT / "docs/natural-assistant/releases/assistant-2b-v4.json").read_text())
assert release["selected_variant"] == "base" and release["adapter_parameters"] == 0
assert release["files"] == []
core_options = types.SimpleNamespace(
    device="cpu", dtype="float32", model=release["base_model"]["repo"],
    model_revision=release["base_model"]["revision"], cache_dir=None,
    local_files_only=True, min_pixels=65536, max_pixels=524288,
)
loads = []


class Loader:
    def __init__(self, value, name):
        self.value, self.name = value, name

    def from_pretrained(self, name, **kwargs):
        assert kwargs["revision"] == release["base_model"]["revision"]
        loads.append({"component": self.name, "model": name, "revision": kwargs["revision"]})
        return self.value


fake_transformers = types.ModuleType("transformers")
fake_transformers.AutoProcessor = Loader(processor, "processor")
fake_transformers.Qwen3VLForConditionalGeneration = Loader(model, "core")
with patch.dict(sys.modules, {"transformers": fake_transformers, "peft": None}):
    loaded_model, loaded_processor = assistant.load_core(core_options, adapter=None, train=False)
    assert loaded_model is model and loaded_processor is processor

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-20-1-v1/verify.py",
    "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    "source": {"path": page["snapshot"], "sha256": page["source_sha256"], "code_sha256": hashlib.sha256(code.encode()).hexdigest()},
    "baseline_stdout": baseline_stdout,
    "exercise_stdout": exercise_stdout,
    "baseline_input": {"history": baseline["history"], "typed_question": baseline["typed_question"], "recognized_question": baseline["recognized_question"]},
    "new_lists_preserve_history": True,
    "exact_notebook_code_matches": True,
    "ui_cpu_mock": ui_result,
    "selected_configuration": {"selected_variant": release["selected_variant"], "adapter_parameters": release["adapter_parameters"], "base_model": release["base_model"], "asr_model": release["asr_model"]},
    "mock_load_core_no_peft_import": loads,
    "scope": "正文原碼和練習實際執行；成品呼叫實作的歷史、影像前處理、轉寫、生成、載入流程使用 CPU mock 模型與 processor。沒有執行 Whisper/Qwen 權重、訓練、GPU、付費 API 或模型能力測試；前端只查官方本專案 source，未執行瀏覽器。",
    "result": "all assertions passed",
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": result["result"], "output": str(OUT), "baseline_stdout": baseline_stdout, "exercise_stdout": exercise_stdout}, ensure_ascii=False))
