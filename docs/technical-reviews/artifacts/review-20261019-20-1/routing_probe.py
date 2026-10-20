"""離線核對文字與語音的匯合點；替身只證明接線，不證明模型能力。"""

import contextlib
import copy
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from scripts import fetch_natural_release as release
from tiny_perceptron import natural_assistant as core
from tiny_perceptron import natural_ui as ui


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/20.1.md"
    code = re.search(r"```python\n(.*?)```", frozen.read_text(), re.S).group(1)
    examples = {}
    for name, program in (
        ("original", code),
        ("typo", code.replace('recognized_question = "請讀出照片裡的招牌。"',
                              'recognized_question = "請讀出照片裡的找牌。"')),
    ):
        namespace = {}
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(program, str(frozen), "exec"), namespace)
        check(len(namespace["history"]) == 1, "原清單被改動")
        check(namespace["history"] is not namespace["typed_chat"], "沒有建立新清單")
        check(namespace["typed_chat"][0] == namespace["history"][0], "先前要求遺失")
        check(len(namespace["spoken_chat"]) == 2, "沒有只增加這輪消息")
        examples[name] = {
            "stdout": stdout.getvalue(),
            "equal": namespace["typed_chat"] == namespace["spoken_chat"],
            "last_content": namespace["spoken_chat"][-1]["content"],
            "history_unchanged": len(namespace["history"]) == 1,
        }
    check(examples["original"]["equal"] is True, "相同文字沒有相等")
    check(examples["typo"]["equal"] is False, "錯字文字沒有分歧")
    check(examples["typo"]["last_content"] == "請讀出照片裡的找牌。", "錯字沒有保留")

    manifest = json.loads((ROOT / "docs/natural-assistant/v4/public-release.json").read_text())
    options = release.student_options(manifest, Path(__file__).parent, device="cpu")
    check(options.adapter is None, "公開配置開啟了修正權重")
    loads = []

    class LoadedModel:
        def to(self, device):
            check(device == "cpu", "裝置不符")
            return self

        def requires_grad_(self, enabled):
            check(enabled is False, "底座沒有凍結")

        def eval(self):
            return self

    class Factory:
        @staticmethod
        def from_pretrained(name, **kwargs):
            loads.append({"name": name, "revision": kwargs["revision"]})
            return LoadedModel()

    with patch.dict(sys.modules, {"transformers": SimpleNamespace(
        AutoProcessor=Factory, Qwen3VLForConditionalGeneration=Factory
    )}):
        core.load_core(options, adapter=options.adapter)
    check(all(x["name"] == manifest["base_model"]["repo"] for x in loads), "底座來源不符")

    # 使用本程式作為存在的路徑；不把其位元組冒充真實影音。
    directory = Path(__file__).parent
    asset_name = Path(__file__).name
    server = ui.NaturalServer.__new__(ui.NaturalServer)
    server.sessions = {}
    server.model, server.processor = object(), object()
    server.options, server.data_root = options, directory
    server.asr = (object(), object())
    history = [
        {"role": "user", "content": "接下來請用繁體中文回答。"},
        {"role": "assistant", "content": [{"type": "text", "text": "好的。"}]},
    ]
    calls = []

    def runner(model, processor, row, data_root, generation_options):
        calls.append(copy.deepcopy(row))
        return {"prediction": "文字回答替身"}

    sessions = []
    for _ in range(2):
        identifier = server.operation("/api/session", {})["session"]
        sessions.append(identifier)
        server.sessions[identifier]["history"] = copy.deepcopy(history)
        server.sessions[identifier]["assets"] = {
            "photo": {"path": asset_name, "kind": "image", "media_type": "image/png"},
            "recording": {"path": asset_name, "kind": "audio", "media_type": "audio/wav"},
        }
    transcript = {"transcript": "請讀出照片裡的招牌。"}
    with patch.object(core, "generate", side_effect=runner), patch.object(core, "transcribe", return_value=transcript):
        server.operation("/api/transcribe", {"session": sessions[1], "audio": "recording"})
        check(not calls, "辨識時已呼叫回答核心")
        server.operation("/api/chat", {"session": sessions[0], "prompt": transcript["transcript"], "image": "photo"})
        server.operation("/api/chat", {"session": sessions[1], "prompt": transcript["transcript"], "image": "photo", "speech": "recording"})
        check(calls[0] == calls[1], "打字與語音送出的核心輸入不同")
        transcript["transcript"] = "請讀出照片裡的找牌。"
        server.operation("/api/transcribe", {"session": sessions[1], "audio": "recording"})
        correction = server.operation("/api/chat", {"session": sessions[1], "prompt": "請讀出照片裡的招牌。", "speech": "recording"})
    check(correction["asr"]["transcript"] == "請讀出照片裡的找牌。", "辨識原稿被改寫")
    check(correction["asr"]["corrected"] is True, "沒有標記更正")
    check(calls[-1]["user"] == correction["asr"]["submitted_text"], "未使用更正後文字")
    check(calls[-1]["history"][0] == history[0], "語言要求遺失")
    check(calls[-1]["history"][2]["content"][0]["type"] == "image", "照片沒有留在歷史")

    messages = core.messages_for(calls[-1], directory)
    processor_calls = []

    class Processor:
        def apply_chat_template(self, turns, **kwargs):
            processor_calls.append({"messages": copy.deepcopy(turns), **kwargs})
            return "模板替身"

        def __call__(self, **kwargs):
            processor_calls.append({"keys": sorted(kwargs), "image_count": len(kwargs.get("images", []))})
            return {"input_ids": torch.tensor([[1, 2]])}

    with patch.object(core.Image, "open", side_effect=lambda path: Image.new("RGB", (2, 2))):
        core.encode_messages(Processor(), messages, directory, generation_prompt=True)
    check(processor_calls[-1]["image_count"] == 1, "前處理器沒有收到歷史照片")
    check(all(set(row) == {"id", "user", "image", "history"} for row in calls), "核心輸入包含其他來源標記")

    asr_calls = []

    class AsrProcessor:
        def __call__(self, waveform, **kwargs):
            return SimpleNamespace(input_features=torch.zeros(1, 2, 3), attention_mask=torch.ones(1, 3))

        def get_decoder_prompt_ids(self, **kwargs):
            return [(1, 2), (2, 3)]

        def batch_decode(self, ids, **kwargs):
            return ["語音轉寫替身"]

    class AsrModel:
        generation_config = SimpleNamespace(decoder_start_token_id=1, eos_token_id=99)

        def generate(self, inputs, **kwargs):
            asr_calls.append({key: value for key, value in kwargs.items() if key != "attention_mask"})
            return SimpleNamespace(sequences=torch.tensor([[1, 2, 3, 10, 99]]))

    with patch("soundfile.read", return_value=(np.zeros((1600, 1), dtype=np.float32), 16000)):
        asr = core.transcribe(AsrModel(), AsrProcessor(), Path(__file__))
    check(asr_calls[0]["task"] == "transcribe", "語音任務不是轉寫")
    check(asr["transcript"] == "語音轉寫替身", "轉寫不是解碼文字")
    check("speechSynthesis" not in ui.PAGE, "介面具有語音合成呼叫")
    check("addTurn('assistant',result.prediction)" in ui.PAGE, "回答沒有顯示為文字")
    check("$('prompt').value=result.asr.transcript" in ui.PAGE, "轉寫沒有填入可編輯欄")
    check("辨識原稿：" in ui.PAGE, "介面沒有辨識原稿")
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-1/routing_probe.py",
        "result": "所有接線斷言通過；原例為 True，錯字練習為 False。",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "models": "未載入；明示替身"},
        "examples": examples,
        "release": {"variant": manifest["selected_variant"], "adapter": options.adapter, "loads": loads},
        "routing": {"typed_and_spoken_core_rows_equal": calls[0] == calls[1], "language_history_retained": True, "photo_history_retained": True, "asr_does_not_generate_answer": True, "correction": correction["asr"], "core_row_keys": sorted(calls[0]), "processor_input": processor_calls[-1]},
        "asr": {"generate_arguments": asr_calls, "transcript": asr["transcript"]},
        "ui": {"editable_transcript_assignment": True, "original_transcript_label": True, "assistant_text_rendering": True, "speech_synthesis_call_present": False},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (frozen, Path(core.__file__), Path(ui.__file__), Path(release.__file__), ROOT / "docs/natural-assistant/v4/public-release.json")},
        "scope": "驗證本機程式接線、消息與公開配置；沒有載入官方模型，沒有辨識真錄音、評估圖片內容、執行瀏覽器 JavaScript 或量測模型品質。",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
