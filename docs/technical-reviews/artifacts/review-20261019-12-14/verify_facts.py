"""保存轉寫、回答比較與共同歷史的獨立核對事實。"""

import argparse
import ast
import hashlib
import itertools
import json
import platform
import re
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def originals():
    selections = [
        {
            "id": "whisper",
            "repository": "openai/whisper",
            "revision": "31243bad24cc746f07d4c8bfdd2d974872cb1803",
            "version": "v20250625",
            "path": "whisper/transcribe.py",
            "ranges": [(38, 40), (122, 125), (253, 256)],
            "inspection": "transcribe 收錄音輸入，回傳 text、segments 與 language；輸出不是聊天回答。",
        },
        {
            "id": "qwen",
            "repository": "QwenLM/Qwen3-VL",
            "revision": "96588727e44c78b25ba03ea03b8e12f7e64fd0da",
            "version": "讀取時的 main，另記不可變 commit",
            "path": "web_demo_mm.py",
            "ranges": [(220, 225), (231, 234), (252, 256), (262, 269), (275, 275)],
            "inspection": "predict 將 task_history 複製為 user／assistant 消息，交給同一模型；模板處理後生成。",
        },
    ]
    result = []
    for selection in selections:
        url = (
            "https://raw.githubusercontent.com/"
            + selection["repository"]
            + "/"
            + selection["revision"]
            + "/"
            + selection["path"]
        )
        command = ["curl", "--fail", "--location", "--silent", "--show-error", url]
        fetched = subprocess.run(command, capture_output=True, check=True)
        lines = fetched.stdout.decode("utf-8").splitlines()
        result.append(
            {
                **selection,
                "url": url,
                "command_argv": command,
                "accessed_on": datetime.now(timezone.utc).date().isoformat(),
                "full_source_sha256": hashlib.sha256(fetched.stdout).hexdigest(),
                "source_line_count": len(lines),
                "excerpts": [
                    {"start_line": start, "end_line": end, "text": "\n".join(lines[start - 1 : end])}
                    for start, end in selection["ranges"]
                ],
                "stderr": fetched.stderr.decode("utf-8"),
            }
        )
    return {"method": "curl 原文只留記憶體，保存少量定位摘錄與全文雜湊", "sources": result}


def brute_distance(reference, prediction):
    """逐一列舉短字串的編輯路徑，不沿用被核對函式的動態規劃。"""
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)
    return min(
        1 + brute_distance(reference[1:], prediction),
        1 + brute_distance(reference, prediction[1:]),
        (reference[0] != prediction[0]) + brute_distance(reference[1:], prediction[1:]),
    )


def local_facts():
    import inspect

    import torch

    from tiny_perceptron.natural_concepts import edit_distance, speech_stages, text_error_report
    from tiny_perceptron.natural_assistant import messages_for

    original = speech_stages("我不吃辣", "我不吃拉", "選清淡的湯麵。", "選清淡的湯麵。")
    corrected = speech_stages("我不吃辣", "我不吃辣", "選清淡的湯麵。", "選清淡的湯麵。")
    identical_wrong = speech_stages("我不吃辣", "我不吃辣", "選超辣湯麵。", "選超辣湯麵。")
    different = speech_stages("我不吃辣", "我不吃辣", "甲", "乙")
    assert original["transcription"]["reference_characters"] == 4
    assert original["transcription"]["edits"] == 1
    assert original["transcription"]["cer"] == 0.25
    assert corrected["transcription"]["cer"] == 0
    assert original["answers_identical"] is True
    assert different["answers_identical"] is False
    assert all(r["answers_correct"] is None for r in (original, corrected, identical_wrong, different))
    assert original["transcription"] == text_error_report("我不吃辣", "我不吃拉")
    signature = list(inspect.signature(speech_stages).parameters)
    assert signature == ["reference", "recognized", "typed_answer", "spoken_answer"]

    strings = ["".join(x) for length in range(4) for x in itertools.product("甲乙", repeat=length)]
    pairs = 0
    for reference in strings:
        for prediction in strings:
            assert edit_distance(reference, prediction) == brute_distance(reference, prediction)
            pairs += 1

    history = [
        {"role": "user", "content": "我不吃辣"},
        {"role": "assistant", "content": "選清淡的湯麵。"},
    ]
    followup = {"user": "還有別的嗎", "history": history}
    typed = messages_for(followup, ROOT)
    spoken = messages_for({"user": "還有別的嗎", "history": history}, ROOT)
    assert typed == spoken
    assert typed[0]["content"][0]["text"] == "我不吃辣"
    assert typed[-1]["content"][0]["text"] == "還有別的嗎"

    release_script = runpy.run_path(str(ROOT / "scripts/fetch_natural_release.py"))
    manifest = json.loads((ROOT / "docs/natural-assistant/v4/public-release.json").read_text())
    options = release_script["student_options"](manifest, ROOT, device="cpu", local_files_only=True)
    assert options.adapter is None
    assert options.model == "Qwen/Qwen3-VL-2B-Instruct"
    assert options.asr_model == "openai/whisper-large-v3-turbo"
    selection = json.loads((ROOT / "docs/natural-assistant/v4/selection.json").read_text())
    assert selection["selected_variant"] == manifest["selected_variant"] == "base"

    chapter = (ROOT / "course/chapters/12.md").read_text()
    sections = dict(re.findall(r"(?ms)^## (12\.\d+) (.*?)(?=^## |\Z)", chapter))
    assert "沒有產生轉寫文字" in sections["12.15"]
    assert "語音那輪保留的是聲音特徵位置" in sections["12.16"]
    assert "語音辨識" in sections["12.13"] and "回答" in sections["12.13"]
    tree = ast.parse((ROOT / "tiny_perceptron/natural_concepts.py").read_text())
    speech_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "speech_stages")
    calls = [ast.unparse(node.func) for node in ast.walk(speech_node) if isinstance(node, ast.Call)]
    assert calls == ["text_error_report"]

    paths = [
        "tiny_perceptron/natural_concepts.py",
        "tiny_perceptron/natural_assistant.py",
        "tiny_perceptron/natural_ui.py",
        "scripts/fetch_natural_release.py",
        "docs/natural-assistant/v4/public-release.json",
        "docs/natural-assistant/v4/selection.json",
        "course/chapters/12.md",
        "course/chapters/20.md",
        "course/figures/rewrite-12-asr-history.svg",
        "docs/course-review-20261019/freeze-01/sources/12.14.md",
        "docs/course-review-20261019/freeze-04.json",
    ]
    return {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-14/verify_facts.py",
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "result": "全部斷言通過；無模型載入、無錄音辨識、無訓練。",
        "signature": signature,
        "original": original,
        "corrected": corrected,
        "identical_but_incompatible_answer": identical_wrong,
        "different_answers": different,
        "independent_edit_distance_comparisons": pairs,
        "history_messages": typed,
        "selected_options": {"model": options.model, "asr_model": options.asr_model, "adapter": options.adapter},
        "next_section_titles": {name: sections[name].splitlines()[0] for name in ("12.13", "12.15", "12.16")},
        "speech_stages_calls": calls,
        "input_sha256": {name: sha(ROOT / name) for name in paths},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", action="store_true")
    args = parser.parse_args()
    print(json.dumps(originals() if args.originals else local_facts(), ensure_ascii=False, indent=2))
