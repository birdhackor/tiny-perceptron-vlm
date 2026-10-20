"""核對聲音資料流、對照輸入與跨輸入方式的歷史保存。"""

import ast
import hashlib
import html
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.course_experiments import modalities
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import AudioEncoder, MultiModalLM, log_mel, tone


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def curl_receipt(url, version, locator, excerpts, inspection, binary=False):
    command = ["curl", "-fsSL", "--max-time", "40", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    if binary:
        assert raw.startswith(b"%PDF"), "來源必須是 PDF"
    else:
        text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw.decode())))
        for excerpt in excerpts:
            assert excerpt in text, "原文未包含指定摘錄"
    return {
        "url": url,
        "version": version,
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "exit_code": completed.returncode,
        "original_bytes": len(raw),
        "original_sha256": digest(raw),
        "locator": locator,
        "excerpts": excerpts,
        "inspection_note": inspection,
        "full_text_saved": False,
    }


torch.set_num_threads(1)
torch.manual_seed(12)
environment = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "platform": platform.platform(),
    "device": "cpu",
    "dtype": "torch.float32",
    "seed": "12",
    "threads": "1",
}
receipts = [
    curl_receipt(
        "https://www.w3.org/TR/2021/REC-webaudio-20210617/",
        "W3C Recommendation, 17 June 2021",
        "The AudioBuffer Interface; AudioBuffer.sampleRate",
        ["32-bit floating-point linear PCM values", "The sample-rate for the PCM audio data in samples per second."],
        "實讀 AudioBuffer 的 PCM 數值表示及每秒樣本數定義；僅用來核對已數位化的單聲道表示。",
    ),
    curl_receipt(
        "https://www.isca-archive.org/interspeech_2019/lugosch19_interspeech.pdf",
        "Interspeech 2019 proceedings, DOI 10.21437/Interspeech.2019-2396",
        "Introduction; Model and Pre-training Strategy; Generalizing to new wordings",
        ["an automatic speech recognition (ASR) module that maps the speech to a text transcript", "without explicitly producing a text transcript"],
        "curl 取得 PDF 原文 bytes；同 URL 的原文另以瀏覽工具抽取文字，實讀 Introduction 的 ASR 與直接意圖路線、Model and Pre-training Strategy 的預訓練條件，以及 Generalizing to new wordings 的泛化限制。只支持任務區別，不把其成績移作本課證據。",
        binary=True,
    ),
    curl_receipt(
        "https://arxiv.org/html/2104.08524v1",
        "arXiv:2104.08524v1, 17 April 2021",
        "MInDS-14: Dataset Collection / Spoken Data Collection; Multilingual and Cross-Lingual Intent Detection: Methodology",
        ["The task is then to provide new spoken utterances associated with the intent class."],
        "實讀原文的真人收集程序、含中文的語言集合、有限銀行意圖，以及其採用成熟 ASR 與預訓練文字編碼器的實驗條件。資料來源支持真人有限需求的設計，論文成績不支持本課從零模型的品質。",
    ),
]
with (ARTIFACT / "original-source-receipts.json").open("x") as handle:
    json.dump({"sources": receipts}, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

waveform = tone(440.0)
mel = log_mel(waveform)
encoder = AudioEncoder()
features = encoder(waveform[None])
features.square().mean().backward()
gradient_names = [name for name, parameter in encoder.named_parameters() if parameter.grad is not None and parameter.grad.abs().sum() > 0]
assert waveform.shape == (1600,), "單音樣本數不符"
assert mel.shape == (16, 11), "固定特徵形狀不符"
assert features.shape == (1, 11, 16), "入口特徵形狀不符"
assert not mel.requires_grad, "示範的輸入未啟用梯度"
assert all(parameter.requires_grad for parameter in encoder.parameters()), "入口參數應可學習"
assert len(gradient_names) == len(list(encoder.parameters())), "入口參數應收到非零梯度"
assert torch.equal(log_mel(waveform), mel), "同一輸入的固定處理結果應相同"

splits = modalities._audio_records()
records = [splits["test"][index] for index in (0, 2, 8, 12)]
ctx = SimpleNamespace(device="cpu")
model = MultiModalLM(TinyLM(ModelConfig(width=8, max_length=128)))
original_generate = modalities.generate_modal
observed_calls = []


def capture_generation(model, prefix, image, waveform, tokens):
    observed_calls.append((prefix.clone(), waveform.clone()))
    return original_generate(model, prefix, image, waveform, tokens)


modalities.generate_modal = capture_generation
controls = {}
try:
    for mode in ("none", "blank", "shuffle"):
        observed_calls.clear()
        result = modalities._evaluate(model, records, ctx, mode, tokens=2)
        assert len(observed_calls) == len(records), "對照不能漏題"
        for index, (prefix, actual) in enumerate(observed_calls):
            donor = (index + len(records) // 2) % len(records) if mode == "shuffle" else index
            expected = modalities._media(records[donor], ctx)[1]
            if mode == "blank":
                expected = torch.zeros_like(expected)
            assert torch.equal(actual, expected), "聲音對照輸入不符"
            assert torch.equal(prefix, modalities._sequence(records[index], ctx, False)[0]), "對照應保留原題文字"
        controls[mode] = {
            "examples": result["examples"],
            "donor_rows": [sample["donor_row"] for sample in result["samples"]],
            "waveform_sha256": [digest(wave.numpy().tobytes()) for _, wave in observed_calls],
            "waveform_lengths": [len(wave) for _, wave in observed_calls],
            "all_expected_inputs_equal": True,
            "generation_errors": result["generation_errors"],
        }
finally:
    modalities.generate_modal = original_generate

runtime = ARTIFACT / "_runtime"
test_command = [
    str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider",
    "--basetemp", str(runtime),
    "tests/test_selftrained_inference.py::test_public_history_has_no_target_or_supervision_and_requires_explicit_multiturn_asset_index",
    "tests/test_selftrained_inference.py::test_voice_actual_first_reply_and_original_audio_survive_typed_rewrite",
    "tests/test_selftrained_model.py::test_audio_retains_temporal_order_and_ignores_only_declared_padding",
]
try:
    completed = subprocess.run(test_command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
    tests = {"command": test_command, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
finally:
    if runtime.exists():
        shutil.rmtree(runtime)

source_symbols = {
    "tiny_perceptron/multimodal.py": ["tone", "log_mel", "AudioEncoder"],
    "scripts/course_experiments/modalities.py": ["_audio_records", "_media", "_evaluate", "run_audio"],
    "tiny_perceptron/selftrained/inference.py": ["ChatSession"],
}
code_receipts = []
for path, names in source_symbols.items():
    raw = (ROOT / path).read_bytes()
    tree = ast.parse(raw)
    for node in tree.body:
        if getattr(node, "name", None) in names:
            code_receipts.append({"path": path, "sha256": digest(raw), "identifier": node.name, "excerpt": ast.get_source_segment(raw.decode(), node)})

source = ROOT / "docs/course-review-20261019/freeze-01/sources/chapter-12.md"
freeze = ROOT / "docs/course-review-20261019/freeze-04.json"
page = next(item for item in json.loads(freeze.read_bytes())["pages"] if item["page_id"] == "chapter-12")
assert digest(source.read_bytes()) == "f8cb81578489c92375630229275c32d053d46db6e9d45bedc46dc0651348fc2b", "凍結原文雜湊不符"
assert digest(freeze.read_bytes()) == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412", "凍結清單雜湊不符"
assert page["figures_sha256"] == {}, "導言不應包含圖"
assert source.read_bytes() == (ROOT / "course/chapters/12.md").read_bytes().split(b"## ", 1)[0], "目前導言與凍結原文不符"
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-chapter-12/verify_intro.py",
    "environment": environment,
    "frozen_page": {"source_sha256": digest(source.read_bytes()), "freeze_sha256": digest(freeze.read_bytes()), "figures_sha256": page["figures_sha256"]},
    "audio_pipeline": {"waveform_shape": list(waveform.shape), "waveform_dtype": str(waveform.dtype), "waveform_first_values": waveform[:3].tolist(), "fixed_log_mel_shape": list(mel.shape), "encoder_output_shape": list(features.shape), "parameters_with_nonzero_gradients": gradient_names},
    "tone_records": {name: {"count": len(rows), "frequencies": sorted({row["frequency"] for row in rows}), "questions": sorted({row["question"] for row in rows})} for name, rows in splits.items()},
    "controls": controls,
    "pytest": tests,
    "code_receipts": code_receipts,
    "scope": "僅核對固定處理、可學參數、對照輸入與歷史保存。模型為隨機初值；沒有訓練、真人錄音推論或品質驗收。",
    "result": "pass" if completed.returncode == 0 else "fail",
}, ensure_ascii=False, indent=2))
assert completed.returncode == 0, "共享歷史與聲音入口測試未通過"
