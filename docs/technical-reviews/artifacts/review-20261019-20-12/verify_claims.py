"""核對字元算例、雙路由、人工更正與固定問句資料的直接證據。"""

import ast
import copy
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron import natural_assistant as assistant
from tiny_perceptron import natural_concepts as concepts
from tiny_perceptron.natural_ui import NaturalServer


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path = ARTIFACT / name
    if path.exists():
        assert path.read_text(encoding="utf-8") == content
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def curl(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return command, result.stdout


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


reference = "請推薦不辣的晚餐。"
omission = "請推薦辣的晚餐。"
substitution = "請推薦不辣的晚飯。"
numeric = []
for name, prediction in (("漏掉否定詞", omission), ("換成同義的晚飯", substitution)):
    actual = concepts.text_error_report(reference, prediction)
    # 用一個明確的編輯給上界，不相等給嚴格正的下界。
    witness = prediction.replace("薦辣", "薦不辣") if prediction == omission else prediction.replace("飯", "餐")
    assert witness == reference
    assert actual["edits"] == 1
    assert actual["reference_characters"] == 9
    assert actual["cer"] == 1 / 9
    assert round(actual["cer"] * 100, 1) == 11.1
    assert actual["exact"] is False
    numeric.append({"case": name, "report": actual, "cer_percent_rounded": round(actual["cer"] * 100, 1), "one_edit_witness": witness})

original_run = ROOT / "outputs/reviewer-tools/runs/20261020T063145561787Z"
section_bundle = {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/20.md#20.12" --execute',
    "files": {},
}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (original_run / name).read_bytes()
    section_bundle["files"][name] = {"sha256": digest(raw), "text": raw.decode("utf-8")}
execution = json.loads(section_bundle["files"]["execution.json"]["text"])
assert execution["exit_code"] == 0
assert execution["source_sha256"] == "ea775243ac2bf65ddf1f90083096f08b897e8884a0a96f9e069e35218587721d"
save("section-execution.json", section_bundle)

model, processor = object(), object()
row = {
    "id": "route-probe",
    "task": "speech_chat",
    "split": "test",
    "audio": "probe.wav",
    "image": "probe.png",
    "user": reference,
    "answer": None,
    "system": "請保留使用者條件。",
    "history": [{"role": "user", "content": "晚餐只吃一份。"}],
}
original_row = copy.deepcopy(row)
calls, writes = [], {}


def generation(actual_model, actual_processor, request, root, options):
    calls.append({
        "same_model": actual_model is model,
        "same_processor": actual_processor is processor,
        "request": copy.deepcopy(request),
        "messages": assistant.messages_for(request, root),
    })
    return {"id": request["id"], "task": request["task"], "user": request["user"], "prediction": "模擬回答", "score": None}


def collect(path, value):
    writes[Path(path).name] = copy.deepcopy(value)


options = SimpleNamespace(manifest="in-memory", data_root=ARTIFACT, max_seconds=60, split="test", adapter=None, output=ARTIFACT)
observation = {"transcript": omission, "truncated": False, "completion_unknown": False, "stop_reason": "eos", "generated_token_count": 8}
with (
    patch.object(assistant, "load_manifest", return_value=({"rows": [], "audio_rows": [row]}, ARTIFACT)),
    patch.object(assistant, "load_core", return_value=(model, processor)),
    patch.object(assistant, "load_asr", return_value=(torch.nn.Identity(), object())),
    patch.object(assistant, "transcribe", return_value=observation.copy()) as transcribe_call,
    patch.object(assistant, "generate", side_effect=generation),
    patch.object(assistant, "asset_path", side_effect=lambda name, root: Path(root) / name),
    patch.object(assistant, "provenance", return_value={}),
    patch.object(assistant, "parameter_counts", return_value={}),
    patch.object(assistant, "write_json", side_effect=collect),
):
    result = assistant.run_evaluate(options, baseline=True)
assert transcribe_call.call_count == 1
assert row == original_row
assert [call["request"]["user"] for call in calls] == [omission, reference]
assert all(call["same_model"] and call["same_processor"] for call in calls)
assert all(call["request"][key] == row[key] for call in calls for key in ("image", "history", "system"))
assert result["status"] == "completed"
assert writes["transcripts.json"][0]["transcript"] == omission
assert writes["transcripts.json"][0]["reference_transcript"] == reference
assert writes["generations.json"][0]["user"] == omission
assert writes["generations.json"][1]["user"] == reference

# 直接呼叫既有 operation，不開 HTTP socket、不建立上傳檔。
server = object.__new__(NaturalServer)
server.model, server.processor, server.options, server.data_root = model, processor, options, ARTIFACT
server.asr = (object(), object())
server.sessions = {"probe": {"history": [], "assets": {"speech": {"kind": "audio", "path": "probe.wav"}}, "transcriptions": {}}}
ui_calls = []


def ui_generation(actual_model, actual_processor, request, root, configuration):
    ui_calls.append(copy.deepcopy(request))
    assert actual_model is model and actual_processor is processor
    return {"prediction": "模擬回答"}


with (
    patch.object(assistant, "transcribe", return_value=observation.copy()),
    patch.object(assistant, "generate", side_effect=ui_generation),
    patch.object(assistant, "asset_path", side_effect=lambda name, root: Path(root) / name),
):
    shown = server.operation("/api/transcribe", {"session": "probe", "audio": "speech"})
    assert ui_calls == []
    submitted = server.operation("/api/chat", {"session": "probe", "prompt": reference, "speech": "speech"})
assert shown["asr"]["transcript"] == omission
assert ui_calls[0]["user"] == reference
assert submitted["asr"]["transcript"] == omission
assert submitted["asr"]["submitted_text"] == reference
assert submitted["asr"]["corrected"] is True
assert server.sessions["probe"]["transcriptions"]["speech"]["transcript"] == omission

source_path = ROOT / "docs/natural-assistant/v4/data/voice-question-sources.json"
document = json.loads(source_path.read_bytes())
selection_path = ROOT / "scripts/prepare_natural_v4_voice_questions.py"
tree = ast.parse(selection_path.read_text())
assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"SPEC", "PIN", "QUESTION_POLICY"}}
spec = {identifier: (split, rubric) for split, identifier, rubric in assignments["SPEC"]}
assert document["revision"] == assignments["PIN"]
assert set(spec) == {item["source_utterance_id"] for item in document["audio_rows"]}
selected_rows = []
for item in document["audio_rows"]:
    assert spec[item["source_utterance_id"]] == (item["split"], item["semantic_rubric"])
    assert item["user"] == "".join(item["original_spaced_source_transcription"].split())
    assert item["answer"] is None and item["history"] == []
    assert item["system"] == assignments["QUESTION_POLICY"]
    assert item["synthetic"] is False and item["channels"] == 1
    selected_rows.append({key: item[key] for key in ("id", "split", "user", "semantic_rubric", "history", "recording_domain", "source_archive")})

code_paths = [
    "tiny_perceptron/natural_concepts.py", "tiny_perceptron/natural_assistant.py", "tiny_perceptron/natural_ui.py",
    "scripts/prepare_natural_v4_voice_questions.py", "scripts/build_natural_v4_assets.py", "scripts/score_natural_v4_test.py",
    "docs/natural-assistant/v4/data/voice-question-sources.json", "docs/natural-assistant/v4/asr-selection.json",
    "course/figures/natural-v4-asr-two-routes.svg",
]
receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
figure_record = next(item for item in receipt["records"] if item["page_id"] == "20.12")
assert digest((ROOT / figure_record["png"]["path"]).read_bytes()) == figure_record["png"]["sha256"]
verification = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-12/verify_claims.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
    "numeric": numeric,
    "reference_characters": list(reference),
    "paired_routing": {"calls": calls, "result": result, "preserved_transcripts": writes["transcripts.json"], "generated_records": writes["generations.json"]},
    "ui_correction": {"shown": shown, "submitted": submitted, "core_requests": ui_calls},
    "dataset_selection": {"rows": selected_rows, "matches_predeclared_spec": True, "scope": document["scoring_policy"]["scope"]},
    "inspected_source_sha256": {name: digest((ROOT / name).read_bytes()) for name in code_paths},
    "figure_receipt": figure_record,
    "limitations": "路由與介面使用注入的辨識及生成器；不下載權重、不解碼錄音、不測模型回答品質。",
    "result": "所有數字、路由、介面與資料選擇斷言通過。",
}
save("verification.json", verification)

external = []
external_specs = [
    ("cer", "https://huggingface.co/spaces/evaluate-metric/cer/resolve/110e440c0dcff8b0e9edf1d6d31732bb63f828db/cer.py", "110e440c0dcff8b0e9edf1d6d31732bb63f828db", False, ["CER = (S + D + I) / N", "N is the number of characters in the reference (N=S+D+C)."]),
    ("aishell_paper", "https://arxiv.org/html/1709.05522v1", "arXiv:1709.05522v1, 2017-09-16", True, ["data annotators are asked to transcribe speech data, utterances with inconsistent raw text and transcription are removed."]),
    ("openslr", "https://www.openslr.org/33/", "SLR33，讀取日的未標版網頁", True, ["speech data and transcripts", "which is conducted in a quiet indoor environment"]),
    ("whisper", "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/README.md", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9", False, ["For speech recognition, the model predicts transcriptions in the *same* language as the audio."]),
    ("qwen", "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/89644892e4d85e24eaac8bacfd4f463576704203/README.md", "89644892e4d85e24eaac8bacfd4f463576704203", False, ['{"type": "text", "text": "Describe this image."}', "generated_ids = model.generate(**inputs, max_new_tokens=128)", "output_text = processor.batch_decode("]),
]
for identifier, url, version, html, excerpts in external_specs:
    command, raw = curl(url)
    text = raw.decode("utf-8")
    if html:
        parser = TextParser()
        parser.feed(text)
        text = " ".join(parser.parts)
    normalized = " ".join(text.split())
    for excerpt in excerpts:
        assert " ".join(excerpt.split()) in normalized, identifier
    external.append({"id": identifier, "url": url, "version": version, "command_argv": command, "accessed_on": datetime.now(UTC).date().isoformat(), "original_bytes_sha256": digest(raw), "original_bytes": len(raw), "excerpts": excerpts, "checked_original": True})
save("original-sources.json", {"sources": external, "storage_scope": "完整第三方原文僅在 curl 與此程序記憶體中讀取；本檔只留少量支持摘錄及原檔指紋。"})
print(json.dumps({"result": verification["result"], "external_sources_checked": len(external)}, ensure_ascii=False))
