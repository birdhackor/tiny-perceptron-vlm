"""以記憶體讀取固定公開資料與權重，保存小型查核結果。"""

import base64
import copy
import hashlib
import importlib.util
import io
import itertools
import json
import platform
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from safetensors.torch import load

from tiny_perceptron.selftrained.dataset import RecordEncoder, perception_loss
from tiny_perceptron.selftrained.inference import InferenceAssistant
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig, decode_ocr
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer, OCR_CHARACTERS

OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.manual_seed(20261020)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def get(url):
    print("讀取", url, flush=True)
    result = subprocess.run(["curl", "-fLsS", "--retry", "2", "--max-time", "180", url], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    return result.stdout


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


receipts = []
manifest = json.loads((ROOT / "docs/selftrained/v2-manifest.json").read_bytes())
package = manifest["package"]
url = f"https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/{package['revision']}/{package['path']}"
raw = get(url)
assert len(raw) == package["bytes"] and sha(raw) == package["sha256"]
receipts.append({"url": url, "bytes": len(raw), "sha256": sha(raw), "stored": False})
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
    files = {entry.name.removeprefix("./"): archive.extractfile(entry).read() for entry in archive.getmembers() if entry.isfile()}
del raw
if "voice-train.jsonl" not in files:
    prefix = next(key[: -len("voice-train.jsonl")] for key in files if key.endswith("voice-train.jsonl"))
    files = {key.removeprefix(prefix): value for key, value in files.items()}
for entry in manifest["records"] + manifest["assets"]:
    assert sha(files[entry["path"]]) == entry["sha256"], entry["path"]

voice_rows = [json.loads(line) for split in ("train", "validation", "test") for line in files[f"voice-{split}.jsonl"].splitlines() if line]
native = [row for row in voice_rows if "augmentation" not in row]
by_audio = {}
for row in native:
    if row["audio"] in by_audio:
        assert row["split"] == by_audio[row["audio"]]["split"]
    by_audio[row["audio"]] = row
selected = list(by_audio.values())
voice_counts = Counter(row["split"] for row in selected)
assert dict(voice_counts) == {"train": 62, "validation": 15, "test": 30}
assert len(selected) == 107
voice_report = {
    "selected_recordings": len(selected), "counts": dict(voice_counts),
    "intents": dict(Counter(row["supervision"]["intent"] for row in selected)),
    "record_count": len(native),
    "count_unit": "排除augmentation欄位後，以audio路徑去重的原始錄音，並檢查不跨split。",
    "speaker_metadata": "原始資料卡只列path/audio/transcription/english_transcription/intent_class/lang_id；prepare_voice不推導speaker。",
}
ocr = {}
for split in ("train", "validation", "test"):
    rows = [json.loads(line) for line in files[f"ocr-{split}.jsonl"].splitlines() if line]
    ocr[split] = rows
words = {split: {r["supervision"]["ocr_text"] for r in rows if len(r["supervision"]["ocr_text"]) > 1} for split, rows in ocr.items()}
ocr_report = {
    "training_fonts": dict(Counter(row["supervision"]["font_family"] for row in ocr["train"])),
    "lengths": {split: dict(Counter(len(row["supervision"]["ocr_text"]) for row in rows)) for split, rows in ocr.items()},
    "composition_overlap": {f"{a}/{b}": sorted(words[a] & words[b]) for a, b in itertools.combinations(words, 2)},
    "characters": "".join(sorted({c for rows in ocr.values() for row in rows for c in row["supervision"]["ocr_text"]})),
}
assert set(ocr_report["training_fonts"]) == {"Sans", "Serif"}
assert not any(ocr_report["composition_overlap"].values())
assert set(ocr_report["characters"]) == set(OCR_CHARACTERS)

figures = {}
figure_paths = ["course/figures/p6-19-modal-fashion.svg", "course/figures/p6-19-modal-ocr.svg"]
expected_images = [
    "images/vision/c98cb897d639c880fde261eafc6d953d.png",
    "images/vision/13913fb756b87b5c3e0f52e4ba194fec.png",
    "images/ocr/3a8d1184bb954b6f30c6fc09b31c678e.png",
]
embedded = []
for relative in figure_paths:
    root = ET.fromstring((ROOT / relative).read_bytes())
    for element in root.iter():
        if element.tag.endswith("}image"):
            href = element.get("href") or element.get("{http://www.w3.org/1999/xlink}href")
            content = base64.b64decode(href.split(",", 1)[1])
            matches = [p for p in expected_images if content == files[p]]
            assert len(matches) == 1
            embedded.append({"svg": relative, "asset": matches[0], "sha256": sha(content), "size": list(Image.open(io.BytesIO(content)).size)})
figures["embedded_pngs"] = embedded

revision = "979cdfacc588ad0536f1c64fff96f264571cf054"
base = f"https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/{revision}/selftrained/v2/moe-joint"
payloads = {}
for name in ("inference-manifest.json", "model-config.json", "tokenizer.json", "model.safetensors"):
    raw = get(base + "/" + name)
    payloads[name] = raw
    receipts.append({"url": base + "/" + name, "bytes": len(raw), "sha256": sha(raw), "stored": False})
export = json.loads(payloads["inference-manifest.json"])
assert sha(payloads["inference-manifest.json"]) == "f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e"
for name, expected in export["files"].items():
    assert sha(payloads[name]) == expected
config = SelftrainedConfig(**json.loads(payloads["model-config.json"]))
model = LimitedAssistant(config).eval()
state = load(payloads["model.safetensors"])
model.load_state_dict(state, strict=True)
tokenizer = CharacterTokenizer.from_dict(json.loads(payloads["tokenizer.json"]))


class MemoryEncoder(RecordEncoder):
    """只替換檔案運輸方式，原前處理照常執行。"""

    def _modality(self, metadata, kind):
        with patch("tiny_perceptron.selftrained.dataset._asset_path", lambda root, relative: io.BytesIO(files[relative])):
            return super()._modality(metadata, kind)


assistant = InferenceAssistant.__new__(InferenceAssistant)
assistant.model = model
assistant.encoder = MemoryEncoder(tokenizer, ROOT, config.max_length, "cpu")
assistant.receipt = {"files": export["files"], "manifest_sha256": sha(payloads["inference-manifest.json"])}
results = []
input_dir = ROOT / "docs/selftrained/examples/v2"
examples = [
    ("vision_clothing", {"image": expected_images[0], "image_layout": json.loads((input_dir / "vision_clothing.layout.json").read_bytes())}, "這是短靴。"),
    ("vision_relation", {"image": expected_images[1], "image_layout": json.loads((input_dir / "vision_relation.layout.json").read_bytes())}, "左邊是褲子。"),
    ("ocr", {"image": expected_images[2], "roi": [18, 50, 90, 86]}, "大小"),
    ("voice_qa", {"audio": "audio/ef25d9a7a3a6ce3790a040ba.wav", "modality_message_index": 3}, "先檢查網路並重新啟動App，仍失敗再詢問官方客服。"),
    ("voice_topic_continuation", {"audio": "audio/ef25d9a7a3a6ce3790a040ba.wav", "modality_message_index": 3}, "先檢查網路並重新啟動App，仍失敗再詢問官方客服。"),
]
for task, metadata, expected in examples:
    messages = json.loads((input_dir / f"{task}.messages.json").read_bytes())
    result = assistant.reply(messages, task=task, max_new_tokens=128, **metadata)
    results.append({"task": task, "expected": expected, "matches": result["answer"] == expected, "result": result})
    print(task, result["answer"], flush=True)
    if task == "voice_topic_continuation":
        first = result
    if task == "voice_qa":
        payload = assistant.encoder._modality({"audio": metadata["audio"]}, "audio")
        with torch.no_grad():
            tokens, logits = model.audio_encoder(payload["values"], payload["valid"])
        audio_source = next(row for row in selected if row["audio"] == metadata["audio"])
        head = {
            "probabilities": logits.softmax(-1).tolist(), "argmax": int(logits.argmax()),
            "reference_intent": audio_source["supervision"]["intent"], "reference_intent_id": audio_source["supervision"]["intent_id"],
            "waveform": {"asset": metadata["audio"], "sha256": sha(files[metadata["audio"]]), "samples": sf.info(io.BytesIO(files[metadata["audio"]])).frames},
            "preprocessed_shape": list(payload["values"].shape), "tokens_shape": list(tokens.shape),
            "ordered_classes": ["address", "app_error", "card_issues"],
        }
continued = first["messages"] + [json.loads((input_dir / "voice-continuation-user.json").read_bytes())]
second = assistant.reply(continued, task="voice_topic_continuation", max_new_tokens=128)
expected = "1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。"
results.append({"task": "voice_topic_continuation_second", "expected": expected, "matches": second["answer"] == expected, "result": second})
assert all(result["matches"] for result in results)
assert second["generations"][0]["prompt_messages"][:-1] == first["messages"]
assert [i for i, message in enumerate(second["messages"]) if message.get("audio")] == [3]

shape_checks = {}
for kind, metadata in [("image", examples[1][1]), ("ocr", examples[2][1])]:
    payload = assistant.encoder._modality(metadata, kind)
    with torch.no_grad():
        tokens, logits = model.vision_encoder(payload["values"], payload["coordinates"]) if kind == "image" else model.ocr_encoder(payload["values"])
    shape_checks[kind] = {"input": list(payload["values"].shape), "tokens": list(tokens.shape), "logits": list(logits.shape)}
    if kind == "image":
        shape_checks[kind]["coordinates"] = payload["coordinates"].tolist()
        shape_checks[kind]["head_argmax"] = logits.argmax(-1).tolist()
    else:
        shape_checks[kind]["greedy_text"] = decode_ocr(logits)

def collapse(ids):
    return tuple(key for key, _ in itertools.groupby(ids) if key)

probabilities = torch.tensor([[0.4, 0.6], [0.4, 0.6]], dtype=torch.float64)
mass = defaultdict(float)
for path in itertools.product(range(2), repeat=2):
    mass[collapse(path)] += float(probabilities[0, path[0]] * probabilities[1, path[1]])
logits = probabilities.log()[:, None]
loss = torch.nn.functional.ctc_loss(logits, torch.tensor([1]), [2], [1], blank=0, reduction="sum")
assert abs(float(loss) + np.log(0.84)) < 1e-12
ctc_report = {
    "probabilities": probabilities.tolist(), "label_probability": [{"labels": list(k), "probability": v} for k, v in mass.items()],
    "ctc_loss": float(loss), "expected_negative_log_0_84": float(-np.log(0.84)),
    "counterexample": "另用兩欄 blank=0.6、大=0.4；貪婪路徑為空字串，大的一字標記的路徑總機率為0.64。",
}
counter = torch.tensor([[0.6, 0.4], [0.6, 0.4]])
ctc_report["counterexample_greedy"] = decode_ocr(counter)
ctc_report["counterexample_one_glyph_probability"] = float(counter[0, 0] * counter[1, 1] + counter[0, 1] * counter[1, 0] + counter[0, 1] * counter[1, 1])
assert ctc_report["counterexample_greedy"] == ""
assert abs(ctc_report["counterexample_one_glyph_probability"] - 0.64) < 1e-6

spec = importlib.util.spec_from_file_location("training_probe", ROOT / "scripts/selftrained/train.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
trainer.set_trainable(model, "joint", True)
trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
assert not any(parameter.requires_grad for name, parameter in model.named_parameters() if ".head." in name or ".cnn." in name or ".temporal." in name or ".sequence." in name)
train_policy = {"trainable_names": trainable, "all_heads_and_backbones_frozen": True, "export_training_options": export.get("training_options"), "export_origin": export["origin"]}
source_paths = [
    "tiny_perceptron/selftrained/model.py", "tiny_perceptron/selftrained/dataset.py", "tiny_perceptron/selftrained/inference.py",
    "tiny_perceptron/selftrained/tokenizer.py", "tiny_perceptron/multimodal.py", "scripts/selftrained/train.py",
    "scripts/selftrained/prepare_voice.py", "scripts/selftrained/prepare_vision_ocr.py", "scripts/selftrained/augment_ocr_v2.py",
    "scripts/selftrained/chat.py", "docs/selftrained/examples/v2/append-voice-continuation.py", "docs/selftrained/v2-manifest.json",
    "docs/selftrained/v2-package-recipe.json",
]
write("probe-results.json", {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-6/probe.py",
    "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": "cpu", "threads": "2", "seed": "20261020"},
    "scope": "只替換權重與資料的檔案運輸；curl下載到記憶體並核對釘選SHA。沒有持久保存權重、完整資料包或第三方全文；生成與前處理使用現行專案函式。",
    "downloads": receipts, "repository_files": {p: sha((ROOT / p).read_bytes()) for p in source_paths},
    "voice_data": voice_report, "ocr_data": ocr_report, "figures": figures,
    "model_config": json.loads(payloads["model-config.json"]), "audio_head": head,
    "shapes": shape_checks, "ctc": ctc_report, "training": train_policy, "examples": results,
})
print("查核完成，結果已保存。", flush=True)
