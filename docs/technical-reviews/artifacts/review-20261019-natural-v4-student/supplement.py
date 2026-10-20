"""補充核對固定配套、輸入邊界與純本機載入參數。"""

import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from scripts import fetch_natural_release as release
from tiny_perceptron import natural_assistant as core
from tiny_perceptron import natural_ui as ui
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig


def fetch(url):
    args = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    result = subprocess.run(args, capture_output=True, check=True)
    return result.stdout, {"url": url, "command": args, "exit_code": result.returncode,
                           "sha256": hashlib.sha256(result.stdout).hexdigest(), "bytes": len(result.stdout),
                           "accessed_on": "2026-10-20"}


raw, install = fetch("https://pytorch.org/get-started/previous-versions/")
import re
import html
text = html.unescape(re.sub("<[^>]+>", "", raw.decode()))
start = text.index("v2.8.0")
section = text[start:text.index("v2.7.1", start)]
install["excerpt"] = "\n".join(line for line in section.splitlines() if "CUDA 12.8" in line or
                                ("torch==2.8.0" in line and ("cu128" in line or "/cpu" in line)))

manifest = json.loads((ROOT / "docs/natural-assistant/v4/public-release.json").read_bytes())
official_models = []
for key in ["base_model", "asr_model"]:
    model = manifest[key]
    raw, receipt = fetch(f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/config.json")
    config = json.loads(raw)
    receipt["configuration"] = {k: config[k] for k in ["model_type", "architectures"]}
    official_models.append(receipt)

def image_data(width, height, animated=False):
    stream = io.BytesIO()
    image = Image.new("RGB", (width, height))
    if animated:
        image.save(stream, format="WEBP", save_all=True, append_images=[Image.new("RGB", (width, height), "red")], duration=20)
    else:
        image.save(stream, format="PNG")
    return stream.getvalue()


def accept(name, kind, raw):
    try:
        content, extension, media = ui.upload_contents({"filename": name, "kind": kind,
                                                        "base64": base64.b64encode(raw).decode()})
        return {"accepted": len(content) == len(raw), "extension": extension, "media": media}
    except ValueError as error:
        return {"accepted": False, "error": str(error)}


image = image_data(8, 8)
boundaries = {
    "image_at_byte_limit": accept("photo.png", "image", image + bytes(ui.MAX_UPLOAD_BYTES-len(image))),
    "image_above_byte_limit": accept("photo.png", "image", image + bytes(ui.MAX_UPLOAD_BYTES-len(image)+1)),
    "image_at_pixel_limit": accept("photo.png", "image", image_data(4000, 4000)),
    "image_above_pixel_limit": accept("photo.png", "image", image_data(4001, 4000)),
    "animated_image": accept("photo.webp", "image", image_data(8, 8, animated=True)),
}
for duration in [30, 30.001]:
    stream = io.BytesIO()
    sf.write(stream, np.zeros(round(duration*16000)), 16000, format="WAV")
    boundaries["audio_seconds_"+str(duration)] = accept("voice.wav", "audio", stream.getvalue())

calls = []
class Model:
    device = "cpu"

    def eval(self):
        return self


def loader(repo, **kwargs):
    calls.append({"repo": repo, "kwargs": {k: str(v) for k, v in kwargs.items()}})
    return Model()


options = release.student_options(manifest, "unused", device="cuda", local_files_only=True)
with patch.dict(sys.modules, {"transformers": SimpleNamespace(
    WhisperProcessor=SimpleNamespace(from_pretrained=loader),
    WhisperForConditionalGeneration=SimpleNamespace(from_pretrained=loader),
)}):
    model, processor = core.load_asr(options)

torch.manual_seed(7)
first = LimitedAssistant(SelftrainedConfig(width=16, layers=1, heads=2, kv_heads=1, ffn_hidden=32))
torch.manual_seed(8)
second = LimitedAssistant(SelftrainedConfig(width=16, layers=1, heads=2, kv_heads=1, ffn_hidden=32))
random_weights = {name: not torch.equal(first.state_dict()[name], second.state_dict()[name]) for name in
                  ["lm.embedding.weight", "vision_encoder.cnn.0.weight", "ocr_encoder.cnn.0.weight", "audio_encoder.temporal.0.weight"]}

page = (ROOT / "docs/course-review-20261019/freeze-01/sources/natural-v4-student.md").read_text()
blocks = re.findall(r"```bash\n(.*?)```", page, re.S)
shell = [subprocess.run(["bash", "-n"], input=block, capture_output=True, text=True).returncode for block in blocks]
ocr_exactness = core.score_output({"answer": "台北咖啡營業中", "references": {"kind": "ocr"}}, "台北咖啡")
print(json.dumps({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
                  "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu"},
                  "pytorch_install": install, "official_models": official_models, "boundaries": boundaries,
                  "asr_loader_calls_with_cuda_chat_options": calls, "asr_default_device": model.device,
                  "v2_seed_changes_weights": random_weights, "bash_syntax_exit_codes": shell,
                  "ocr_partial_name_does_not_prove_exact_transcription": ocr_exactness,
                  "scope": "下載小型公開配置，不下載模型權重；邊界檢查與 v2 隨機初始化在本機執行，Whisper 載入器以記錄替身核對參數。"},
                 ensure_ascii=False, indent=2))
