"""核對 12.2 明確連結的 T.6 音訊格式契約；不載入或訓練模型。"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.audio_utils import load_mono_audio
from tiny_perceptron.modal_data import modal_example

RAW = Path("/workspace/work/reference/omni-technical-technical_12_2-v1")
ROOT = Path(__file__).resolve().parents[4]
cases = [
    ("valid", np.zeros(160, dtype="float32"), 16000, True),
    ("wrong-rate", np.zeros(160, dtype="float32"), 8000, False),
    ("stereo", np.zeros((160, 2), dtype="float32"), 16000, False),
    ("empty", np.zeros(0, dtype="float32"), 16000, False),
]
rows = []
for name, samples, rate, valid in cases:
    path = RAW / f"contract-{name}.wav"
    sf.write(path, samples, rate, subtype="FLOAT")
    for loader in ("inference", "own-data-training"):
        try:
            if loader == "inference":
                waveform, metadata = load_mono_audio(path)
                shape = list(waveform.shape)
                assert metadata["sample_rate"] == 16000 and not metadata["resampled"]
            else:
                record = {"audio": path.name, "question": "digit?", "answer": "0"}
                _, _, _, waveform = modal_example(record, RAW, "audio", device="cpu")
                shape = list(waveform.shape)
            assert valid and shape == [160]
            rows.append({"case": name, "loader": loader, "accepted": True, "shape": shape})
        except ValueError as error:
            assert not valid
            rows.append({"case": name, "loader": loader, "accepted": False, "error": str(error)})
print(json.dumps({
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_12_2-v1/format-contract.py",
    "environment": {"python": sys.version, "torch": torch.__version__, "soundfile": sf.__version__, "device": "cpu"},
    "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in [
        "scripts/audio_utils.py", "scripts/infer_modal.py", "tiny_perceptron/modal_data.py"]},
    "results": rows,
    "scope": "只核非空、單聲道、16kHz 的輸入契約。未載入模型、未更新權重。",
    "result": "PASS: both loaders accepted valid mono 16kHz and rejected all three invalid formats",
}, ensure_ascii=False, indent=2))
