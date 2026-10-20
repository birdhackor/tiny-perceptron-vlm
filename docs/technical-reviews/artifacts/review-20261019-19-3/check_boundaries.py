"""以小型公開輸入核對 metadata 邊界與 manifest 檔案驗證。"""

import copy
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from scripts.selftrained.train_local_stage import verified_file
from tiny_perceptron.selftrained.dataset import RecordEncoder
from tiny_perceptron.selftrained.inference import attach_public_metadata
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer


def main():
    result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-3/check_boundaries.py", "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "soundfile": sf.__version__, "device": "cpu"}}
    with tempfile.TemporaryDirectory(dir=ARTIFACT) as directory:
        root = Path(directory)
        sample_rate = 8000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        sf.write(root / "input.wav", 0.1 * np.sin(2 * np.pi * 440 * t), sample_rate)
        tokenizer = CharacterTokenizer.build(["請回答聲音問題。正確答案隱藏文字。"])
        encoder = RecordEncoder(tokenizer, root)
        record = {"id": "source", "group_id": "group", "split": "test", "task": "voice_qa", "audio": "input.wav", "transcription": "隱藏文字", "supervision": {"intent": "label"}, "messages": [{"role": "user", "content": "請回答聲音問題。"}, {"role": "assistant", "content": "正確答案。"}]}
        baseline = encoder.encode(record, generation=True)
        altered = copy.deepcopy(record)
        altered.update(id="different", group_id="different", transcription="changed transcript", intent="changed label", supervision={"intent": "changed label"})
        altered["messages"][-1]["content"] = "changed answer"
        encoded = encoder.encode(altered, generation=True)
        assert baseline["input_ids"] == encoded["input_ids"]
        assert len(baseline["modalities"]) == len(encoded["modalities"]) == 1
        assert torch.equal(baseline["modalities"][0]["values"], encoded["modalities"][0]["values"])
        result["voice_generation_metadata_invariance"] = {"same_input_ids": True, "same_waveform_features": True, "altered_fields": ["id", "group_id", "transcription", "intent", "supervision", "final assistant target"], "feature_shape": list(baseline["modalities"][0]["values"].shape), "waveform": "1 s, 8 kHz, 440 Hz sine; only checks input boundary, not speech capability"}
        image = Image.fromarray(np.arange(192 * 96, dtype=np.uint8).reshape(96, 192))
        image.save(root / "image.png")
        ocr = {"id": "ocr", "task": "ocr", "image": "image.png", "roi": [0, 0, 72, 36], "messages": [{"role": "user", "content": "請回答。"}, {"role": "assistant", "content": "答案。"}]}
        cropped = encoder.encode(ocr, generation=True)
        result["ocr_public_roi"] = {"roi": ocr["roi"], "output_shape": list(cropped["modalities"][0]["values"].shape)}
        del ocr["roi"]
        try:
            encoder.encode(ocr, generation=True)
        except ValueError as error:
            result["ocr_without_roi"] = str(error)
        else:
            raise AssertionError("OCR without public ROI was accepted")
        try:
            attach_public_metadata([{"role": "user", "content": "請回答。", "transcription": "隱藏文字"}])
        except ValueError as error:
            result["public_history_rejects_transcript"] = str(error)
        else:
            raise AssertionError("Hidden transcript was accepted in public messages")
        path = root / "fixed.txt"
        path.write_bytes(b"fixed-data")
        item = {"path": "fixed.txt", "bytes": 10, "sha256": hashlib.sha256(b"fixed-data").hexdigest()}
        assert verified_file(root, item) == path
        checks = {"correct_file": "accepted"}
        for name, value in [("wrong_sha", {**item, "sha256": "0" * 64}), ("wrong_bytes", {**item, "bytes": 9})]:
            try:
                verified_file(root, value)
            except ValueError as error:
                checks[name] = str(error)
            else:
                raise AssertionError("Modified fixed data was accepted")
        result["wrapper_verified_file"] = checks
    with (ARTIFACT / "boundary-results.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2); handle.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
