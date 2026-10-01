"""一筆圖片／音訊問答的讀取與對齊；檔案路徑相對於 JSONL 所在資料夾。"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from tiny_perceptron.data import IGNORE, ByteTokenizer


def modal_example(record, directory, task, device="cpu", include_answer=True):
    tok = ByteTokenizer()
    ids = [tok.bos_id, tok.user_id]
    image = waveform = None
    if task in ("vision", "joint"):
        with Image.open(Path(directory) / record["image"]) as source:
            pixels = np.array(source.convert("RGB").resize((16, 16)), copy=True)
        image = torch.from_numpy(pixels).permute(2, 0, 1).float().div(255).to(device)
        ids.append(tok.image_id)
    if task in ("audio", "joint"):
        wave, rate = sf.read(Path(directory) / record["audio"], dtype="float32")
        if rate != 16000 or wave.ndim != 1 or wave.size == 0:
            raise ValueError("音訊必須是非空的 16 kHz 單聲道；先明確轉換資料")
        waveform = torch.from_numpy(wave.copy()).to(device)
        ids.append(tok.audio_id)
    ids += tok.encode(record["question"]) + [tok.eos_id, tok.assistant_id]
    labels = [IGNORE] * len(ids)
    if include_answer:
        answer = tok.encode(record["answer"]) + [tok.eos_id]
        ids += answer
        labels += answer
    return torch.tensor(ids, device=device), torch.tensor(labels, device=device), image, waveform
