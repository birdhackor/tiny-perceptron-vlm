"""載入真實.modal.pt，用圖片／音訊生成；預設可重現的課堂合成輸入。"""

import argparse
import json

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, generate_modal, scene, tone
from tiny_perceptron.training import choose_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--image")
    p.add_argument("--audio")
    p.add_argument("--color", choices=["red", "green", "blue"], default="red")
    p.add_argument("--shape", choices=["circle", "square"], default="square")
    p.add_argument("--frequency", type=float, default=220.0)
    p.add_argument("--prompt")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.tokens < 1:
        raise ValueError("tokens 必須為正")
    device = choose_device(args.device)
    tok = ByteTokenizer()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MultiModalLM(TinyLM(ModelConfig(**saved["config"])))
    model.load_state_dict(saved["model"], strict=True)
    model.to(device)
    task = saved["task"]
    image = wave = None
    markers = []
    if task not in ("vision", "audio", "joint"):
        raise ValueError("此檔案不是模態checkpoint")
    if task != "audio":
        if args.image:
            pixels = np.asarray(Image.open(args.image).convert("RGB").resize((16, 16)), dtype=np.float32) / 255
            image = torch.from_numpy(pixels).permute(2, 0, 1)
        else:
            image = scene(args.color, args.shape)
        image = image.to(device)
        markers.append(tok.image_id)
    if task != "vision":
        if args.audio:
            samples, rate = sf.read(args.audio, dtype="float32")
            if rate != 16000 or samples.ndim != 1 or samples.size == 0:
                raise ValueError("這個課堂encoder需真正16kHz mono；請先重取樣，不只改metadata")
            wave = torch.from_numpy(samples)
        else:
            wave = tone(args.frequency)
        wave = wave.to(device)
        markers.append(tok.audio_id)
    question = args.prompt or {"vision": "shape?", "audio": "pitch?", "joint": "joint?"}[task]
    ids = [tok.bos_id, tok.user_id] + markers + tok.encode(question) + [tok.eos_id, tok.assistant_id]
    prefix = torch.tensor(ids, device=device)
    output = generate_modal(model, prefix, image, wave, args.tokens)
    print(
        json.dumps(
            {
                "task": task,
                "answer": tok.decode(output[len(ids) :].tolist()),
                "note": "16×16合成規則模型；自然圖片/語音品質不能由此保證。",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
