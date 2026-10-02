"""載入完整多模態 checkpoint；省略圖片／音訊檔案時使用課堂合成輸入。"""

import argparse
import json

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import AudioEncoder, MultiModalLM, VisionEncoder, generate_modal, scene, tone
from tiny_perceptron.tokenization import generation_report, load_tokenizer
from tiny_perceptron.training import choose_device, load_checkpoint


def load_modal_checkpoint(path, device="cpu"):
    """完整格式用共用 loader；舊 sidecar 從權重形狀還原 encoder。"""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") == "multimodal-v1":
        model, saved = load_checkpoint(path, device)
    elif "format_version" not in saved and all(key in saved for key in ("config", "model", "task")):
        weights = saved["model"]
        vision_width, pixels = weights["vision.projection.weight"].shape
        patches = weights["vision.position"].shape[1]
        patch_size = int((pixels // 3) ** 0.5)
        image_size = int(patches**0.5) * patch_size
        if 3 * patch_size**2 != pixels or (image_size // patch_size) ** 2 != patches:
            raise ValueError("舊 checkpoint 的影像 patch 形狀無法還原")
        audio_width, bands = weights["audio.projection.weight"].shape
        model = MultiModalLM(TinyLM(ModelConfig(**saved["config"])), vision_width, audio_width)
        model.vision = VisionEncoder(vision_width, image_size, patch_size)
        model.audio = AudioEncoder(bands, audio_width)
        model.load_state_dict(weights, strict=True)
        model.to(device)
    else:
        raise ValueError("需要完整 multimodal-v1 或舊 .modal.pt checkpoint")
    task = saved.get("task") or saved.get("metadata", {}).get("task")
    if task not in ("vision", "audio", "joint"):
        raise ValueError("多模態 checkpoint 需要明確的 vision/audio/joint 任務")
    load_tokenizer(None, model.language.config.vocab_size, saved)
    return model, saved, task


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
    model, saved, task = load_modal_checkpoint(args.checkpoint, device)
    if (task == "audio" and args.image) or (task == "vision" and args.audio):
        raise ValueError("輸入檔案的模態與 checkpoint 任務不相容")
    image = wave = None
    markers = []
    sources = {}
    image_size = model.vision.image_size
    if task != "audio":
        if args.image:
            with Image.open(args.image) as source:
                pixels = (
                    np.array(source.convert("RGB").resize((image_size, image_size)), dtype=np.float32, copy=True) / 255
                )
            image = torch.from_numpy(pixels).permute(2, 0, 1)
            sources["image"] = {"type": "file", "path": args.image, "resized_to": [image_size, image_size]}
        else:
            image = scene(args.color, args.shape, size=image_size)
            sources["image"] = {"type": "synthetic", "color": args.color, "shape": args.shape, "size": image_size}
        image = image.to(device)
        markers.append(tok.image_id)
    if task != "vision":
        if args.audio:
            samples, rate = sf.read(args.audio, dtype="float32")
            if rate != 16000 or samples.ndim != 1 or samples.size == 0:
                raise ValueError("這個課堂encoder需真正16kHz mono；請先重取樣，不只改metadata")
            wave = torch.from_numpy(samples)
            sources["audio"] = {"type": "file", "path": args.audio, "sample_rate": rate}
        else:
            wave = tone(args.frequency)
            sources["audio"] = {"type": "synthetic-tone", "frequency": args.frequency, "sample_rate": 16000}
        wave = wave.to(device)
        markers.append(tok.audio_id)
    question = args.prompt or {"vision": "shape?", "audio": "pitch?", "joint": "joint?"}[task]
    ids = [tok.bos_id, tok.user_id] + markers + tok.encode(question) + [tok.eos_id, tok.assistant_id]
    prefix = torch.tensor(ids, device=device)
    output = generate_modal(model, prefix, image, wave, args.tokens)
    report = generation_report(tok, output[len(ids) :].tolist())
    print(
        json.dumps(
            {
                "task": task,
                **report,
                "input_sources": sources,
                "visual_tokens": 0 if image is None else (image_size // model.vision.patch_size) ** 2,
                "checkpoint_metadata": {
                    key: saved.get("metadata", {})[key]
                    for key in ("experiment", "scope")
                    if key in saved.get("metadata", {})
                },
                "note": "合成 scene/tone 是操作練習，不能當作真實圖片、數字語音或音效辨識的評估結果。",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
