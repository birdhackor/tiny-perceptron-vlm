"""同一份留出問答，比較正常／遮蔽／打亂模態；保存逐題生成，沒有預填品質。"""

import argparse
import json
import random
from pathlib import Path

import torch

from tiny_perceptron.data import ByteTokenizer, load_jsonl
from tiny_perceptron.modal_data import modal_example
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, generate_modal
from tiny_perceptron.training import choose_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ablation", choices=["none", "blank", "shuffle"], default="none")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=Path("outputs/modal-evaluation.json"))
    args = p.parse_args()
    if min(args.limit, args.tokens) < 1:
        raise ValueError("limit/tokens 必須為正")
    device, tok = choose_device(args.device), ByteTokenizer()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MultiModalLM(TinyLM(ModelConfig(**payload["config"]))).to(device)
    model.load_state_dict(payload["model"], strict=True)
    records = load_jsonl(args.data)[: args.limit]
    if not records:
        raise ValueError("沒有評估資料")
    examples = [modal_example(r, args.data.parent, payload["task"], device, False) for r in records]
    order = list(range(len(examples)))
    random.Random(args.seed).shuffle(order)
    if args.ablation == "shuffle" and len(order) < 2:
        raise ValueError("打亂模態至少需要兩筆")
    # 強制無自配對；不保證換後答案不同，因此逐題保存配對索引。
    donors = {index: order[(i + 1) % len(order)] for i, index in enumerate(order)}
    samples = []
    for i, record in enumerate(records):
        prefix, _, image, wave = examples[i]
        if args.ablation == "blank":
            image = None if image is None else torch.zeros_like(image)
            wave = None if wave is None else torch.zeros_like(wave)
        elif args.ablation == "shuffle":
            _, _, image, wave = examples[donors[i]]
        result = generate_modal(model, prefix, image, wave, args.tokens)
        answer = tok.decode(result[len(prefix) :].tolist())
        samples.append(
            {
                "row": i,
                "target": record["answer"],
                "generated": answer,
                "exact_match": answer.strip() == record["answer"].strip(),
                "donor_row": donors[i] if args.ablation == "shuffle" else None,
            }
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "ablation": args.ablation,
        "examples": len(samples),
        "exact_match": sum(s["exact_match"] for s in samples) / len(samples),
        "samples": samples,
        "note": "檢查同題的模態依賴；低分本身不能證明模型用了或沒用圖片。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
