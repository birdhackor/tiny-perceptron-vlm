"""匯出真正 packed 的 Linear 權重；參考推論會反量化，不承諾加速。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from tiny_perceptron.quantization import replace_linear_layers
from tiny_perceptron.training import load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--bits", type=int, choices=[4, 8], default=4)
    p.add_argument("--output", type=Path, default=Path("checkpoints/quantized.pt"))
    args = p.parse_args()
    if args.checkpoint.resolve() == args.output.resolve():
        raise ValueError("量化輸出請用另一個檔名，保留原始 checkpoint 供比較")
    model, source = load_checkpoint(args.checkpoint)
    if source.get("format_version") != 1:
        raise ValueError("請從原始浮點 checkpoint 量化，避免反覆量化")
    was_tied = model.config.tied
    replace_linear_layers(model, args.bits)
    model.config.tied = False
    metadata = {
        "source": str(args.checkpoint),
        "input_output_sharing_removed": was_tied,
        "scope": "Linear weight-only; embeddings/norms remain float; dequantized reference forward",
    }
    state = model.state_dict()
    payload = {
        "format_version": "quantized-v1",
        "config": asdict(model.config),
        "bits": args.bits,
        "model": state,
        "optimizer": None,
        "tokenizer": source["tokenizer"],
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report = {
        "bits": args.bits,
        "source_file_bytes": args.checkpoint.stat().st_size,
        "quantized_file_bytes": args.output.stat().st_size,
        "model_tensor_storage_bytes": sum(t.numel() * t.element_size() for t in state.values()),
        "note": "原檔可能含 optimizer；檔案大小比例不是純權重壓縮率，也不是執行記憶體。",
        **metadata,
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
