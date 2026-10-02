"""載入本專案 checkpoint 生成文字；沒有權重就不假裝已學會回答。"""

import argparse
import json

import torch

from tiny_perceptron.model import TinyLM, generate
from tiny_perceptron.tokenization import generation_report, load_tokenizer
from tiny_perceptron.training import choose_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="顏色=")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--tokenizer", help="模型訓練使用的 BPE JSON；264-ID byte 模型可省略")
    parser.add_argument("--json", action="store_true", help="一併列出原始 IDs、EOS 與非法控制標記")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    if not isinstance(model, TinyLM):
        raise ValueError("多模態 checkpoint 請使用 scripts/infer_modal.py")
    if args.tokens < 1:
        raise ValueError("tokens 必須為正")
    tok = load_tokenizer(args.tokenizer, model.config.vocab_size, payload)
    ids = [tok.bos_id] + tok.encode(args.prompt)
    if args.chat:
        ids = [tok.bos_id, tok.user_id] + tok.encode(args.prompt) + [tok.eos_id, tok.assistant_id]
    output = generate(
        model,
        torch.tensor([ids], device=device),
        args.tokens,
        args.temperature,
        eos_id=tok.eos_id,
        use_cache=args.cache,
    )
    report = generation_report(tok, output[0, len(ids) :].tolist())
    print(json.dumps(report, ensure_ascii=False) if args.json else report["answer"])


if __name__ == "__main__":
    main()
