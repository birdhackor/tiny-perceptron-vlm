"""載入本專案 checkpoint 生成文字；沒有權重就不假裝已學會回答。"""

import argparse

import torch

from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import generate
from tiny_perceptron.training import choose_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="顏色=")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    tok = ByteTokenizer()
    if model.config.vocab_size != tok.vocab_size or args.tokens < 1:
        raise ValueError("需要相容的 byte 詞表與正數生成長度")
    ids = [tok.bos_id] + tok.encode(args.prompt)
    if args.chat:
        ids = [tok.bos_id, tok.user_id] + tok.encode(args.prompt) + [tok.eos_id, tok.assistant_id]
    output = generate(model, torch.tensor([ids], device=device), args.tokens, args.temperature, use_cache=args.cache)
    print(tok.decode(output[0, len(ids) :].tolist()))


if __name__ == "__main__":
    main()
