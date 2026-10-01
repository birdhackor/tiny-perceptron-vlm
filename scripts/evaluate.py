"""離線評估指定文字checkpoint；同時記錄loss、BPB與自由生成，資料切分由讀者宣告。"""

import argparse
import json
import math
from pathlib import Path

import torch

from tiny_perceptron.data import ByteTokenizer, load_jsonl, render_chat, shifted, toy_conversations, toy_documents
from tiny_perceptron.model import generate, loss_sum
from tiny_perceptron.training import choose_device, load_checkpoint


def chat_prompt(messages, tok):
    ids = [tok.bos_id]
    roles = {"user": tok.user_id, "assistant": tok.assistant_id, "system": tok.system_id}
    for message in messages:
        ids += [roles[message["role"]]] + tok.encode(message["content"]) + [tok.eos_id]
    return ids + [tok.assistant_id]


@torch.no_grad()
def evaluate(model, records, mode="text", max_new_tokens=32):
    tok = ByteTokenizer()
    if model.config.vocab_size != tok.vocab_size or max_new_tokens < 1:
        raise ValueError("需要相容的 byte 詞表與正數生成長度")
    model.eval()
    device = next(model.parameters()).device
    total_nll, total_tokens, text_bytes = 0.0, 0, 0
    samples, skipped = [], []
    for index, record in enumerate(records):
        if mode == "text":
            text = record["text"]
            ids = [tok.bos_id] + tok.encode(text) + [tok.eos_id]
            if len(ids) < 2:
                continue
            text_bytes += len(text.encode("utf-8"))
            for start in range(0, len(ids) - 1, model.config.max_length):
                x, y = shifted(ids[start : start + model.config.max_length + 1])
                logits = model(x[None].to(device))["logits"]
                total, count = loss_sum(logits, y[None].to(device))
                total_nll += total.item()
                total_tokens += count.item()
            prompt_text = text[: min(4, len(text))]
            prefix = [tok.bos_id] + tok.encode(prompt_text)
            if len(prefix) < model.config.max_length:
                output = generate(model, torch.tensor([prefix], device=device), max_new_tokens)
                samples.append(
                    {"row": index, "prompt": prompt_text, "generated": tok.decode(output[0, len(prefix) :].tolist())}
                )
        else:
            messages = record.get("messages")
            if messages is None and "question" in record and "answer" in record:
                messages = [
                    {"role": "user", "content": record["question"]},
                    {"role": "assistant", "content": record["answer"]},
                ]
            if not messages or messages[-1]["role"] != "assistant":
                raise ValueError(f"第{index}筆SFT評估需要完整且以assistant結束的messages")
            x, y = render_chat(messages, tok)
            prefix = chat_prompt(messages[:-1], tok)
            if len(x) > model.config.max_length or len(prefix) >= model.config.max_length:
                skipped.append({"row": index, "reason": "context_too_long"})
                continue
            total, count = loss_sum(model(x[None].to(device))["logits"], y[None].to(device))
            total_nll += total.item()
            total_tokens += count.item()
            output = generate(model, torch.tensor([prefix], device=device), max_new_tokens)
            answer = tok.decode(output[0, len(prefix) :].tolist())
            target = messages[-1]["content"]
            samples.append(
                {"row": index, "target": target, "generated": answer, "exact_match": answer.strip() == target.strip()}
            )
    if total_tokens == 0:
        raise ValueError("没有有效評估token；不能報成零loss或成功")
    return {
        "mean_token_nll": total_nll / total_tokens,
        "effective_tokens": total_tokens,
        "bpb_including_bos_eos_boundary_targets": total_nll / (text_bytes * math.log(2))
        if mode == "text" and text_bytes
        else None,
        "exact_match": sum(s["exact_match"] for s in samples) / len(samples) if mode == "sft" and samples else None,
        "skipped": skipped,
        "samples": samples,
        "note": "自由生成、內容與格式另看樣本；toy分數不代表開放世界能力。",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--mode", choices=["text", "sft"], default="text")
    p.add_argument("--split-label", choices=["train-diagnostic", "validation", "test"], default="validation")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    args = p.parse_args()
    if args.limit < 1:
        raise ValueError("limit 必須為正")
    model, _ = load_checkpoint(args.checkpoint, choose_device(args.device))
    records = (
        load_jsonl(args.data)
        if args.data
        else (
            [{"text": s} for s in toy_documents()]
            if args.mode == "text"
            else [{"messages": m} for m in toy_conversations()]
        )
    )
    report = evaluate(model, records[: args.limit], args.mode, args.tokens)
    report.update(checkpoint=str(args.checkpoint), data=str(args.data), declared_split=args.split_label)
    if args.data is None:
        report["declared_split"] = "train-diagnostic"
        report["note"] += " 預設toy例也用於訓練，這是diagnostic，不是held-out。"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
