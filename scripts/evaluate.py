"""離線評估文字 checkpoint；分開核對內容、正常結束與實際評估分母。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from tiny_perceptron.data import ByteTokenizer, load_jsonl, render_chat, shifted, toy_conversations, toy_documents
from tiny_perceptron.model import TinyLM, generate, loss_sum
from tiny_perceptron.tokenization import generation_report, load_tokenizer
from tiny_perceptron.training import choose_device, load_checkpoint


def chat_prompt(messages, tok):
    ids = [tok.bos_id]
    roles = {"user": tok.user_id, "assistant": tok.assistant_id, "system": tok.system_id}
    for message in messages:
        ids += [roles[message["role"]]] + tok.encode(message["content"]) + [tok.eos_id]
    return ids + [tok.assistant_id]


def answer_sample(tok, generated_ids, target):
    """只移除正常末尾的 EOS；其他控制 ID 和首尾空白都參與核對。"""
    report = generation_report(tok, generated_ids)
    ids = report["generated_ids"]
    ended = bool(ids and ids[-1] == tok.eos_id and ids.count(tok.eos_id) == 1)
    content_ids = ids[:-1] if ended else ids
    expected = tok.encode(target)
    exact = content_ids == expected and not report["invalid_special_tokens"]
    report["generated"] = report.pop("answer")
    return {
        "target": target,
        "expected_content_ids": expected,
        **report,
        "exact_match": exact,
        "completed_exact_match": exact and ended,
    }


@torch.no_grad()
def evaluate(model, records, mode="text", max_new_tokens=32, tokenizer=None):
    tok = tokenizer or ByteTokenizer()
    if not isinstance(model, TinyLM):
        raise ValueError("此入口只評估 TinyLM 文字 checkpoint；多模態模型請使用對應評估入口")
    if model.config.vocab_size != tok.vocab_size:
        raise ValueError("tokenizer 詞表與模型 config 不相容；非 byte 模型需指定配對 tokenizer")
    if max_new_tokens < 1 or mode not in ("text", "sft"):
        raise ValueError("需要正數生成長度與 text/sft 模式")
    records = list(records)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_nll, total_tokens, text_bytes, loss_records = 0.0, 0, 0, 0
    samples, skipped = [], []
    try:
        for index, record in enumerate(records):
            if mode == "text":
                text = record["text"]
                ids = [tok.bos_id] + tok.encode(text) + [tok.eos_id]
                text_bytes += len(text.encode("utf-8"))
                # 每個目標只計一次。BOS 是第一格輸入，目標是文字單位與 EOS。
                for start in range(0, len(ids) - 1, model.config.max_length):
                    x, y = shifted(ids[start : start + model.config.max_length + 1])
                    logits = model(x[None].to(device))["logits"]
                    total, count = loss_sum(logits, y[None].to(device))
                    total_nll += total.item()
                    total_tokens += int(count.item())
                loss_records += 1
                prompt_text = text[: min(4, len(text))]
                prefix = [tok.bos_id] + tok.encode(prompt_text)
                target = text[len(prompt_text) :]
                if len(prefix) >= model.config.max_length:
                    skipped.append(
                        {
                            "row": index,
                            "phase": "generation",
                            "reason": "prompt_context_too_long",
                            "source": record,
                            "prompt": prompt_text,
                        }
                    )
                    continue
            else:
                messages = record.get("messages")
                if messages is None and "question" in record and "answer" in record:
                    messages = [
                        {"role": "user", "content": record["question"]},
                        {"role": "assistant", "content": record["answer"]},
                    ]
                if not messages or messages[-1]["role"] != "assistant":
                    raise ValueError(f"第 {index} 筆 SFT 評估需要完整且以 assistant 結束的 messages")
                x, y = render_chat(messages, tok)
                prefix = chat_prompt(messages[:-1], tok)
                if len(x) > model.config.max_length or len(prefix) >= model.config.max_length:
                    skipped.append(
                        {"row": index, "phase": "loss_and_generation", "reason": "context_too_long", "source": record}
                    )
                    continue
                total, count = loss_sum(model(x[None].to(device))["logits"], y[None].to(device))
                total_nll += total.item()
                total_tokens += int(count.item())
                loss_records += 1
                target = messages[-1]["content"]
            output = generate(model, torch.tensor([prefix], device=device), max_new_tokens, eos_id=tok.eos_id)
            sample = {"row": index, **answer_sample(tok, output[0, len(prefix) :].tolist(), target)}
            if mode == "text":
                sample["prompt"] = prompt_text
            samples.append(sample)
        if total_tokens == 0:
            raise ValueError("沒有有效評估 token；不能報成零 loss 或成功")
    finally:
        model.train(was_training)
    denominator = len(samples)
    return {
        "mean_token_nll": total_nll / total_tokens,
        "effective_tokens": total_tokens,
        "bpb_including_eos_boundary_targets": total_nll / (text_bytes * math.log(2))
        if mode == "text" and text_bytes
        else None,
        "raw_text_bytes": text_bytes if mode == "text" else None,
        "exact_match": sum(sample["exact_match"] for sample in samples) / denominator
        if mode == "sft" and denominator
        else None,
        "completed_exact_match": sum(sample["completed_exact_match"] for sample in samples) / denominator
        if mode == "sft" and denominator
        else None,
        "eos_rate": sum(sample["eos"] for sample in samples) / denominator if denominator else None,
        "records_read": len(records),
        "records_selected": len(records),
        "loss_evaluated_records": loss_records,
        "generation_evaluated_records": denominator,
        "loss_skipped_records": sum(item["phase"] == "loss_and_generation" for item in skipped),
        "generation_skipped_records": len(skipped),
        "metric_denominators": {
            "mean_token_nll": total_tokens,
            "bpb_including_eos_boundary_targets": text_bytes if mode == "text" else None,
            "exact_match": denominator if mode == "sft" else None,
            "completed_exact_match": denominator if mode == "sft" else None,
            "eos_rate": denominator,
        },
        "skipped": skipped,
        "samples": samples,
        "note": "exact_match 比較原始內容 ID，不刪空白或非法控制；completed_exact_match 另要求正常 EOS 結束。"
        "生成比例以實際生成題數為分母；跳過題及未選取題另列，不算答對。文字 NLL 分窗重置上下文，"
        "BPB 以原文 UTF-8 bytes 為分母，目標含 EOS、不含 BOS。toy 分數不代表開放世界能力。",
    }


def parse_limit(value):
    if value == "all":
        return None
    try:
        limit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit 必須是正整數或 all") from error
    if limit < 1:
        raise argparse.ArgumentTypeError("limit 必須是正整數或 all")
    return limit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--mode", choices=["text", "sft"], default="text")
    p.add_argument("--split-label", choices=["train-diagnostic", "validation", "test"], default="validation")
    p.add_argument("--limit", type=parse_limit, default=20, help="最多選取的資料筆數；all 選取整份資料（預設 20）")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--tokenizer", type=Path, help="訓練時的 BPE／byte JSON；核對詞表、特殊 ID 與 checkpoint SHA 綁定")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    args = p.parse_args()
    model, payload = load_checkpoint(args.checkpoint, choose_device(args.device))
    if not isinstance(model, TinyLM):
        raise ValueError("此入口只評估 TinyLM 文字 checkpoint；多模態模型請使用對應評估入口")
    tok = load_tokenizer(args.tokenizer, model.config.vocab_size, payload)
    records = (
        load_jsonl(args.data)
        if args.data
        else (
            [{"text": text} for text in toy_documents()]
            if args.mode == "text"
            else [{"messages": messages} for messages in toy_conversations()]
        )
    )
    selected = records if args.limit is None else records[: args.limit]
    report = evaluate(model, selected, args.mode, args.tokens, tokenizer=tok)
    report.update(
        checkpoint=str(args.checkpoint),
        data=str(args.data) if args.data else None,
        declared_split=args.split_label,
        records_read=len(records),
        unselected_records=len(records) - len(selected),
        limit="all" if args.limit is None else args.limit,
        tokenizer={
            "file": str(args.tokenizer) if args.tokenizer else None,
            "sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() if args.tokenizer else None,
            "vocab_size": tok.vocab_size,
        },
    )
    if args.data is None:
        report["declared_split"] = "train-diagnostic"
        report["note"] += " 預設 toy 例也用於訓練，這是 diagnostic，不是 held-out。"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
