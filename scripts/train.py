"""GPU 訓練入口；不加 --train 時只做 forward/backward，不更新權重。"""

import argparse
import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tiny_perceptron.alignment import distillation_loss, dpo_loss, sequence_log_probability
from tiny_perceptron.data import (
    ByteTokenizer,
    load_jsonl,
    pad_batch,
    render_chat,
    shifted,
    toy_conversations,
    toy_documents,
)
from tiny_perceptron.modal_data import modal_example
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene, tone
from tiny_perceptron.training import choose_device, learning_rate, load_checkpoint, save_checkpoint, seed_everything


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["text", "sft", "dpo", "distill", "vision", "audio", "joint"], default="text")
    p.add_argument("--train", action="store_true", help="明確開啟權重更新；預設只檢查梯度")
    p.add_argument("--data", type=Path, help="text/messages 或 chosen/rejected JSONL；預設使用課程合成規則")
    p.add_argument("--checkpoint", type=Path, help="載入相容文字基模；--resume 同時恢復 optimizer/RNG")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--teacher", type=Path)
    p.add_argument("--freeze", choices=["none", "projector", "partial"], default="none")
    p.add_argument("--vision-encoder", type=Path)
    p.add_argument("--audio-encoder", type=Path)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--save-every", type=int, default=100, help="text/sft 的週期存檔間隔")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--experts", type=int, default=0)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--rotary", action="store_true")
    p.add_argument("--norm", choices=["layer", "rms"], default="layer")
    p.add_argument("--activation", choices=["gelu", "relu2", "swiglu"], default="gelu")
    p.add_argument("--backend", choices=["manual", "sdpa"], default="manual")
    p.add_argument("--tied", action="store_true")
    p.add_argument("--kv-heads", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--output", type=Path, default=Path("checkpoints/text.pt"))
    return p


def prepare_examples(args, tokenizer):
    records = load_jsonl(args.data) if args.data else None
    if records is not None and not records:
        raise ValueError("資料檔案不能是空的；不會自動改用 toy 資料")
    if args.task == "text":
        documents = [r["text"] for r in records] if records is not None else toy_documents()
        examples = []
        for text in documents:
            ids = [tokenizer.bos_id] + tokenizer.encode(text) + [tokenizer.eos_id]
            for start in range(0, len(ids) - 1, args.max_length):
                examples.append(shifted(ids[start : start + args.max_length + 1]))
        return examples
    if args.task in ("sft", "distill"):
        messages = []
        for r in records or []:
            if "messages" in r:
                messages.append(r["messages"])
            elif "question" in r and "answer" in r:
                messages.append(
                    [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}]
                )
            else:
                raise ValueError("SFT JSONL 需要 messages 或 question/answer")
        return [render_chat(m, tokenizer) for m in messages or toy_conversations()]
    if args.task == "dpo":
        pairs = records or [{"prompt": "1+1=?", "chosen": "2", "rejected": "3"}]
        result = []
        for r in pairs:
            converted = []
            for key in ("chosen", "rejected"):
                value = r[key]
                messages = (
                    value
                    if isinstance(value, list)
                    else [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": value}]
                )
                converted.append(render_chat(messages, tokenizer))
            result.append(tuple(converted))
        return result
    return None


def modal_loss(model, task, tok, device, record=None, directory=None):
    if record is not None:
        ids, labels, image, wave = modal_example(record, directory, task, device)
        output = model(ids, labels, image=image, waveform=wave)
        return masked_loss(output["logits"], output["labels"]) + 0.01 * output["auxiliary"]
    color, shape, high = (
        random.choice(("red", "green", "blue")),
        random.choice(("circle", "square")),
        random.choice((False, True)),
    )
    modalities = ([] if task == "audio" else [tok.image_id]) + ([] if task == "vision" else [tok.audio_id])
    answer = (
        shape
        if task == "vision"
        else ("high" if high else "low")
        if task == "audio"
        else f"{shape},{'high' if high else 'low'}"
    )
    question = {"vision": "shape?", "audio": "pitch?", "joint": "joint?"}[task]
    ids = [tok.bos_id, tok.user_id] + modalities + tok.encode(question) + [tok.eos_id, tok.assistant_id]
    labels = [-100] * len(ids)
    tail = tok.encode(answer) + [tok.eos_id]
    ids += tail
    labels += tail
    output = model(
        torch.tensor(ids, device=device),
        torch.tensor(labels, device=device),
        image=None if task == "audio" else scene(color, shape).to(device),
        waveform=None if task == "vision" else tone(880.0 if high else 220.0).to(device),
    )
    return masked_loss(output["logits"], output["labels"]) + 0.01 * output["auxiliary"]


def main():
    args = parser().parse_args()
    if args.steps < 1 or args.save_every < 1 or args.batch_size < 1 or args.max_length < 2:
        raise ValueError("steps/batch-size 為正，max-length 至少2")
    if args.resume and args.checkpoint is None:
        raise ValueError("--resume 必須配合 --checkpoint")
    if args.resume and args.task in ("vision", "audio", "joint", "dpo", "distill"):
        raise ValueError("最小 resume 入口支援 text/sft；進階任務請明確重建各階段 optimizer")
    if args.task == "distill" and args.train and args.teacher is None:
        raise ValueError("正式蒸餾需要 --teacher；沒有教師權重時只能做 dry-run")
    seed_everything(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device, tok = choose_device(args.device), ByteTokenizer()
    payload = None
    if args.checkpoint:
        model, payload = load_checkpoint(args.checkpoint, device, restore_rng=args.resume)
        if payload["format_version"] != 1:
            raise ValueError("訓練請載入浮點 checkpoint；packed 版本供推論與評估")
        args.max_length = model.config.max_length
    else:
        config = ModelConfig(
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            max_length=args.max_length,
            experts=args.experts,
            top_k=args.top_k,
            rotary=args.rotary,
            norm=args.norm,
            activation=args.activation,
            backend=args.backend,
            tied=args.tied,
            kv_heads=args.kv_heads,
        )
        model = TinyLM(config).to(device)
    if model.config.vocab_size != tok.vocab_size:
        raise ValueError("CLI 的 byte tokenizer 與這個 checkpoint 詞表不相容")
    reference = teacher = None
    if args.task == "dpo":
        reference = copy.deepcopy(model).eval().requires_grad_(False)
    if args.task == "distill":
        teacher = load_checkpoint(args.teacher, device)[0] if args.teacher else copy.deepcopy(model)
        if teacher.config.vocab_size != tok.vocab_size or teacher.config.max_length < model.config.max_length:
            raise ValueError("教師詞表不相容或 context 太短")
        teacher.eval().requires_grad_(False)
    train_model = MultiModalLM(model).to(device) if args.task in ("vision", "audio", "joint") else model
    if train_model is model and (args.freeze != "none" or args.vision_encoder or args.audio_encoder):
        raise ValueError("encoder與freeze選項只適用模態任務")
    if train_model is not model:
        for path, encoder in ((args.vision_encoder, train_model.vision), (args.audio_encoder, train_model.audio)):
            if path:
                encoder.load_state_dict(torch.load(path, map_location=device, weights_only=True)["encoder"])
        if args.freeze != "none":
            train_model.requires_grad_(False)
            train_model.image_projector.requires_grad_(True)
            train_model.audio_projector.requires_grad_(True)
            if args.freeze == "partial":
                train_model.language.blocks[0].requires_grad_(True)
                train_model.language.blocks[-1].requires_grad_(True)
    examples = prepare_examples(args, tok)
    modal_records = load_jsonl(args.data) if examples is None and args.data else None
    if modal_records is not None and not modal_records:
        raise ValueError("模態資料不能是空檔案")
    optimizer = torch.optim.AdamW([p for p in train_model.parameters() if p.requires_grad], lr=args.lr)
    start = 0
    if args.resume:
        if payload["optimizer"] is None:
            raise ValueError("checkpoint 没有 optimizer 狀態")
        optimizer.load_state_dict(payload["optimizer"])
        start = payload["step"]
    # --steps為這個排程的總步數；resume只跑還沒完成的部分。
    steps = max(0, args.steps - start) if args.train else 1
    if args.train and steps == 0:
        raise ValueError("checkpoint已達指定總步數；增加 --steps 才有後續更新")
    history, began = [], time.perf_counter()
    for step in range(start, start + steps):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.lr)
        if examples is None:
            # 為了先看清楚展開與 labels，逐筆處理；這是等例權重的平均。
            losses = [
                modal_loss(
                    train_model,
                    args.task,
                    tok,
                    device,
                    random.choice(modal_records) if modal_records else None,
                    args.data.parent if args.data else None,
                )
                for _ in range(args.batch_size)
            ]
            loss = torch.stack(losses).mean()
            effective = None
        else:
            selected = random.choices(examples, k=args.batch_size)
            if args.task == "dpo":
                scores = []
                for side in (0, 1):
                    x, y, valid = (
                        t.to(device) for t in pad_batch([e[side] for e in selected], max_length=args.max_length)
                    )
                    policy = sequence_log_probability(model(x, valid=valid)["logits"], y)
                    with torch.no_grad():
                        baseline = sequence_log_probability(reference(x, valid=valid)["logits"], y)
                    scores.append((policy, baseline))
                loss = dpo_loss(scores[0][0], scores[1][0], scores[0][1], scores[1][1], args.beta)
                effective = None
            else:
                x, y, valid = (t.to(device) for t in pad_batch(selected, max_length=args.max_length))
                output = model(x, valid=valid)
                if teacher is None:
                    loss = masked_loss(output["logits"], y)
                else:
                    with torch.no_grad():
                        teaching = teacher(x, valid=valid)["logits"]
                    loss = distillation_loss(output["logits"], teaching, y, args.alpha, args.temperature)
                loss = loss + 0.01 * output["auxiliary"]
                effective = int((y != -100).sum())
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0))
        if args.train:
            optimizer.step()
        entry = {"step": step + 1, "loss": float(loss.detach()), "effective_tokens": effective, "grad_norm": grad_norm}
        history.append(entry)
        print(json.dumps(entry))
        if args.train and args.task in ("text", "sft") and (step + 1) % args.save_every == 0:
            save_checkpoint(
                args.output,
                model,
                optimizer,
                step + 1,
                {
                    "task": args.task,
                    "seed": args.seed,
                    "data": str(args.data),
                    "schedule_steps": args.steps,
                    "peak_lr": args.lr,
                },
            )
    report = {
        "mode": "train" if args.train else "dry-run-no-weight-update",
        "task": args.task,
        "device": str(device),
        "seconds": time.perf_counter() - began,
        "config": asdict(model.config),
        "parameters": sum(p.numel() for p in train_model.parameters()),
        "history": history,
    }
    if args.train:
        save_checkpoint(
            args.output,
            model,
            optimizer if train_model is model else None,
            start + steps,
            {"task": args.task, "seed": args.seed, "data": str(args.data), "report": report},
        )
        if train_model is not model:
            torch.save(
                {"model": train_model.state_dict(), "config": asdict(model.config), "task": args.task},
                args.output.with_suffix(".modal.pt"),
            )
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
