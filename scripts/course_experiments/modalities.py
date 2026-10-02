"""圖像、聲音與跨模態的受控實驗；測量值與人工生成規則分開保存。"""

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import masked_loss
from tiny_perceptron.multimodal import AudioEncoder, MultiModalLM, VisionEncoder, generate_modal, scene, tone
from tiny_perceptron.training import load_checkpoint, save_checkpoint, seed_everything

from .common import extract_asset, write_json


def _steps(ctx, normal=300, smoke=12):
    return smoke if str(ctx.device).startswith("cpu") else normal


def _sync(ctx):
    if str(ctx.device).startswith("cuda"):
        torch.cuda.synchronize()


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _manifest(ctx, name, splits, policy):
    identities = {key: {row["family"] for row in rows} for key, rows in splits.items()}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if identities[first] & identities[second]:
            raise ValueError(f"{name}: {first}/{second} 題目家族洩漏")
    result = {
        "seed": ctx.seed,
        "split_policy": policy,
        "splits": {key: {"count": len(rows), "sha256": _hash(rows), "records": rows} for key, rows in splits.items()},
    }
    write_json(ctx.output / f"{name}-data.json", result)
    write_json(ctx.output / "dataset.json", splits)
    return result


def _summary(model):
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if isinstance(model, MultiModalLM):
        result["config"] = asdict(model.language.config)
        result["modal_config"] = {
            "vision_width": model.vision.projection.out_features,
            "audio_width": model.audio.projection.out_features,
            "image_size": model.vision.image_size,
            "patch_size": model.vision.patch_size,
            "bands": model.audio.bands,
        }
    return result


def _fit(model, loss_fn, ctx, steps, name="model", task=None, lr=0.003, token_budget=None):
    """loss_fn 回傳 loss 與有效目標數；固定 probe 不消耗訓練抽樣的 RNG。"""
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("没有可訓練參數")
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    originals = [p.detach().clone() for p in parameters]
    model.eval()
    with torch.no_grad():
        initial_loss = float(loss_fn(-1)[0])
    model.train()
    history, effective = [], 0
    _sync(ctx)
    started = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, count = loss_fn(step)
        if not bool(torch.isfinite(loss)):
            raise ValueError("訓練 loss 非有限值")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        if not math.isfinite(norm):
            raise ValueError("訓練梯度非有限值")
        optimizer.step()
        effective += int(count)
        history.append(
            {"step": step + 1, "loss": float(loss.detach()), "grad_norm": norm, "effective_targets": int(count)}
        )
        if isinstance(model, MultiModalLM) and (step + 1) % max(1, steps // 4) == 0:
            save_checkpoint(
                ctx.output / f"{name}-step-{step + 1}.pt",
                model,
                optimizer,
                step + 1,
                {"task": task, "experiment": name, "effective_tokens": effective, "lr": lr, "schedule": "constant"},
            )
        if token_budget is not None and effective >= token_budget:
            break
    _sync(ctx)
    seconds = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        final_loss = float(loss_fn(-1)[0])
    changed = any(not torch.equal(old, p.detach()) for old, p in zip(originals, parameters, strict=True))
    result = {
        **_summary(model),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_probe": "same deterministic batch before and after training",
        "history": history,
        "steps": len(history),
        "seconds": seconds,
        "elapsed": seconds,
        "timing_scope": "optimizer loop including logging, periodic local checkpoints and CUDA synchronization; excludes probes, evaluation and final file write",
        "effective_tokens": effective if isinstance(model, MultiModalLM) else None,
        "effective_targets": effective,
        "weights_changed": changed,
        "nonzero_gradient_seen": any(row["grad_norm"] > 0 for row in history),
        "cpu_smoke": str(ctx.device).startswith("cpu"),
        "checkpoint": f"{name}.pt",
    }
    if token_budget is not None:
        result.update(requested_effective_tokens=token_budget, budget_excess=effective - token_budget)
    ctx.output.mkdir(parents=True, exist_ok=True)
    path = ctx.output / f"{name}.pt"
    if isinstance(model, MultiModalLM):
        save_checkpoint(
            path,
            model,
            optimizer,
            len(history),
            {
                "task": task,
                "experiment": name,
                "report": result,
                "seed": ctx.seed,
                "lr": lr,
                "schedule": "constant",
                "records_sha256": getattr(loss_fn, "records_sha256", None),
            },
        )
    else:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": len(history),
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "report": result,
            },
            path,
        )
    if not changed:
        raise ValueError("optimizer 執行後權重沒有改變")
    return result


def _vision_records(questions=("describe",)):
    splits = {}
    for split, offsets in (("train", (-2, -1, 0)), ("validation", (1,)), ("test", (2,))):
        rows = []
        for color in ("red", "green", "blue"):
            for shape in ("circle", "square"):
                for offset in offsets:
                    for question in questions:
                        answer = {"describe": f"{color} {shape}", "shape?": shape, "color?": color}[question]
                        rows.append(
                            {
                                "modality": "vision",
                                "color": color,
                                "shape": shape,
                                "offset": offset,
                                "question": question,
                                "answer": answer,
                                "family": f"{color}:{shape}:{offset}",
                            }
                        )
        splits[split] = rows
    return splits


def _audio_records():
    splits = {}
    frequencies = {
        "train": (140, 180, 220, 260, 340, 380, 420, 460),
        "validation": (160, 240, 360, 440),
        "test": (200, 280, 290, 300, 310, 320, 400),
    }
    for split, values in frequencies.items():
        splits[split] = [
            {
                "modality": "audio",
                "frequency": f,
                "amplitude": amplitude,
                "seconds": seconds,
                "question": "pitch?",
                "answer": "high" if f > 300 else "low",
                "family": str(f),
            }
            for f in values
            for amplitude, seconds in ((0.25, 0.1), (0.5, 0.12))
        ]
    return splits


def _media(row, ctx):
    image = waveform = None
    if row["modality"] in ("vision", "joint"):
        if "image" in row:
            with Image.open(row["image"]) as source:
                pixels = np.array(source.convert("RGB").resize((16, 16)), copy=True)
            image = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
        elif "digits" in row:
            from scripts.prepare_ocr import draw_digits

            pixels = np.array(draw_digits(row["digits"], row["offset"]), copy=True)
            image = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
        else:
            image = scene(row["color"], row["shape"], offset=row["offset"])
    if row["modality"] in ("audio", "joint"):
        if "audio" in row:
            values, sample_rate = sf.read(row["audio"], dtype="float32")
            if values.ndim != 1 or not len(values):
                raise ValueError("錄音必須是非空單聲道")
            waveform = torch.from_numpy(values.copy())
            if sample_rate != 16000:
                raise ValueError("必須先真正重採樣為 16 kHz")
        else:
            waveform = tone(row["frequency"], seconds=row.get("seconds", 0.1)) * (row.get("amplitude", 0.5) / 0.5)
    return (None if image is None else image.to(ctx.device), None if waveform is None else waveform.to(ctx.device))


def _sequence(row, ctx, answer=True):
    tok = ByteTokenizer()
    markers = ([] if row["modality"] == "audio" else [tok.image_id]) + (
        [] if row["modality"] == "vision" else [tok.audio_id]
    )
    prefix = [tok.bos_id, tok.user_id] + markers + tok.encode(row["question"]) + [tok.eos_id, tok.assistant_id]
    tail = tok.encode(row["answer"]) + [tok.eos_id] if answer else []
    return (
        torch.tensor(prefix + tail, device=ctx.device),
        torch.tensor([-100] * len(prefix) + tail, device=ctx.device),
        len(tail),
    )


def _loss_fn(model, records, ctx, batch=4, replay=None, replay_ratio=0):
    def calculate(step):
        rng = random.Random(ctx.seed + step)
        totals, counts = [], []
        for _ in range(batch):
            if replay and rng.random() < replay_ratio:
                from tiny_perceptron.data import render_chat

                x, y = render_chat(rng.choice(replay)["messages"])
                y = y.to(ctx.device)[None]
                output = model.language(x.to(ctx.device)[None])
                count = int((y != -100).sum())
            else:
                row = rng.choice(records)
                ids, labels, count = _sequence(row, ctx)
                image, waveform = _media(row, ctx)
                output = model(ids, labels, image=image, waveform=waveform)
                y = output["labels"]
            totals.append(masked_loss(output["logits"], y) * count)
            counts.append(count)
        return torch.stack(totals).sum() / sum(counts), sum(counts)

    calculate.records_sha256 = _hash(records)
    return calculate


@torch.no_grad()
def _evaluate(model, records, ctx, ablation="none", tokens=16):
    model.eval()
    tok = ByteTokenizer()
    samples, total_nll, effective = [], 0.0, 0
    for index, row in enumerate(records):
        image, waveform = _media(row, ctx)
        donor = None
        if ablation in ("blank", "blank_image") and image is not None:
            image = torch.zeros_like(image)
        if ablation in ("blank", "blank_audio") and waveform is not None:
            waveform = torch.zeros_like(waveform)
        if ablation == "shuffle":
            donor = (index + max(1, len(records) // 2)) % len(records)
            image, waveform = _media(records[donor], ctx)
        ids, labels, count = _sequence(row, ctx)
        output = model(ids, labels, image=image, waveform=waveform)
        total_nll += float(masked_loss(output["logits"], output["labels"])) * count
        effective += count
        prefix = _sequence(row, ctx, False)[0]
        generation_error = None
        try:
            generated = generate_modal(model, prefix, image, waveform, tokens)
            generated_ids = generated[len(prefix) :].tolist()
        except ValueError as error:
            if "placeholder 缺少配對" not in str(error):
                raise
            # 未學模型產生錯誤模態標記也是失敗，不可略過這道題。
            generation_error = str(error)
            generated_ids = []
        raw = generated_ids[: generated_ids.index(tok.eos_id)] if tok.eos_id in generated_ids else generated_ids
        answer = tok.decode(raw)
        samples.append(
            {
                "row": index,
                "family": row["family"],
                "question": row["question"],
                "target": row["answer"],
                "generated": answer,
                "generated_ids": generated_ids,
                "exact_match": raw == tok.encode(row["answer"]),
                "eos": tok.eos_id in generated_ids,
                "generation_error": generation_error,
                "invalid_special_tokens": sum(token < 8 for token in raw),
                "donor_row": donor,
            }
        )
    if not samples or not effective:
        raise ValueError("評估集沒有有效題目")
    result = {
        "examples": len(samples),
        "correct": sum(row["exact_match"] for row in samples),
        "exact_match": sum(row["exact_match"] for row in samples) / len(samples),
        "mean_token_nll": total_nll / effective,
        "effective_tokens": effective,
        "eos_rate": sum(row["eos"] for row in samples) / len(samples),
        "generation_errors": sum(row["generation_error"] is not None for row in samples),
        "invalid_special_tokens": sum(row["invalid_special_tokens"] for row in samples),
        "ablation": ablation,
        "skipped": [],
        "samples": samples,
    }
    if records[0]["modality"] == "joint":
        for part, label in ((0, "shape"), (1, "pitch")):
            correct = sum(
                len(s["generated"].split(",")) == 2 and s["generated"].split(",")[part] == s["target"].split(",")[part]
                for s in samples
            )
            result[f"{label}_correct"] = correct
            result[f"{label}_accuracy"] = correct / len(samples)
    groups = {}
    for row, sample in zip(records, samples, strict=True):
        key = row.get("question", "all")
        group = groups.setdefault(key, {"correct": 0, "count": 0})
        group["count"] += 1
        group["correct"] += int(sample["exact_match"])
    result["groups"] = groups
    result["macro_accuracy"] = sum(g["correct"] / g["count"] for g in groups.values()) / len(groups)
    return result


def _modal(ctx, modality="vision", pretrained=True):
    language, _ = load_checkpoint(ctx.dependency("sft"), ctx.device)
    model = MultiModalLM(language).to(ctx.device)
    if pretrained:
        for name in ("vision", "audio"):
            saved = torch.load(ctx.dependency("encoders", f"{name}.pt"), map_location="cpu", weights_only=True)
            getattr(model, name).load_state_dict(saved["encoder"], strict=True)
    return model


def _freeze(model, scope, modality="vision"):
    model.requires_grad_(scope == "all")
    if modality in ("vision", "joint"):
        model.image_projector.requires_grad_(True)
    if modality in ("audio", "joint"):
        model.audio_projector.requires_grad_(True)
    if scope == "partial":
        model.language.blocks[-1].requires_grad_(True)


def run_encoders(ctx):
    seed_everything(ctx.seed)
    ctx.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for modality in ("vision", "audio"):
        splits = _vision_records() if modality == "vision" else _audio_records()
        manifest = _manifest(
            ctx, modality, splits, "vision offset groups; audio frequency groups; variants stay together"
        )
        encoder = (VisionEncoder() if modality == "vision" else AudioEncoder()).to(ctx.device)
        classes = (
            [f"{color} {shape}" for color in ("red", "green", "blue") for shape in ("circle", "square")]
            if modality == "vision"
            else ["low", "high"]
        )
        classifier = nn.Linear(16, len(classes)).to(ctx.device)
        # classifier 一併註冊，使 optimizer 確實更新兩端。
        model = nn.ModuleDict({"encoder": encoder, "classifier": classifier})

        def classification(rows):
            inputs = [_media(row, ctx)[0 if modality == "vision" else 1] for row in rows]
            if modality == "vision":
                features = encoder(torch.stack(inputs)).mean(1)
            else:
                # 不同時長錄音各自編碼，避免把補零框平均成聲學特徵。
                features = torch.cat([encoder(wave[None]).mean(1) for wave in inputs])
            logits = classifier(features)
            labels = torch.tensor([classes.index(row["answer"]) for row in rows], device=ctx.device)
            return logits, labels

        def loss_fn(step):
            rows = random.Random(ctx.seed + step).choices(splits["train"], k=8)
            logits, labels = classification(rows)
            return F.cross_entropy(logits, labels), len(rows)

        def evaluate(rows):
            with torch.no_grad():
                logits, labels = classification(rows)
                selected = logits.argmax(-1).tolist()
            samples = [
                {
                    "row": i,
                    "family": row["family"],
                    "target": row["answer"],
                    "predicted": classes[prediction],
                    "correct": classes[prediction] == row["answer"],
                }
                for i, (row, prediction) in enumerate(zip(rows, selected, strict=True))
            ]
            return {
                "count": len(rows),
                "correct": sum(s["correct"] for s in samples),
                "accuracy": sum(s["correct"] for s in samples) / len(samples),
                "samples": samples,
            }

        before = evaluate(splits["test"])
        trained = _fit(model, loss_fn, ctx, _steps(ctx, 250, 20), name=f"{modality}-training")

        def confidence_metrics(logits, labels, temperature):
            probabilities = (logits / temperature).softmax(-1)
            # 正溫度保持 logits 的排名；用 logits 避免 softmax 捨入造成假平手。
            predicted = (logits / temperature).argmax(-1)
            correct = predicted == labels
            confidence = probabilities.max(-1).values
            one_hot = F.one_hot(labels, num_classes=len(classes)).to(probabilities.dtype)
            bins, ece = [], 0.0
            # [0,.2), [.2,.4), ..., [.8,1]；每題只進一格，空格也保存。
            bin_ids = (confidence * 5).floor().long().clamp(max=4)
            for bin_id in range(5):
                selected = bin_ids == bin_id
                count = int(selected.sum())
                mean_confidence = float(confidence[selected].mean()) if count else None
                accuracy = int(correct[selected].sum()) / count if count else None
                if count:
                    ece += count / len(labels) * abs(accuracy - mean_confidence)
                bins.append(
                    {
                        "lower": bin_id / 5,
                        "upper": (bin_id + 1) / 5,
                        "upper_inclusive": bin_id == 4,
                        "count": count,
                        "mean_confidence": mean_confidence,
                        "accuracy": accuracy,
                    }
                )
            return {
                "temperature": temperature,
                "count": len(labels),
                "correct": int(correct.sum()),
                "accuracy": int(correct.sum()) / len(labels),
                "mean_confidence": float(confidence.mean()),
                "nll": float(F.cross_entropy(logits / temperature, labels)),
                "brier": float((probabilities - one_hot).square().sum(-1).mean()),
                "brier_definition": "mean over examples of sum over classes (probability - one_hot_label)^2",
                "ece": ece,
                "ece_definition": "sum over five equal-width confidence bins: count/N * abs(bin accuracy - bin mean confidence)",
                "bins": bins,
                "probabilities": probabilities.tolist(),
                "predicted_labels": predicted.tolist(),
                "confidence": confidence.tolist(),
            }

        with torch.no_grad():
            validation_logits, validation_labels = classification(splits["validation"])
        validation_logits = validation_logits.detach().float().cpu()
        validation_labels = validation_labels.detach().cpu()
        temperature_grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
        selection = [
            {
                "temperature": value,
                "validation_nll": float(F.cross_entropy(validation_logits / value, validation_labels)),
            }
            for value in temperature_grid
        ]
        # 到這裡才固定溫度；test logits/labels 沒有參與任何選溫判準。
        chosen_temperature = min(selection, key=lambda candidate: candidate["validation_nll"])["temperature"]
        with torch.no_grad():
            test_logits, test_labels = classification(splits["test"])
        test_logits = test_logits.detach().float().cpu()
        test_labels = test_labels.detach().cpu()
        if not torch.equal(test_logits.argmax(-1), (test_logits / chosen_temperature).argmax(-1)):
            raise AssertionError("正溫度校準不應改變這批分類的 argmax")
        calibration = {
            "method": "post-hoc positive temperature scaling; fixed-grid teaching approximation",
            "source": "Guo et al. (2017), On Calibration of Modern Neural Networks",
            "source_url": "https://proceedings.mlr.press/v70/guo17a.html",
            "scope": "confidence for one of a fixed finite set of classifier labels, not probability an entire generated answer is correct",
            "temperature_grid": temperature_grid,
            "selection_split": "validation only",
            "selection_rule": "minimum mean validation NLL; exact ties use first item in the predefined grid",
            "selection": selection,
            "chosen_temperature": chosen_temperature,
            "validation": {
                "count": len(validation_labels),
                "logits": validation_logits.tolist(),
                "labels": validation_labels.tolist(),
                "families": [row["family"] for row in splits["validation"]],
                "original": confidence_metrics(validation_logits, validation_labels, 1.0),
                "calibrated": confidence_metrics(validation_logits, validation_labels, chosen_temperature),
            },
            "test": {
                "count": len(test_labels),
                "logits": test_logits.tolist(),
                "labels": test_labels.tolist(),
                "families": [row["family"] for row in splits["test"]],
                "original": confidence_metrics(test_logits, test_labels, 1.0),
                "calibrated": confidence_metrics(test_logits, test_labels, chosen_temperature),
            },
            "test_argmax_invariant": True,
            "limitation": "held-out sets are very small and synthetic; bin estimates and temperature choice are unstable, and validation improvement does not guarantee test improvement or generalization",
        }
        write_json(ctx.output / f"{modality}-calibration.json", calibration)
        result = {
            "training": trained,
            "before": before,
            "validation": evaluate(splits["validation"]),
            "test": evaluate(splits["test"]),
            "data": manifest,
            "config": {"width": 16, "image_size": 16, "patch_size": 4}
            if modality == "vision"
            else {"width": 16, "bands": 16},
            "classes": classes,
            "calibration": calibration,
            "scope": "synthetic colour/shape classification or tone >300 Hz; not natural perception",
        }
        training_payload = torch.load(ctx.output / f"{modality}-training.pt", map_location="cpu", weights_only=True)
        torch.save(
            {
                **training_payload,
                "encoder": encoder.state_dict(),
                "classifier": classifier.state_dict(),
                "config": result["config"],
                "classes": classes,
                "report": result,
            },
            ctx.output / f"{modality}.pt",
        )
        results[modality] = result
    return results


class _Contrastive(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = VisionEncoder()
        self.text = nn.Embedding(264, 16)

    def scores(self, images, texts):
        tok = ByteTokenizer()
        image_features = F.normalize(self.vision(images).mean(1), dim=-1)
        text_features = torch.stack(
            [self.text(torch.tensor(tok.encode(text), device=images.device)).mean(0) for text in texts]
        )
        return image_features @ F.normalize(text_features, dim=-1).T / 0.1


def run_contrastive(ctx):
    seed_everything(ctx.seed)
    splits = _vision_records()
    manifest = _manifest(
        ctx, "contrastive", splits, "offset-held-out; each batch contains one image per unique colour/shape description"
    )
    labels = [f"{c} {s}" for c in ("red", "green", "blue") for s in ("circle", "square")]
    reports = {}
    for direction in ("one_way", "two_way"):
        seed_everything(ctx.seed)
        model = _Contrastive().to(ctx.device)

        def loss_fn(step):
            rng = random.Random(ctx.seed + step)
            rows = [rng.choice([r for r in splits["train"] if r["answer"] == label]) for label in labels]
            scores = model.scores(torch.stack([_media(row, ctx)[0] for row in rows]), labels)
            targets = torch.arange(len(labels), device=ctx.device)
            loss = F.cross_entropy(scores, targets)
            if direction == "two_way":
                loss = (loss + F.cross_entropy(scores.T, targets)) / 2
            return loss, len(labels)

        def evaluate(rows):
            with torch.no_grad():
                scores = model.scores(torch.stack([_media(row, ctx)[0] for row in rows]), labels)
            image_samples = [
                {"row": i, "target": row["answer"], "predicted": labels[j], "correct": labels[j] == row["answer"]}
                for i, (row, j) in enumerate(zip(rows, scores.argmax(-1).tolist(), strict=True))
            ]
            text_samples = [
                {
                    "query": label,
                    "image_row": j,
                    "target": label,
                    "retrieved": rows[j]["answer"],
                    "correct": rows[j]["answer"] == label,
                }
                for label, j in zip(labels, scores.argmax(0).tolist(), strict=True)
            ]
            return {
                "image_queries": len(image_samples),
                "text_queries": len(text_samples),
                "image_to_text_accuracy": sum(s["correct"] for s in image_samples) / len(image_samples),
                "text_to_image_accuracy": sum(s["correct"] for s in text_samples) / len(text_samples),
                "image_samples": image_samples,
                "text_samples": text_samples,
            }

        before = evaluate(splits["test"])
        training = _fit(model, loss_fn, ctx, _steps(ctx, 250, 20), name=direction)
        reports[direction] = {
            "training": training,
            "before": before,
            "validation": evaluate(splits["validation"]),
            "test": evaluate(splits["test"]),
        }
        if direction == "two_way":
            (ctx.output / "model.pt").write_bytes((ctx.output / "two_way.pt").read_bytes())
    return {
        "data": manifest,
        "variants": reports,
        "config": {
            "image_size": 16,
            "patch_size": 4,
            "width": 16,
            "temperature": 0.1,
            "text_encoder": "mean byte embeddings",
        },
        "scope": "six known descriptions, unseen image positions; not free-form caption generation",
    }


def run_projector(ctx):
    seed_everything(ctx.seed)
    splits = _vision_records()
    manifest = _manifest(ctx, "projector", splits, "all prompts derived from an image stay in its offset family")
    model = _modal(ctx)
    _freeze(model, "projector")
    before = _evaluate(model, splits["test"], ctx)
    language_before = {k: v.detach().clone() for k, v in model.language.state_dict().items()}
    training = _fit(model, _loss_fn(model, splits["train"], ctx), ctx, _steps(ctx, 300), task="vision")
    return {
        "training": training,
        "before": before,
        "validation": _evaluate(model, splits["validation"], ctx),
        "test": _evaluate(model, splits["test"], ctx),
        "data": manifest,
        "language_weights_unchanged": all(
            torch.equal(language_before[k], v) for k, v in model.language.state_dict().items()
        ),
        "scope": "trained projector with frozen self-trained encoders/language; failures remain evidence",
    }


def _replay_records():
    return [
        {
            "messages": [
                {"role": "user", "content": f"color={c};shape={s};pitch=low;shape?"},
                {"role": "assistant", "content": s},
            ]
        }
        for c in ("red", "green", "blue")
        for s in ("circle", "square")
    ]


def run_vqa(ctx):
    seed_everything(ctx.seed)
    splits = _vision_records(("shape?", "color?"))
    manifest = _manifest(ctx, "vqa", splits, "all questions of one colour/shape/offset stay together")
    initial, upstream = load_checkpoint(ctx.dependency("projector"), ctx.device)
    if not isinstance(initial, MultiModalLM):
        raise ValueError("projector 必須是包含已學接頭的完整多模態 checkpoint")
    variants = {}
    text_splits = json.loads(ctx.dependency("sft", "dataset.json").read_text())
    replay = text_splits["train"]
    text_holdout = text_splits["test"]
    text_validation = text_splits["validation"]
    from .common import evaluate_lm

    for name, scope, ratio in (
        ("projector_only", "projector", 0),
        ("partial", "partial", 0),
        ("all", "all", 0),
        ("all_replay", "all", 0.5),
    ):
        seed_everything(ctx.seed)
        model = copy.deepcopy(initial)
        _freeze(model, scope)
        before = _evaluate(model, splits["test"], ctx)
        text_before = evaluate_lm(model.language, text_holdout, mode="sft")
        training = _fit(
            model,
            _loss_fn(model, splits["train"], ctx, replay=replay, replay_ratio=ratio),
            ctx,
            _steps(ctx, 160, 8),
            name=name,
            task="vision",
        )
        variants[name] = {
            "training": training,
            "before": before,
            "validation": _evaluate(model, splits["validation"], ctx),
            "test": _evaluate(model, splits["test"], ctx),
            "text_before": text_before,
            "text_after": evaluate_lm(model.language, text_holdout, mode="sft"),
            "text_validation": evaluate_lm(model.language, text_validation, mode="sft"),
            "text_probe_is_training_diagnostic": False,
            "freeze_scope": "last language block + image projector" if scope == "partial" else scope,
            "replay_probability_per_example": ratio,
        }
        if name == "all":
            (ctx.output / "model.pt").write_bytes((ctx.output / "all.pt").read_bytes())
    seed_everything(ctx.seed)
    direct = _modal(ctx)
    _freeze(direct, "all")
    before = _evaluate(direct, splits["test"], ctx)
    budget = variants["all"]["training"]["effective_tokens"] + upstream["metadata"]["report"]["effective_tokens"]
    training = _fit(
        direct,
        _loss_fn(direct, splits["train"], ctx),
        ctx,
        _steps(ctx, 850, 48),
        name="direct_vqa",
        task="vision",
        token_budget=budget,
    )
    variants["direct_vqa"] = {
        "training": training,
        "before": before,
        "validation": _evaluate(direct, splits["validation"], ctx),
        "test": _evaluate(direct, splits["test"], ctx),
    }
    return {
        "data": manifest,
        "variants": variants,
        "two_stage_alignment_training": upstream["metadata"]["report"],
        "text_data": {name: {"count": len(rows), "sha256": _hash(rows)} for name, rows in text_splits.items()},
        "direct_vs_two_stage_budget": {
            "target_effective_tokens": budget,
            "direct_effective_tokens": training["effective_tokens"],
            "matched_to_within_final_batch": 0 <= training["budget_excess"] <= 40,
        },
        "scope": "controlled colour/shape VQA; pure-text retention uses the original SFT held-out test families, replay uses train only",
    }


def run_vision_ablation(ctx):
    splits = _vision_records(("shape?", "color?"))
    manifest = _manifest(ctx, "vision-ablation", splits, "held-out offset families, paired test interventions")
    source, _ = load_checkpoint(ctx.dependency("vqa"), ctx.device)
    seed_everything(ctx.seed)
    model = copy.deepcopy(source)
    _freeze(model, "all")
    before = _evaluate(model, splits["test"], ctx)
    training = _fit(model, _loss_fn(model, splits["train"], ctx), ctx, _steps(ctx, 100, 8), task="vision")
    results = {ablation: _evaluate(model, splits["test"], ctx, ablation) for ablation in ("none", "blank", "shuffle")}
    # 真替圖須改 target；這與打亂配對後保持原 target 的消融不同。
    swapped = [
        dict(
            row,
            shape="square" if row["shape"] == "circle" else "circle",
            answer=("square" if row["shape"] == "circle" else "circle")
            if row["question"] == "shape?"
            else row["answer"],
        )
        for row in splits["test"]
    ]
    results["shape_swap_relabelled"] = _evaluate(model, swapped, ctx)
    crop_rows = []
    ctx.output.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(splits["test"]):
        image, _ = _media(dict(row, offset=7), ctx)
        image = image.cpu()[:, 4:12, 4:12]
        path = ctx.output / f"crop-{i}.png"
        Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype("uint8")).resize((16, 16)).save(path)
        crop_rows.append(dict(row, image=str(path)))
    results["edge_cropped"] = _evaluate(model, crop_rows, ctx)
    token_variants = {}
    for patch in (4, 8):
        seed_everything(ctx.seed)
        candidate = _modal(ctx, pretrained=False)
        candidate.vision = VisionEncoder(patch_size=patch).to(ctx.device)
        _freeze(candidate, "all")
        report = _fit(
            candidate,
            _loss_fn(candidate, splits["train"], ctx),
            ctx,
            _steps(ctx, 200, 8),
            name=f"patch-{patch}",
            task="vision",
        )
        token_variants[str(patch)] = {
            "visual_tokens": (16 // patch) ** 2,
            "training": report,
            "test": _evaluate(candidate, splits["test"], ctx),
        }
    return {
        "training": training,
        "before": before,
        "data": manifest,
        "interventions": results,
        "patch_variants": token_variants,
        "patch_budget": "same initial language checkpoint, fresh patch-specific encoders, same updates/data; parameter counts differ",
        "crop_note": "edge object shifted then centre cropped; this is input-information loss, not guaranteed generalization",
    }


def run_ocr(ctx):
    seed_everything(ctx.seed)
    families = list(range(100))
    random.Random(ctx.seed).shuffle(families)
    splits = {}
    for split, values in (("train", families[:80]), ("validation", families[80:90]), ("test", families[90:])):
        splits[split] = [
            {
                "modality": "vision",
                "digits": str(value),
                "offset": offset,
                "family": str(value),
                "question": "read digits",
                "answer": str(value),
            }
            for value in values
            for offset in (-1, 0, 1)
        ]
    manifest = _manifest(ctx, "ocr", splits, "digit-string family; all position variants stay in the same split")
    model = _modal(ctx)
    _freeze(model, "all")
    before = _evaluate(model, splits["test"], ctx, tokens=5)
    training = _fit(model, _loss_fn(model, splits["train"], ctx, batch=8), ctx, _steps(ctx, 500, 20), task="vision")
    test = _evaluate(model, splits["test"], ctx, tokens=5)

    def edit_distance(a, b):
        previous = list(range(len(b) + 1))
        for i, left in enumerate(a, 1):
            current = [i]
            for j, right in enumerate(b, 1):
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
            previous = current
        return previous[-1]

    test["character_edits"] = sum(edit_distance(s["generated"], s["target"]) for s in test["samples"])
    test["reference_characters"] = sum(len(s["target"]) for s in test["samples"])
    test["character_error_rate"] = test["character_edits"] / test["reference_characters"]
    test["character_error_scope"] = (
        "decoded text; raw-token exact match rejects invalid special tokens counted separately"
    )
    return {
        "training": training,
        "before": before,
        "validation": _evaluate(model, splits["validation"], ctx, tokens=5),
        "test": test,
        "blank": _evaluate(model, splits["test"], ctx, "blank", tokens=5),
        "data": manifest,
        "scope": "one/two digit strings in a fixed 5x7 font, not handwriting or documents",
    }


def run_audio(ctx):
    seed_everything(ctx.seed)
    splits = _audio_records()
    manifest = _manifest(
        ctx,
        "audio-qa",
        splits,
        "frequency family; amplitudes and durations remain together; high iff frequency >300 Hz",
    )
    model = _modal(ctx, "audio")
    _freeze(model, "all", "audio")
    before = _evaluate(model, splits["test"], ctx, tokens=6)
    training = _fit(model, _loss_fn(model, splits["train"], ctx), ctx, _steps(ctx, 300), task="audio")
    return {
        "training": training,
        "before": before,
        "validation": _evaluate(model, splits["validation"], ctx, tokens=6),
        "test": _evaluate(model, splits["test"], ctx, tokens=6),
        "blank": _evaluate(model, splits["test"], ctx, "blank", tokens=6),
        "shuffle": _evaluate(model, splits["test"], ctx, "shuffle", tokens=6),
        "data": manifest,
        "text_only_majority_accuracy": max(
            sum(r["answer"] == label for r in splits["test"]) for label in ("low", "high")
        )
        / len(splits["test"]),
        "scope": "tone classification with explicit 300 Hz boundary; not speech transcription",
    }


def run_joint(ctx):
    seed_everything(ctx.seed)
    splits = {}
    for split, offset, frequencies in (
        ("train", 0, (180, 220, 380, 420)),
        ("validation", 1, (200, 400)),
        ("test", 2, (260, 340)),
    ):
        splits[split] = [
            {
                "modality": "joint",
                "color": color,
                "shape": shape,
                "offset": offset,
                "frequency": frequency,
                "question": "joint?",
                "answer": f"{shape},{'high' if frequency > 300 else 'low'}",
                "family": f"{color}:{shape}:{offset}:{frequency}",
            }
            for color in ("red", "green", "blue")
            for shape in ("circle", "square")
            for frequency in frequencies
        ]
    manifest = _manifest(
        ctx, "joint", splits, "held-out position and frequency; independently balanced shape and pitch"
    )
    model = _modal(ctx, "joint")
    _freeze(model, "all", "joint")
    before = _evaluate(model, splits["test"], ctx)
    training = _fit(model, _loss_fn(model, splits["train"], ctx), ctx, _steps(ctx, 400, 16), task="joint")
    results = {a: _evaluate(model, splits["test"], ctx, a) for a in ("none", "blank_image", "blank_audio", "blank")}
    for name in ("swap_image", "swap_audio"):
        changed = []
        for row in splits["test"]:
            other = dict(row)
            if name == "swap_image":
                other["shape"] = "square" if row["shape"] == "circle" else "circle"
            else:
                other["frequency"] = 340 if row["frequency"] <= 300 else 260
            other["answer"] = f"{other['shape']},{'high' if other['frequency'] > 300 else 'low'}"
            changed.append(other)
        results[name] = _evaluate(model, changed, ctx)
    return {
        "training": training,
        "before": before,
        "validation": _evaluate(model, splits["validation"], ctx),
        "evaluations": results,
        "data": manifest,
        "constant_guess_joint_accuracy": 0.25,
        "scope": "two independent required attributes; individual masking/swapping tests actual modal use",
    }


def _find(root, name):
    choices = list(Path(root).rglob(name))
    if len(choices) != 1:
        raise ValueError(f"期待唯一 {name}，找到 {len(choices)} 個")
    return choices[0]


def modal_inputs(row, ctx, include_answer=True):
    """供蒸餾共用：同一筆固定參數產生 ids、labels、image、waveform、答案 token 數。"""
    ids, labels, count = _sequence(row, ctx, include_answer)
    image, waveform = _media(row, ctx)
    return ids, labels, image, waveform, count


def evaluate_modal(model, records, ctx, ablation="none", tokens=16):
    """完整評估指定 split，不預先填答案或略過難題。"""
    return _evaluate(model, records, ctx, ablation, tokens)


def _resample_8_to_16(values):
    times = torch.arange(len(values) * 2, dtype=torch.float64) / 2
    indices = times.floor().long()[:, None] + torch.arange(-15, 17)[None]
    distance = times[:, None] - indices
    valid = (indices >= 0) & (indices < len(values))
    window = torch.where(distance.abs() < 16, 0.5 + 0.5 * torch.cos(math.pi * distance / 16), 0)
    kernel = torch.sinc(distance) * window * valid
    source = torch.from_numpy(values.astype("float64"))[indices.clamp(0, len(values) - 1)]
    return ((kernel * source).sum(-1) / kernel.sum(-1).clamp(min=1e-12)).numpy().astype("float32")


def run_real_modal(ctx):
    seed_everything(ctx.seed)
    results = {}
    for asset in ("fashion-mnist", "fsdd"):
        root = Path(extract_asset(ctx, asset))
        manifest_entry = next(
            a for a in json.loads((ctx.assets / "manifest.json").read_text())["assets"] if a["id"] == asset
        )
        root = root / manifest_entry["target"]
        source = _find(root, "train.jsonl")
        rows = [json.loads(line) for line in source.read_text().splitlines() if line]
        splits = {key: [] for key in ("train", "validation", "test")}
        conversions = []
        if asset == "fashion-mnist":
            grouped = {}
            for row in rows:
                grouped.setdefault(row.get("label", row.get("label_text_en")), []).append(row)
            for group in grouped.values():
                for index, row in enumerate(group):
                    split = "train" if index < 3 else "validation" if index == 3 else "test"
                    path = source.parent / row["image"]
                    splits[split].append(
                        {
                            "modality": "vision",
                            "image": str(path),
                            "question": row.get("question", "clothing?"),
                            "answer": str(row.get("answer", row["label_text_en"])),
                            "family": str(path.name),
                            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
            modality = "vision"
        else:
            for split in splits:
                source = _find(root, f"{split}.jsonl")
                for row in [json.loads(line) for line in source.read_text().splitlines() if line]:
                    relative = row.get("audio", row.get("path", row.get("file")))
                    if relative is None:
                        raise ValueError("FSDD record 缺少 audio 路徑")
                    path = source.parent / relative
                    values, rate = sf.read(path, dtype="float32")
                    if values.ndim != 1 or rate != 8000:
                        raise ValueError("固定 FSDD 原檔應為 8 kHz mono")
                    # 帶限 sinc 插值；8 ->16 kHz 只補取樣點，不新增高頻資訊。
                    destination = ctx.output / "resampled" / path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(destination, _resample_8_to_16(values), 16000, subtype="FLOAT")
                    conversion = {
                        "source": path.name,
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "derived_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        "source_rate": 8000,
                        "target_rate": 16000,
                        "samples_before": len(values),
                        "samples_after": len(values) * 2,
                        "method": "windowed-sinc interpolation, 16-sample radius; no new above-Nyquist information",
                    }
                    conversions.append(conversion)
                    splits[split].append(
                        {
                            "modality": "audio",
                            "audio": str(destination),
                            "question": row.get("question", "digit?"),
                            "answer": str(row.get("answer", row.get("label", path.name.split("_")[0]))),
                            "family": path.name,
                            "derived_sha256": conversion["derived_sha256"],
                        }
                    )
            modality = "audio"
        manifest = _manifest(
            ctx,
            asset,
            splits,
            "Fashion custom 3/1/1 per label; FSDD fixed speaker train/validation/test, never an official benchmark",
        )
        model = _modal(ctx, modality, pretrained=False)
        # 真實錄音框數有變化；擴大 context 並明確重建較長絕對位置表。
        if modality == "audio":
            maximum = max(len(_media(row, ctx)[1]) for side in splits.values() for row in side)
            required = math.ceil(maximum / 160) + 100
            if required > model.language.config.max_length:
                old = model.language.position
                model.language.config.max_length = required
                if old is not None:
                    position = nn.Embedding(required, old.embedding_dim).to(ctx.device)
                    with torch.no_grad():
                        position.weight[: len(old.weight)].copy_(old.weight)
                    model.language.position = position
        _freeze(model, "all", modality)
        before = _evaluate(model, splits["test"], ctx, tokens=32)
        training = _fit(model, _loss_fn(model, splits["train"], ctx), ctx, _steps(ctx, 250), name=asset, task=modality)
        results[asset] = {
            "training": training,
            "before": before,
            "validation": _evaluate(model, splits["validation"], ctx, tokens=32),
            "test": _evaluate(model, splits["test"], ctx, tokens=32),
            "data": manifest,
            "resampling": conversions,
            "scope": "very small real-data pipeline with independent held-out examples; no natural-world capability claim",
        }
    (ctx.output / "model.pt").write_bytes((ctx.output / "fsdd.pt").read_bytes())
    return results
