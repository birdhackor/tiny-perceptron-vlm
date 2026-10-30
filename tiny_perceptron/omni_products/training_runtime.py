"""有上界的 development／正式下游訓練；本檔不開 sealed final。"""

import hashlib
import io
import json
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .av import AVConfig, AVModel, collate_av
from .choice import ChoiceModel, collate_questions
from .data import av_pair, fingerprint, sound_splits, sound_wave, split_a
from .questions import panel_c, training_questions
from .routing import training_routes
from .signal_metrics import active_reference_snr, program_success
from .sound import CodecConfig, ProgramTalker, SoundCodec, program_schedule

TRAINING_REVISION = "omni-training-v2"
MODEL_SOURCES = ("av.py", "choice.py", "sound.py")


def source_binding():
    root = Path(__file__).parent
    names = (*MODEL_SOURCES, "data.py", "questions.py", "routing.py", "signal_metrics.py", "training_runtime.py")
    return {name: file_sha(root / name) for name in names}


def json_file(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_sha(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def deterministic(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def save_checkpoint(path, model, optimizer, generator, step, metadata, auxiliary=None):
    """只有 stdlib／tensor／純 dict；可 weights_only=True 讀回自己的固定 SHA。"""
    value = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler_rng": generator.get_state(),
        "metadata": metadata,
        "auxiliary": auxiliary,
        "numpy_rng": "not used by this trainer or data generator",
    }
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
    return file_sha(path)


def restore_checkpoint(path, sha256, model, optimizer, generator, device="cpu", *, expected_metadata=None):
    if file_sha(path) != sha256:
        raise ValueError("continuation checkpoint SHA 不符")
    value = torch.load(path, weights_only=True, map_location="cpu")
    if expected_metadata is not None and value["metadata"] != expected_metadata:
        raise ValueError("continuation 的來源／revision／runtime／recipe 與本次固定實作不同")
    model.load_state_dict(value["model"])
    model.to(device)
    optimizer.load_state_dict(value["optimizer"])
    for state in optimizer.state.values():
        for key, tensor in state.items():
            if torch.is_tensor(tensor) and key != "step":
                state[key] = tensor.to(device)
    # 必須在新 model／optimizer 初始化與載入之後才還原 RNG。
    random.setstate(value["python_rng"])
    torch.set_rng_state(value["torch_rng"])
    if value["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA continuation 不可假稱 CPU exact resume")
        torch.cuda.set_rng_state_all(value["cuda_rng"])
    generator.set_state(value["sampler_rng"])
    metadata = dict(value["metadata"])
    if value.get("auxiliary") is not None:
        metadata["_continuation"] = value["auxiliary"]
    return value["step"], metadata


def export_weights(path, model):
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}, str(path))
    return file_sha(path)


def learning_rate(step, total, peak):
    warmup = max(1, min(200, total // 10))
    if step < warmup:
        return peak * (step + 1) / warmup
    phase = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * phase)) / 2)


def reset_codebook_moments(optimizer, parameter, unused):
    """重啟的碼不是上一個向量；不能把上一個向量的 AdamW 動量帶過去。"""
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = optimizer.state.get(parameter, {}).get(name)
        if value is not None:
            value[unused] = 0


def run_updates(model, make_batch, loss_function, validate, config, output, resume=None):
    implementation = source_binding()
    if implementation != LOADED_SOURCE_BINDING:
        raise RuntimeError("訓練模組載入後來源已變，必須另啟乾淨process")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(config["seed"] + 100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    initial = parameter_sha(model)
    start, logs, best, complete = time.monotonic(), [], None, True
    usage = torch.zeros(model.config.codes, device=device) if isinstance(model, SoundCodec) else None
    best_checkpoint_bytes = None
    continuation_metadata = {
        "config": config,
        "training_revision": TRAINING_REVISION,
        "implementation_sha256": implementation,
        "runtime": {"device": str(device), "torch": str(torch.__version__)},
    }
    latest_step = 0
    if resume is not None:
        latest_step, restored = restore_checkpoint(
            *resume, model, optimizer, generator, device, expected_metadata=continuation_metadata
        )
        auxiliary = restored["_continuation"]
        logs, best, initial = auxiliary["logs"], auxiliary["best"], auxiliary["initial_parameter_sha256"]
        if usage is not None:
            usage.copy_(auxiliary["code_usage"].to(device))
        best_checkpoint_bytes = auxiliary.get("best_checkpoint_bytes")
        if best is not None:
            if best_checkpoint_bytes is None:
                if best["update"] != latest_step:
                    raise ValueError("continuation缺少既有best，不能重設best或改挑新權重")
                best_checkpoint_bytes = Path(resume[0]).read_bytes()
            selected = torch.load(io.BytesIO(best_checkpoint_bytes), weights_only=True, map_location="cpu")
            if selected["metadata"] != continuation_metadata or selected["step"] != best["update"]:
                raise ValueError("continuation內的best權重／來源不符")
            (output / "best-continuation.pt").write_bytes(best_checkpoint_bytes)
            save_file(
                {name: value.contiguous() for name, value in selected["model"].items()},
                str(output / "weights.safetensors"),
            )

    def checkpoint(path, step):
        return save_checkpoint(
            path,
            model,
            optimizer,
            generator,
            step,
            continuation_metadata,
            {
                "logs": logs,
                "best": best,
                "initial_parameter_sha256": initial,
                "code_usage": usage.detach().cpu() if usage is not None else None,
                # best檔本身不包含歷代best bytes；latest才內嵌一份，避免遞迴膨脹。
                "best_checkpoint_bytes": best_checkpoint_bytes if Path(path).name != "best-continuation.pt" else None,
            },
        )

    for step in range(latest_step, config["updates"]):
        if time.monotonic() - start > config["training_seconds"]:
            complete = False
            break
        model.train()
        batch = make_batch(generator)
        optimizer.zero_grad(set_to_none=True)
        result = loss_function(model, batch)
        loss = result["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("訓練 loss 非有限，不能跳過此失敗")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, config["updates"], config["learning_rate"])
        optimizer.step()
        latest_step = step + 1
        if usage is not None:
            valid_codes = result["codes"].detach().flatten()
            usage += torch.bincount(valid_codes[valid_codes >= 0], minlength=model.config.codes)
            if (
                config.get("training_dead_code_reset_every")
                and (step + 1) % config["training_dead_code_reset_every"] == 0
            ):
                unused = (usage == 0).nonzero().flatten()
                reset = model.reset_unused_training_codes(usage, result["encoded"])
                reset_codebook_moments(optimizer, model.codebook.weight, unused)
                usage.zero_()
            else:
                reset = 0
        else:
            reset = None
        if (step + 1) % config["validate_every"] == 0 or step + 1 == config["updates"]:
            model.eval()
            with torch.no_grad():
                measured = validate(model)
                score = measured["selection_score"] if isinstance(measured, dict) else measured
            row = {
                "update": step + 1,
                "loss": float(loss.detach()),
                "dev_score": score,
                "elapsed_seconds": time.monotonic() - start,
                "training_codes_reset": reset,
            }
            if isinstance(measured, dict):
                row["dev_metrics"] = measured
            logs.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
            if best is None or score > best["dev_score"]:
                best = row
                checkpoint(output / "best-continuation.pt", step + 1)
                best_checkpoint_bytes = (output / "best-continuation.pt").read_bytes()
                export_weights(output / "weights.safetensors", model)
            checkpoint(output / "latest-continuation.pt", step + 1)
            json_file(output / "training-log.json", logs)
    if latest_step and (not logs or logs[-1]["update"] != latest_step):
        checkpoint(output / "latest-continuation.pt", latest_step)
    if source_binding() != implementation:
        raise RuntimeError("訓練期間來源改動；保留診斷，但不得把權重列作固定實作的正式產物")
    result = {
        "complete": complete and latest_step == config["updates"],
        "updates": latest_step,
        "config": config,
        "elapsed_seconds": time.monotonic() - start,
        "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
        "initial_parameter_sha256": initial,
        "latest_parameter_sha256": parameter_sha(model),
        "optimizer_updates_observed": latest_step > 0 and bool(optimizer.state),
        "best": best,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha(path)}
            for path in output.iterdir()
            if path.is_file() and path.name != "training-result.json"
        },
        "final_opened": False,
        "training_revision": TRAINING_REVISION,
        "implementation_sha256": implementation,
        "source_unchanged_during_training": True,
        "runtime": {"device": str(device), "torch": str(torch.__version__), "threads": torch.get_num_threads()},
    }
    json_file(output / "training-result.json", result)
    return result


def av_bank(split, variants, config, device):
    examples = []
    for pattern in split_a()[split]:
        for variant in range(variants):
            for panel in range(4):
                for item, label in av_pair(pattern, variant, panel):
                    examples.append((item, label))
                    if variant == 0:
                        examples.extend(
                            ((replace(item, image_present=False), 2), (replace(item, audio_present=False), 2))
                        )
    return collate_av(examples, device, config)


def av_accuracy(model, bank, batch_size=64):
    correct, total = 0, len(bank["labels"])
    for start in range(0, total, batch_size):
        batch = {key: value[start : start + batch_size] for key, value in bank.items()}
        correct += int((model(batch).argmax(-1) == batch["labels"]).sum())
    return correct / total


def av_development_metrics(model, bank, batch_size=64):
    """每組兩個移時題都對才算；缺線索成績只作同分決選。"""
    predictions = []
    for start in range(0, len(bank["labels"]), batch_size):
        batch = {key: value[start : start + batch_size] for key, value in bank.items()}
        predictions.append(model(batch).argmax(-1))
    correct = torch.cat(predictions) == bank["labels"]
    if len(correct) % 6:
        raise ValueError("development bank 應是 pair 各含完整／刪圖／刪聲的六列")
    paired = float((correct[0::6] & correct[3::6]).float().mean())
    remove_image, remove_audio = float(correct[1::3].float().mean()), float(correct[2::3].float().mean())
    missing = (remove_image + remove_audio) / 2
    return {
        "paired_accuracy": paired,
        "remove_image_accuracy": remove_image,
        "remove_audio_accuracy": remove_audio,
        "selection_score": paired + 1e-6 * missing,
        "selection_rule": "paired accuracy first; mean missing-cue accuracy breaks ties; earlier checkpoint breaks remaining ties",
    }


def train_av(config, output, device, resume=None):
    deterministic(config["seed"])
    model_config = AVConfig(representation=config["representation"], width=config.get("width", 64))
    model = AVModel(model_config).to(device)
    train = av_bank("train", config["variants"], model_config, device)
    dev = av_bank("dev", 1, model_config, device)
    config = {
        **config,
        "model": asdict(model_config),
        "train_families": 38,
        "dev_families": 16,
        "train_examples": len(train["labels"]),
    }

    def batch(generator):
        indices = torch.randint(len(train["labels"]), (config["batch_size"],), generator=generator).to(device)
        return {key: value[indices] for key, value in train.items()}

    return run_updates(
        model,
        batch,
        lambda model, row: {"loss": F.cross_entropy(model(row), row["labels"])},
        lambda model: av_development_metrics(model, dev),
        config,
        output,
        resume=resume,
    )


def choice_training_core(mode="augmented"):
    rows = training_questions()
    if mode == "augmented":
        return rows
    if mode == "compact":
        # 每家族三種 source 各四型題；對照僅保留原先 variant0 的四型。
        # 不重新產生已刪除的欄位，也不改同家族的 panel。
        return [row for index, row in enumerate(rows) if index % 12 < 4]
    raise ValueError("C data_mode 只支援 augmented／compact")


def train_choice(config, output, device, resume=None):
    deterministic(config["seed"])
    model = ChoiceModel().to(device)
    core, routes = choice_training_core(config.get("data_mode", "augmented")), training_routes()
    # 每批一半核心問題、一半路由；路由四個類別等權，避免 903 個 C 狀態壓過其他類別。
    train = core + routes
    route_groups = [[row for row in routes if row.gold == name] for name in ("A", "B", "C", "none")]
    dev = [question for row in panel_c("dev") for question in row["questions"].values()]
    config = {**config, "model": asdict(model.config), "train_questions": len(train), "dev_questions": len(dev)}

    def batch(generator):
        selected = [
            core[index]
            for index in torch.randint(len(core), (config["batch_size"] // 2,), generator=generator).tolist()
        ]
        for _ in range(config["batch_size"] - len(selected)):
            group = route_groups[int(torch.randint(4, (), generator=generator))]
            selected.append(group[int(torch.randint(len(group), (), generator=generator))])
        return collate_questions(selected, device)

    def loss(model, row):
        logits = model(row, recompute=config["recompute"])["logits"]
        return {"loss": F.cross_entropy(logits, row["labels"])}

    def validate(model):
        correct = 0
        for start in range(0, len(dev), 32):
            row = collate_questions(dev[start : start + 32], device)
            correct += int((model(row)["logits"].argmax(-1) == row["labels"]).sum())
        return correct / len(dev)

    return run_updates(model, batch, loss, validate, config, output, resume=resume)


def codec_blocks(config, device):
    rows = []
    for index, program in enumerate(sound_splits()["train"]):
        for variant in range(config["variants"]):
            rng = random.Random(int(fingerprint([index, variant])[:8], 16))
            waveform = sound_wave(program, rng.uniform(0, 2 * math.pi), (0.4, 0.6, 0.8)[variant % 3])
            rows.append(waveform.reshape(-1, config["hop"]))
    return torch.cat(rows).to(device)


def train_codec(config, output, device, resume=None):
    deterministic(config["seed"])
    model = SoundCodec(CodecConfig(hop=config["hop"], width=config["width"], codes=config["codes"])).to(device)
    blocks = codec_blocks(config, device)
    # 低於固定 RMS 的靜音塊與有聲塊平衡抽樣，避免大量停頓支配 reconstruction。
    active = blocks.square().mean(-1).sqrt() > 0.01
    yes, no = active.nonzero().flatten(), (~active).nonzero().flatten()
    dev = sound_splits()["dev"][:32]
    config = {
        **config,
        "model": asdict(model.config),
        "train_blocks": len(blocks),
        "sampling": "each batch 3/4 active,1/4 silence; training-only dead-code restart",
        "dev_snr_programs": 32,
    }

    def batch(generator):
        n = config["batch_size"]
        a = yes[torch.randint(len(yes), (3 * n // 4,), generator=generator).to(device)]
        b = no[torch.randint(len(no), (n - len(a),), generator=generator).to(device)]
        return blocks[torch.cat((a, b))]

    def validate(model):
        values = []
        for program in dev:
            reference = sound_wave(program).to(device)
            codes, length = model.encode_wave(reference)
            reconstructed = model.decode_codes(codes)[:length]
            values.append(active_reference_snr(reference, reconstructed, program))
        return sum(values) / len(values)

    return run_updates(model, batch, lambda model, row: model(row), validate, config, output, resume=resume)


def talker_targets(codec, programs):
    if codec.training or any(parameter.requires_grad for parameter in codec.parameters()):
        raise ValueError("Talker targets 前必須凍結 codec eval／所有權重")
    before = parameter_sha(codec)
    device = codec.codebook.weight.device
    rows = []
    for program in programs:
        waveform = sound_wave(program).to(device)
        codes, _ = codec.encode_wave(waveform)
        rows.append((program, codes.detach().cpu(), program_schedule(program, codec.config.hop)))
    if parameter_sha(codec) != before:
        raise RuntimeError("targets 產生污染 codec")
    return rows, before


def train_talker(config, output, device, codec, resume=None):
    deterministic(config["seed"])
    codec.eval().requires_grad_(False)
    rows, codec_sha = talker_targets(codec, sound_splits()["train"])
    model = ProgramTalker(codec.config.codes, config["width"], codec.config.hop).to(device)
    config = {
        **config,
        "model": model.config(),
        "codec_parameter_sha256": codec_sha,
        "targets_families": len(rows),
        "train_targets_sha256": fingerprint(
            [
                {"program": fingerprint([event.__dict__ for event in program]), "codes": codes.tolist()}
                for program, codes, _ in rows
            ]
        ),
    }
    dev = sound_splits()["dev"][:32]

    def batch(generator):
        indices = torch.randint(len(rows), (config["batch_size"],), generator=generator).tolist()
        selected = [rows[i] for i in indices]
        length = max(len(codes) + 1 for _, codes, _ in selected)
        previous = torch.full((len(selected), length), model.pad, dtype=torch.long)
        target = torch.full((len(selected), length), -100, dtype=torch.long)
        schedule = torch.tensor([4, 0, 1]).expand(len(selected), length, 3).clone()
        for i, (_, codes, requested) in enumerate(selected):
            n = len(codes)
            previous[i, : n + 1] = torch.cat((torch.tensor([model.bos]), codes))
            target[i, : n + 1] = torch.cat((codes, torch.tensor([model.eos])))
            schedule[i, : n + 1] = requested
        return previous.to(device), target.to(device), schedule.to(device)

    def loss(model, row):
        previous, target, schedule = row
        return {"loss": F.cross_entropy(model(previous, schedule).transpose(1, 2), target)}

    def validate(model):
        correct = 0
        for program in dev:
            generated = model.generate(program)
            if generated["eos"] and len(generated["codes"]):
                wave = codec.decode_codes(generated["codes"])
                correct += program_success(wave, program)["success"]
        return correct / len(dev)

    return run_updates(model, batch, loss, validate, config, output, resume=resume)


# 在 import 完成時記錄；訓練開始與結束均核對，不以結束時的新檔案冒充載入版本。
LOADED_SOURCE_BINDING = source_binding()
