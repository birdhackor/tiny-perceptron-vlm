"""只以train更新、policy選checkpoint；最後test由另外的凍結評估入口處理。"""

import hashlib
import platform
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import torch

from .data import read_dataset
from .model import CandidateScorer, Config, LoopedSolver, Tokenizer
from .training import checkpoint, encoded_data, metrics, predict, restore, save_json, state_digest, step, verify_resume


def policy_report(model, data, reference_depth):
    logits = predict(model, data, loops=reference_depth)
    labels = data["labels"].cpu()
    rows = data["rows"]
    groups = {}
    if data["product"] == "b":
        for task in ("boxes", "sum"):
            groups[task] = [i for i, row in enumerate(rows) if row["task"] == task]
        for length in range(1, 5):
            groups[f"boxes_length_{length}"] = [
                i for i, row in enumerate(rows) if row["task"] == "boxes" and row["strata"]["length"] == length
            ]
        primary = "boxes"
    else:
        for question in ("choice", "info", "rubric"):
            groups[question] = [i for i, row in enumerate(rows) if row["question"] == question]
        primary = "choice"
    report = {name: metrics(logits[indices], labels[indices]) for name, indices in groups.items() if indices}
    if primary not in report:
        raise ValueError("policy缺少預先指定的核心題型")
    return report, (report[primary]["accuracy"], -report[primary]["cross_entropy"])


def train(
    data_directory,
    output,
    product,
    mode,
    max_steps=6000,
    max_seconds=1200,
    batch_size=128,
    learning_rate=0.001,
    seed=7421,
    resume=None,
    resume_best=None,
    device="cpu",
    depth_weights=(1, 1, 1, 1, 1),
    on_checkpoint=None,
):
    if product not in ("b", "c") or mode not in (("multi", "fixed3", "odd") if product == "b" else ("candidate",)):
        raise ValueError("需明定B的多輪／固定3／奇數末端，或C的candidate目標")
    if len(depth_weights) != 5 or any(weight <= 0 for weight in depth_weights):
        raise ValueError("產品多輪loss需要五個正權重，並正規化成加權平均")
    if not 1 <= max_steps <= 50000 or not 1 <= max_seconds <= 1500 or not 1 <= batch_size <= 256:
        raise ValueError("超出有界教學訓練規模")
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data_directory = Path(data_directory)
    data_sha = hashlib.sha256((data_directory / "manifest.json").read_bytes()).hexdigest()
    device = torch.device(device)
    torch.set_num_threads(2 if device.type == "cuda" else 1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    tokenizer = Tokenizer()
    tokenizer.save(output / "tokenizer.json")
    tokenizer_sha = hashlib.sha256((output / "tokenizer.json").read_bytes()).hexdigest()
    config = Config(len(tokenizer), width=128 if product == "b" else 64, ff=256 if product == "b" else 128)
    if resume:
        model, optimizer, generator, metadata = restore(resume, product, device, learning_rate)
        for key, expected in (
            ("product", product),
            ("mode", mode),
            ("data_manifest_sha256", data_sha),
            ("tokenizer_sha256", tokenizer_sha),
        ):
            if metadata.get(key) != expected:
                raise ValueError("續訓的資料或目標契約不符：" + key)
        if metadata.get("depth_weights") != list(depth_weights):
            raise ValueError("續訓不可暗換多輪監督權重")
        initial_hash = metadata["initial_weights_sha256"]
        start_step = metadata["step"]
        seed = metadata["seed"]
        learning_rate = optimizer.param_groups[0]["lr"]
        best_source = Path(resume_best or resume)
        best_value = torch.load(best_source, map_location="cpu", weights_only=True)
        stable = (
            "run_identity",
            "product",
            "mode",
            "seed",
            "initial_weights_sha256",
            "data_manifest_sha256",
            "tokenizer_sha256",
            "depth_weights",
            "reference_depth",
            "training_recipe",
        )
        if (
            not metadata.get("run_identity")
            or any(best_value["metadata"].get(key) != metadata.get(key) for key in stable)
            or best_value["config"] != asdict(model.config)
            or best_value["metadata"]["step"] != metadata["best_step"]
            or best_value["metadata"]["best_key"] != metadata["best_key"]
            or state_digest(best_value["model"]) != metadata.get("best_weights_sha256")
        ):
            raise ValueError("last checkpoint需搭配同一份原policy最佳checkpoint，不可用last覆蓋best")
        if metadata["training_recipe"] != {"batch_size": batch_size, "learning_rate": learning_rate}:
            raise ValueError("續訓不可暗換batch或optimizer learning rate")
        run_identity = metadata["run_identity"]
        best_weights_hash = metadata["best_weights_sha256"]
    else:
        model = (LoopedSolver if product == "b" else CandidateScorer)(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, foreach=False)
        generator = torch.Generator().manual_seed(seed + 1)
        initial_hash = state_digest(model.state_dict())
        best_weights_hash = initial_hash
        run_identity = uuid.uuid4().hex
        start_step = 0
    config = model.config
    config.save(output / "config.json")
    train_data = encoded_data(read_dataset(data_directory, product, "train"), product, tokenizer, config, device)
    policy_data = encoded_data(read_dataset(data_directory, product, "policy"), product, tokenizer, config, device)
    prepared_at = time.monotonic()
    reference_depth = (5 if mode == "multi" else 3) if product == "b" else 0
    before_policy, before_key = policy_report(model, policy_data, reference_depth)
    best_key = tuple(metadata.get("best_key", before_key)) if resume else before_key
    best_step = metadata.get("best_step", start_step) if resume else start_step
    # 一次attempt自己的起始權重與從隨機初始化的原始權重分開記錄。
    attempt_start_hash = state_digest(model.state_dict())
    history = []
    work = {"block_sequence_work": 0, "block_token_work": 0}
    options = {"batch_size": batch_size, "mode": mode, "depth_weights": list(depth_weights), "depth_seed": seed}
    metadata = {
        "run_identity": run_identity,
        "product": product,
        "mode": mode,
        "seed": seed,
        "data_manifest_sha256": data_sha,
        "initial_weights_sha256": initial_hash,
        "tokenizer_sha256": tokenizer_sha,
        "depth_weights": list(depth_weights),
        "step": start_step,
        "best_key": list(best_key),
        "best_step": best_step,
        "reference_depth": reference_depth,
        "best_weights_sha256": best_weights_hash,
        "training_recipe": {"batch_size": batch_size, "learning_rate": learning_rate},
    }
    if resume:
        if best_source.resolve() != (output / "best.pt").resolve():
            shutil.copyfile(best_source, output / "best.pt")
    else:
        checkpoint(output / "best.pt", model, optimizer, generator, metadata)
    stop_reason = "max_steps"
    successful_evaluations = 0
    last_step = start_step
    for current in range(start_step + 1, start_step + max_steps + 1):
        if time.monotonic() - started >= max_seconds:
            stop_reason = "wall_budget"
            break
        measurement = step(model, optimizer, train_data, generator, step_number=current, **options)
        last_step = current
        for name in work:
            work[name] += measurement[name]
        if current == start_step + 1 or current % 50 == 0:
            history.append({"step": current, "elapsed_seconds": time.monotonic() - started, **measurement})
        if current == start_step + 20 or current % 200 == 0:
            report, key = policy_report(model, policy_data, reference_depth)
            if key > best_key:
                best_key, best_step = key, current
                metadata["best_weights_sha256"] = state_digest(model.state_dict())
                metadata.update(step=current, best_key=list(best_key), best_step=best_step)
                checkpoint(output / "best.pt", model, optimizer, generator, metadata)
            metadata.update(step=current, best_key=list(best_key), best_step=best_step)
            checkpoint(output / "last.pt", model, optimizer, generator, metadata)
            history.append({"step": current, "policy": report, "best_step": best_step})
            save_json(output / "progress.json", {"metadata": metadata, "history": history, "work": work})
            if on_checkpoint:
                on_checkpoint(output / "last.pt", current)
            if product == "b":
                target = report["boxes"]["accuracy"] >= 0.98 and report["boxes_length_4"]["accuracy"] >= 0.96
            else:
                target = all(report[q]["accuracy"] >= 0.995 for q in ("choice", "info", "rubric"))
            successful_evaluations = successful_evaluations + 1 if target else 0
            if mode in ("multi", "candidate") and current - start_step >= 800 and successful_evaluations >= 2:
                stop_reason = "development_target"
                break
    metadata.update(step=last_step, best_key=list(best_key), best_step=best_step)
    checkpoint(output / "last.pt", model, optimizer, generator, metadata)
    # 最佳checkpoint是policy選出的；此處不計算test。
    selected, _, _, selected_metadata = restore(output / "best.pt", product, device)
    selected_policy, _ = policy_report(selected, policy_data, reference_depth)
    resume_receipt = verify_resume(output / "last.pt", product, device, train_data, options)
    selected_hash = state_digest(selected.state_dict())
    if selected_hash == initial_hash or state_digest(model.state_dict()) == attempt_start_hash:
        raise ValueError("未得到實際更新且經policy選出的權重")
    torch.save(selected.state_dict(), output / "model.pt")
    norms = None
    if product == "b":
        selected.eval()
        with torch.inference_mode():
            _, norms = selected(train_data["tokens"][:32, 0], loops=5, diagnostics=True)
    receipt = {
        "run_identity": run_identity,
        "product": product,
        "mode": mode,
        "random_initialization": True,
        "resume_source": str(resume) if resume else None,
        "steps_this_attempt": last_step - start_step,
        "last_step": last_step,
        "selected_step": selected_metadata["step"],
        "stop_reason": stop_reason,
        "before_policy": before_policy,
        "selected_policy": selected_policy,
        "reference_depth": reference_depth,
        "depth_weights": list(depth_weights),
        "work": work,
        "preparation_seconds": prepared_at - started,
        "total_seconds": time.monotonic() - started,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "parameters": sum(p.numel() for p in selected.parameters()),
        "data_manifest_sha256": data_sha,
        "initial_weights_sha256": initial_hash,
        "selected_weights_sha256": selected_hash,
        "weights_changed": selected_hash != initial_hash,
        "resume_verification": resume_receipt,
        "selected_hidden_rms_by_depth": norms,
        "history": history,
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "scope": "Training and development-policy results only. No final-test score; capstone gates remain unverified until settings freeze and independent heldout evaluation. Work counts padded block tokens/sequences, not FLOPs or dollars; validation/resume/upload excluded from training-step work but included in wall time where applicable.",
    }
    save_json(output / "training-receipt.json", receipt)
    return receipt
