"""核對參數、未使用模態、資料切分與既有實驗紀錄；不重跑訓練。"""

import ast
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.modalities import _freeze, _sequence, _vision_records
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM, scene


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (ARTIFACTS / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def modes(model, order):
    observed = {}
    for mode in order:
        model.requires_grad_(mode == "all")
        model.image_projector.requires_grad_(True)
        if mode == "partial":
            model.language.blocks[-1].requires_grad_(True)
        observed[mode] = {
            "count": count(model),
            "names": [n for n, p in model.named_parameters() if p.requires_grad],
        }
    return observed


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "platform": platform.platform(),
    }
    helper_command = [
        ".venv/bin/python", "docs/review-tools/section_facts.py",
        "course/chapters/11.md#11.5", "--execute",
    ]
    if "--reuse-section-execution" in sys.argv:
        saved = json.loads((ARTIFACTS / "section-execution.json").read_bytes())
        receipt = json.loads(saved["stdout"])
        copied = saved["copied_files"]
    else:
        helper = subprocess.run(
            helper_command, cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True,
        )
        receipt = json.loads(helper.stdout)
        run = Path(receipt["artifact_directory"])
        assert run.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
        copied = {}
        for name in [
            "execution.json", "environment.json", "stdout.txt", "stderr.txt",
            "extraction.json", "fence-1.py",
        ]:
            path = run / name
            copied[name] = {"sha256": sha(path), "content": path.read_text()}
        save("section-execution.json", {
            "command": "PYTHONDONTWRITEBYTECODE=1 " + " ".join(helper_command),
            "returncode": helper.returncode, "stdout": helper.stdout,
            "stderr": helper.stderr, "copied_files": copied,
            "environment": environment,
        })
    assert receipt["source_sha256"] == "f2f1afd7faee1faf3c4f7930028fc7332cc3f93af1c67dc9d08fe4f691977e37"
    assert copied["stdout.txt"]["content"] == "projector 136\npartial 976\nall 8296\n"

    small = MultiModalLM(TinyLM(ModelConfig(width=8)))
    small_modes = modes(small, ["projector", "partial", "all"])
    reverse = modes(small, ["all", "partial", "projector"])
    deep_modes = modes(MultiModalLM(TinyLM(ModelConfig(width=8, layers=2))), ["projector", "partial", "all"])
    assert [small_modes[x]["count"] for x in small_modes] == [136, 976, 8296]
    assert [deep_modes[x]["count"] for x in deep_modes] == [136, 976, 9136]
    assert all(small_modes[x] == reverse[x] for x in small_modes)
    components = {name: sum(p.numel() for p in child.parameters()) for name, child in small.named_children()}
    assert components["audio"] + components["audio_projector"] == 712

    gradient_model = MultiModalLM(TinyLM(ModelConfig(width=8)))
    gradient_model.requires_grad_(True)
    optimizer = torch.optim.AdamW(gradient_model.parameters(), lr=0.003)
    optimizer.zero_grad(set_to_none=True)
    originals = {n: p.detach().clone() for n, p in gradient_model.named_parameters()}
    row = _vision_records(("shape?", "color?"))["train"][0]
    ids, labels, effective = _sequence(row, SimpleNamespace(device="cpu"))
    output = gradient_model(ids, labels, image=scene(row["color"], row["shape"], offset=row["offset"]))
    loss = masked_loss(output["logits"], output["labels"])
    loss.backward()
    gradients = {
        name: {
            "parameters": sum(p.numel() for p in child.parameters()),
            "gradient_numel": sum(p.grad.numel() for p in child.parameters() if p.grad is not None),
            "all_gradients_none": all(p.grad is None for p in child.parameters()),
        }
        for name, child in gradient_model.named_children()
    }
    optimizer.step()
    updates = {}
    for name, child in gradient_model.named_children():
        updates[name] = {
            "changed_tensors": sum(
                not torch.equal(originals[name + "." + n], p.detach()) for n, p in child.named_parameters()
            ),
            "optimizer_state_numel": {
                key: sum(optimizer.state[p].get(key, torch.empty(0)).numel() for p in child.parameters())
                for key in ["exp_avg", "exp_avg_sq"]
            },
        }
    assert gradients["audio"]["all_gradients_none"]
    assert gradients["audio_projector"]["all_gradients_none"]
    assert updates["audio"]["changed_tensors"] == updates["audio_projector"]["changed_tensors"] == 0
    assert sum(x["gradient_numel"] for x in gradients.values()) == 8296 - 712
    for name, result in gradients.items():
        assert all(v == result["gradient_numel"] for v in updates[name]["optimizer_state_numel"].values())

    report_path = ROOT / "docs/course-experiments/results/vqa.json"
    report = json.loads(report_path.read_bytes())
    result = report["results"]
    matching_files = {}
    for name in [
        "tiny_perceptron/model.py", "tiny_perceptron/multimodal.py",
        "tiny_perceptron/attention.py", "tiny_perceptron/modern.py",
        "scripts/course_experiments/modalities.py", "tiny_perceptron/data.py",
        "tiny_perceptron/training.py",
    ]:
        current = sha(ROOT / name)
        matching_files[name] = {"current_sha256": current, "recorded_sha256": report["code_sha256"][name]}
        assert current == report["code_sha256"][name]
    expected_splits = _vision_records(("shape?", "color?"))
    split_summary = {}
    family_sets = {}
    for name, split in result["data"]["splits"].items():
        rows = split["records"]
        assert rows == expected_splits[name]
        pairs = Counter(x["family"] for x in rows)
        assert set(pairs.values()) == {2}
        assert len(rows) == split["count"]
        assert all(x["answer"] == x["shape" if x["question"] == "shape?" else "color"] for x in rows)
        canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
        assert hashlib.sha256(canonical).hexdigest() == split["sha256"]
        family_sets[name] = set(pairs)
        split_summary[name] = {
            "count": len(rows), "families": len(pairs), "sha256": split["sha256"],
            "offsets": sorted({x["offset"] for x in rows}),
            "color_shape_combinations": sorted({x["color"] + ":" + x["shape"] for x in rows}),
            "questions": sorted({x["question"] for x in rows}),
            "answer_vocabulary": sorted({x["answer"] for x in rows}),
            "two_questions_per_image": True,
        }
    assert all(not family_sets[a] & family_sets[b] for a, b in [("train", "test"), ("train", "validation"), ("validation", "test")])

    # 依實際抽樣規則計數答案與結束符；忽略 prompt 與圖片插入位置。
    recomputed_history = []
    for step in range(160):
        rng = random.Random(report["seed"] + step)
        rows = []
        for _ in range(4):
            # 傳入非空 replay 時，機率為零也會先消耗一次亂數。
            rng.random()
            rows.append(rng.choice(expected_splits["train"]))
        recomputed_history.append(sum(len(ByteTokenizer().encode(x["answer"])) + 1 for x in rows))
    assert sum(recomputed_history) == 3830
    variant_summary = {}
    for name, expected in [("projector_only", (1088, 3, 3)), ("partial", (50816, 10, 12)), ("all", (145664, 12, 9))]:
        variant = result["variants"][name]
        training = variant["training"]
        model = MultiModalLM(TinyLM(ModelConfig(**training["config"])), vision_width=16, audio_width=16)
        _freeze(model, "projector" if name == "projector_only" else name)
        assert count(model) == training["trainable_parameters"] == expected[0]
        assert training["steps"] == len(training["history"]) == 160
        assert [x["effective_targets"] for x in training["history"]] == recomputed_history
        assert training["effective_targets"] == sum(recomputed_history)
        recounted = {}
        for split, expected_correct in [("validation", expected[1]), ("test", expected[2])]:
            evaluation = variant[split]
            samples = evaluation["samples"]
            assert len(samples) == evaluation["examples"] == 12
            assert [x["family"] for x in samples] == [x["family"] for x in expected_splits[split]]
            for sample, data in zip(samples, expected_splits[split], strict=True):
                assert sample["target"] == data["answer"]
                raw = sample["generated_ids"]
                raw = raw[:raw.index(2)] if 2 in raw else raw
                assert sample["exact_match"] == (raw == ByteTokenizer().encode(sample["target"]))
                assert sample["generated"] == ByteTokenizer().decode(raw)
            correct = sum(x["exact_match"] for x in samples)
            assert correct == evaluation["correct"] == expected_correct
            recounted[split] = {"correct": correct, "examples": len(samples), "effective_tokens": evaluation["effective_tokens"]}
        variant_summary[name] = {
            "trainable_parameters": count(model), "steps": training["steps"],
            "effective_targets": training["effective_targets"],
            "freeze_scope": variant["freeze_scope"], "replay_probability_per_example": variant["replay_probability_per_example"],
            "validation": recounted["validation"], "test": recounted["test"],
            "before_sha256": hashlib.sha256(json.dumps(variant["before"], sort_keys=True).encode()).hexdigest(),
            "text_before": {k: variant["text_before"][k] for k in ["matches", "records", "effective_tokens"]},
            "text_after": {k: variant["text_after"][k] for k in ["matches", "records", "effective_tokens"]},
        }
    assert len({v["before_sha256"] for v in variant_summary.values()}) == 1
    projector_path = ROOT / "docs/course-experiments/results/projector.json"
    projector = json.loads(projector_path.read_bytes())
    assert result["two_stage_alignment_training"] == projector["results"]["training"]
    assert projector["results"]["language_weights_unchanged"] is True
    assert projector["results"]["training"]["trainable_parameters"] == 1088
    assert all(x["question"] == "describe" and x["answer"] == f'{x["color"]} {x["shape"]}' for x in projector["results"]["data"]["splits"]["train"]["records"])

    # 抽出既有 CLI 的凍結敘述，以三區塊模型辨別首末區塊與只開末區塊。
    tree = ast.parse((ROOT / "scripts/train.py").read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "args.freeze != 'none'")
    cli = MultiModalLM(TinyLM(ModelConfig(width=8, layers=3)))
    namespace = {"train_model": cli, "args": SimpleNamespace(freeze="partial")}
    exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), "scripts/train.py:freeze", "exec"), namespace)
    cli_blocks = [all(p.requires_grad for p in b.parameters()) for b in cli.language.blocks]
    experimental = MultiModalLM(TinyLM(ModelConfig(width=8, layers=3)))
    _freeze(experimental, "partial")
    experiment_blocks = [all(p.requires_grad for p in b.parameters()) for b in experimental.language.blocks]
    assert cli_blocks == [True, False, True] and experiment_blocks == [False, False, True]

    save("audit-output.json", {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " " + " ".join(sys.argv[1:]),
        "result": "所有斷言通過；僅執行一次前向、反向與更新，未重跑實驗訓練。",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(), "environment": environment,
        "source_sha256": receipt["source_sha256"], "small_modes": small_modes,
        "reverse_modes_equal": True, "two_layer_modes": deep_modes, "small_components": components,
        "image_only": {"loss": float(loss.detach()), "effective_targets": effective, "gradients": gradients, "updates": updates},
        "recorded_experiment": {
            "path": str(report_path.relative_to(ROOT)), "sha256": sha(report_path),
            "revision": report["revision"], "seed": report["seed"], "device": report["device"],
            "torch_version": report["torch_version"], "python_version": report["python_version"], "gpu": report["gpu"],
            "step_scale": report["step_scale"], "status": report["status"], "evidence_status": report["evidence_status"],
            "code_hash_comparison": matching_files, "splits": split_summary,
            "recomputed_target_count": sum(recomputed_history), "variants": variant_summary,
            "projector_record": {"path": str(projector_path.relative_to(ROOT)), "sha256": sha(projector_path), "upstream_training_equal": True, "language_weights_unchanged": True, "description_answers_verified": True},
        },
        "cli_partial": {"code": ast.unparse(node), "three_block_flags": cli_blocks, "experiment_three_block_flags": experiment_blocks, "both_projectors_enabled": all(p.requires_grad for module in [cli.image_projector, cli.audio_projector] for p in module.parameters())},
        "limitations": ["實測品質由保存的 GPU 實驗逐筆紀錄核對，沒有重新生成答案。", "CPU 機制驗證不支持 GPU 效能或記憶體實測。", "requires_grad 參數數量不是精確 FLOPs、耗時或尖峰記憶體；AdamW 預設 amsgrad=False 另有每 tensor 的 step 計數。", "未使用參數不更新的驗證使用新 optimizer 與 grad=None；不涵蓋手動留下舊梯度的流程。"],
    })
    print(json.dumps({"result": "所有核對通過", "environment": environment, "small_counts": {k: v["count"] for k, v in small_modes.items()}, "deep_counts": {k: v["count"] for k, v in deep_modes.items()}, "effective_targets": sum(recomputed_history)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
