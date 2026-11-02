"""T.6 independent, short CPU mechanism inspection; never executes a course recipe."""

import argparse
import ast
import contextlib
import hashlib
import importlib
import io
import json
import platform
import random
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from scripts import evaluate, evaluate_modal, infer_modal, train
from scripts.course_experiments import modalities, run
from scripts.course_experiments.common import Context
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.modal_data import modal_example
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, VisionEncoder, scene, tone
from tiny_perceptron.training import load_checkpoint, restore_checkpoint_rng, save_checkpoint, seed_everything


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(function, arguments):
    buffer = io.StringIO()
    with patch.object(sys, "argv", ["mechanism-only", *arguments]), contextlib.redirect_stdout(buffer):
        function()
    return json.loads(buffer.getvalue())


torch.set_num_threads(2)
seed_everything(42)
freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-numbered-source-freeze-v8.json"
page = next(p for p in json.loads(freeze_path.read_bytes())["pages"] if p["page_id"] == "T.6")
body = (ROOT / page["snapshot"]).read_text()
current = re.search(r"^## T\.6 .*?(?=^## |\Z)", (ROOT / page["source"]).read_text(), re.M | re.S)[0]
assert body == current
assert sha(ROOT / page["snapshot"]) == page["source_sha256"]
out = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_p6_technical_t6",
    "scope": "Short synthetic CPU mechanism/CLI parsing and preserved JSON recomputation only. No student training recipe, formal training, frozen final, teacher forward, GPU, Modal, downloads of weights, or publication.",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "numpy": np.__version__, "soundfile": sf.__version__},
    "freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha(freeze_path), **page, "bytes": len(body.encode()), "current_equals_snapshot": True},
}

# Parse every complete command with its real parser and stop before any operation.
class ParseBoundary(Exception):
    pass


original_parse = argparse.ArgumentParser.parse_args
parsed = []
for block in re.findall(r"```bash\n(.*?)```", body, re.S):
    for line in block.strip().splitlines():
        tokens = shlex.split(line)
        if tokens[1] == "-m":
            module, arguments = tokens[2], tokens[3:]
        else:
            module, arguments = tokens[1][:-3].replace("/", "."), tokens[2:]
        target = importlib.import_module(module)
        record = {"command": line, "module": module}

        def boundary(parser, args=None, namespace=None):
            result = original_parse(parser, args, namespace)
            record["parsed"] = vars(result)
            raise ParseBoundary

        with patch.object(sys, "argv", [tokens[1], *arguments]), patch.object(argparse.ArgumentParser, "parse_args", boundary):
            try:
                target.main()
            except ParseBoundary:
                record["stopped_before_execution"] = True
        assert record.get("stopped_before_execution")
        parsed.append(record)
out["exact_commands_parse_only"] = parsed
out["cli_help"] = []
for module in ("scripts.pretrain_encoders", "scripts.train", "scripts.infer_modal", "scripts.fetch_training_assets", "scripts.course_experiments.run", "scripts.evaluate_modal", "scripts.prepare_ocr"):
    command = [sys.executable, "-m", module, "--help"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    out["cli_help"].append({"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})

# Execute only the data construction statements from the actual pretraining source.
source = (ROOT / "scripts/pretrain_encoders.py").read_text()
main = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "main")
report_node = next(node.value for node in main.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "report" for t in node.targets))
mode_expr = next(v for k, v in zip(report_node.keys, report_node.values) if isinstance(k, ast.Constant) and k.value == "mode")
loop_node = next(node for node in main.body if isinstance(node, ast.For))
out["pretrain_control_expressions"] = []
for enabled in (False, True):
    args = SimpleNamespace(train=enabled, steps=300)
    out["pretrain_control_expressions"].append({"train_flag": enabled, "mode": eval(compile(ast.Expression(mode_expr), "mode-only", "eval"), {"args": args}), "scheduled_iterations_without_executing_them": len(eval(compile(ast.Expression(loop_node.iter), "loop-bound-only", "eval"), {"args": args}))})
nodes = [node for node in main.body if (isinstance(node, ast.If) and ast.unparse(node.test).startswith("args.modality ==")) or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("training", "holdout") for t in node.targets))]
out["encoder_data"] = {}
for modality in ("vision", "audio"):
    space = {"args": SimpleNamespace(modality=modality), "scene": scene, "tone": tone}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pretrain-data-only", "exec"), space)
    items = space["items"]
    if modality == "vision":
        identities = [{"color": c, "shape": s, "offset": o, "label": int(s == "circle"), "holdout": c == "blue" and o == 1} for c in ("red", "green", "blue") for s in ("square", "circle") for o in (-1, 0, 1)]
        translated = all(torch.equal(scene("blue", shape, offset=1)[:, :, 1:], scene("blue", shape, offset=0)[:, :, :-1]) for shape in ("square", "circle"))
        assert translated
    else:
        identities = [{"frequency": f, "label": int(f > 500), "holdout": f in (180, 1000)} for f in (180, 220, 260, 780, 880, 1000)]
    out["encoder_data"][modality] = {"items": identities, "train_examples": len(space["training"]), "holdout_examples": len(space["holdout"]), "input_shape": list(items[0][0].shape), "input_dtype": str(items[0][0].dtype), "right_shift_exactly_one_pixel": translated if modality == "vision" else None}
assert [out["encoder_data"][m]["holdout_examples"] for m in ("vision", "audio")] == [2, 2]
out["accuracy_arithmetic"] = {str(correct): correct / 2 for correct in (0, 1, 2)}

# Apply the exact freeze branch from train.main to newly initialized tiny models.
tree = ast.parse((ROOT / "scripts/train.py").read_text())
main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
freeze_node = next(node for node in main.body if isinstance(node, ast.If) and ast.unparse(node.test) == "train_model is not model")
freeze_code = compile(ast.Module(body=freeze_node.body, type_ignores=[]), "train-freeze-only", "exec")
out["manual_freeze"] = {}
for scope in ("projector", "partial", "none"):
    seed_everything(42)
    lm = TinyLM(ModelConfig(width=16, layers=3, heads=1, max_length=128))
    model = MultiModalLM(lm)
    args = SimpleNamespace(freeze=scope, vision_encoder=None, audio_encoder=None)
    exec(freeze_code, {"train_model": model, "args": args, "torch": torch})
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    expected = [n for n, p in model.named_parameters() if scope == "none" or n.startswith(("image_projector.", "audio_projector.")) or (scope == "partial" and n.startswith(("language.blocks.0.", "language.blocks.2.")))]
    assert names == expected
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    optimizer.zero_grad(set_to_none=True)
    loss = train.modal_loss(model, "vision", ByteTokenizer(), "cpu")
    loss.backward()
    no_gradient = [n for n, p in model.named_parameters() if p.grad is None]
    optimizer.step()  # One artificial update, not a course training/evaluation recipe.
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert all(not n.startswith(("audio.", "audio_projector.")) for n in changed)
    assert all(n in names for n in changed)
    out["manual_freeze"][scope] = {"layers": 3, "synthetic_task": "vision", "update_steps": 1, "trainable": names, "no_gradient": no_gradient, "changed": changed, "unused_audio_unchanged": True, "loss": loss.item()}

# Fixed recipe freeze/data/CPU status expressions, without entering run.execute.
fixed = MultiModalLM(TinyLM(ModelConfig(width=16, layers=3, heads=1)))
modalities._freeze(fixed, "partial")
fixed_names = [n for n, p in fixed.named_parameters() if p.requires_grad]
assert all(n.startswith(("image_projector.", "language.blocks.2.")) for n in fixed_names)
out["fixed_recipe"] = {"partial_trainable": fixed_names, "specs": {name: run.experiment_spec(name) for name in ("sft", "encoders", "projector", "vqa")}}
split = modalities._vision_records(("shape?", "color?"))
families = {k: set(r["family"] for r in rows) for k, rows in split.items()}
assert all(not families[a] & families[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
out["fixed_recipe"]["vqa_splits"] = {k: {"examples": len(v), "families": len(families[k]), "offsets": sorted(set(r["offset"] for r in v))} for k, v in split.items()}
execute_node = next(node for node in ast.parse((ROOT / "scripts/course_experiments/run.py").read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "execute")
result_node = next(node.value for node in execute_node.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "result" for t in node.targets))
status_expr = next(v for k, v in zip(result_node.keys, result_node.values) if isinstance(k, ast.Constant) and k.value == "evidence_status")
status = eval(compile(ast.Expression(status_expr), "status-only", "eval"), {"spec": run.experiment_spec("vqa"), "device": "cpu", "step_scale": 1.0, "unfinished": [], "results": {}})
assert status == "interface_smoke_only"
out["fixed_recipe"]["cpu_status_expression"] = status

with tempfile.TemporaryDirectory(prefix="t6-mechanism-") as name:
    temporary = Path(name)
    context = Context("cpu", temporary, temporary, ROOT / "assets/training")
    try:
        context.dependency("sft")
    except FileNotFoundError as error:
        out["fixed_recipe"]["missing_dependency"] = str(error).replace(name, "<temporary>")
    else:
        raise AssertionError("dependency silently created")

    # Real inference on random, tiny reviewer-created models: routing only, no quality claim.
    out["infer_routes"] = []
    tok = ByteTokenizer()
    for task, image_size, flags in (("vision", 16, ["--color", "blue", "--shape", "circle", "--prompt", "shape?"]), ("audio", 16, ["--frequency", "880", "--prompt", "pitch?"]), ("joint", 16, ["--color", "red", "--shape", "square", "--frequency", "220", "--prompt", "joint?"]), ("vision", 32, ["--color", "blue", "--shape", "circle", "--prompt", "shape?"])):
        seed_everything(42)
        tiny = MultiModalLM(TinyLM(ModelConfig(width=16, layers=1, heads=1, max_length=128)))
        tiny.vision = VisionEncoder(image_size=image_size)
        checkpoint = temporary / f"{task}.pt"
        save_checkpoint(checkpoint, tiny, metadata={"task": task})
        real_generate = infer_modal.generate_modal
        captured = {}

        def observed_generate(model, prefix, image, wave, tokens):
            captured.update(prefix_ids=prefix.tolist(), image_shape=None if image is None else list(image.shape), audio_samples=None if wave is None else len(wave))
            return real_generate(model, prefix, image, wave, tokens)

        with patch.object(infer_modal, "generate_modal", observed_generate):
            report = capture(infer_modal.main, [str(checkpoint), *flags, "--tokens", "16", "--device", "cpu"])
        markers = ([] if task == "audio" else [tok.image_id]) + ([] if task == "vision" else [tok.audio_id])
        expected = [tok.bos_id, tok.user_id] + markers + tok.encode(report["prompt"]) + [tok.eos_id, tok.assistant_id]
        assert expected == captured["prefix_ids"]
        out["infer_routes"].append({"task": task, "image_size": image_size, "random_untrained_model": True, **captured, "prefix_only_prompt_and_role_modality_ids": True, "report": report})

    # A checkpoint made in this mechanism probe has the full resume state.
    checkpoint = temporary / "full.pt"
    save_checkpoint(checkpoint, model, optimizer, 1, {"task": "vision", "schedule_steps": 2, "freeze": "none"})
    expected_random = [random.random(), torch.rand(()).item()]
    loaded, payload = load_checkpoint(checkpoint)
    restore_checkpoint_rng(payload)
    resumed_random = [random.random(), torch.rand(()).item()]
    assert expected_random == resumed_random
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    out["checkpoint"] = {"format_version": payload["format_version"], "keys": sorted(payload), "step": payload["step"], "optimizer_state_entries": len(payload["optimizer"]["state"]), "trainable_names_restored": payload["trainable_parameters"] == [n for n, p in loaded.named_parameters() if p.requires_grad], "random_sequence_restored": expected_random == resumed_random, "weights_restored": True}
    try:
        capture(evaluate.main, [str(checkpoint), "--device", "cpu"])
    except ValueError as error:
        out["text_evaluator_rejection"] = str(error)
    else:
        raise AssertionError("multimodal accepted by text evaluator")

    # JSONL image and audio loading; tiny temporary inputs are not training data.
    (temporary / "images").mkdir()
    Image.new("L", (24, 24), 255).save(temporary / "images/12.png")
    record = {"image": "images/12.png", "question": "read digits", "answer": "12"}
    ids, labels, image, _ = modal_example(record, temporary, "vision")
    out["own_files"] = {"image_relative_path": True, "image_shape": list(image.shape), "image_dtype": str(image.dtype), "image_min_max": [image.min().item(), image.max().item()]}
    sf.write(temporary / "mono16.wav", np.zeros(1600, dtype="float32"), 16000)
    sf.write(temporary / "mono8.wav", np.zeros(800, dtype="float32"), 8000)
    sf.write(temporary / "stereo16.wav", np.zeros((1600, 2), dtype="float32"), 16000)
    sf.write(temporary / "empty16.wav", np.zeros(0, dtype="float32"), 16000)
    out["own_files"]["audio"] = {}
    for filename in ("mono16.wav", "mono8.wav", "stereo16.wav", "empty16.wav"):
        try:
            _, _, _, wave = modal_example({"audio": filename, "question": "pitch?", "answer": "low"}, temporary, "audio")
            out["own_files"]["audio"][filename] = {"accepted": True, "samples": len(wave)}
        except ValueError as error:
            out["own_files"]["audio"][filename] = {"accepted": False, "error": str(error)}
    assert out["own_files"]["audio"]["mono16.wav"]["accepted"]
    assert not out["own_files"]["audio"]["mono8.wav"]["accepted"]
    assert not out["own_files"]["audio"]["stereo16.wav"]["accepted"]
    assert not out["own_files"]["audio"]["empty16.wav"]["accepted"]
    for extra in (["--resume"], ["--resume", "--checkpoint", str(checkpoint), "--vision-encoder", "unused.pt"]):
        try:
            capture(train.main, ["--task", "vision", "--device", "cpu", *extra])
        except ValueError as error:
            out.setdefault("resume_guards", []).append(str(error))
        else:
            raise AssertionError("resume guard absent")

    prepare_ocr = importlib.import_module("scripts.prepare_ocr")
    manifest = capture(prepare_ocr.main, ["--output", str(temporary / "ocr")])
    all_records = [json.loads(line) for line in (temporary / "ocr/test.jsonl").read_text().splitlines()]
    records = [all_records[0], next(row for row in all_records if row["answer"] != all_records[0]["answer"])]
    dataset = temporary / "ocr/probe.jsonl"
    dataset.write_text("".join(json.dumps(r) + "\n" for r in records))
    out["ocr"] = {"generated_manifest": manifest, "evaluations": {}}
    for ablation in ("none", "blank", "shuffle"):
        result_path = temporary / f"eval-{ablation}.json"
        capture(evaluate_modal.main, [str(checkpoint), "--data", str(dataset), "--ablation", ablation, "--tokens", "1", "--device", "cpu", "--output", str(result_path)])
        result = json.loads(result_path.read_bytes())
        assert [s["target"] for s in result["samples"]] == [r["answer"] for r in records]
        if ablation == "shuffle":
            assert [s["donor_row"] for s in result["samples"]] == [1, 0]
        out["ocr"]["evaluations"][ablation] = {k: result[k] for k in ("ablation", "examples", "exact_match", "samples")}

# Recompute preserved historical receipts, never load their model weights.
out["historical"] = {}
for name in ("encoders", "projector", "vqa"):
    path = ROOT / f"docs/course-experiments/results/{name}.json"
    raw = json.loads(path.read_bytes())
    out["historical"][name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size, **{k: raw[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "elapsed_seconds", "timing_scope", "step_scale", "evidence_status", "modal")}, "rounded_one_decimal_seconds": round(raw["elapsed_seconds"], 1)}
vqa = json.loads((ROOT / "docs/course-experiments/results/vqa.json").read_bytes())["results"]
partial = vqa["variants"]["partial"]
split_checks = {}
for name, values in vqa["data"]["splits"].items():
    rows = values["records"]
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert digest == values["sha256"]
    assert rows == split[name]
    split_checks[name] = {"record_count": len(rows), "reported_count": values["count"], "sha256_recomputed": digest, "equals_current_generator": True}
recounts = {}
for key, flag, target in (("test", "exact_match", "target"), ("text_before", "exact", "expected"), ("text_after", "exact", "expected")):
    values = partial[key]
    samples = values["samples"]
    exact = []
    for sample in samples:
        ids = sample["generated_ids"]
        content = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        actual = content == tok.encode(sample[target])
        assert actual == sample[flag]
        exact.append(actual)
    correct = sum(exact)
    assert correct == values["correct" if key == "test" else "matches"]
    assert len(samples) == values["examples"]
    recounts[key] = {"correct_recomputed_from_raw_ids": correct, "examples_recounted": len(samples), "effective_tokens": values["effective_tokens"], "all_reported_flags_match_raw_ids": True, "samples": samples}
assert [s["messages"] for s in partial["text_before"]["samples"]] == [s["messages"] for s in partial["text_after"]["samples"]]
out["historical"]["vqa"].update(split_checks=split_checks, partial_training={k: partial["training"][k] for k in ("steps", "effective_tokens", "config", "modal_config")}, partial_freeze_scope=partial["freeze_scope"], recounts=recounts, same_text_questions_before_after=True, text_data=vqa["text_data"])

paths = ["scripts/pretrain_encoders.py", "scripts/train.py", "scripts/infer_modal.py", "scripts/evaluate.py", "scripts/evaluate_modal.py", "scripts/prepare_ocr.py", "scripts/audio_utils.py", "scripts/fetch_training_assets.py", "scripts/course_experiments/run.py", "scripts/course_experiments/common.py", "scripts/course_experiments/modalities.py", "scripts/course_experiments/text.py", "tiny_perceptron/multimodal.py", "tiny_perceptron/modal_data.py", "tiny_perceptron/training.py", "docs/course-experiments/plan.json", "assets/training/manifest.json"]
out["inspected_source_sha256"] = {path: sha(ROOT / path) for path in paths}
out["result"] = "All assertions passed; mechanism and raw-receipt scope only."
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
