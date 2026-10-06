"""Independent bounded CPU checks of the actual training entry point for T.2."""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from tiny_perceptron.data import ByteTokenizer, toy_conversations, toy_documents
from tiny_perceptron.multimodal import scene

assert torch.version.cuda is None
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("actual_train_entry", ROOT / "scripts/train.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
WORK = ART / "run-workspace"
WORK.mkdir(exist_ok=True)
os.chdir(WORK)

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def digest(parameters):
    result = hashlib.sha256()
    for parameter in parameters:
        result.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()

DATA = ART / "custom-data"
DATA.mkdir(exist_ok=True)
(DATA / "text.jsonl").write_text('{"text":"CUSTOM QZ9."}\n', encoding="utf-8")
(DATA / "sft.jsonl").write_text('{"messages":[{"role":"user","content":"Q?"},{"role":"assistant","content":"XYZ"}]}\n', encoding="utf-8")
(DATA / "sft-qa.jsonl").write_text('{"question":"Q?","answer":"XYZ"}\n', encoding="utf-8")
Image.new("RGB", (16, 16), (64, 128, 192)).save(DATA / "custom.png")
(DATA / "vision.jsonl").write_text('{"image":"custom.png","question":"Q?","answer":"XYZ"}\n', encoding="utf-8")
(DATA / "empty.jsonl").write_text("", encoding="utf-8")

original_optimizer = torch.optim.AdamW
original_save = entry.save_checkpoint
original_docs, original_chats, original_scene = entry.toy_documents, entry.toy_conversations, entry.scene
results = []

def run(task, extra=(), label=None, expected_error=None):
    track = {"optimizer_step_calls": 0, "checkpoint_calls": 0, "toy_documents_calls": 0,
             "toy_conversations_calls": 0, "scene_calls": 0}
    parameters, initial = [], []
    def factory(values, *args, **kwargs):
        parameters.extend(list(values))
        initial.extend([p.detach().clone() for p in parameters])
        optimizer = original_optimizer(parameters, *args, **kwargs)
        original_step = optimizer.step
        def counted_step(*a, **k):
            track["optimizer_step_calls"] += 1
            return original_step(*a, **k)
        optimizer.step = counted_step
        return optimizer
    def counted_save(*a, **k):
        track["checkpoint_calls"] += 1
        return original_save(*a, **k)
    def counted_docs():
        track["toy_documents_calls"] += 1
        return original_docs()
    def counted_chats():
        track["toy_conversations_calls"] += 1
        return original_chats()
    def counted_scene(*a, **k):
        track["scene_calls"] += 1
        return original_scene(*a, **k)
    argv = [str(ROOT / "scripts/train.py"), "--task", task, "--device", "cpu", *extra]
    before_files = sorted(str(p.relative_to(WORK)) for p in WORK.rglob("*") if p.is_file())
    stdout = io.StringIO()
    error = None
    with patch.object(sys, "argv", argv), patch.object(torch.optim, "AdamW", factory), \
         patch.object(entry, "save_checkpoint", counted_save), patch.object(entry, "toy_documents", counted_docs), \
         patch.object(entry, "toy_conversations", counted_chats), patch.object(entry, "scene", counted_scene), \
         contextlib.redirect_stdout(stdout):
        try:
            entry.main()
        except Exception as e:
            error = {"type": type(e).__name__, "message": str(e)}
    name = label or task
    (ART / f"stdout-{name}.jsonl").write_text(stdout.getvalue(), encoding="utf-8")
    raw_lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    report = raw_lines[-1] if raw_lines else None
    changed = sum(not torch.equal(a, b) for a, b in zip(initial, parameters))
    zero_grad = sum(p.grad is None or not bool(torch.count_nonzero(p.grad)) for p in parameters)
    after_files = sorted(str(p.relative_to(WORK)) for p in WORK.rglob("*") if p.is_file())
    summary = {"label": name, "argv": [sys.executable, *argv], "cwd": str(WORK), "observed": track,
               "parameter_tensor_count": len(parameters), "changed_parameter_tensors": changed,
               "zero_or_absent_gradient_parameter_tensors": zero_grad,
               "before_parameters_sha256": digest(initial), "after_parameters_sha256": digest(parameters),
               "new_files": sorted(set(after_files) - set(before_files)), "error": error, "report": report}
    if expected_error:
        assert error and error["type"] == expected_error, summary
    else:
        assert error is None and report["task"] == task and report["device"] == "cpu", summary
        assert len(report["history"]) == 1
        item = report["history"][0]
        assert math.isfinite(item["loss"]) and math.isfinite(item["grad_norm"]) and item["grad_norm"] > 0
        if task == "vision":
            assert item["effective_tokens"] is None
        else:
            assert item["effective_tokens"] > 0
        training = "--train" in extra
        assert report["mode"] == ("train" if training else "dry-run-no-weight-update")
        assert track["optimizer_step_calls"] == int(training)
        if training:
            assert changed > 0 and track["checkpoint_calls"] > 0 and summary["new_files"]
        else:
            assert changed == 0 and track["checkpoint_calls"] == 0 and not summary["new_files"]
        if "--data" in extra:
            assert track["toy_documents_calls"] == 0 and track["toy_conversations_calls"] == 0 and track["scene_calls"] == 0
    results.append(summary)

for task in ("text", "sft", "vision"):
    run(task, label=f"default-{task}")
for task in ("text", "sft", "vision"):
    run(task, ("--train", "--steps", "1", "--batch-size", "1", "--width", "8", "--max-length", "64",
               "--output", str(WORK / f"trained-{task}.pt")), label=f"one-step-{task}")
for task in ("text", "sft", "vision"):
    run(task, ("--data", str(DATA / f"{task}.jsonl")), label=f"custom-{task}")
run("sft", ("--data", str(DATA / "sft-qa.jsonl")), label="custom-sft-qa")
run("text", ("--data", str(DATA / "empty.jsonl")), label="empty-text", expected_error="ValueError")
run("vision", ("--data", str(DATA / "empty.jsonl")), label="empty-vision", expected_error="ValueError")

documents = toy_documents()
conversations = toy_conversations()
assert len(documents) == 12 and len(conversations) == 16
assert all(conversation[1]["content"] == str(int(conversation[0]["content"][0]) + int(conversation[0]["content"][2])) for conversation in conversations)
synthetic = {shape: {"shape": list(scene("red", shape).shape), "colored_pixels": int((scene("red", shape)[0] > 0).sum())} for shape in ("circle", "square")}
assert synthetic["circle"]["colored_pixels"] == 49 and synthetic["square"]["colored_pixels"] == 81
tok = ByteTokenizer()
custom_text_examples = entry.prepare_examples(entry.parser().parse_args(["--task", "text", "--data", str(DATA / "text.jsonl")]), tok)
assert tok.decode(custom_text_examples[0][0]) == "CUSTOM QZ9."
custom_chat_examples = entry.prepare_examples(entry.parser().parse_args(["--task", "sft", "--data", str(DATA / "sft.jsonl")]), tok)
assert tok.decode(custom_chat_examples[0][1][custom_chat_examples[0][1] >= 0]) == "XYZ"

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
loss = x.square() + 0.0 * y
loss.backward()
total_norm = float(torch.nn.utils.clip_grad_norm_([x, y], 100.0))
assert total_norm == 6.0 and x.item() == 3.0 and y.item() == 2.0 and y.grad.item() == 0.0
gradient_example = {"loss": float(loss.detach()), "x": float(x.detach()), "y": float(y.detach()),
                    "x_grad": float(x.grad), "y_grad": float(y.grad), "global_grad_norm": total_norm,
                    "parameters_changed_by_backward": False}
environment = {"python": sys.version, "python_executable": sys.executable, "torch": torch.__version__,
               "torch_git_version": torch.version.git_version, "cuda_build": str(torch.version.cuda),
               "device": "cpu", "repo": str(ROOT), "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset")}
write_json(ART / "results.json", {"environment": environment, "runs": results,
           "materials": {"documents": documents, "conversations": conversations, "synthetic_shapes": synthetic},
           "custom_data_decoding": {"text": "CUSTOM QZ9.", "assistant_scored_text": "XYZ"},
           "gradient_counterexample": gradient_example,
           "scope": "12 short CPU entry invocations: three exact default argv, three one-step tiny updates, four custom-data dry runs, and two empty-file errors. Instrumentation counts optimizer/checkpoint/material calls and compares real parameter bytes; it preserves all computations and actual optimizer step/save behavior. No GPU, heldout evaluation, external model, or full training."})
print(json.dumps({"runs": len(results), "status": "all assertions passed", "environment": environment}, ensure_ascii=False))
