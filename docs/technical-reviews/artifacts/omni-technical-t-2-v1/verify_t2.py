"""T.2 reviewer evidence: exact frozen commands plus separately labelled CPU probes."""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.2.md"
EXPECTED_SOURCE = "f592deace89a1efcc79d55bf7223e7314a6ef3c52d42de0334349fea31006a47"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def parsed(stdout, task):
    lines = stdout.strip().splitlines()
    report = json.loads(lines[-1])
    assert len(lines) == 2, lines
    assert report["mode"] == "dry-run-no-weight-update"
    assert report["task"] == task and report["device"] == "cpu"
    assert len(report["history"]) == 1
    assert json.loads(lines[0]) == report["history"][0]
    entry = report["history"][0]
    assert math.isfinite(entry["loss"])
    assert math.isfinite(entry["grad_norm"]) and entry["grad_norm"] > 0
    assert entry["effective_tokens"] is None if task == "vision" else entry["effective_tokens"] > 0
    return report


def main():
    assert sha(SNAPSHOT) == EXPECTED_SOURCE
    frozen = SNAPSHOT.read_text(encoding="utf-8")
    block = frozen.split("```bash\n", 1)[1].split("```", 1)[0]
    commands = block.strip().splitlines()
    expected = [f".venv/bin/python scripts/train.py --task {task} --device cpu" for task in ("text", "sft", "vision")]
    assert commands == expected
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_sha256": sha(SNAPSHOT),
        "train_sha256": sha(ROOT / "scripts/train.py"),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_git": torch.version.git_version, "platform": platform.platform(),
            "device": "cpu", "executable": sys.executable,
        },
        "exact_frozen_commands": [], "instrumented_entrypoint_probes": [],
    }
    for command, task in zip(commands, ("text", "sft", "vision"), strict=True):
        completed = subprocess.run(command.split(), cwd=ROOT, text=True, capture_output=True, timeout=60)
        assert completed.returncode == 0, completed.stderr
        report = parsed(completed.stdout, task)
        result["exact_frozen_commands"].append({
            "command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr, "report": report,
        })

    # Independent observation around the unchanged train.main entrypoint.
    # These probes are not the exact subprocess runs recorded above.
    spec = importlib.util.spec_from_file_location("t2_train_under_review", ROOT / "scripts/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_init = torch.optim.AdamW.__init__
    original_step = torch.optim.AdamW.step
    for task in ("text", "sft", "vision"):
        observed = {"optimizer_step_calls": 0, "save_checkpoint_calls": 0, "torch_save_calls": 0}
        captured = {}

        def init(optimizer, params, *args, **kwargs):
            parameters = list(params)
            captured["parameters"] = parameters
            captured["before"] = [p.detach().clone() for p in parameters]
            original_init(optimizer, parameters, *args, **kwargs)

        def step(optimizer, *args, **kwargs):
            observed["optimizer_step_calls"] += 1
            return original_step(optimizer, *args, **kwargs)

        def no_checkpoint(*args, **kwargs):
            observed["save_checkpoint_calls"] += 1
            raise AssertionError("dry run attempted save_checkpoint")

        def no_save(*args, **kwargs):
            observed["torch_save_calls"] += 1
            raise AssertionError("dry run attempted torch.save")

        stdout = io.StringIO()
        with patch.object(torch.optim.AdamW, "__init__", init), patch.object(torch.optim.AdamW, "step", step), patch.object(module, "save_checkpoint", no_checkpoint), patch.object(torch, "save", no_save), patch.object(sys, "argv", ["scripts/train.py", "--task", task, "--device", "cpu"]), contextlib.redirect_stdout(stdout):
            module.main()
        report = parsed(stdout.getvalue(), task)
        parameters = captured["parameters"]
        observed.update({
            "parameter_tensors": len(parameters),
            "parameter_tensors_changed": sum(not torch.equal(before, after.detach()) for before, after in zip(captured["before"], parameters, strict=True)),
            "parameter_tensors_with_grad": sum(p.grad is not None for p in parameters),
            "parameter_tensors_with_nonzero_grad": sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in parameters),
            "gradient_norm_after_clip": float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in parameters if p.grad is not None]))),
        })
        assert observed["optimizer_step_calls"] == observed["save_checkpoint_calls"] == observed["torch_save_calls"] == observed["parameter_tensors_changed"] == 0
        result["instrumented_entrypoint_probes"].append({"task": task, "report": report, "observed": observed})

    # One valid user-owned JSONL record for each advertised task, using --data.
    fixtures = {
        "text": {"text": "測試文字。"},
        "sft": {"messages": [{"role": "user", "content": "1+2=?"}, {"role": "assistant", "content": "3"}]},
        "vision": {"image": "custom-image.png", "question": "shape?", "answer": "square"},
    }
    image = Image.new("RGB", (16, 16), "black")
    ImageDraw.Draw(image).rectangle((4, 4, 12, 12), fill="red")
    image.save(HERE / "custom-image.png")
    result["custom_data_commands"] = []
    for task, record in fixtures.items():
        fixture = HERE / f"custom-{task}.jsonl"
        with fixture.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        command = [".venv/bin/python", "scripts/train.py", "--task", task, "--device", "cpu", "--data", str(fixture.relative_to(ROOT))]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
        assert completed.returncode == 0, completed.stderr
        report = parsed(completed.stdout, task)
        result["custom_data_commands"].append({"command": " ".join(command), "input": record, "input_sha256": sha(fixture), "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "report": report})

    # Inspect concrete target masks, dtypes, shapes, and one expanded vision target.
    from tiny_perceptron.data import ByteTokenizer, pad_batch, render_chat, toy_conversations, toy_documents
    from tiny_perceptron.multimodal import expand_modalities

    tok = ByteTokenizer()
    x, y, valid = pad_batch([render_chat(toy_conversations()[0]), render_chat(toy_conversations()[-1])])
    assert y.tolist() == [[-100] * 8 + [56, 2]] * 1 + [[-100] * 8 + [62, 2]] * 1
    logits = torch.zeros(1, 3, tok.vocab_size, requires_grad=True)
    labels = torch.tensor([[-100, tok.encode("3")[0], 2]])
    from tiny_perceptron.model import masked_loss
    loss = masked_loss(logits, labels)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert torch.allclose(loss.detach(), torch.tensor(math.log(tok.vocab_size)))
    vision_ids = torch.tensor([1, 5, 4, 73, 2])
    vision_labels = torch.tensor([-100, -100, -100, 73, 2])
    expanded, target = expand_modalities(vision_ids, vision_labels, torch.nn.Embedding(264, 8), {5: torch.randn(3, 8)}, {5})
    assert expanded.shape == (1, 6, 8) and target.tolist() == [[-100, -100, -100, -100, 73, 2]]
    result["input_and_mask_probes"] = {
        "toy_documents_count": len(toy_documents()), "toy_document_first": toy_documents()[0],
        "toy_document_last": toy_documents()[-1], "toy_conversations_count": len(toy_conversations()),
        "toy_conversation_first": toy_conversations()[0], "toy_conversation_last": toy_conversations()[-1],
        "sft_input_shape": list(x.shape), "sft_input_dtype": str(x.dtype), "sft_labels": y.tolist(),
        "sft_target_dtype": str(y.dtype), "sft_valid_dtype": str(valid.dtype),
        "sft_effective_tokens": int((y != -100).sum()),
        "masked_uniform_logits_shape": list(logits.shape), "masked_labels": labels.tolist(),
        "masked_loss": float(loss.detach()), "uniform_expected_log_vocab": math.log(tok.vocab_size),
        "ignored_logits_grad_nonzero_count": int(torch.count_nonzero(logits.grad[0, 0])),
        "vision_expanded_shape": list(expanded.shape), "vision_expanded_labels": target.tolist(),
    }
    result["limits"] = "All actual executions are bounded CPU dry-runs. No --train, optimizer update, GPU, paid job, model/data publishing, quality evaluation or cross-platform execution was performed."
    write_new(HERE / "execution.json", result)
    print(json.dumps({"result": "all assertions passed", "artifact": str((HERE / 'execution.json').relative_to(ROOT)), "sha256": sha(HERE / "execution.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
