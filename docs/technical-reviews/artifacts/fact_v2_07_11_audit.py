"""Bounded CPU review of lesson 7.11; never reruns the GPU training schedules."""

import contextlib
import copy
import hashlib
import inspect
import io
import json
import platform
import random
import re
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.check_technical_reviews import sections
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, pad_batch, render_chat, shifted, toy_conversations
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.training import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_07_11_"
torch.set_num_threads(2)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save_json(name, value):
    path = OUT / (PREFIX + name)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


body = dict(sections(ROOT / "course/chapters/07.md"))["7.11"]
(OUT / (PREFIX + "section.txt")).write_text(body, encoding="utf-8")
code = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(code) == 1
(OUT / (PREFIX + "original_code.txt")).write_text(code[0], encoding="utf-8")
namespace = {}
original_stdout = io.StringIO()
with contextlib.redirect_stdout(original_stdout):
    exec(compile(code[0], "course/chapters/07.md#7.11", "exec"), namespace)
(OUT / (PREFIX + "original_stdout.txt")).write_text(original_stdout.getvalue(), encoding="utf-8")
x, y, valid, model, loss = (namespace[name] for name in ("x", "y", "valid", "model", "loss"))
logits = model(x, valid=valid)["logits"]
mask = y != -100
manual_nll = -logits.log_softmax(-1)[mask].gather(1, y[mask, None]).mean()
torch.manual_seed(42)
unchanged = TinyLM(ModelConfig(width=8))
assert all(torch.equal(value, unchanged.state_dict()[name]) for name, value in model.state_dict().items())
grads = {name: float(parameter.grad.abs().max()) for name, parameter in model.named_parameters()}
assert all(parameter.grad.isfinite().all() for parameter in model.parameters())
assert max(grads.values()) > 0
assert x.shape == y.shape == valid.shape == (2, 10)
assert logits.shape == (2, 10, 264)
assert int(mask.sum()) == 4
assert y[mask].tolist() == [56, 2, 57, 2]
assert torch.allclose(loss.detach(), manual_nll.detach(), atol=1e-6, rtol=0)
assert loss.item() > 0

messages = copy.deepcopy(toy_conversations()[0])
original_x, original_y = render_chat(messages)
messages[-1]["content"] = "3"
changed_x, changed_y = render_chat(messages)
assert original_y[original_y != -100].tolist() == [56, 2]
assert changed_y[changed_y != -100].tolist() == [59, 2]
assert torch.equal(original_x[:-1], changed_x[:-1])
assert toy_conversations()[0][-1]["content"] == "0"

# An unequal-length batch exercises the valid mask, beyond the all-valid lesson batch.
longer = copy.deepcopy(toy_conversations()[0])
longer[-1]["content"] = "12"
px, py, pv = pad_batch([render_chat(toy_conversations()[0]), render_chat(longer)])
padded = model(px, valid=pv)["logits"]
base = model(original_x[None])["logits"]
padding_difference = float((base - padded[:1, : len(original_x)]).abs().max().detach())
assert padding_difference < 1e-6
assert px[0, -1].item() == 0 and py[0, -1].item() == -100 and not pv[0, -1]

environment = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "device": "cpu",
    "cuda_available": str(torch.cuda.is_available()),
    "cpu_threads": str(torch.get_num_threads()),
    "platform": platform.platform(),
}
snapshots = []
for name, obj, relative in (
    ("torch_cross_entropy.txt", F.cross_entropy, "torch/nn/functional.py"),
    ("torch_loss.txt", torch.nn.CrossEntropyLoss, "torch/nn/modules/loss.py"),
    ("torch_backward.txt", torch.Tensor.backward, "torch/_tensor.py"),
):
    source, start = inspect.getsourcelines(obj)
    original_file = Path(inspect.getfile(obj))
    header = (
        f"Installed official PyTorch source, version {torch.__version__}\n"
        f"Git commit {torch.version.git_version}, {relative}, starting line {start}\n"
        f"Whole installed source SHA-256: {digest(original_file.read_bytes())}\n\n"
    )
    (OUT / (PREFIX + name)).write_text(header + "".join(source), encoding="utf-8")
    snapshots.append({"path": relative, "start_line": start, "sha256": digest(original_file.read_bytes())})

# Only one text update and one SFT update, with generated temporary checkpoints.
with tempfile.TemporaryDirectory(prefix=PREFIX) as temporary:
    directory = Path(temporary)
    torch.manual_seed(42)
    base_model = TinyLM(ModelConfig(width=8))
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=0.001)
    tx, ty = shifted([1] + ByteTokenizer().encode("0+0=?0") + [2])
    masked_loss(base_model(tx[None])["logits"], ty[None]).backward()
    optimizer.step()
    source_checkpoint = directory / "text.pt"
    save_checkpoint(source_checkpoint, base_model, optimizer, step=1, metadata={"task": "text", "seed": 42})
    before = {name: value.detach().clone() for name, value in base_model.state_dict().items()}
    dry_output, trained_output = directory / "dry.pt", directory / "sft.pt"
    common = [str(ROOT / ".venv/bin/python"), "scripts/train.py", "--task", "sft", "--checkpoint",
              str(source_checkpoint), "--device", "cpu", "--steps", "1", "--batch-size", "2"]
    cli_runs = []
    for stage, flags in (("dry", ["--output", str(dry_output)]),
                         ("train", ["--train", "--output", str(trained_output)])):
        command = common + flags
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        report = json.loads(result.stdout.splitlines()[-1])
        cli_runs.append({"stage": stage, "command_argv": command, "returncode": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr, "report": report})
    assert not dry_output.exists()
    trained, payload = load_checkpoint(trained_output)
    changed_parameters = [name for name, value in trained.state_dict().items() if not torch.equal(value, before[name])]
    assert changed_parameters
    assert payload["metadata"]["task"] == "sft" and payload["step"] == 1
    assert payload["config"]["width"] == 8 and payload["config"]["vocab_size"] == 264
    assert cli_runs[0]["report"]["mode"] == "dry-run-no-weight-update"
    assert cli_runs[1]["report"]["mode"] == "train"
    assert cli_runs[1]["report"]["history"][0]["effective_tokens"] == 4
    mismatch_path = directory / "mismatch.pt"
    save_checkpoint(mismatch_path, TinyLM(ModelConfig(vocab_size=5, width=8)))
    mismatch_command = common.copy()
    mismatch_command[mismatch_command.index(str(source_checkpoint))] = str(mismatch_path)
    mismatch = subprocess.run(mismatch_command, cwd=ROOT, capture_output=True, text=True)
    assert mismatch.returncode != 0 and "tokenizer" in mismatch.stderr
    cli = {"runs": cli_runs, "changed_parameters": changed_parameters,
           "fresh_stage_step": payload["step"], "loaded_config": payload["config"],
           "mismatch": {"command_argv": mismatch_command, "returncode": mismatch.returncode,
                        "stdout": mismatch.stdout, "stderr": mismatch.stderr}}

formal_path = ROOT / "docs/course-experiments/results/sft.json"
formal = json.loads(formal_path.read_text(encoding="utf-8"))
results = formal["results"]
parts = split_records(generate_records("attributes-sft"), seed=42)
text_rows = [{"text": row["messages"][0]["content"] + row["messages"][1]["content"],
              "family": row["family"]} for row in parts["train"]]
split_stats = {}
for split, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    assert digest(raw) == results["data"][split]["sha256"]
    split_stats[split] = {"records": len(rows), "families": len({row["family"] for row in rows}),
                          "jsonl_sha256": digest(raw),
                          "effective_tokens": sum(int((labels != -100).sum()) for _, labels in text_examples(rows, "sft"))}
assert records_sha256(parts["train"]) == results["training"]["records_sha256"]
assert records_sha256(text_rows) == results["pretrain_then_sft"]["pretraining"]["records_sha256"]
save_json("reconstructed_dataset.json", {"parts": parts, "pretraining_text_rows": text_rows})

budgets = {}
for name, records, mode, steps, entry in (
    ("direct_sft", parts["train"], "sft", 900, results["training"]),
    ("pretraining_text", text_rows, "text", 250, results["pretrain_then_sft"]["pretraining"]),
    ("continued_sft", parts["train"], "sft", 900, results["pretrain_then_sft"]["sft"]),
):
    examples = text_examples(records, mode)
    sampler = random.Random(42)
    total = sum(int((labels != -100).sum())
                for _ in range(steps) for _, labels in sampler.choices(examples, k=16))
    assert total == entry["effective_tokens"] and steps == entry["steps"]
    budgets[name] = {"steps": steps, "batch_size": 16, "lr": 0.003, "effective_tokens": total,
                     "records": len(records), "records_sha256": records_sha256(records)}

sample_audits = []
for phase_name, phase in (("direct_before", results["before"]), ("direct_after", results["after"]),
                          ("pretraining_before_sft", results["pretrain_then_sft"]["before_sft"]),
                          ("pretraining_after_sft", results["pretrain_then_sft"]["after_sft"])):
    for split, evaluation in phase.items():
        assert len(evaluation["samples"]) == evaluation["records"] == len(parts[split])
        recomputed_matches = ended = 0
        for row, sample in zip(parts[split], evaluation["samples"], strict=True):
            assert sample["messages"] == row["messages"][:-1]
            assert sample["expected"] == row["messages"][-1]["content"]
            generated = sample["generated_ids"]
            raw = generated[:generated.index(2)] if 2 in generated else generated
            exact, eos = raw == ByteTokenizer().encode(sample["expected"]), 2 in generated
            assert exact == sample["exact"] and eos == sample["eos"]
            assert ByteTokenizer().decode(raw) == sample["generated"]
            assert len(generated) <= 32
            recomputed_matches += exact
            ended += eos
            sample_audits.append({"phase": phase_name, "split": split, "family": row["family"], **sample})
        assert recomputed_matches == evaluation["matches"]
        assert ended / len(parts[split]) == evaluation["eos_rate"]
        assert split_stats[split]["effective_tokens"] == evaluation["effective_tokens"]
        assert abs(evaluation["nll_sum"] / evaluation["effective_tokens"] - evaluation["nll"]) < 1e-12
save_json("all_sft_raw_samples.json", sample_audits)

revision = formal["revision"]
historical_sources = {}
for relative in ("scripts/course_experiments/run.py", "scripts/prepare_data.py"):
    raw = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)
    name = "formal_" + Path(relative).stem + ".txt"
    (OUT / (PREFIX + name)).write_bytes(raw)
    historical_sources[relative] = {"git_revision": revision, "sha256": digest(raw),
                                    "matches_current": raw == (ROOT / relative).read_bytes()}
    if relative in formal["code_sha256"]:
        assert digest(raw) == formal["code_sha256"][relative]

save_json("audit.json", {
    "command": ".venv/bin/python -m docs.technical-reviews.artifacts.fact_v2_07_11_audit",
    "environment": environment,
    "source_sha256": digest(body.encode()),
    "original_stdout": original_stdout.getvalue(),
    "interface": {"x": x.tolist(), "y": y.tolist(), "valid": valid.tolist(),
                  "x_dtype": str(x.dtype), "y_dtype": str(y.dtype), "valid_dtype": str(valid.dtype),
                  "logits_shape": list(logits.shape), "logits_dtype": str(logits.dtype),
                  "effective_targets": y[mask].tolist(), "loss": float(loss.detach()),
                  "independently_gathered_nll": float(manual_nll.detach()),
                  "gradient_maxima": grads, "weights_unchanged_after_backward": True,
                  "padding_max_difference": padding_difference},
    "exercise": {"before": original_y[original_y != -100].tolist(),
                 "after": changed_y[changed_y != -100].tolist(), "unchanged_question_and_roles": True},
    "cli": cli,
    "formal": {"result_path": str(formal_path.relative_to(ROOT)), "result_sha256": digest(formal_path.read_bytes()),
               "revision": revision, "device": formal["device"], "gpu": formal["gpu"],
               "seed": formal["seed"], "step_scale": formal["step_scale"],
               "evidence_status": formal["evidence_status"], "data": split_stats, "budgets": budgets,
               "all_raw_samples_checked": len(sample_audits), "historical_sources": historical_sources},
    "official_source_snapshots": snapshots,
    "result": "All bounded CPU assertions passed; existing GPU records audited without rerunning training.",
})
print(original_stdout.getvalue(), end="")
print(json.dumps({"result": "passed", "effective_targets": y[mask].tolist(),
                  "loss": float(loss.detach()), "padding_max_difference": padding_difference,
                  "changed_parameters_in_cli_step": len(changed_parameters),
                  "all_raw_sft_samples_checked": len(sample_audits), "budgets": budgets}, ensure_ascii=False))
