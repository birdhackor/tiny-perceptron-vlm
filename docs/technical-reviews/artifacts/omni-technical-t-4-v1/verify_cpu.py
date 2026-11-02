"""T.4 independent review: exact embedded snippet and bounded teaching CPU commands."""

import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.evaluate import answer_sample, evaluate
from tiny_perceptron.data import ByteTokenizer, load_jsonl, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, generate
from tiny_perceptron.tokenization import load_tokenizer
from tiny_perceptron.training import load_checkpoint, seed_everything

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.4.md"
RUN = Path("/workspace/work/omni-technical-t-4-v1/run")
RUN.mkdir(parents=True, exist_ok=False)
(RUN / "outputs").mkdir()
(RUN / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
raw = SNAPSHOT.read_text()
python_blocks = re.findall(r"```python\n(.*?)```", raw, re.S)
assert len(python_blocks) == 1
snippet = python_blocks[0]
(RUN / "outputs/save_start.py").write_text(snippet)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
interpreter = ["uv", "run", "--project", str(ROOT), "--extra", "cpu", "python"]
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "source": str(SNAPSHOT.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
    "embedded_python_sha256": hashlib.sha256(snippet.encode()).hexdigest(),
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git": torch.version.git_version, "device": "cpu", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "platform": platform.platform(), "workdir": str(RUN)},
    "commands": [],
    "limitations": ["Only this lesson's 32-wide one-layer short CPU routes are trained; no fixed course recipe, GPU, paid service, download or publication is run.", "Same seed is checked within this installed environment, not a guarantee across versions or devices.", "Binary checkpoints remain outside Git. All small evaluation reports and full command stdout/stderr are retained here once."],
}


def run(args, expected_exit=0, label=None):
    command = interpreter + args
    started = datetime.now(UTC).isoformat()
    process = subprocess.run(command, cwd=RUN, env=env, text=True, capture_output=True, timeout=90)
    entry = {"label": label, "command": shlex.join(command), "cwd": str(RUN), "started_at": started, "completed_at": datetime.now(UTC).isoformat(), "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
    result["commands"].append(entry)
    (ARTIFACTS / "cpu-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    assert process.returncode == expected_exit, entry
    print(label or shlex.join(args), process.returncode, flush=True)
    return process


bash = re.findall(r"```bash\n(.*?)```", raw, re.S)
assert len(bash) == 5
for line in bash[0].strip().splitlines():
    args = shlex.split(line)
    assert args.pop(0) == ".venv/bin/python"
    run(args, label="exact lesson data preparation")

result["data"] = {}
for kind, sizes in (("toy-text", [9, 1, 2]), ("attributes-sft", [45, 5, 10])):
    folder = RUN / "data/generated" / kind
    manifest = json.loads((folder / "manifest.json").read_text())
    groups = {}
    for split, size in zip(("train", "validation", "test"), sizes, strict=True):
        rows = load_jsonl(folder / f"{split}.jsonl")
        assert len(rows) == size
        groups[split] = sorted({r["family"] for r in rows})
        if kind == "attributes-sft":
            assert all(sum(r["family"] == family for r in rows) == 5 for family in groups[split])
    assert not set(groups["train"]) & set(groups["validation"])
    assert not set(groups["train"]) & set(groups["test"])
    assert not set(groups["validation"]) & set(groups["test"])
    result["data"][kind] = {"manifest": manifest, "family_assignments": groups}
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    run(["scripts/prepare_data.py", "--kind", kind, "--seed", "42"], label="repeat seed determinism")
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    assert before == after
    result["data"][kind]["repeat_exact_hashes"] = before

run(["outputs/save_start.py"], label="exact embedded Python from frozen T.4")
initial, payload = load_checkpoint(RUN / "checkpoints/start.pt")
seed_everything(42)
repeated = TinyLM(ModelConfig(width=32, layers=1, heads=1, max_length=128))
equal = all(torch.equal(value, repeated.state_dict()[name]) for name, value in initial.state_dict().items())
assert equal and payload["step"] == 0 and payload["optimizer"] is None
result["baseline"] = {"config": payload["config"], "step": payload["step"], "optimizer": payload["optimizer"], "model_equal_to_repeated_seed": equal, "checkpoint_sha256": hashlib.sha256((RUN / "checkpoints/start.pt").read_bytes()).hexdigest()}

tok = ByteTokenizer()
messages = [{"role": "user", "content": "color=red;shape=circle;pitch=low;shape?"}, {"role": "assistant", "content": "circle"}]
x, y = render_chat(messages)
actual_targets = y[y != -100].tolist()
assert actual_targets == tok.encode("circle") + [tok.eos_id]
result["supervision_probe"] = {"messages": messages, "inputs": x.tolist(), "labels": y.tolist(), "effective_target_ids": actual_targets, "effective_target_count": len(actual_targets), "utf8_example": {"text": "貓", "bytes": list("貓".encode()), "token_ids": tok.encode("貓")}}

for block in bash[1:4]:
    for line in block.strip().splitlines():
        args = shlex.split(line)
        assert args.pop(0) == ".venv/bin/python"
        run(args, label="exact teaching CPU route; interpreter changed to uv --extra cpu only")

result["evaluations"] = {p.name: json.loads(p.read_text()) for p in sorted((RUN / "outputs").glob("*.json"))}
result["training_summaries"] = {}
for name in ("text", "attributes", "text-then-sft"):
    saved = torch.load(RUN / f"checkpoints/{name}.pt", map_location="cpu", weights_only=True)
    report = saved["metadata"]["report"]
    result["training_summaries"][name] = {"step": saved["step"], "metadata": {k:v for k,v in saved["metadata"].items() if k != "report"}, "config": saved["config"], "optimizer_present": saved["optimizer"] is not None, "optimizer_step_values": sorted({int(v["step"].item()) for v in saved["optimizer"]["state"].values()}), "loop_seconds": report["seconds"], "actual_update_count": len(report["history"]), "sample_draws": 4 * len(report["history"]), "effective_training_tokens": sum(row["effective_tokens"] for row in report["history"]), "first_step": report["history"][0], "last_step": report["history"][-1], "checkpoint_sha256": hashlib.sha256((RUN / f"checkpoints/{name}.pt").read_bytes()).hexdigest()}

result["answer_edge_probes"] = {}
cases = {"correct_and_eos": tok.encode("circle") + [tok.eos_id], "correct_without_eos": tok.encode("circle"), "leading_space": tok.encode(" circle") + [tok.eos_id], "illegal_user": [tok.user_id] + tok.encode("circle") + [tok.eos_id]}
for label, ids in cases.items():
    result["answer_edge_probes"][label] = answer_sample(tok, ids, "circle")
assert result["answer_edge_probes"]["correct_and_eos"]["completed_exact_match"]
assert result["answer_edge_probes"]["correct_without_eos"]["exact_match"]
assert not result["answer_edge_probes"]["correct_without_eos"]["completed_exact_match"]
assert not result["answer_edge_probes"]["leading_space"]["exact_match"]
assert not result["answer_edge_probes"]["illegal_user"]["exact_match"]
assert "<user>" in result["answer_edge_probes"]["illegal_user"]["generated"]

truncated = {"messages": [{"role":"user", "content":"x" * 130}, {"role":"assistant", "content":"circle"}]}
normal = {"messages": [{"role":"user", "content":"Q"}, {"role":"assistant", "content":"circle"}]}
result["skip_probe"] = evaluate(initial, [normal, truncated], "sft", 2)
assert result["skip_probe"]["records_read"] == 2
assert result["skip_probe"]["loss_evaluated_records"] == 1
assert result["skip_probe"]["generation_evaluated_records"] == 1
assert result["skip_probe"]["metric_denominators"]["mean_token_nll"] == 7
assert result["skip_probe"]["samples"][0]["row"] == 0
assert result["skip_probe"]["skipped"][0]["row"] == 1
generated = generate(initial, torch.tensor([[tok.bos_id]]), max_new_tokens=2)
assert 1 <= generated.shape[1] - 1 <= 2
result["new_token_budget_probe"] = {"prompt_ids": [tok.bos_id], "max_new_tokens": 2, "output_ids": generated.tolist(), "new_token_count": generated.shape[1] - 1}
try:
    load_tokenizer(None, 512)
except ValueError as e:
    result["missing_bpe_pair_guard"] = str(e)
else:
    raise AssertionError("non-byte vocabulary accepted without tokenizer")

run(["scripts/train.py", "--task", "sft", "--data", "data/generated/attributes-sft/train.jsonl", "--checkpoint", "checkpoints/text.pt", "--resume", "--steps", "500", "--device", "cpu"], expected_exit=1, label="resume cannot masquerade as a changed training stage; fails before updates")
run(["scripts/train.py", "--task", "text", "--data", "data/generated/toy-text/train.jsonl", "--checkpoint", "checkpoints/start.pt", "--resume", "--steps", "200", "--device", "cpu"], expected_exit=1, label="untrained saved baseline has no optimizer; cannot resume")
run(["scripts/train.py", "--help"], label="training CLI")
run(["scripts/evaluate.py", "--help"], label="evaluation CLI")
run(["-m", "scripts.course_experiments.run", "--help"], label="fixed recipe CLI only; no recipe execution")
run(["scripts/fetch_training_assets.py", "--help"], label="asset fetch CLI help only; no download")
for experiment in ("text_foundation", "real_text", "tokenizer", "sft", "sft_ablation"):
    run(["-m", "scripts.course_experiments.run", "--list-assets", experiment], label="fixed recipe asset listing only")
run(["-m", "scripts.course_experiments.run", "--experiment", "sft", "--device", "mps"], expected_exit=2, label="fixed recipe rejects MPS before training")

result["completed_at"] = datetime.now(UTC).isoformat()
(ARTIFACTS / "cpu-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": "all assertions passed", "evaluations": {name: {k: r[k] for k in ("mean_token_nll", "effective_tokens", "exact_match", "completed_exact_match", "eos_rate", "metric_denominators")} for name, r in result["evaluations"].items()}, "training_summaries": result["training_summaries"]}, ensure_ascii=False))
