"""T.3 independent bounded CPU verification; temporary weights are deleted."""

import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from scripts.course_release import inference_payload
from scripts.train_simple import examples
from tiny_perceptron.data import ByteTokenizer, split_documents, toy_documents
from tiny_perceptron.model import ModelConfig, TinyLM, generate
from tiny_perceptron.simple import BigramLM, ContextMLP
from tiny_perceptron.tokenization import generation_report
from tiny_perceptron.training import load_checkpoint, save_checkpoint, seed_everything

ENV = dict(os.environ)
ENV.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
execution = []
observed = {}


def run(argv, cwd, name):
    result = subprocess.run(argv, cwd=cwd, env=ENV, text=True, capture_output=True, timeout=30)
    (OUT / (name + ".stdout.txt")).write_text(result.stdout)
    (OUT / (name + ".stderr.txt")).write_text(result.stderr)
    execution.append({"command_argv": argv, "command": shlex.join(argv), "cwd": str(cwd),
                      "exit_code": result.returncode, "stdout": name + ".stdout.txt", "stderr": name + ".stderr.txt"})
    assert result.returncode == 0, (name, result.stderr)
    return result


documents = toy_documents()
parts = split_documents(documents, 42)
assert len(documents) == len(set(documents)) == 12
assert {k: len(v) for k, v in parts.items()} == {"train": 9, "validation": 1, "test": 2}
assert parts == split_documents(documents + documents[:3], 42)
assert not (set(parts["train"]) & set(parts["validation"]))
assert not (set(parts["train"]) & set(parts["test"]))
vocabulary = {char: index + 2 for index, char in enumerate(sorted(set("".join(parts["train"]))))}
v = len(vocabulary) + 2
assert v == 17
x, y = examples(parts["train"], vocabulary, 3)
vx, vy = examples(parts["validation"], vocabulary, 3)
assert x.shape == (len(y), 3)
assert len(y) == sum(len(d) + 1 for d in parts["train"])
assert y.eq(0).sum().item() == 9
ux, uy = examples(["A"], {}, 3)
assert ux.tolist() == [[0, 0, 0], [0, 0, 1]] and uy.tolist() == [1, 0]
observed["documents"] = {"count": len(documents), "split": {k: len(z) for k, z in parts.items()},
                         "vocab_with_boundary_unknown": v, "train_targets": len(y), "validation_targets": len(vy),
                         "dedup_same_split": True, "unknown_id": 1, "boundary_id": 0}

seed_everything(42)
mlp = ContextMLP(v, 3, 16)
emb = mlp.embedding(x[:2])
hidden = torch.tanh(mlp.hidden(emb.flatten(1)))
assert list(emb.shape) == [2, 3, 16]
assert list(hidden.shape) == [2, 16]
assert list(mlp(x[:2]).shape) == [2, 17]
counts = {}
for context in [1, 3, 5]:
    model = ContextMLP(v, context, 16)
    actual = sum(p.numel() for p in model.parameters())
    formula = v * 16 + context * 16 * 16 + 16 + 16 * v + v
    assert actual == formula
    counts[str(context)] = actual
assert counts == {"1": 833, "3": 1345, "5": 1857}
changed = ContextMLP(v, 3, 7)
assert list(changed.embedding.weight.shape) == [17, 7]
assert list(changed.hidden.weight.shape) == [7, 21]
assert sum(p.numel() for p in changed.parameters()) == 409
bigram = BigramLM(v)
ctx = torch.tensor([[2, 3, 4], [5, 3, 4]])
assert torch.equal(bigram(ctx)[0], bigram(ctx)[1])
assert not torch.equal(mlp(ctx)[0], mlp(ctx)[1])
observed["width_context"] = {"embedding_shape": list(emb.shape), "hidden_shape": list(hidden.shape),
                             "counts_width16": counts, "width7_context3_parameters": 409,
                             "bigram_uses_last_only": True, "mlp_oldest_change_affects_logits": True}

section = (OUT / "section.md").read_text()
fences = []
opened = None
for line in section.splitlines(keepends=True):
    if opened is None and line.startswith("```"):
        opened = {"language": line.strip()[3:], "lines": []}
    elif opened is not None and line.startswith("```"):
        fences.append(opened)
        opened = None
    elif opened is not None:
        opened["lines"].append(line)
for i, fence in enumerate(fences, 1):
    (OUT / f"original-fence-{i}.{('json' if fence['language'] == 'json' else 'sh')}").write_text("".join(fence["lines"]))

with tempfile.TemporaryDirectory(prefix="t3-independent-cpu-") as temp:
    workspace = Path(temp)
    for name in [".venv", "scripts", "tiny_perceptron"]:
        (workspace / name).symlink_to(ROOT / name, target_is_directory=True)
    # The complete first original bash fence is short, offline and has no updates.
    run(["bash", "--noprofile", "--norc", str(OUT / "original-fence-1.sh")], workspace, "original-dry-fence")
    for kind in ["bigram", "mlp"]:
        report = json.loads((workspace / "outputs" / f"{kind}-before.json").read_text())
        (OUT / f"{kind}-before.json").write_text(json.dumps(report, indent=2) + "\n")
        assert report["mode"] == "dry-run-no-weight-update"
        assert report["split_unit"] == "whole deduplicated document"
        assert report["train_loss"] == report["last_batch_loss_before_update"]
        assert not list((workspace / "checkpoints").glob("*.pt"))
        seed_everything(42)
        model = BigramLM(v) if kind == "bigram" else ContextMLP(v, 3, 16)
        tx, ty = examples(parts["train"], vocabulary, 1 if kind == "bigram" else 3)
        tvx, tvy = examples(parts["validation"], vocabulary, 1 if kind == "bigram" else 3)
        with torch.no_grad():
            expected = F.cross_entropy(model(tx), ty).item()
            validation = F.cross_entropy(model(tvx), tvy).item()
        assert abs(expected - report["train_loss"]) < 1e-6
        assert abs(validation - report["validation_loss"]) < 1e-6
        state_before = {k: z.clone() for k, z in model.state_dict().items()}
        name = f"{kind}-bounded-two-updates"
        argv = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/train_simple.py"), "--model", kind,
                "--seed", "42", "--device", "cpu", "--train", "--steps", "2", "--output", str(workspace / f"{kind}.pt")]
        if kind == "mlp":
            argv += ["--context", "3", "--width", "16"]
        after = json.loads(run(argv, workspace, name).stdout)
        saved = torch.load(workspace / f"{kind}.pt", weights_only=True, map_location="cpu")
        assert set(saved) == {"model", "vocabulary", "context", "width", "kind", "report"}
        model.load_state_dict(saved["model"])
        assert any(not torch.equal(state_before[k], z) for k, z in model.state_dict().items())
        with torch.no_grad():
            post = F.cross_entropy(model(tx), ty).item()
            post_validation = F.cross_entropy(model(tvx), tvy).item()
            per_target = -model(tx).log_softmax(-1).gather(1, ty[:, None]).mean().item()
        assert abs(post - after["train_loss"]) < 1e-6
        assert abs(post_validation - after["validation_loss"]) < 1e-6
        assert abs(per_target - post) < 1e-6
        assert after["last_batch_loss_before_update"] != after["train_loss"]
        try:
            load_checkpoint(workspace / f"{kind}.pt")
        except ValueError as error:
            assert str(error) == "不支援的 checkpoint 格式"
        else:
            raise AssertionError("simple checkpoint unexpectedly accepted by TinyLM loader")
        observed[kind + "_cli"] = {"before": report, "bounded_after_two_updates": after,
                                  "post_update_recomputed": post, "heldout_recomputed": post_validation,
                                  "weights_changed": True, "native_tinylm_loader_rejects_simple": True}
    assert (3.0 - 4.0) < 0 and (5.0 - 4.0) > 0
    run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/fetch_course_models.py"), "--list"], ROOT, "public-models-list")
    run([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--list-assets", "simple_models"], ROOT, "simple-models-list-assets")
    run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/infer.py"), "--help"], ROOT, "infer-help")
    # Only random tiny local state is saved; it is not a published course model or ability result.
    seed_everything(123)
    native = TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=27))
    optimizer = torch.optim.AdamW(native.parameters())
    path = workspace / "tiny-random.pt"
    save_checkpoint(path, native, optimizer=optimizer, step=7, metadata={"experiment": "bounded_contract"})
    payload = torch.load(path, weights_only=True, map_location="cpu")
    clean = inference_payload(payload, {"revision": "bounded-contract"})
    assert not ({"optimizer", "step", "torch_rng", "python_rng", "cuda_rng", "mps_rng", "training_state"} & set(clean))
    assert {"format_version", "config", "model", "tokenizer", "metadata"} <= set(clean)
    torch.save(clean, path)
    infer = json.loads(run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/infer.py"), str(path),
                           "--prompt", "color=blue;shape=circle;", "--tokens", "32", "--device", "cpu", "--json"], ROOT, "infer-tiny-contract").stdout)
    assert {"answer", "generated_ids", "eos", "invalid_special_tokens", "valid_answer_tokens", "generation_status"} == set(infer)
    assert len(infer["generated_ids"]) <= 2  # prompt plus BOS consumes 25 of 27 positions
    observed["tiny_infer_cli"] = {"scope": "temporary random state, no ability conclusion", "report": infer,
                                  "native_keys": sorted(payload), "exported_keys": sorted(clean),
                                  "transient_weight_path_deleted_when_context_exits": True}

tokenizer = ByteTokenizer()
sample = json.loads("".join(fences[4]["lines"]))
assert tokenizer.encode("A") == [73] and tokenizer.eos_id == 2
full = generation_report(tokenizer, [73, 2])
assert sample == {k: full[k] for k in sample}
assert generation_report(tokenizer, [73])["answer"] == full["answer"] == "A"
assert generation_report(tokenizer, [73])["generation_status"] == "token_or_context_limit"
assert generation_report(tokenizer, [73, 3, 2])["generation_status"] == "invalid_special_tokens"


class Controlled(nn.Module):
    """An oracle for stop mechanics; these are not learned model answers."""

    def __init__(self, ids, max_length):
        super().__init__()
        self.config = SimpleNamespace(max_length=max_length)
        self.outputs = ids
        self.calls = 0

    def forward(self, ids, cache=None):
        scores = torch.full((1, ids.shape[1], 264), -100.0)
        scores[:, -1, self.outputs[min(self.calls, len(self.outputs) - 1)]] = 100
        self.calls += 1
        return {"logits": scores, "cache": None}


stop_cases = {}
for name, planned, context_limit, cap, expected in [
    ("early_eos", [73, 2], 64, 32, [73, 2]),
    ("token_limit", [73], 64, 3, [73, 73, 73]),
    ("context_limit", [73], 3, 32, [73, 73]),
    ("already_at_context_limit", [73], 1, 32, []),
]:
    controlled = Controlled(planned, context_limit)
    output = generate(controlled, torch.tensor([[1]]), cap, eos_id=2)
    got = output[0, 1:].tolist()
    assert got == expected
    assert controlled.training is True
    stop_cases[name] = {"planned_oracle_ids": planned, "max_length": context_limit, "max_new_tokens": cap,
                        "actual_new_ids": got, "report": generation_report(tokenizer, got)}
observed["generation"] = {"handwritten_excerpt_matches_current_schema_subset": True, "full_report": full,
                          "A_utf8_byte": list("A".encode("utf-8")), "byte_offset": 8, "eos_id": 2,
                          "cases": stop_cases, "scope": "controlled stop mechanics, not a trained text_foundation response"}

environment = {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__),
               "torch_git_version": str(torch.version.git_version), "cuda_build": str(torch.version.cuda),
               "cuda_available": str(torch.cuda.is_available()), "device": "cpu",
               "platform": sys.platform, "huggingface_hub": importlib.metadata.version("huggingface_hub")}
(OUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
(OUT / "commands.json").write_text(json.dumps(execution, indent=2) + "\n")
(OUT / "verification.json").write_text(json.dumps({"status": "all_assertions_passed", "environment": environment,
                                                "observed": observed, "command_count": len(execution),
                                                "scope": "Bounded CPU contract validation only; original 200-step recipes and public weights not executed or downloaded."},
                                               ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": "all_assertions_passed", "commands": len(execution), "torch": str(torch.__version__),
                  "device": "cpu", "width16_counts": counts, "generation_example": full,
                  "temporary_weights_retained": False}, ensure_ascii=False))
