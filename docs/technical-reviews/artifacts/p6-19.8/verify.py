"""Independent bounded CPU checks; no optimizer, training run, or held-out evaluation."""
import hashlib
import json
import platform
import sys
import tarfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.selftrained.train import (
    language_loss_weights,
    objective,
    set_trainable,
    stage_records,
    weighted_language_loss,
)
from tiny_perceptron.selftrained.dataset import RecordEncoder
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer
from tiny_perceptron.posttraining import FiniteResponsePolicy

BASE = Path(__file__).parent
torch.set_num_threads(2)
torch.manual_seed(1908)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


result = {
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "scope": "Existing receipt/data accounting and synthetic forward/gradient checks only; no optimizer step, training, test scoring, or paid work.",
    "receipt_checks": [],
}
receipt_root = ROOT / "docs/selftrained/results/training-raw"
for architecture in ("moe", "dense"):
    previous = None
    stages = ["pretrain", "sft", "vision", "ocr", "audio", "joint", "weighted"]
    if architecture == "moe":
        stages.append("native")
    for label in stages:
        run = f"{architecture}-{label}"
        paths = [receipt_root / run / s for s in ("receipt.json", "raw/execution.json", "raw/train-receipt.json")]
        outer, execution, train = [json.loads(p.read_bytes()) for p in paths]
        files = {f["path"]: f for f in outer["files"]}
        assert files["execution.json"]["sha256"] == sha(paths[1].read_bytes())
        assert files["train-receipt.json"]["sha256"] == sha(paths[2].read_bytes())
        assert outer["status"] == execution["status"] == "completed"
        assert outer["returncode"] == execution["returncode"] == 0
        assert train["completed_requested_steps"] and not train["interrupted"]
        assert train["steps"] == execution["job"]["steps"]
        assert train["architecture"] == architecture
        assert not train["test_used_for_selection"]
        assert train["selection"] == "validation_loss (teacher-forced; not generation success)"
        assert train["origin"]["kind"] == "all-neural-weights-random"
        if previous:
            init = execution["job"]["init_checkpoint"]
            assert init["sha256"] == previous["bestpt_sha256"]
            last = train["stage_history"][-1]
            assert last["checkpoint_sha256"] == init["sha256"]
            assert last["selection"] == "validation_loss"
            assert train["tokenizer_sha256"] == previous["tokenizer_sha256"]
            previous["selected_step_seen_in_child"] = last["step"]
        weights = {k: train.get(k, 1.0) for k in ("tool_loss_weight", "numeric_run_loss_weight", "native_voice_loss_weight")}
        if label == "weighted":
            assert list(weights.values()) == [4.0, 4.0, 1.0]
        if label == "native":
            assert list(weights.values()) == [4.0, 1.0, 4.0]
        item = {
            "run": run, "stage": train["stage"], "completed_steps": train["steps"],
            "train_records": train["train_records"], "validation_records": train["validation_records"],
            "input_tokens": train["tokens"], "target_tokens": train["target_tokens"],
            "seed": train["seed"], "bestpt_sha256": files["best.pt"]["sha256"],
            "tokenizer_sha256": train["tokenizer_sha256"], "weights": weights,
            "selection": train["selection"], "files": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in paths},
        }
        result["receipt_checks"].append(item)
        previous = item

# Count original JSONL records through the actual stage filter, without scoring any held-out answer.
archive = ROOT / "assets/training/selftrained-v2.tar.gz"
archive_sha = sha(archive.read_bytes())
assert archive_sha == "0976073a3bc7c331a65cedb4c31d54f5c6e9a5014ff2443698a8e2b8cad7e78a"
records = []
raw_samples = []
members = []
with tarfile.open(archive, "r:gz") as handle:
    for member in handle.getmembers():
        if not member.isfile() or not member.name.endswith(".jsonl"):
            continue
        raw = handle.extractfile(member).read()
        members.append({"path": member.name, "sha256": sha(raw), "bytes": len(raw)})
        lines = raw.splitlines(keepends=True)
        rows = [json.loads(line) for line in lines if line.strip()]
        records.extend(rows)
        if member.name == "text-tools-train.jsonl":
            for task in ("text_pretrain", "text", "tool_call"):
                index = next(i for i, row in enumerate(rows) if row["task"] == task)
                sample_name = f"data-sample-{task}.jsonl"
                (BASE / sample_name).write_bytes(lines[index])
                raw_samples.append({"parent_member": member.name, "parent_sha256": sha(raw), "line_1based": index + 1, "sample_path": sample_name, "sample_sha256": sha(lines[index])})
counts = {stage: {split: len(stage_records(records, stage, split)) for split in ("train", "validation")} for stage in ("pretrain", "sft", "vision", "ocr", "audio", "joint")}
for item in result["receipt_checks"]:
    assert item["train_records"] == counts[item["stage"]]["train"]
    assert item["validation_records"] == counts[item["stage"]]["validation"]
result["data_accounting"] = {"archive_sha256": archive_sha, "members": members, "counts": counts, "raw_samples": raw_samples}

# Synthetic messages verify the encoder's next-character labels and assistant-only SFT mask.
messages = [{"role": "user", "content": "問甲"}, {"role": "assistant", "content": "12甲"}]
tokenizer = CharacterTokenizer.build([m["content"] for m in messages])
encoder = RecordEncoder(tokenizer, BASE, context=64, device="cpu")
record = {"id": "synthetic-19.8", "task": "text", "split": "train", "messages": messages}
pretrain = encoder.encode(record, pretrain=True)
sft = encoder.encode(record)
assert sum(x != -100 for x in pretrain["labels"]) == len(pretrain["input_ids"]) - 1
assert [x for x in sft["labels"] if x != -100] == tokenizer.encode("12甲") + [tokenizer.eos_id]
config = SelftrainedConfig(vocab_size=tokenizer.vocab_size, architecture="dense", width=16, layers=1, heads=2, kv_heads=1, ffn_hidden=32, max_length=64)
model = LimitedAssistant(config)
saved = {k: v.clone() for k, v in model.state_dict().items()}
continuation = LimitedAssistant(config)
continuation.load_state_dict(saved)
assert all(torch.equal(v, continuation.state_dict()[k]) for k, v in saved.items())
set_trainable(continuation, "sft")
assert all(p.requires_grad == name.startswith("lm.") for name, p in continuation.named_parameters())
losses = {}
for stage in ("pretrain", "sft", "joint"):
    loss, detail = objective(continuation, encoder, [record], stage, 1.0, 0.01)
    assert torch.isfinite(loss)
    batch = encoder.batch([record], pretrain=stage == "pretrain")
    output = continuation(**batch)
    independent = torch.nn.functional.cross_entropy(output["logits"].flatten(0, 1), batch["labels"].flatten(), ignore_index=-100)
    torch.testing.assert_close(loss, independent, rtol=1e-6, atol=1e-6)
    losses[stage] = {"target_tokens": detail["target_tokens"], "loss": float(loss.detach())}
result["mask_and_continuation"] = {"pretrain_labels": pretrain["labels"], "sft_labels": sft["labels"], "losses": losses, "exact_state_load": True, "sft_lm_only_trainable": True}

# Independent weighted-CE value and gradient checks, without parameter updates.
labels = torch.tensor([tokenizer.encode("12甲") + [tokenizer.eos_id, -100]] * 4)
rows = [{"task": "tool_call", "split": "train"}, {"task": "voice_qa", "split": "train"}, {"task": "text", "split": "train"}, {"task": "voice_qa", "split": "train", "augmentation": {}}]
weighted_results = []
for policy, expected in [((4, 4, 1), [[16, 16, 16, 4, 0], [4, 4, 4, 1, 0], [4, 4, 4, 1, 0], [4, 4, 4, 1, 0]]), ((4, 1, 4), [[4, 4, 4, 4, 0], [4, 4, 4, 4, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])]:
    weights = language_loss_weights(labels, rows, tokenizer, *policy)
    torch.testing.assert_close(weights, torch.tensor(expected).float(), rtol=0, atol=0)
    logits = torch.randn(4, 5, tokenizer.vocab_size, requires_grad=True)
    actual = weighted_language_loss({"logits": logits}, labels, rows, tokenizer, *policy)
    valid = labels.ne(-100)
    independent = -(logits.log_softmax(-1).gather(-1, labels.clamp_min(0)[..., None]).squeeze(-1) * torch.tensor(expected) * valid).sum() / torch.tensor(expected).sum()
    torch.testing.assert_close(actual, independent, rtol=1e-6, atol=1e-6)
    grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(independent, logits)[0]
    torch.testing.assert_close(grad, expected_grad, rtol=1e-5, atol=1e-7)
    weighted_results.append({"policy_tool_numeric_native": policy, "weights": weights.tolist(), "loss": float(actual.detach()), "value_and_gradient_match": True})
result["weighted_ce"] = weighted_results
finite = FiniteResponsePolicy()
assert finite(torch.zeros(1, 4)).shape == (1, 4)
assert continuation(encoder.batch([record])["input_ids"])["logits"].shape[-1] == tokenizer.vocab_size
result["separate_models"] = {"chapter13_class": type(finite).__name__, "chapter13_output_shape": [1, 4], "public_trainer_class": type(continuation).__name__, "public_output_vocabulary": tokenizer.vocab_size, "same_instance": finite is continuation}
(BASE / "verification-output.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"environment": result["environment"], "receipt_runs": len(result["receipt_checks"]), "data_counts": counts, "mask_losses": losses, "weighted_value_and_gradient_checks": len(weighted_results), "result": "PASS"}, ensure_ascii=False, indent=2))
