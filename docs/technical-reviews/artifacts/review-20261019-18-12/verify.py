"""核對模型介面、遮罩、參數與故事蒸餾紀錄；不訓練模型。"""

import collections
import copy
import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.course_experiments.compression import _evaluate
from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import DenseFFN, MoEFFN


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


torch.set_num_threads(1)
torch.manual_seed(0)
teacher = TinyLM(ModelConfig(width=16, layers=1, experts=3, top_k=2)).eval().requires_grad_(False)
student = TinyLM(ModelConfig(width=8, layers=1))
ids = torch.tensor([[1, 2]])
labels = torch.tensor([[1, 2]])
routes = []
handle = teacher.blocks[0].ffn.register_forward_hook(lambda module, args, output: routes.append(output[2].tolist()))
with torch.no_grad():
    t = teacher(ids)["logits"]
handle.remove()
s = student(ids)["logits"]
loss = distillation_kl(s, t, labels, temperature=1.0)
p, logq = t.softmax(-1), s.log_softmax(-1)
direct = (p * (p.log() - logq)).sum(-1).mean()
loss.backward()
assert torch.allclose(loss, direct, atol=1e-6, rtol=1e-6)
assert isinstance(teacher.blocks[0].ffn, MoEFFN)
assert isinstance(student.blocks[0].ffn, DenseFFN)
assert student.output.weight.grad.abs().sum() > 0
assert all(parameter.grad is None for parameter in teacher.parameters())
assert tuple(t.shape) == tuple(s.shape) == (1, 2, 264)
assert teacher.description()["parameters"] == 18048
assert student.description()["parameters"] == 6104
changed_labels = torch.tensor([[201, 230]])
assert torch.equal(distillation_kl(s, t, changed_labels, 1.0), loss.detach())
masked_labels = torch.tensor([[IGNORE, 2]])
masked_loss = distillation_kl(s, t, masked_labels, 1.0)
masked_direct = (p * (p.log() - logq)).sum(-1)[0, 1]
assert torch.allclose(masked_loss, masked_direct, atol=1e-6, rtol=1e-6)
bad = TinyLM(ModelConfig(width=8, layers=1, vocab_size=265))(ids)["logits"]
try:
    distillation_kl(bad, t, labels, 1.0)
except ValueError as exc:
    shape_error = str(exc)
else:
    raise AssertionError("不同字表大小應被拒絕")
alternate = TinyLM(ModelConfig(width=8, layers=2))(ids)["logits"]
assert torch.isfinite(distillation_kl(alternate, t, labels, 1.0))
real_config = ModelConfig(width=64, layers=2, heads=4, experts=4, top_k=2)
real_teacher = TinyLM(real_config)
assert real_teacher.description()["parameters"] == 340608
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-12/verify.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "interface": {
        "teacher_config": teacher.description()["config"],
        "student_config": student.description()["config"],
        "shapes": [list(t.shape), list(s.shape)],
        "parameters": [18048, 6104],
        "selected_experts": routes,
        "third_expert_stored_parameters": sum(p.numel() for p in teacher.blocks[0].ffn.experts[2].parameters()),
        "kl": float(loss.detach()),
        "direct_kl": float(direct.detach()),
        "valid_target_count": int((labels != IGNORE).sum()),
        "positive_student_output_gradient": bool(student.output.weight.grad.abs().sum() > 0),
        "teacher_has_no_gradient": all(p.grad is None for p in teacher.parameters()),
        "changing_valid_label_values_keeps_kl": True,
        "one_valid_position_kl": float(masked_loss.detach()),
        "shape_mismatch_shapes": [list(bad.shape), list(t.shape)],
        "shape_error": shape_error,
        "different_layer_count_works": True,
        "experiment_teacher_parameters": real_teacher.description()["parameters"],
    },
}
paths = ["tiny_perceptron/model.py", "tiny_perceptron/modern.py", "tiny_perceptron/alignment.py", "tiny_perceptron/data.py", "scripts/course_experiments/common.py", "scripts/course_experiments/compression.py", "scripts/course_experiments/architecture.py", "docs/course-experiments/results/distillation.json", "docs/course-experiments/results/moe.json"]
result["source_sha256"] = {path: sha((ROOT / path).read_bytes()) for path in paths}
d = json.loads((ROOT / "docs/course-experiments/results/distillation.json").read_bytes())
m = d["results"]["tasks"]["moe_to_dense"]
ce, kd = m["runs"]["w32_ce"], m["runs"]["w32_ce_kl"]
evaluations = {"teacher": m["teacher_test"], "ce": ce["test"], "ce_kl": kd["test"]}
metrics = {}
tok = ByteTokenizer()
for name, record in evaluations.items():
    calculated = record["nll_sum"] / record["supervised_tokens"]
    assert abs(calculated - record["answer_nll"]) < 1e-12
    assert record["supervised_tokens"] == 41914
    assert record["examples"] == len(record["generated_samples"]) == 52
    assert record["answer_bytes"] + record["examples"] == record["supervised_tokens"]
    assert all(tok.decode(row["generated_ids"]) == row["generated"] for row in record["generated_samples"])
    metrics[name] = {"nll_sum": record["nll_sum"], "denominator": record["supervised_tokens"], "nll_recomputed": calculated, "rounded": format(calculated, ".4f"), "stories": record["examples"], "chunks": record["nll_sequence_chunks"], "story_bytes": record["answer_bytes"], "first_generated_sample": record["generated_samples"][0], "all_generated_ids_decode_consistently": True}
assert [metrics[n]["rounded"] for n in ("ce", "ce_kl", "teacher")] == ["2.4601", "2.3890", "2.3017"]
assert metrics["teacher"]["nll_recomputed"] < metrics["ce_kl"]["nll_recomputed"] < metrics["ce"]["nll_recomputed"]
matched_fields = ("initialization_sha256", "batch_plan_sha256", "steps", "optimizer_updates", "training_examples", "training_sequence_chunks", "effective_supervised_tokens", "batch_size", "learning_rate")
assert all(ce["training"][k] == kd["training"][k] for k in matched_fields)
assert ce["storage"]["parameter_count"] == kd["storage"]["parameter_count"] == 33632
moe = json.loads((ROOT / "docs/course-experiments/results/moe.json").read_bytes())["results"]
assert moe["teacher_variant"] == "top2_aux0.01"
assert moe["variants"]["top2_aux0.01"]["heldout"]["test"]["nll_sum"] == m["teacher_test"]["nll_sum"]
assert moe["dataset"]["train"]["records"] == 409
assert moe["dataset"]["train"]["sha256"] == m["teacher_provenance"]["metadata"]["records_sha256"]
result["experiment"] = {
    "historical_environment": {k: d[k] for k in ("revision", "seed", "python_version", "torch_version", "device", "gpu")},
    "test": metrics,
    "matched_student_training": {k: ce["training"][k] for k in matched_fields},
    "ce_objective": ce["training"]["objective"],
    "kd_objective": kd["training"]["objective"],
    "alpha": kd["training"]["alpha"],
    "temperature": kd["training"]["temperature"],
    "data": m["data"],
    "teacher_matches_moe_experiment": True,
    "teacher_config": m["teacher_provenance"]["config"],
    "source_code_hash_comparison": {path: {"recorded": d["code_sha256"].get(path), "current": result["source_sha256"][path]} for path in paths if path in d["code_sha256"]},
    "raw_ids_scope": "generated_samples 保存逐篇續寫 generated_ids；data 保存資料集路徑與雜湊，沒有完整原故事的 token ID 陣列。",
}
stories = [{"text": "a" * 131}, {"text": "bc"}]
examples = text_examples(stories, max_length=128)
observed = torch.cat([y for _, y in examples]).tolist()
expected = sum([tok.encode(row["text"]) + [tok.eos_id] for row in stories], [])
assert observed == expected
dummy = TinyLM(ModelConfig(width=8, layers=1))
toy_eval = _evaluate(dummy, stories, "cpu")
assert toy_eval["supervised_tokens"] == len(expected) == 135
result["whole_story_mask"] = {"bytes": 133, "stories": 2, "chunks": len(examples), "supervised_tokens": 135, "target_ids_equal_all_story_bytes_plus_each_eos": True, "evaluation_denominator": toy_eval["supervised_tokens"], "includes_bos_as_target": False, "chunk_limit": 128}
archive = ROOT / "assets/training/tinystories-v1.tar.gz"
asset_bytes = archive.read_bytes()
if asset_bytes.startswith(b"version https://git-lfs.github.com/spec/v1"):
    url = "https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/main/assets/training/tinystories-v1.tar.gz"
    fetch = subprocess.run(["curl", "-fsSL", "--max-time", "45", url], capture_output=True)
    result["data_fetch"] = {"command": "curl -fsSL --max-time 45 " + url, "exit_code": fetch.returncode, "stderr": fetch.stderr.decode(errors="replace"), "local_lfs_pointer": asset_bytes.decode()}
    asset_bytes = fetch.stdout if fetch.returncode == 0 else b""
if asset_bytes:
    try:
        with tarfile.open(fileobj=io.BytesIO(asset_bytes)) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("tinystories-train-512.jsonl"))
            records = [json.loads(line) for line in tar.extractfile(member).read().splitlines() if line.strip()]
        for row in records:
            row["family"] = row.get("text_sha256", records_sha256([{"text": row["text"]}]))
        parts = split_records(records, 42)
        split_hashes = {k: records_sha256(v) for k, v in parts.items()}
        assert all(split_hashes[k] == moe["dataset"][k]["sha256"] for k in parts)
        targets = text_examples(parts["test"], max_length=128)
        count = sum(len(y) for _, y in targets)
        assert count == 41914
        assert len(targets) == 352
        result["reconstructed_dataset"] = {"archive_sha256": sha(asset_bytes), "split_counts": {k: len(v) for k, v in parts.items()}, "split_hashes": split_hashes, "test_supervised_tokens": count, "test_chunks": len(targets), "first_32_train_chunks": len(text_examples(parts["train"][:32], max_length=128))}
    except (tarfile.TarError, StopIteration) as exc:
        result["reconstructed_dataset"] = {"status": "unavailable", "reason": str(exc)}
else:
    result["reconstructed_dataset"] = {"status": "unavailable", "reason": "本機只有 LFS 指標，網路取檔沒有成功。"}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
