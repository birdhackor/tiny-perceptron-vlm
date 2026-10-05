"""Bounded CPU checks for 18.12; no training, checkpoint loading, or new model scoring."""
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from tiny_perceptron.alignment import distillation_kl
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import DenseFFN, MoEFFN

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.manual_seed(0)
teacher = TinyLM(ModelConfig(width=16, experts=3, top_k=2)).eval().requires_grad_(False)
student = TinyLM(ModelConfig(width=8))
ids = torch.tensor([[1, 2]])
with torch.no_grad():
    t = teacher(ids)["logits"]
s = student(ids)["logits"]
labels = torch.tensor([[1, 2]])
before = {name: p.detach().clone() for name, p in student.named_parameters()}
loss = distillation_kl(s, t, labels, temperature=1.0)
p, logq = t.float().softmax(-1), s.float().log_softmax(-1)
manual_per_position = (p * (p.log() - logq)).sum(-1)
assert torch.allclose(loss, manual_per_position.mean(), atol=1e-7, rtol=1e-6)
loss.backward()
assert all(torch.equal(before[name], p.detach()) for name, p in student.named_parameters())
assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
assert student.output.weight.grad.abs().sum().item() > 0
assert isinstance(teacher.blocks[0].ffn, MoEFFN)
assert isinstance(student.blocks[0].ffn, DenseFFN)
assert tuple(t.shape) == tuple(s.shape) == (1, 2, 264)

def parameter_formula(width, experts):
    # Untied embedding/output; 128 positions; 1 attention block, four unbiased
    # width x width projections; three affine LayerNorms; default FFN hidden=4w.
    one_ffn = 8 * width**2 + 5 * width
    components = {
        "embedding_and_output": 2 * 264 * width,
        "position_table": 128 * width,
        "attention": 4 * width**2,
        "three_layer_norms": 6 * width,
        "ffn": one_ffn * (experts or 1),
        "router": width * experts,
    }
    return components, sum(components.values())

toy = {
    "shape_axes": "batch=1, input positions=2, candidate IDs=264",
    "teacher_shape": list(t.shape), "student_shape": list(s.shape),
    "student_output_gradient_absolute_sum": student.output.weight.grad.abs().sum().item(),
    "teacher_gradients_absent": True, "student_parameters_unchanged_after_backward": True,
    "kl": loss.item(), "manual_kl": manual_per_position.mean().item(),
    "valid_positions": 2, "unit": "natural-log divergence per valid position",
}
for name, model, width, experts, expected in [
    ("teacher", teacher, 16, 3, 18048), ("student", student, 8, 0, 6104)
]:
    components, count = parameter_formula(width, experts)
    assert count == model.description()["parameters"] == expected
    toy[name + "_parameter_formula"] = components
    toy[name + "_parameters"] = count

masked = torch.tensor([[-100, 2]])
assert torch.allclose(distillation_kl(s.detach(), t, masked, 1.0), manual_per_position[0, 1], atol=1e-7)
arbitrary_valid = torch.tensor([[999, 999]])
assert torch.equal(distillation_kl(s.detach(), t, arbitrary_valid, 1.0), loss.detach())
toy["labels_are_validity_only"] = True
toy["one_valid_position_kl"] = distillation_kl(s.detach(), t, masked, 1.0).item()

wrong = TinyLM(ModelConfig(width=8, vocab_size=265))(ids)["logits"]
assert tuple(wrong.shape) == (1, 2, 265)
try:
    distillation_kl(wrong, t, labels, 1.0)
except ValueError as error:
    toy["vocab_265_error"] = str(error)
else:
    raise AssertionError("shape mismatch was accepted")
restored = TinyLM(ModelConfig(width=8, vocab_size=264))(ids)["logits"]
assert torch.isfinite(distillation_kl(restored, t, labels, 1.0))
other_depth = TinyLM(ModelConfig(width=12, layers=2))(ids)["logits"]
assert tuple(other_depth.shape) == tuple(t.shape)
assert torch.isfinite(distillation_kl(other_depth, t, labels, 1.0))
toy["restored_shape"] = list(restored.shape)
toy["different_depth_shape"] = list(other_depth.shape)
_, _, chosen = teacher.blocks[0].ffn(torch.zeros(1, 2, 16))
assert tuple(chosen.shape) == (2, 2)
toy["chosen_expert_shape"] = list(chosen.shape)
toy["stored_experts"] = len(teacher.blocks[0].ffn.experts)

raw = (HERE / "inputs/distillation.json").read_bytes()
report = json.loads(raw)
case = report["results"]["tasks"]["moe_to_dense"]
moe = json.loads((HERE / "inputs/moe.json").read_bytes())
provenance = case["teacher_provenance"]
assert provenance["config"] == moe["results"]["variants"]["top2_aux0.01"]["model"]["config"]
assert provenance["metadata"]["records_sha256"] == moe["results"]["dataset"]["train"]["sha256"]
assert moe["results"]["dataset"]["train"]["records"] == case["data"]["counts"]["train"] == 409
assert case["data"]["student_training_records"] == 32
assert next(x["sha256"] for x in moe["artifacts"] if x["path"] == "model.pt") == provenance["sha256"]
assert case["teacher_frozen_and_unchanged"] is True
assert hashlib.sha256((HERE / "inputs/compression-provenance.py").read_bytes()).hexdigest() == report["code_sha256"]["scripts/course_experiments/compression.py"]
assert hashlib.sha256((HERE / "inputs/architecture-teacher-provenance.py").read_bytes()).hexdigest() == moe["code_sha256"]["scripts/course_experiments/architecture.py"]
ce, kd = (case["runs"][x]["training"] for x in ("w32_ce", "w32_ce_kl"))
for key in ["initialization_sha256", "batch_plan_sha256", "steps", "optimizer_updates", "training_examples", "training_sequence_chunks", "effective_supervised_tokens"]:
    assert ce[key] == kd[key]
assert ce["steps"] == 300 and ce["training_examples"] == 32
assert ce["alpha"] == 0 and kd["alpha"] == 0.5 and kd["temperature"] == 2
for train in [ce, kd]:
    for entry in train["loss_trace"]:
        expected = (1-train["alpha"])*entry["ce"] + train["alpha"]*entry["kl"]
        assert math.isclose(entry["loss"], expected, rel_tol=1e-6, abs_tol=1e-6)
tok = ByteTokenizer()
summaries = {}
for name in ["w32_ce", "w32_ce_kl"]:
    run = case["runs"][name]
    summaries[name] = {}
    for split in ["validation", "test"]:
        measured = run[split]
        samples = measured["generated_samples"]
        assert len(samples) == measured["examples"] == case["data"]["counts"][split]
        exact, ended = 0, 0
        for sample in samples:
            ids = sample["generated_ids"]
            has_eos = tok.eos_id in ids
            output = ids[:ids.index(tok.eos_id)] if has_eos else ids
            hit = output == tok.encode(sample["expected"])
            assert hit == sample["exact"] and has_eos == sample["ended_with_eos"]
            assert tok.decode(output) == sample["generated"]
            assert len(ids) <= measured["max_new_tokens"] == 24
            exact += int(hit); ended += int(has_eos)
        assert exact == measured["correct"] == 0
        assert ended == measured["eos_count"] == 0
        assert measured["exact_match"] == exact / len(samples)
        nll = measured["nll_sum"] / measured["supervised_tokens"]
        bpb = measured["nll_sum"] / measured["answer_bytes"] / math.log(2)
        assert math.isclose(nll, measured["answer_nll"], rel_tol=0, abs_tol=1e-12)
        assert math.isclose(bpb, measured["answer_bpb"], rel_tol=0, abs_tol=1e-12)
        summaries[name][split] = {
            "samples": len(samples), "supervised_tokens": measured["supervised_tokens"],
            "bytes": measured["answer_bytes"], "nll_sequence_chunks": measured["nll_sequence_chunks"],
            "nll_recomputed": nll, "bpb_recomputed": bpb, "exact_recomputed": exact,
            "eos_recomputed": ended, "first_three_raw_generations": [x["generated"] for x in samples[:3]],
        }
    assert [x["family"] for x in run["test"]["generated_samples"]] == [x["family"] for x in case["teacher_test"]["generated_samples"]]
assert summaries["w32_ce_kl"]["test"]["nll_recomputed"] < summaries["w32_ce"]["test"]["nll_recomputed"]

result = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version), "device": "cpu", "threads": 1},
    "toy": toy,
    "existing_evidence_only": {"revision": report["revision"], "seed": report["seed"],
        "recorded_torch": report["torch_version"], "teacher_original_training_records": 409,
        "student_training_records": 32, "student_steps": 300, "student_sequence_chunks": 213,
        "student_effective_supervised_tokens": ce["effective_supervised_tokens"],
        "same_initialization": True, "same_batch_plan": True,
        "teacher_checkpoint_sha256": provenance["sha256"], "summaries": summaries,
        "scope": "Recomputed recorded sums, token IDs and raw flags; no model/checkpoint loaded or newly scored."},
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
