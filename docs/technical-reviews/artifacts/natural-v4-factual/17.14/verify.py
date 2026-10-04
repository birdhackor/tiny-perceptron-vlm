"""Bounded independent CPU checks; no training, downloads, or checkpoint writes."""
import copy
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import torch

from scripts.prepare_data import generate_records
from scripts.course_experiments.common import split_records
from scripts.course_experiments.compression import _QATLinear, _qat_layers
from tiny_perceptron.data import ByteTokenizer, render_chat, pad_batch
from tiny_perceptron.quantization import fake_quantize, quantize_symmetric, QuantizedLinear
from tiny_perceptron.training import load_checkpoint

torch.set_num_threads(2)
root = Path(__file__).resolve().parents[5]
official = json.loads((root / "docs/course-experiments/results/qat.json").read_text())
record = official["results"]
out = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                       "torch_git_version": torch.version.git_version, "device": "cpu"},
       "scope": "Arithmetic, original-run record recomputation, and inference from existing public exports. No optimizer updates or GPU replication."}
started = time.perf_counter()

# The two literal lesson losses, using fresh leaves rather than accumulating gradients.
toy = []
for squared in (False, True):
    x = torch.tensor([0.1, 0.4, 0.9], requires_grad=True)
    y = fake_quantize(x, bits=4)
    (y.square().sum() if squared else y.sum()).backward()
    hard = x.detach().clone().requires_grad_()
    plain = (hard / (0.9 / 7)).round() * (0.9 / 7)
    plain.sum().backward()
    q, scale = quantize_symmetric(x.detach(), 4)
    expected = torch.tensor([0.9 / 7, 2.7 / 7, 0.9])
    assert torch.allclose(y, expected, atol=1e-7, rtol=0)
    assert q.tolist() == [1, 3, 7]
    assert torch.equal(x.grad, 2 * y.detach() if squared else torch.ones_like(x))
    assert torch.equal(hard.grad, torch.zeros_like(hard))
    assert hard.is_leaf and hard.grad_fn is None and hard.data_ptr() != x.data_ptr()
    toy.append({"loss": "y.square().sum()" if squared else "y.sum()", "scale": scale.item(),
                "integers": q.tolist(), "forward": y.detach().tolist(), "ste_gradient": x.grad.tolist(),
                "plain_round_forward": plain.detach().tolist(), "plain_round_gradient": hard.grad.tolist(),
                "independent_leaf_and_storage": True})
out["toy"] = toy

# Independently build the original small synthetic dataset, then check repository generator.
rows = []
for color in ("red", "green", "blue"):
    for shape in ("circle", "square"):
        for pitch in ("low", "high"):
            for query, answer in (("describe", shape), ("shape?", shape), ("color?", color),
                                  ("pitch?", pitch), ("joint?", shape + "," + pitch)):
                rows.append({"messages": [{"role": "user", "content": f"color={color};shape={shape};pitch={pitch};{query}"},
                                          {"role": "assistant", "content": answer}],
                             "family": f"{color}:{shape}:{pitch}", "source": "course-generated", "license": "MIT"})
families = sorted({r["family"] for r in rows})
random.Random(42).shuffle(families)
parts = {s: [r for f in fs for r in rows if r["family"] == f]
         for s, fs in (("train", families[:9]), ("validation", families[9:10]), ("test", families[10:]))}
assert parts == split_records(generate_records("attributes-sft"), seed=42)
dataset_bytes = (json.dumps(parts, ensure_ascii=False, indent=2) + "\n").encode()
dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
assert dataset_sha == record["data"]["sha256"]
tok = ByteTokenizer()
target_counts = {s: [len(r["messages"][-1]["content"].encode()) + 1 for r in rs] for s, rs in parts.items()}
rng = random.Random(42)
plan = [[rng.randrange(45) for _ in range(16)] for _ in range(350)]
plan_sha = hashlib.sha256(json.dumps(plan).encode()).hexdigest()
effective = sum(target_counts["train"][i] for indices in plan for i in indices)
for branch in record["training"].values():
    assert branch["initialization_sha256"] == record["initialization_sha256"]
    assert branch["batch_plan_sha256"] == plan_sha
    assert branch["effective_supervised_tokens"] == effective == 39348
    assert branch["steps"] == branch["optimizer_updates"] == 350
    assert branch["batch_size"] == 16 and branch["learning_rate"] == 0.003
out["dataset_and_sampling"] = {"dataset_sha256": dataset_sha, "batch_plan_sha256": plan_sha, "seed": 42,
    "counts": {s: len(rs) for s, rs in parts.items()},
    "families": {s: sorted({r["family"] for r in rs}) for s, rs in parts.items()},
    "target_counts": {s: sum(cs) for s, cs in target_counts.items()},
    "optimizer_updates_each_as_recorded": 350, "sampled_answers_each": 350 * 16,
    "sampled_answer_and_eos_targets_each": effective,
    "recorded_device": official["device"], "recorded_gpu": official["gpu"],
    "recorded_training_seconds": {s: v["seconds"] for s, v in record["training"].items()},
    "recorded_total_seconds": official["elapsed_seconds"], "recorded_timing_scope": official["timing_scope"]}

def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

@torch.no_grad()
def independently_evaluate(model, records):
    examples = [render_chat(r["messages"]) for r in records]
    x, labels, valid = pad_batch(examples)
    logits = model(x, valid=valid)["logits"]
    mask = labels != -100
    log_probs = logits.log_softmax(-1)
    nll_sum = -log_probs[mask].gather(-1, labels[mask, None]).sum().item()
    count = int(mask.sum())
    samples = []
    for r in records:
        question, expected = r["messages"][0]["content"], r["messages"][-1]["content"]
        prompt = [1, 3] + tok.encode(question) + [2, 4]
        context = torch.tensor([prompt])
        generated = []
        for _ in range(24):
            if context.shape[1] >= model.config.max_length:
                break
            token = model(context)["logits"][:, -1].argmax(-1, keepdim=True)
            generated.append(int(token.item()))
            if generated[-1] == 2:
                break
            context = torch.cat((context, token), -1)
        raw = generated[:-1] if generated and generated[-1] == 2 else generated
        samples.append({"question": question, "expected": expected, "generated_ids": generated,
                        "generated": tok.decode(raw), "exact": raw == tok.encode(expected),
                        "eos": bool(generated and generated[-1] == 2),
                        "invalid_control_ids": [i for i in raw if i < 8]})
    return {"nll_sum": nll_sum, "targets": count, "nll": nll_sum / count, "examples": len(records),
            "correct": sum(s["exact"] for s in samples), "eos": sum(s["eos"] for s in samples),
            "invalid_control_count": sum(len(s["invalid_control_ids"]) for s in samples), "samples": samples}

out["runs"] = {}
public = root / "checkpoints/course/qat"
manifest = json.loads((public / "download-manifest.json").read_text())
manifest_files = {f["output"]: f for f in manifest["files"]}
names = {"initial_fp32": "initial_fp32.pt", "initial_ptq4": "initial_ptq4.pt", "fp_finetuned": "fp_finetuned.pt",
         "matched_ptq4": "matched_ptq4.pt", "qat_float": "qat_float.pt", "qat_packed4": "model.pt"}
for variant, filename in names.items():
    path = public / filename
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert file_sha == manifest_files[filename]["sha256"]
    model, payload = load_checkpoint(path, "cpu")
    model.eval()
    state_sha = state_hash(model)
    expected_state = {"initial_fp32": record["initialization_sha256"],
                      "fp_finetuned": record["training"]["fp_finetuning"]["final_sha256"],
                      "qat_float": record["training"]["qat"]["final_sha256"]}.get(variant)
    if expected_state is not None:
        assert state_sha == expected_state
    results = {}
    for split in ("validation", "test"):
        stored = record["runs"][variant][split]
        stored_samples = stored["generated_samples"]
        assert len(stored_samples) == stored["examples"] == len(parts[split])
        assert sum(len(s["expected"].encode()) + 1 for s in stored_samples) == stored["supervised_tokens"]
        assert sum(s["generated_ids"][:-1] == tok.encode(s["expected"]) for s in stored_samples) == stored["correct"]
        assert all(s["generated_ids"][-1] == 2 and not any(i < 8 for i in s["generated_ids"][:-1]) for s in stored_samples)
        recomputed = stored["nll_sum"] / stored["supervised_tokens"]
        assert abs(recomputed - stored["answer_nll"]) < 1e-12
        observed = independently_evaluate(model, parts[split])
        assert observed["correct"] == stored["correct"]
        assert observed["targets"] == stored["supervised_tokens"]
        assert observed["eos"] == stored["eos_count"] and observed["invalid_control_count"] == 0
        assert abs(observed["nll"] - stored["answer_nll"]) < 3e-5
        assert [s["generated_ids"] for s in observed["samples"]] == [s["generated_ids"] for s in stored_samples]
        results[split] = {"stored_sum": stored["nll_sum"], "stored_recomputed_nll": recomputed,
                         "stored_nll_4dp": f"{recomputed:.4f}", **observed}
    tensor_bytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
    assert tensor_bytes == record["runs"][variant]["storage"]["tensor_bytes"]
    out["runs"][variant] = {"existing_export": str(path.relative_to(root)), "export_sha256": file_sha,
        "state_sha256": state_sha, "format": str(payload["format_version"]), "tensor_bytes": tensor_bytes,
        "export_file_bytes": path.stat().st_size, "original_run_file_bytes": record["runs"][variant]["storage"]["file_bytes"],
        **results}

float_model, _ = load_checkpoint(public / "qat_float.pt")
fake_model = _qat_layers(copy.deepcopy(float_model)).eval()
packed_model, _ = load_checkpoint(public / "model.pt")
packed_model.eval()
x, _ = render_chat(parts["test"][0]["messages"])
with torch.no_grad():
    difference = (fake_model(x[None])["logits"] - packed_model(x[None])["logits"]).abs().max().item()
out["fake_vs_packed"] = {"recorded_max_logit_difference": record["fake_deployed_max_logit_difference"],
    "cpu_existing_export_max_logit_difference": difference, "input": parts["test"][0],
    "input_positions": len(x), "includes_gold_answer_prefix": True}
assert difference <= 1e-4
out["check_seconds"] = time.perf_counter() - started
print(json.dumps(out, indent=2, ensure_ascii=False))
