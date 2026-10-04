"""Independent bounded CPU audit of section 17.14; no experiment training rerun."""

import copy
import ast
import hashlib
import json
import platform
import random
import sys
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import Context, split_records
from scripts.course_experiments.compression import (
    _example, _examples, _packed, _parameter_hash, _qat_layers, _storage,
)
from scripts.course_experiments.run import experiment_spec
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.quantization import fake_quantize, quantize_symmetric
from tiny_perceptron.training import load_checkpoint, save_checkpoint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    original = json.loads((ROOT / "docs/course-experiments/results/qat.json").read_text())
    result = original["results"]
    parts = split_records(generate_records("attributes-sft"), seed=42)
    raw = json.dumps(parts, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    dataset_hash = hashlib.sha256(raw.encode()).hexdigest()
    assert dataset_hash == result["data"]["sha256"]
    denominator = {}
    for split, records in parts.items():
        examples = _examples(records, 128)
        denominator[split] = {
            "examples": len(records),
            "families": sorted({r["family"] for r in records}),
            "answer_and_eos_targets": sum(int((y != IGNORE).sum()) for _, y in examples),
            "answer_bytes": sum(len(r["messages"][-1]["content"].encode()) for r in records),
            "largest_input_length": max(len(x) for x, _ in examples),
        }
    assert denominator["test"]["answer_and_eos_targets"] == 69
    assert denominator["validation"]["answer_and_eos_targets"] == 36
    assert all(not (set(denominator[a]["families"]) & set(denominator[b]["families"]))
               for a, b in [("train", "test"), ("train", "validation"), ("test", "validation")])
    examples = _examples(parts["train"], 128)
    sampler = random.Random(42)
    plan = [[sampler.randrange(len(examples)) for _ in range(16)] for _ in range(350)]
    batch_hash = hashlib.sha256(json.dumps(plan).encode()).hexdigest()
    target_count = sum(int((examples[i][1] != IGNORE).sum()) for batch in plan for i in batch)
    assert target_count == 39348
    for training in result["training"].values():
        assert training["steps"] == training["optimizer_updates"] == 350
        assert training["batch_size"] == 16 and training["learning_rate"] == 0.003
        assert training["initialization_sha256"] == result["initialization_sha256"]
        assert training["batch_plan_sha256"] == batch_hash
        assert training["effective_supervised_tokens"] == target_count
    tok = ByteTokenizer()
    runs = {}
    expected_correct = [5, 4, 6, 6, 4, 4]
    expected_nll = [0.5058, 0.5262, 0.4896, 0.4066, 0.5398, 0.5979]
    for (name, run), correct, rounded_nll in zip(result["runs"].items(), expected_correct, expected_nll):
        runs[name] = {}
        for split in ("validation", "test"):
            measured = run[split]
            samples = measured["generated_samples"]
            assert len(samples) == denominator[split]["examples"] == measured["examples"]
            assert measured["supervised_tokens"] == denominator[split]["answer_and_eos_targets"]
            nll = measured["nll_sum"] / measured["supervised_tokens"]
            assert nll == measured["answer_nll"]
            hits, eos, illegal = 0, 0, []
            for sample, record in zip(samples, parts[split]):
                assert sample["family"] == record["family"]
                assert sample["question"] == record["messages"][-2]["content"]
                assert sample["expected"] == record["messages"][-1]["content"]
                ids = sample["generated_ids"]
                content = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
                hit = content == tok.encode(sample["expected"])
                assert sample["exact"] == hit
                assert sample["generated"] == tok.decode(content)
                hits += hit
                eos += tok.eos_id in ids
                illegal.extend(i for i in ids if i < 8 and i != tok.eos_id)
                assert len(ids) <= 24 and ids[-1] == tok.eos_id
            assert hits == measured["correct"]
            assert eos == measured["eos_count"] == len(samples)
            assert illegal == []
            runs[name][split] = {"correct": hits, "examples": len(samples),
                                  "nll_recalculated": nll, "targets": measured["supervised_tokens"],
                                  "eos_count": eos, "illegal_control_ids": illegal}
        assert runs[name]["test"]["correct"] == correct
        assert abs(runs[name]["test"]["nll_recalculated"] - rounded_nll) < 0.00005
        storage = run["storage"]
        assert storage["tensor_bytes"] == storage["parameter_tensor_bytes"] + storage["buffer_tensor_bytes"]
        assert sum(storage["buffers"].values()) == storage["buffer_tensor_bytes"]
        assert storage["parameter_count"] == 141568
        runs[name]["storage"] = {k: storage[k] for k in ("tensor_bytes", "file_bytes", "parameter_count", "forward")}
    assert result["fake_deployed_max_logit_difference"] == 0
    delta = result["runs"]["qat_packed4"]["test"]["answer_nll"] - result["runs"]["matched_ptq4"]["test"]["answer_nll"]
    assert delta == result["qat_vs_matched_ptq_test_nll_delta"]
    prompt = "color=red;shape=square;pitch=high;describe"
    recorded_prompt = {
        name: next(s for s in result["runs"][name]["test"]["generated_samples"] if s["question"] == prompt)
        for name in ("qat_packed4", "matched_ptq4")
    }
    assert recorded_prompt["qat_packed4"]["generated"] == "circle"
    assert recorded_prompt["matched_ptq4"]["generated"] == "square"
    assert result["runs"]["qat_packed4"]["storage"]["tensor_bytes"] == 168736
    assert result["runs"]["qat_packed4"]["storage"]["file_bytes"] == 183045

    vector = torch.tensor([0.1, 0.4, 0.9], requires_grad=True)
    quantized, scale = quantize_symmetric(vector.detach(), 4)
    simulated = fake_quantize(vector, bits=4)
    simulated.square().sum().backward()
    assert quantized.tolist() == [1, 3, 7]
    assert torch.allclose(vector.grad, 2 * simulated.detach(), atol=1e-7, rtol=0)
    hard = vector.detach().clone().requires_grad_()
    assert hard.is_leaf and hard.data_ptr() != vector.data_ptr()
    ((hard / (0.9 / 7)).round() * (0.9 / 7)).sum().backward()
    assert hard.grad.tolist() == [0, 0, 0]

    # One optimizer update checks wrapper and save/load mechanics, not trained quality.
    config = ModelConfig(**result["teacher_provenance"]["config"])
    base = TinyLM(config)
    master = _qat_layers(copy.deepcopy(base)).eval()
    assert _parameter_hash(base) == _parameter_hash(master)
    optimizer = torch.optim.AdamW(master.parameters(), lr=0.003)
    x, y = _example(parts["test"][0], config.max_length)
    optimizer.zero_grad()
    loss = masked_loss(master(x[None])["logits"], y[None])
    loss.backward()
    gradient_norm = float(master.output.weight.grad.norm())
    assert gradient_norm > 0
    optimizer.step()
    with tempfile.TemporaryDirectory(prefix="tool-choice-dep-1714-") as directory:
        directory = Path(directory)
        checkpoint = directory / "native-training.pt"
        save_checkpoint(checkpoint, master, optimizer, step=1)
        loaded, payload = load_checkpoint(checkpoint)
        assert payload["optimizer"] is not None and payload["step"] == 1
        assert _parameter_hash(master) == _parameter_hash(loaded)
        assert type(loaded.output).__name__ == "Linear"
        loaded = _qat_layers(loaded).eval()
        restored_optimizer = torch.optim.AdamW(loaded.parameters(), lr=0.003)
        restored_optimizer.load_state_dict(payload["optimizer"])
        plain_master = _qat_layers(copy.deepcopy(master), remove=True)
        ctx = Context("cpu", directory, directory, ROOT / "assets/training", 42)
        packed, packed_path = _packed(ctx, plain_master, "deployment", 4)
        with torch.no_grad():
            a = master(x[None])["logits"]
            b = packed(x[None])["logits"]
            c = loaded(x[None])["logits"]
        maximum = float((a-b).abs().max())
        wrapper_reload_maximum = float((a-c).abs().max())
        assert maximum <= 1e-4 and wrapper_reload_maximum == 0
        local_storage = _storage(packed, packed_path)
        assert local_storage["tensor_bytes"] == 168736

    spec = experiment_spec("qat")
    assert spec["module"] == "compression" and spec["function"] == "run_qat"
    assert spec["dependencies"] == ["sft"] and "17.14" in spec["lessons"]
    assert experiment_spec("tool_choice")["lessons"] == ["B.5", "B.6", "B.7", "B.8"]
    source_hashes = {path: {"reported": value, "current": sha(ROOT / path)}
                     for path, value in original["code_sha256"].items()
                     if path in {"scripts/course_experiments/compression.py", "scripts/course_experiments/common.py",
                                 "scripts/course_experiments/run.py", "tiny_perceptron/quantization.py",
                                 "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/training.py"}}
    unchanged_functions = {}
    comparison_functions = {
        "scripts/course_experiments/compression.py": {
            "_sha", "_sync", "_teacher", "_parameter_hash", "_chat", "_example", "_examples", "_steps",
            "_dataset", "_prompt", "_answer", "_storage", "_save_float", "_packed", "_evaluate", "_fit_text",
            "_QATLinear", "_qat_layers", "run_qat",
        },
    }
    for path, hashes in source_hashes.items():
        if hashes["reported"] == hashes["current"]:
            continue
        old = subprocess.check_output(["git", "show", original["revision"] + ":" + path], cwd=ROOT)
        assert hashlib.sha256(old).hexdigest() == hashes["reported"]
        if path in comparison_functions:
            def nodes(source):
                return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            old_nodes, new_nodes = nodes(old), nodes((ROOT / path).read_bytes())
            checked = sorted(comparison_functions[path])
            assert all(old_nodes[n] == new_nodes[n] for n in checked)
            unchanged_functions[path] = checked
        elif path == "scripts/course_experiments/run.py":
            old_plan = json.loads(subprocess.check_output(
                ["git", "show", original["revision"] + ":docs/course-experiments/plan.json"], cwd=ROOT))
            old_spec = next(v for v in old_plan["sequence"] if v["id"] == "qat")
            assert {k: v for k, v in old_spec.items() if k not in {"status", "evidence"}} == {
                k: v for k, v in spec.items() if k not in {"status", "evidence"}
            }
            # The inspected runner changes broaden spec lookup and mechanism-probe metadata.
            assert "kind" not in spec and spec["module"] == "compression"
            unchanged_functions[path] = ["qat execution spec unchanged; status planned->complete_run and evidence path recorded after experiment; no mechanism_probe kind; full step_scale=1 keeps complete_run"]
        else:
            raise AssertionError("Uninspected formal-source change: " + path)
    output = {
        "reviewer_task": "/root/technical_dep_1714",
        "command": ".venv/bin/python docs/technical-reviews/artifacts/tool-choice-dep-1714-audit.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
        "original_experiment_environment": {k: original[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status", "status")},
        "input_sha256": {"qat_result": sha(ROOT / "docs/course-experiments/results/qat.json"), "plan": sha(ROOT / "docs/course-experiments/plan.json")},
        "dataset_regenerated_sha256": dataset_hash,
        "denominators": denominator,
        "training_plan": {"seed": 42, "steps_per_branch": 350, "batch_size": 16, "sample_draws_per_branch": 5600, "batch_plan_sha256": batch_hash, "answer_and_eos_targets_per_branch": target_count},
        "formal_runs_recalculated_from_original_result": runs,
        "qat_minus_matched_ptq_test_nll": delta,
        "recorded_test_prompt": recorded_prompt,
        "recorded_original_fake_deployed_max_difference": result["fake_deployed_max_logit_difference"],
        "exercise": {"scale": float(scale), "integer_codes": quantized.tolist(), "forward": simulated.detach().tolist(), "squared_loss_ste_gradient": vector.grad.tolist(), "direct_round_gradient": hard.grad.tolist(), "hard_is_independent_leaf": True},
        "bounded_random_model_one_update": {"first_output_weight_gradient_norm": gradient_norm, "fake_vs_reloaded_packed_max_logit_difference": maximum, "restored_wrapper_max_logit_difference": wrapper_reload_maximum, "packed_tensor_bytes": local_storage["tensor_bytes"], "optimizer_state_present_and_reloaded": True},
        "current_qat_spec": spec,
        "formal_code_hashes_compared_with_current": source_hashes,
        "changed_files_semantically_checked": unchanged_functions,
        "limitations": ["No formal QAT/FP32 training was rerun; formal L4 metrics were read and arithmetically checked from qat.json.", "Original private checkpoints are unavailable locally and were not downloaded; original 183045 file bytes and 0 logit difference are recorded-run evidence, not independently rerun checkpoint measurements.", "The one-update model is freshly random and supports wrapper, packing, storage and reload mechanics only.", "CPU mechanism checks do not establish GPU speed, memory improvements or general QAT quality.", "CLI flags and conditional SFT dependency paths inspected; full experiment and original infer command not rerun."],
        "status": "all bounded checks passed",
    }
    destination = ROOT / "docs/technical-reviews/artifacts/tool-choice-dep-1714-audit.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "artifact": str(destination.relative_to(ROOT)), "targets_per_branch": target_count, "test_targets": denominator["test"]["answer_and_eos_targets"], "nll_delta": delta, "mechanism_max_difference": maximum}))


if __name__ == "__main__":
    main()
