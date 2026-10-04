"""CPU evidence for section 17.1; no retraining or private checkpoint claims."""

import copy
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

from scripts.check_technical_reviews import sections
from scripts.course_experiments.compression import _evaluate, _parameter_hash, _storage
from scripts.course_release import inference_payload, validate_export
from tiny_perceptron.quantization import QuantizedLinear, replace_linear_layers
from tiny_perceptron.training import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    section = dict(sections(ROOT / "course/chapters/17.md"))["17.1"]
    code = re.search(r"```python\n(.*?)```", section, re.S)[1]
    programs = {"original": code, "exercise": code.replace("zeros(64, 32)", "zeros(128, 32)")}
    program_runs = {}
    for name, program in programs.items():
        run = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True)
        program_runs[name] = {"code": program, "stdout": run.stdout, "stderr": run.stderr, "exit_code": run.returncode}
    report_path = ROOT / "docs/course-experiments/results/quantization.json"
    report = json.loads(report_path.read_text())
    (OUT / "fact_v2_17_01_quantization_report.json").write_bytes(report_path.read_bytes())
    formal = report["results"]
    export_dir = ROOT / "checkpoints/course/quantization"
    export_manifest = json.loads((export_dir / "export-manifest.json").read_text())
    (OUT / "fact_v2_17_01_export_manifest.json").write_bytes((export_dir / "export-manifest.json").read_bytes())
    models, payloads, storage = {}, {}, {}
    filenames = {"fp32": "fp32.pt", "packed4": "model.pt", "packed8": "packed8.pt"}
    for name, filename in filenames.items():
        path = export_dir / filename
        entry = next(item for item in export_manifest["files"] if item["output"] == filename)
        assert digest(path) == entry["sha256"]
        assert path.stat().st_size == entry["bytes"]
        original = next(item for item in report["artifacts"] if item["path"] == filename)
        assert entry["source_sha256"] == original["sha256"]
        model, payload = load_checkpoint(path)
        models[name], payloads[name] = model.eval(), payload
        detail = _storage(model, path)
        assert detail["tensor_bytes"] == formal["runs"][name]["storage"]["tensor_bytes"]
        assert detail["parameter_count"] == 141568
        packed = [layer for layer in model.modules() if isinstance(layer, QuantizedLinear)]
        detail.update({
            "file_sha256": digest(path),
            "private_source_sha256": entry["source_sha256"],
            "private_recorded_file_bytes": original["bytes"],
            "logical_quantized_weight_count": sum(math.prod(layer.shape) for layer in packed),
            "fp32_scale_count": sum(layer.scale.numel() for layer in packed),
            "fp32_bias_count": sum(0 if layer.bias is None else layer.bias.numel() for layer in packed),
            "torch_parameter_count": sum(p.numel() for p in model.parameters()),
            "payload_keys": list(payload),
            "payload_model_bytes": sum(v.numel() * v.element_size() for v in payload["model"].values()),
            "optimizer_present": "optimizer" in payload,
            "rng_present": any(k.endswith("_rng") for k in payload),
        })
        storage[name] = detail
    fp32_hash = _parameter_hash(models["fp32"])
    assert fp32_hash == formal["training"]["final_sha256"]
    packing_checks = {}
    for name, bits in (("packed4", 4), ("packed8", 8)):
        converted = replace_linear_layers(copy.deepcopy(models["fp32"]), bits)
        comparisons = {}
        for key, value in converted.state_dict().items():
            saved_value = payloads[name]["model"][key]
            exact = torch.equal(value, saved_value)
            maximum_error = float((value.float() - saved_value.float()).abs().max())
            matches = torch.allclose(value, saved_value, atol=1e-8, rtol=1e-6) if key.endswith(".scale") else exact
            assert matches
            comparisons[key] = {"exact": exact, "max_abs_error": maximum_error, "verified": matches}
        packing_checks[name] = comparisons
        assert storage[name]["logical_quantized_weight_count"] == 115200
        assert storage[name]["fp32_scale_count"] == 1416
        assert storage[name]["fp32_bias_count"] == 640
        assert storage[name]["torch_parameter_count"] == 25728
    evaluated = {}
    for name, model in models.items():
        evaluated[name] = {}
        for split in ("validation", "test"):
            previous = formal["runs"][name][split]
            # Inputs are reconstructed from the original report's per-question records.
            records = [{"family": row["family"], "question": row["question"], "answer": row["expected"]} for row in previous["generated_samples"]]
            measured = _evaluate(model, records, "cpu")
            assert measured["supervised_tokens"] == previous["supervised_tokens"]
            assert measured["answer_bytes"] == previous["answer_bytes"]
            assert measured["examples"] == previous["examples"]
            assert abs(measured["nll_sum"] - previous["nll_sum"]) < 1e-4
            comparisons = []
            for old, new in zip(previous["generated_samples"], measured["generated_samples"], strict=True):
                same = old["generated_ids"] == new["generated_ids"]
                assert same
                comparisons.append({"question": old["question"], "expected": old["expected"], "generated_ids_match": same})
            measured["original_nll_sum"] = previous["nll_sum"]
            measured["per_question_comparisons"] = comparisons
            measured["input_scope"] = "Reconstructed heldout questions/answers from the formal report; original private dataset SHA is not independently remeasured."
            evaluated[name][split] = measured
    with tempfile.TemporaryDirectory(prefix="fact_v2_17_01_") as scratch:
        fixture = Path(scratch) / "native.pt"
        save_checkpoint(fixture, models["fp32"], metadata={"review_fixture": "17.1"})
        saved = torch.load(fixture, map_location="cpu", weights_only=True)
        cleaned = inference_payload(saved, {"scope": "17.1 review fixture"})
        public = Path(scratch) / "public.pt"
        torch.save(cleaned, public)
        version = validate_export(public)
        release_check = {
            "native_keys": list(saved),
            "native_optimizer_is_none": saved["optimizer"] is None,
            "native_torch_rng_bytes": saved["torch_rng"].numel() * saved["torch_rng"].element_size(),
            "clean_keys": list(cleaned),
            "clean_has_optimizer_or_rng": any(k == "optimizer" or k.endswith("_rng") for k in cleaned),
            "native_fixture_bytes": fixture.stat().st_size,
            "clean_fixture_bytes": public.stat().st_size,
            "validated_format": str(version),
            "scope": "Fresh CPU fixture demonstrates the current export contract; these are not original GPU-run file sizes.",
        }
        assert not release_check["clean_has_optimizer_or_rng"]
    probe = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(probe.parameters())
    activation = probe(torch.ones(1, 2))
    activation.square().sum().backward()
    optimizer.step()
    memory_probe = {"parameter_bytes": sum(p.numel() * p.element_size() for p in probe.parameters()), "activation_bytes": activation.numel() * activation.element_size(), "gradient_bytes": sum(p.grad.numel() * p.grad.element_size() for p in probe.parameters()), "optimizer_state_keys": sorted({k for state in optimizer.state.values() for k in state})}
    with torch.no_grad():
        result = models["fp32"](torch.tensor([[1, 2, 3]]))
    memory_probe["kv_cache_bytes_three_positions"] = sum(t.numel() * t.element_size() for pair in result["cache"] for t in pair)
    code_paths = ("tiny_perceptron/model.py", "tiny_perceptron/quantization.py", "tiny_perceptron/training.py", "scripts/course_experiments/compression.py", "scripts/course_release.py")
    output = {
        "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_17_01_audit.py",
        "result": "All explicit CPU assertions passed; 45 heldout generation comparisons matched the formal report.",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "threads": "1", "torch_git_version": str(torch.version.git_version)},
        "source_sha256": hashlib.sha256(section.encode()).hexdigest(),
        "code_sha256": {path: digest(ROOT / path) for path in code_paths},
        "original_experiment_compression_sha256": digest(OUT / "fact_v2_17_01_compression_run.txt"),
        "original_compression_hash_matches_report": digest(OUT / "fact_v2_17_01_compression_run.txt") == report["code_sha256"]["scripts/course_experiments/compression.py"],
        "program_runs": program_runs,
        "cast_point_seven": torch.tensor([0.7]).to(torch.int8).tolist(),
        "formal_environment": {k: report[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "elapsed_seconds", "timing_scope", "step_scale", "evidence_status", "peak_memory_scope")},
        "formal_provenance": formal["teacher_provenance"],
        "formal_data": formal["data"],
        "formal_training": formal["training"],
        "formal_timing": {name: run["timing"] for name, run in formal["runs"].items()},
        "storage": storage,
        "fp32_parameter_hash_matches_formal_final": fp32_hash,
        "packing_tensor_identity": packing_checks,
        "heldout_cpu_evaluation": evaluated,
        "release_fixture": release_check,
        "memory_probe": memory_probe,
        "limits": ["No private original checkpoint was loaded or remeasured.", "No training was rerun; original training configuration and 13610 effective tokens are inspected recorded results.", "CPU timings are not GPU speed or peak-memory reproduction.", "Logical parameter count is distinct from nn.Parameter count after packed weights become buffers.", "Recomputed CPU scales differ from saved GPU scales by at most 7.45e-9; scale comparison uses atol=1e-8, rtol=1e-6 while packed values and retained tensors require exact identity."],
    }
    assert output["original_compression_hash_matches_report"]
    destination = OUT / "fact_v2_17_01_audit.json"
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output["result"])
    print(json.dumps({"storage": {k: {field: v[field] for field in ("tensor_bytes", "file_bytes", "logical_quantized_weight_count", "fp32_scale_count", "fp32_bias_count", "torch_parameter_count")} for k, v in storage.items()}, "release_fixture": release_check, "memory_probe": memory_probe}, indent=2))


if __name__ == "__main__":
    main()
