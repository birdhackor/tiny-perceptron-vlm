"""Fresh 19.11 audit: tiny temporary state tests and existing measurement audit.

No network, no model download, no complete training, no new model scores, and no
permanent weights. Existing public/cache bytes are read only. Saved JSON contains
provenance, hashes, shapes and a few necessary state values, not reusable weights.
"""
import copy
import contextlib
import hashlib
import io
import json
import os
import random
import re
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
A = Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import torch
from huggingface_hub import try_to_load_from_cache
from tiny_perceptron.capstone import (
    CapstoneModel, TOK, build_dataset, calculator_runtime, default_config,
    frozen_reference, load_capstone, parse_action, save_capstone,
)
from tiny_perceptron.capstone_quantization import quantize_capstone, load_quantized_capstone
from scripts.capstone_release import clean_capstone_payload, public_stage_id, validate_capstone_payload, validate_manifest
from scripts.fetch_capstone import fetch_capstone
from scripts.course_experiments.capstone import _balanced_sample, train_stage
from scripts.capstone import main as cli_main

assert not torch.cuda.is_available() and torch.version.cuda is None
torch.set_num_threads(1)

def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def js(p):
    return json.loads(Path(p).read_bytes())

def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

environment = {"python": sys.version, "python_executable": sys.executable,
               "torch": str(torch.__version__), "torch_git": str(torch.version.git_version),
               "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
               "device": "cpu", "network": "not used; offline environment", "cwd": str(ROOT)}
(A / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
print("ENV", json.dumps(environment))

# Execute the original, unedited fence, including its TemporaryDirectory.
exec(compile((A / "fence-1.py").read_bytes(), "course/chapters/19.md#19.11:fence-1", "exec"), {})
results = {"original_fence": {"format_version": "capstone-v1", "step": 0,
                            "inference_only": True, "description_equal": True}}

with TemporaryDirectory(prefix="review19_11-", dir="/tmp") as td:
    td = Path(td)
    torch.manual_seed(123)
    random.seed(123)
    tiny = CapstoneModel(replace(default_config(dense=True), width=8, layers=1))
    original_reference = frozen_reference(tiny)
    optimizer = torch.optim.AdamW(tiny.parameters(), lr=0.003)
    # One bounded optimizer transaction with synthetic gradients, not task training.
    for p in tiny.parameters():
        p.grad = torch.full_like(p, 0.01)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sampler = random.Random(42)
    rows = [{"task": t, "id": i} for i, t in enumerate(["a", "a", "b", "b"])]
    _balanced_sample(rows, sampler, 3)
    state = {"sampler_rng": sampler.getstate(), "batch_size": 2, "history": [],
             "effective_tokens": 0, "elapsed_training_seconds": 0.0}
    p = td / "training.pt"
    save_capstone(p, tiny, stage="dpo", step=1, optimizer=optimizer,
                  state=state, reference=original_reference)
    expected_python = random.random()
    expected_torch = torch.rand(3)
    expected_sample = [r["id"] for r in _balanced_sample(rows, sampler, 5)]
    restored, saved = load_capstone(p)
    assert same(tiny.state_dict(), restored.state_dict())
    assert tiny.description() == restored.description()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=99)
    restored_optimizer.load_state_dict(saved["optimizer"])
    assert same(optimizer.state_dict(), restored_optimizer.state_dict())
    torch.set_rng_state(saved["torch_rng"])
    random.setstate(saved["python_rng"])
    observed_python, observed_torch = random.random(), torch.rand(3)
    sampler2 = random.Random(42)
    sampler2.setstate(saved["training_state"]["sampler_rng"])
    observed_sample = [r["id"] for r in _balanced_sample(rows, sampler2, 5)]
    assert observed_python == expected_python and torch.equal(observed_torch, expected_torch)
    assert observed_sample == expected_sample
    assert same(saved["reference"], original_reference.state_dict())
    assert not same(saved["reference"], tiny.state_dict())
    first = next(iter(saved["model"]))
    model_values = saved["model"][first].flatten()[:3].tolist()
    reference_values = saved["reference"][first].flatten()[:3].tolist()
    with torch.no_grad():
        next(restored.parameters()).add_(1)
    assert tiny.description() == restored.description() and not same(tiny.state_dict(), restored.state_dict())
    results["tiny_save_restore"] = {
        "parameters": tiny.description()["parameters"], "optimizer_transaction_count": 1,
        "task_training_updates": 0, "tensor_equal_before_mutation": True,
        "description_equal_after_tensor_mutation": True, "optimizer_state_exact": True,
        "torch_next_3": observed_torch.tolist(), "python_next": observed_python,
        "sampler_next_5_ids": observed_sample, "sampler_continues_after_saved_progress": True,
        "reference_restored_exactly": True, "reference_differs_from_current_policy": True,
        "example_state_key": first, "policy_first3": model_values, "reference_first3": reference_values,
        "serialized_sha256": h(p), "temporary_weight_deleted_after_run": True,
    }
    inf = td / "inference.pt"
    save_capstone(inf, tiny, stage="joint", step=0, inference_only=True)
    _, inference = load_capstone(inf)
    omitted = ["optimizer", "training_state", "torch_rng", "python_rng", "cuda_rng", "reference"]
    assert all(k not in inference for k in omitted)
    results["inference_state"] = {"keys": sorted(inference), "omitted": omitted,
                                  "step": inference["step"], "inference_only": inference["inference_only"]}

    # Actual original rejection branches execute before the training loop.
    _, manifest = build_dataset(42)
    code_hashes = {f: h(ROOT / f) for f in ["tiny_perceptron/capstone.py", "scripts/course_experiments/capstone.py"]}
    metadata = {"data_manifest": manifest, "requested_steps": 3, "code_sha256": code_hashes,
                "schedule_completed": False, "seed": 42}
    rejection_base = td / "base.pt"
    save_capstone(rejection_base, tiny, stage="pretrain", step=1, optimizer=optimizer,
                  state=state, metadata=metadata)
    base = torch.load(rejection_base, map_location="cpu", weights_only=True)
    rejections = []
    def reject(name, payload, *, stage="pretrain", steps=3, batch_size=2, parent=False, expected):
        file = td / (name + ".pt")
        torch.save(payload, file)
        arguments = {"input_checkpoint" if parent else "resume": file}
        try:
            train_stage(stage, td / name, steps=steps, seconds=0.001, seed=42,
                        batch_size=batch_size, validation=False, **arguments)
        except ValueError as e:
            assert expected in str(e), (name, str(e), expected)
            rejections.append({"case": name, "error": str(e), "training_loop_reached": False})
        else:
            raise AssertionError("Expected rejection: " + name)
    bad = copy.deepcopy(base); bad["metadata"]["data_manifest"] = {}
    reject("changed-data", bad, expected="frozen training data differ")
    bad = copy.deepcopy(base); bad["inference_only"] = True
    reject("inference-resume", bad, expected="training checkpoint for the same stage")
    reject("wrong-stage", base, stage="sft", expected="training checkpoint for the same stage")
    reject("changed-schedule", base, steps=4, expected="fixed update schedule")
    reject("changed-batch", base, batch_size=3, expected="batch size must match")
    bad = copy.deepcopy(base); bad["optimizer"] = None
    reject("missing-optimizer", bad, expected="no optimizer state")
    bad = copy.deepcopy(base); bad["metadata"]["code_sha256"] = {}
    reject("changed-code", bad, expected="code hashes changed")
    bad = copy.deepcopy(base); bad["stage"] = "dpo"; bad["reference"] = None
    reject("missing-dpo-reference", bad, stage="dpo", expected="original frozen reference")
    reject("incomplete-parent", base, stage="sft", parent=True, expected="completed immediate predecessor")
    public_payload = copy.deepcopy(inference)
    reject("public-parent-no-manifest", public_payload, stage="dpo", parent=True,
           expected="frozen training data differ")
    results["resume_rejections"] = rejections

    # Tiny quantization loads into float32; no model answer or benchmark is run.
    qs = []
    for bits in [4, 8]:
        qfile = td / f"quant{bits}.pt"
        receipt = quantize_capstone(inf, qfile, bits)
        loaded_q, qp = load_quantized_capstone(qfile)
        assert all(x.dtype == torch.float32 for x in loaded_q.state_dict().values())
        try:
            load_capstone(qfile)
        except ValueError as e:
            ordinary_error = str(e)
        else:
            raise AssertionError("FP32 loader should reject PTQ format")
        qs.append({"bits": bits, "storage_value_dtypes": sorted({str(x["values"].dtype) for x in qp["quantized"].values()}),
                   "loaded_parameter_dtypes": ["torch.float32"], "compute": qp["quantization"]["compute"],
                   "fp32_loader_error": ordinary_error, "temporary_sha256": receipt["sha256"]})
    results["tiny_quantized_load"] = qs
    # Execute the actual infer dispatch with a tiny PTQ file, while replacing
    # generation with a recording stub. This is a loader/CLI check, not a score.
    infer_calls = []
    def record_assistant(model, row, max_new_tokens=64):
        infer_calls.append({"row": row, "max_new_tokens": max_new_tokens,
                            "model_dtypes": sorted({str(p.dtype) for p in model.parameters()})})
        return {"generation_stub": True}
    with patch("tiny_perceptron.capstone.run_assistant", record_assistant), contextlib.redirect_stdout(io.StringIO()):
        cli_main(["infer", "--checkpoint", str(td / "quant4.pt"),
                  "--prompt", "1+2等於多少？", "--device", "cpu"])
    assert len(infer_calls) == 1 and infer_calls[0]["model_dtypes"] == ["torch.float32"]
    from tiny_perceptron.capstone_ui import serve
    try:
        serve(td / "quant4.pt")
    except ValueError as e:
        serve_error = str(e)
    else:
        raise AssertionError("serve should reject PTQ before creating a server")
    results["cli_quantized_infer_and_serve"] = {"infer_dispatch": infer_calls,
                                               "new_model_generations": 0,
                                               "serve_ptq_error": serve_error,
                                               "server_started": False}

    # Parse the five original training commands through the real CLI and record
    # its call arguments. The training function is replaced and never executes.
    training_calls = []
    def record_train(stage, output, **kwargs):
        training_calls.append({"stage": stage, "output": output, **kwargs})
        return {"training_stub": True}
    for block in re.findall(r"```bash\n(.*?)```", (A / "section.md").read_text(), re.S):
        for line in block.splitlines():
            words = shlex.split(line)
            if len(words) > 2 and words[1] == "scripts/capstone.py" and words[2] == "train":
                with patch("scripts.course_experiments.capstone.train_stage", record_train), contextlib.redirect_stdout(io.StringIO()):
                    cli_main(words[2:])
    assert [x["steps"] for x in training_calls] == [300, 1400, 600, 100, 600]
    assert all(x["seed"] == 42 and x["batch_size"] == 24 and x["seconds"] == 540 and x["device"] == "cpu" for x in training_calls)
    assert training_calls[-1]["resume"].endswith("joint/model-training.pt")
    results["training_cli_argument_dispatch"] = {"calls": training_calls, "training_function_executed": False}

    # No-network fixture exercises actual atomic fetch validation and token=False.
    fixture = td / "fixture"; fixture.mkdir()
    clean = clean_capstone_payload(inference, {}, {"architecture": {"type": "CapstoneModel"}})
    torch.save(clean, fixture / "model.pt")
    for name in ["README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "data.json"]:
        (fixture / name).write_text("{}\n" if name.endswith("json") else "Temporary no-network test fixture\n")
    files = [{"path": "fixture/" + p.name, "output": p.name, "sha256": h(p),
              "bytes": p.stat().st_size, "license": "MIT"} for p in sorted(fixture.iterdir())]
    fm = {"schema_version": 1, "repo": "birdhackor/tiny-perceptron-course-models",
          "revision": "33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed",
          "models": [{"id": "joint", "format_version": "capstone-v1", "checkpoint": "model.pt", "files": files}]}
    calls = []
    def fake_download(repo, path, *, revision, token):
        assert repo == fm["repo"] and revision == fm["revision"] and token is False
        calls.append({"path": path, "revision": revision, "token": token})
        return str(fixture / Path(path).name)
    with patch("huggingface_hub.hf_hub_download", fake_download):
        destination = fetch_capstone(fm, "joint", td / "success")
        assert sorted(p.name for p in destination.iterdir()) == sorted(p.name for p in fixture.iterdir())
        try:
            fetch_capstone(fm, "joint", td / "success")
        except ValueError as e:
            existing_target_error = str(e)
        else:
            raise AssertionError("Existing destination must not be overwritten")
        bad = copy.deepcopy(fm); bad["models"][0]["files"][0]["sha256"] = "0" * 64
        try:
            fetch_capstone(bad, "joint", td / "corrupt")
        except ValueError as e:
            corrupt_error = str(e)
        else:
            raise AssertionError("Corrupt file must be rejected")
        assert not (td / "corrupt" / "joint").exists()
    results["fetch_fixture"] = {"network_used": False, "fixture_calls": calls,
                                "five_files_placed_after_validation": True,
                                "existing_target_error": existing_target_error,
                                "corrupt_error": corrupt_error, "corrupt_target_absent": True}

# Original --list executes offline; no weight download or training.
listed = subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/fetch_capstone.py", "--list"],
                        check=True, capture_output=True, text=True, timeout=30)
(A / "list.stdout.txt").write_text(listed.stdout)
list_rows = [json.loads(x) for x in listed.stdout.splitlines()]
public = validate_manifest(js(ROOT / "docs/course-experiments/capstone-public.json"))
assert len(list_rows) == 11 and [x["stage"] for x in list_rows] == [x["id"] for x in public["models"]]
results["public_list"] = {"rows": len(list_rows), "stage_ids": [x["stage"] for x in list_rows],
                          "revision": public["revision"], "network_used": False}

# Audit original raw measurement files, not author result comments.
measure = js(A / "inputs/docs/course-experiments/student-checks/capstone-public-tensors.json")
assert len(measure["records"]) == 11 and len({x["stage"] for x in measure["records"]}) == 11
assert measure["public_revision"] == public["revision"]
assert h(ROOT / measure["verification_program"]["path"]) == measure["verification_program"]["sha256"]
source_paths = {}
for exp, basepath in [
    ("capstone_deployment", "outputs/integration-runs/v2-deployment/capstone-review/capstone_deployment/gha-37169529991-1"),
    ("capstone_student", "outputs/integration-runs/v2-student/capstone-review/capstone_student/gha-37170200956-1"),
]:
    approval = js(A / f"inputs/docs/course-experiments/releases/{exp}.json")
    for f in approval["files"]:
        if f["kind"] == "checkpoint":
            source_paths[Path(f["output"]).parts[0]] = (ROOT / basepath / f["path"], f["sha256"])
audits, companions = [], {}
for model in public["models"]:
    for f in model["files"]:
        cp = try_to_load_from_cache(public["repo"], f["path"], revision=public["revision"])
        assert isinstance(cp, str) and Path(cp).is_file(), (model["id"], f["path"])
        assert h(cp) == f["sha256"] and Path(cp).stat().st_size == f["bytes"]
        companions[f["path"]] = {"bytes": f["bytes"], "sha256": f["sha256"], "output": f["output"],
                                  "original_url": f"https://huggingface.co/{public['repo']}/resolve/{public['revision']}/{f['path']}"}
        if f["output"] == "model.pt":
            pub = torch.load(cp, map_location="cpu", weights_only=True)
            sp, sh = source_paths[model["id"]]
            assert h(sp) == sh
            src = torch.load(sp, map_location="cpu", weights_only=True)
            keys = ["config", "model", "stage", "step", "tokenizer", "data_version"]
            if "quantized" in pub:
                keys += ["quantized", "quantization"]
            assert all(same(pub[k], src[k]) for k in keys)
            assert public_stage_id(pub) == model["id"] and pub["inference_only"] is True
            # Existing original schema/loader contract: one finite-shape forward,
            # no decoding, no score and no capability measurement.
            assert validate_capstone_payload(pub) == model["format_version"]
            assert not {"optimizer", "training_state", "torch_rng", "python_rng", "cuda_rng", "reference"} & pub.keys()
            assert not {"data_manifest", "schedule_completed", "requested_steps"} & pub["metadata"].keys()
            historical = next(x for x in measure["records"] if x["stage"] == model["id"])
            assert historical["public_sha256"] == f["sha256"] and historical["approved_source_sha256"] == sh
            assert historical["passed"] is True and historical["exact_tensor_and_config_fields"] == keys
            tensor_summary = {}
            for k, value in pub["model"].items():
                raw = value.contiguous().view(torch.uint8).numpy().tobytes()
                tensor_summary[k] = {"dtype": str(value.dtype), "shape": list(value.shape),
                                     "tensor_value_sha256": hashlib.sha256(raw).hexdigest()}
            first = next(iter(pub["model"]))
            audits.append({"id": model["id"], "bytes": f["bytes"], "sha256": f["sha256"],
                           "source_checkpoint_sha256": sh, "source_fields_equal": keys,
                           "inference_only": True, "format_version": pub["format_version"],
                           "stage": pub["stage"], "step": pub["step"], "config": pub["config"],
                           "tokenizer": pub["tokenizer"], "metadata_keys": sorted(pub["metadata"]),
                           "provenance": {k: pub["metadata"][k] for k in ["source_checkpoint_sha256", "dataset_manifest_sha256", "student_branch", "teacher_checkpoint_sha256", "source_revision"] if k in pub["metadata"]},
                           "tensor_summary": tensor_summary, "example_state_key": first,
                           "example_first3": pub["model"][first].flatten()[:3].tolist(),
                           "source_original_location": str(sp.relative_to(ROOT)),
                           "source_published_url": companions[f["path"]]["original_url"],
                           "original_validator_cpu_shape_check": f"finite (1,5,{TOK.vocab_size}) logits; no decoding or score",
                           "network_used": False, "new_score_evaluation": False})
assert len(audits) == 11
results["public_existing_bytes"] = {"models": len(audits), "unique_files": len(companions),
                                    "all_11_exact_tensors_config_match_source": True,
                                    "new_model_downloads": 0, "new_model_score_evaluations": 0}
(A / "public-existing-byte-audit.json").write_text(json.dumps({"models": audits, "files": companions}, indent=2, ensure_ascii=False) + "\n")

selection = js(A / "inputs/docs/course-experiments/capstone-selection.json")
counts = {}
frozen_cases = {}
for stage in ["joint", "dpo"]:
    v = js(A / f"inputs/docs/course-experiments/capstone-evidence/{stage}/validation.json")
    assert v["count"] == len(v["records"]) == 84
    assert len({r["id"] for r in v["records"]}) == 84
    frozen_cases[stage] = [{k: r[k] for k in ["id", "family", "task", "expected_action", "expected_final"]} for r in v["records"]]
    by_task = {}
    for r in v["records"]:
        ac = r["action_trace"]["eos"] and r["action_trace"]["raw"] == r["expected_action"]
        ec = ac and r["answer"] == r["expected_final"]
        assert r["action_correct"] == ac and r["end_to_end_correct"] == ec
        totals = by_task.setdefault(r["task"], {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
        totals["count"] += 1; totals["action_correct"] += int(ac); totals["end_to_end_correct"] += int(ec)
    assert by_task == v["by_task"]
    correct = sum(r["end_to_end_correct"] for r in v["records"])
    assert v["end_to_end_correct"] == correct == selection["candidates"][stage]["validation_end_to_end_correct"]
    counts[stage] = {"count": 84, "exact_end_to_end_correct": correct, "score": correct / 84}
assert selection["selected_stage"] == "joint" and counts["joint"]["exact_end_to_end_correct"] > counts["dpo"]["exact_end_to_end_correct"]
assert frozen_cases["joint"] == frozen_cases["dpo"]
results["existing_selection_audit"] = {"counts": counts, "selected_stage": "joint",
                                        "same_84_unique_cases_and_answer_rules": True,
                                        "selected_before_test_generation_record": selection["selected_before_test_generation"],
                                        "new_generations": 0}
student = js(A / "inputs/docs/course-experiments/student-checks/capstone-public-student.json")
r = student["record"]
assert student["public_revision"] == public["revision"] and student["stage"] == "student-kd-int4"
assert student["checkpoint_sha256"] == next(x["sha256"] for x in audits if x["id"] == "student-kd-int4")
assert r["action_trace"]["raw"] == "TOOL:calculator:2+9" and r["action_trace"]["eos"]
assert parse_action(r["action_trace"]) == r["parsed_action"]
assert calculator_runtime(r["parsed_action"]) == r["runtime"] == {"status": "ok", "result": "11"}
assert r["final_trace"]["raw"] == "DIRECT:12" and r["final_trace"]["eos"] and r["answer"] == "12"
results["existing_student_failure"] = {"sample_count": 1, "user_prompt_from_recorded_command": "1+2等於多少？",
                                        "action": r["action_trace"]["raw"], "tool_result": "11", "final": "12",
                                        "original_sha256": student["checkpoint_sha256"], "new_generations": 0}
(A / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print("RESULTS", json.dumps(results, ensure_ascii=False))
assert not list(A.rglob("*.pt"))
print("PASS: original fence, bounded CPU state tests, 10 rejection branches, offline list/fetch fixture, 11 existing byte comparisons, original validation/failure records; no permanent weights")
