"""5.7 專用離線 CPU 查證；不跑正式訓練或 GPU 實驗。"""

import ast
import contextlib
import hashlib
import inspect
import io
import json
import os
import pathlib
import platform
import random
import re
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.prepare_data import generate_records
from scripts.train import parser, training_metadata
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM
from tiny_perceptron.training import load_checkpoint, save_checkpoint

HERE = pathlib.Path(__file__).resolve().parent
SNAPSHOT = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15/sources/5.7.md"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git": str(torch.version.git_version),
        "device": "cpu",
        "platform": platform.platform(),
        "threads": str(torch.get_num_threads()),
        "network": "UV_OFFLINE=1; no remote calls",
    }
    print("ENVIRONMENT", json.dumps(environment, ensure_ascii=False))
    source = SNAPSHOT.read_text()
    code = re.findall(r"```python\n(.*?)\n```", source, re.S)[0]
    namespace = {}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(code, str(SNAPSHOT), "exec"), namespace)
    print("EXACT_FROZEN_CELL_STDOUT")
    print(captured.getvalue(), end="")
    model, loaded, x = (namespace[k] for k in ("model", "loaded", "x"))
    with torch.no_grad():
        max_difference = float((model(x)["logits"] - loaded(x)["logits"]).abs().max())
    results = {
        "environment": environment,
        "snapshot_sha256": digest(SNAPSHOT),
        "exact_cell": {
            "both_assertions_passed": True,
            "stdout": captured.getvalue(),
            "logits_shape": list(model(x)["logits"].shape),
            "input_dtype": str(x.dtype),
            "max_logit_difference_after_next_update": max_difference,
            "temporary_directory_removed": not namespace["path"].parent.exists(),
        },
    }

    omission = code.replace('    restored_optimizer.load_state_dict(payload["optimizer"])',
                            '    # restored_optimizer.load_state_dict(payload["optimizer"])')
    assert omission != code
    omitted_namespace = {}
    try:
        exec(compile(omission, "5.7-omit-optimizer-state", "exec"), omitted_namespace)
    except AssertionError as error:
        failing_line = error.__traceback__.tb_next.tb_lineno
        assert omission.splitlines()[failing_line - 1].strip() == 'assert torch.equal(model(x)["logits"], loaded(x)["logits"])'
        assert failing_line == max(i for i, line in enumerate(omission.splitlines(), 1) if "assert torch.equal" in line)
    else:
        raise AssertionError("省略最佳化器狀態後，第二個 assert 應失敗")
    with torch.no_grad():
        a = omitted_namespace["model"](omitted_namespace["x"])["logits"]
        b = omitted_namespace["loaded"](omitted_namespace["x"])["logits"]
    results["omission_exercise"] = {
        "first_assertion_passed": True,
        "second_assertion_failed": True,
        "failing_cell_line": failing_line,
        "max_logit_difference": float((a - b).abs().max()),
    }

    with tempfile.TemporaryDirectory(prefix="technical-save57-") as folder:
        path = pathlib.Path(folder) / "state.pt"
        optimizer = namespace["optimizer"]
        model.embedding.weight.requires_grad_(False)
        save_checkpoint(path, model, optimizer, step=2,
                        metadata={"chars": ["。", "狗", "看", "貓", "，"]},
                        training_state={"cursor": 3})
        expected_torch = torch.rand(4)
        expected_python = [random.random() for _ in range(3)]
        rebuilt, payload = load_checkpoint(path, restore_rng=True)
        assert torch.equal(expected_torch, torch.rand(4))
        assert expected_python == [random.random() for _ in range(3)]
        assert not rebuilt.embedding.weight.requires_grad
        assert payload["training_state"] == {"cursor": 3}
        assert payload["step"] == 2
        assert payload["tokenizer"] == {"type": "unspecified", "vocab_size": 5}
        state_keys = sorted(next(iter(payload["optimizer"]["state"].values())))
        assert state_keys == ["exp_avg", "exp_avg_sq", "step"]
        results["field_and_rng_contract"] = {
            "fields": sorted(payload), "optimizer_state_keys": state_keys,
            "optimizer_steps": sorted({int(v["step"].item()) for v in payload["optimizer"]["state"].values()}),
            "python_and_torch_rng_repeated": True,
            "freeze_restored": True, "training_state_restored": True,
            "tokenizer": payload["tokenizer"],
        }

        before = digest(path)
        def interrupted_save(_payload, target):
            pathlib.Path(target).write_bytes(b"partial-write")
            raise OSError("controlled interruption before rename")
        with patch("tiny_perceptron.training.torch.save", side_effect=interrupted_save):
            try:
                save_checkpoint(path, model)
            except OSError:
                pass
            else:
                raise AssertionError("controlled interruption did not run")
        assert digest(path) == before
        assert path.with_name("state.pt.tmp").read_bytes() == b"partial-write"
        results["interrupted_write"] = {
            "old_complete_file_unchanged": True, "partial_data_only_in_tmp": True,
            "scope": "controlled write error before Path.replace; no power-loss or durability test",
        }

        invalid = pathlib.Path(folder) / "invalid.pt"
        bad = dict(payload, format_version="not-a-supported-format")
        torch.save(bad, invalid)
        try:
            load_checkpoint(invalid)
        except ValueError as error:
            assert str(error) == "不支援的 checkpoint 格式"
            version_error = str(error)
        else:
            raise AssertionError("unknown format was accepted")
        bad = dict(payload, model={k: v for k, v in payload["model"].items() if k != "output.weight"})
        torch.save(bad, invalid)
        try:
            load_checkpoint(invalid)
        except RuntimeError as error:
            assert 'Missing key(s)' in str(error) and 'output.weight' in str(error)
            strict_error = str(error)
        else:
            raise AssertionError("missing model key was accepted")
        torch.save({"opaque": object()}, invalid)
        try:
            load_checkpoint(invalid)
        except Exception as error:
            assert type(error).__name__ == "UnpicklingError"
            restricted_error = type(error).__name__
        else:
            raise AssertionError("unallowlisted object was accepted")
        results["loader_rejection"] = {
            "unknown_version": version_error, "strict_missing_key": strict_error,
            "unallowlisted_builtin_object": restricted_error,
            "scope": "harmless local fixtures; not a comprehensive security audit",
        }

        multi = MultiModalLM(TinyLM(ModelConfig(vocab_size=5, width=8, max_length=8)))
        save_checkpoint(path, multi, metadata={"task": "joint"})
        _, multi_payload = load_checkpoint(path)
        assert multi_payload["format_version"] == "multimodal-v1"
        assert multi_payload["task"] == "joint"
        results["multimodal_extra_fields"] = {key: multi_payload[key] for key in ("format_version", "task", "modal_config")}

        args = parser().parse_args(["--task", "text", "--steps", "80", "--lr", "0.003", "--freeze", "none"])
        cli_model = TinyLM(ModelConfig(width=8, max_length=8))
        cli_optimizer = torch.optim.AdamW(cli_model.parameters(), lr=args.lr)
        save_checkpoint(path, cli_model, cli_optimizer, step=1, metadata=training_metadata(args))
        cli_checks = []
        for option, value, key in [("--steps", "81", "schedule_steps"), ("--lr", "0.004", "peak_lr")]:
            cmd = [sys.executable, "scripts/train.py", "--task", "text", "--device", "cpu", "--checkpoint", str(path), "--resume", "--steps", "80", "--lr", "0.003", option, value]
            completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            assert completed.returncode != 0
            assert f"resume 的 {key} 與原訓練不同" in completed.stderr
            assert not completed.stdout
            cli_checks.append({"argv": cmd, "returncode": completed.returncode,
                               "stdout": completed.stdout, "stderr": completed.stderr})
        results["resume_schedule_rejection_no_training"] = cli_checks

    record_path = ROOT / "docs/course-experiments/results/text_foundation.json"
    record = json.loads(record_path.read_text())
    resume = record["results"]["resume"]
    assert resume["total_steps"] == 80 and resume["saved_step"] == 40
    assert resume["max_weight_difference"] == 0.0
    assert record["device"] == "cuda" and record["evidence_status"] == "complete_run"
    hashes = {}
    for name in ["scripts/course_experiments/text.py", "scripts/course_experiments/run.py", "scripts/course_experiments/common.py"]:
        completed = subprocess.run(["git", "show", f'{record["revision"]}:{name}'], cwd=ROOT, capture_output=True, check=True)
        observed = hashlib.sha256(completed.stdout).hexdigest()
        assert observed == record["code_sha256"][name]
        hashes[name] = observed
        if name.endswith("text.py"):
            historical = completed.stdout.decode()
            current = (ROOT / name).read_text()
            def probe(content):
                node = next(n for n in ast.parse(content).body if isinstance(n, ast.FunctionDef) and n.name == "_resume_probe")
                return ast.get_source_segment(content, node)
            assert probe(historical) == probe(current)
    results["gpu_record_audit_no_reexecution"] = {
        "record_sha256": digest(record_path), "revision": record["revision"],
        "recorded_environment": {key: record[key] for key in ("device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status")},
        "resume": resume, "historical_source_hashes": hashes,
        "resume_probe_unchanged": True,
        "probe_source_records": len(generate_records("toy-text")), "batch_size": 4,
        "scope": "audited recorded metric and pinned implementation; did not load GPU weight artifacts or rerun GPU experiment",
    }
    result_path = HERE / "execution-results-v1.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("RESULTS", json.dumps(results, ensure_ascii=False))
    print("RESULT_SHA256", digest(result_path))


if __name__ == "__main__":
    main()
