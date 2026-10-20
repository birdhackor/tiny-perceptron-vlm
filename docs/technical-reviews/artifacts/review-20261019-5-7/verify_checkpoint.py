import contextlib
import copy
import hashlib
import io
import json
import platform
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts import train
from scripts.course_experiments.common import Context
from scripts.course_experiments.text import _resume_probe
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM
from tiny_perceptron.training import load_checkpoint, save_checkpoint


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def difference(a, b):
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def rejected(action):
    try:
        action()
    except Exception as error:
        return {"type": type(error).__name__, "message": str(error)}
    raise AssertionError("預期拒絕的操作卻完成了")


def cli(arguments):
    stream = io.StringIO()
    result = {"call": "scripts.train.main()", "argv": ["scripts/train.py", *arguments]}
    with patch.object(sys, "argv", result["argv"]), contextlib.redirect_stdout(stream):
        try:
            train.main()
            result["completed"] = True
        except Exception as error:
            result.update(completed=False, error_type=type(error).__name__, error=str(error))
    result["stdout"] = stream.getvalue()
    return result


def main():
    torch.set_num_threads(1)
    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-7/verify_checkpoint.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": str(torch.get_num_threads()), "cuda_available": str(torch.cuda.is_available()), "mps_available": str(torch.backends.mps.is_available())},
    }
    section = (ROOT / "docs/course-review-20261019/freeze-01/sources/5.7.md").read_bytes()
    freeze = (ROOT / "docs/course-review-20261019/freeze-04.json").read_bytes()
    output["freeze"] = {"source_sha256": digest(section), "manifest_sha256": digest(freeze)}
    run = ROOT / "outputs/reviewer-tools/runs/20261020T055728777469Z"
    output["section_execution"] = {
        "invoked_command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/05.md#5.7\" --execute",
        "files": {name: {"sha256": digest((run / name).read_bytes()), "content": (run / name).read_text()} for name in ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py", "bootstrap.py"]},
    }
    fence = re.search(r"```python\n(.*?)```", section.decode(), re.S)[1]
    variant = fence.replace('    restored_optimizer.load_state_dict(payload["optimizer"])\n', "")
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        work = Path(directory)
        variant_stdout = io.StringIO()
        with contextlib.redirect_stdout(variant_stdout):
            exercise_error = rejected(lambda: exec(compile(variant, "checkpoint_exercise", "exec"), {}))
        assert exercise_error["type"] == "AssertionError"
        output["exercise"] = {"change": "只移除 restored_optimizer.load_state_dict 呼叫", "result": exercise_error, "stdout": variant_stdout.getvalue()}

        torch.manual_seed(42)
        random.seed(42)
        model = TinyLM(ModelConfig(vocab_size=5, width=8, max_length=8))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
        masked_loss(model(x)["logits"], y).backward()
        optimizer.step()
        path = work / "state.pt"
        chars = ["。", "狗", "看", "貓", "，"]
        save_checkpoint(path, model, optimizer, step=1, metadata={"chars": chars}, training_state={"sampler": {"cursor": 3}})
        loaded, payload = load_checkpoint(path, restore_rng=True)
        fresh, _ = load_checkpoint(path)
        restored = torch.optim.AdamW(loaded.parameters(), lr=0.0001)
        restored.load_state_dict(payload["optimizer"])
        new_optimizer = torch.optim.AdamW(fresh.parameters(), lr=0.01)
        initial_equal = torch.equal(model(x)["logits"], loaded(x)["logits"]) and torch.equal(model(x)["logits"], fresh(x)["logits"])
        state_keys = sorted(next(iter(payload["optimizer"]["state"].values())))
        steps = sorted({int(s["step"]) for s in payload["optimizer"]["state"].values()})
        for candidate, opt in [(model, optimizer), (loaded, restored), (fresh, new_optimizer)]:
            opt.zero_grad()
            masked_loss(candidate(x)["logits"], y).backward()
            opt.step()
        restored_difference = float((model(x)["logits"] - loaded(x)["logits"]).abs().max())
        fresh_difference = float((model(x)["logits"] - fresh(x)["logits"]).abs().max())
        assert initial_equal and restored_difference == 0 and fresh_difference > 0
        output["optimizer_roundtrip"] = {"x": x.tolist(), "y": y.tolist(), "dtype": str(next(model.parameters()).dtype), "initial_logits_equal": initial_equal, "restored_next_logits_max_difference": restored_difference, "fresh_next_logits_max_difference": fresh_difference, "payload_keys": sorted(payload), "config_matches": payload["config"] == asdict(model.config), "optimizer_state_keys": state_keys, "optimizer_steps": steps, "restored_lr": restored.param_groups[0]["lr"], "step": payload["step"], "metadata": payload["metadata"], "training_state": payload["training_state"], "tokenizer": payload["tokenizer"]}

        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
        expected_torch, expected_python = torch.rand(5), [random.random() for _ in range(5)]
        torch.rand(17)
        random.random()
        load_checkpoint(path, restore_rng=True)
        actual_torch, actual_python = torch.rand(5), [random.random() for _ in range(5)]
        assert torch.equal(expected_torch, actual_torch) and expected_python == actual_python
        output["rng_roundtrip"] = {"torch_expected": expected_torch.tolist(), "torch_observed": actual_torch.tolist(), "python_expected": expected_python, "python_observed": actual_python, "note": "模型重建後的下一次 CPU/Python 抽樣精確相同；未執行 CUDA/MPS"}

        model.embedding.weight.requires_grad_(False)
        save_checkpoint(path, model, optimizer)
        frozen, frozen_payload = load_checkpoint(path)
        assert not frozen.embedding.weight.requires_grad
        output["trainable_roundtrip"] = {"embedding_requires_grad": frozen.embedding.weight.requires_grad, "names_equal": [n for n, p in frozen.named_parameters() if p.requires_grad] == frozen_payload["trainable_parameters"]}
        save_checkpoint(path, model)
        inference, inference_payload = load_checkpoint(path)
        assert inference_payload["optimizer"] is None and torch.equal(model(x)["logits"], inference(x)["logits"])
        output["without_optimizer"] = {"optimizer": inference_payload["optimizer"], "logits_equal": True}

        invalid = copy.deepcopy(payload)
        invalid["format_version"] = "unsupported"
        torch.save(invalid, path)
        invalid_format = rejected(lambda: load_checkpoint(path))
        invalid = copy.deepcopy(payload)
        del invalid["model"]["embedding.weight"]
        torch.save(invalid, path)
        invalid_keys = rejected(lambda: load_checkpoint(path))
        output["rejections"] = {"unknown_format": invalid_format, "missing_model_key": invalid_keys}

        multi = MultiModalLM(TinyLM(ModelConfig(width=8)), vision_width=4, audio_width=4)
        save_checkpoint(path, multi, metadata={"task": "vision"})
        multimodal, multi_payload = load_checkpoint(path)
        assert isinstance(multimodal, MultiModalLM)
        output["multimodal"] = {"format_version": multi_payload["format_version"], "modal_config": multi_payload["modal_config"], "task": multi_payload["task"], "weights_max_difference": difference(multi.state_dict(), multimodal.state_dict())}

        checkpoint = work / "atomic.pt"
        checkpoint.write_bytes(b"previous complete checkpoint")
        previous = checkpoint.read_bytes()

        def interrupted_save(data, target):
            Path(target).write_bytes(b"incomplete checkpoint")
            raise RuntimeError("模擬寫入途中中斷")

        with patch("tiny_perceptron.training.torch.save", interrupted_save):
            rejected(lambda: save_checkpoint(checkpoint, model))
            preserved = checkpoint.read_bytes() == previous
            checkpoint.unlink()
            rejected(lambda: save_checkpoint(checkpoint, model))
            absent = not checkpoint.exists()
        assert preserved and absent
        output["interrupted_write"] = {"previous_preserved": preserved, "new_final_absent": absent, "temporary_content": checkpoint.with_name(checkpoint.name + ".tmp").read_text(), "scope": "同目錄、單一寫入者，模擬 torch.save 未完成；未測斷電或儲存裝置損壞"}

        base = ["--device", "cpu", "--train", "--steps", "4", "--width", "8", "--max-length", "8", "--batch-size", "1", "--seed", "42", "--lr", "0.001"]
        full, middle, resumed = work / "full.pt", work / "middle.pt", work / "resumed.pt"
        calls = [cli([*base, "--output", str(full)]), cli([*base, "--stop-after", "2", "--output", str(middle)]), cli([*base, "--checkpoint", str(middle), "--resume", "--output", str(resumed)])]
        for call in calls:
            assert call["completed"]
        full_model, full_payload = load_checkpoint(full)
        resumed_model, resumed_payload = load_checkpoint(resumed)
        cli_difference = difference(full_model.state_dict(), resumed_model.state_dict())
        assert cli_difference == 0
        wrong_steps = cli([*base, "--steps", "5", "--checkpoint", str(middle), "--resume", "--output", str(work / "wrong-steps.pt")])
        wrong_lr = cli([*base, "--lr", "0.002", "--checkpoint", str(middle), "--resume", "--output", str(work / "wrong-lr.pt")])
        assert not wrong_steps["completed"] and "schedule_steps" in wrong_steps["error"]
        assert not wrong_lr["completed"] and "peak_lr" in wrong_lr["error"]
        output["training_entrypoint"] = {"calls": calls, "steps": resumed_payload["step"], "metadata_schedule_steps": resumed_payload["metadata"]["schedule_steps"], "metadata_peak_lr": resumed_payload["metadata"]["peak_lr"], "max_weight_difference": cli_difference, "wrong_steps": wrong_steps, "wrong_lr": wrong_lr}

        ctx = Context("cpu", work, work, ROOT / "assets/training", seed=42)
        output["cpu_resume_probe"] = _resume_probe(ctx)

    record_path = ROOT / "docs/course-experiments/results/text_foundation.json"
    record = json.loads(record_path.read_bytes())
    old_text = subprocess.check_output(["git", "show", "6c84f3fb1af8d7f316bdfe3e40f95064357456f5:scripts/course_experiments/text.py"], cwd=ROOT)
    old_common = subprocess.check_output(["git", "show", "6c84f3fb1af8d7f316bdfe3e40f95064357456f5:scripts/course_experiments/common.py"], cwd=ROOT)
    current_text = (ROOT / "scripts/course_experiments/text.py").read_bytes()
    extract_probe = lambda raw: raw.decode().split("def _resume_probe(ctx):", 1)[1].split("\ndef run_text_foundation(ctx):", 1)[0]
    assert digest(old_text) == record["code_sha256"]["scripts/course_experiments/text.py"]
    assert digest(old_common) == record["code_sha256"]["scripts/course_experiments/common.py"]
    assert extract_probe(old_text) == extract_probe(current_text)
    output["gpu_record_audit"] = {"path": str(record_path.relative_to(ROOT)), "sha256": digest(record_path.read_bytes()), "device": record["device"], "gpu": record["gpu"], "python": record["python_version"], "torch": record["torch_version"], "seed": record["seed"], "step_scale": record["step_scale"], "resume": record["results"]["resume"], "separate_text_training_steps": record["results"]["training"]["steps"], "historical_source_commit": "6c84f3fb1af8d7f316bdfe3e40f95064357456f5", "historical_code_hashes_match_record": True, "resume_probe_unchanged": True, "probe_source": "def _resume_probe(ctx):" + extract_probe(old_text), "limitation": "本回合核對原有 GPU 執行紀錄、對應原碼與 CPU 重跑；未重跑 GPU 或下載 checkpoint。紀錄 revision 的原 commit 已無法以本地 Git 讀取，改以全檔 SHA-256 找到相同原碼。"}
    output["repository_sources"] = {p: digest((ROOT / p).read_bytes()) for p in ["tiny_perceptron/training.py", "tiny_perceptron/model.py", "scripts/train.py", "scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "scripts/course_experiments/run.py", "course/training.md"]}
    output["result"] = "全部斷言通過"
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
