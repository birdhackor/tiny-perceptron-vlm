"""有界 development 工作；不開 final，結果先封包，等待控制端取回後停止。"""

import json
import os
import random
import signal
import time
import traceback
import zipfile
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402

from tiny_perceptron.omni_products import training_runtime as trainer  # noqa: E402


def recipe(updates, seconds, **extra):
    return {
        "seed": 20261029,
        "updates": updates,
        "training_seconds": seconds,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validate_every": updates,
        **extra,
    }


def gpu_resume_probe(output):
    """CUDA 上真做四步，與二步儲存／新實例／讀回再二步比較。"""

    def make():
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 1)).cuda()
        return model, torch.optim.AdamW(model.parameters(), lr=0.01), torch.Generator().manual_seed(11)

    def update(model, optimizer, generator, n):
        for _ in range(n):
            inputs = torch.randn(8, 4, device="cuda") * random.random()
            target = torch.randint(2, (8, 1), generator=generator).float().cuda()
            optimizer.zero_grad()
            (model(inputs) - target).square().mean().backward()
            optimizer.step()

    trainer.deterministic(17)
    direct, opt, gen = make()
    update(direct, opt, gen, 4)
    expected = trainer.parameter_sha(direct)
    trainer.deterministic(17)
    partial, partial_opt, partial_gen = make()
    update(partial, partial_opt, partial_gen, 2)
    path = output / "gpu-resume.pt"
    sha = trainer.save_checkpoint(path, partial, partial_opt, partial_gen, 2, {"scope": "actual CUDA exact resume"})
    trainer.deterministic(999)
    actual, actual_opt, actual_gen = make()
    step, _ = trainer.restore_checkpoint(path, sha, actual, actual_opt, actual_gen, "cuda")
    update(actual, actual_opt, actual_gen, 2)
    assert step == 2 and trainer.parameter_sha(actual) == expected
    for key, state in opt.state_dict()["state"].items():
        for field, value in state.items():
            observed = actual_opt.state_dict()["state"][key][field]
            assert torch.equal(value, observed) if torch.is_tensor(value) else value == observed
    result = {
        "passed": True,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "direct_and_resumed_parameter_sha256": expected,
        "checkpoint_sha256": sha,
        "optimizer_equal": True,
        "restored_rng": ["Python", "torch CPU", "all CUDA", "sampler"],
    }
    trainer.json_file(output / "gpu-resume.json", result)


def development(job, output):
    if job == "dev-av":
        gpu_resume_probe(output)
        return {
            name: trainer.train_av(recipe(160, 250, representation=name, variants=2), output / name, "cuda")
            for name in ("additive", "time-rope")
        }
    if job == "dev-choice":
        return {
            name: trainer.train_choice(recipe(120, 250, recompute=redo, batch_size=16), output / name, "cuda")
            for name, redo in (("shared", False), ("recompute", True))
        }
    if job == "dev-sound":
        rows = {}
        for hop, codes in ((80, 128), (80, 256), (40, 256)):
            name = f"hop{hop}-k{codes}"
            rows[name] = trainer.train_codec(
                recipe(
                    1200,
                    180,
                    hop=hop,
                    codes=codes,
                    width=32,
                    variants=1,
                    batch_size=512,
                    learning_rate=0.002,
                    training_dead_code_reset_every=200,
                ),
                output / name,
                "cuda",
            )
        return rows
    raise ValueError("未凍結的工作不得進入此 development worker")


def main():
    config = json.loads(Path("job.json").read_bytes())
    output = Path("result")
    output.mkdir(exist_ok=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("bounded worker deadline")))
    signal.alarm(config["sandbox_seconds"] - 90)
    result = {
        "job": config["job"],
        "config_sha256": trainer.file_sha("job.json"),
        "final_opened": False,
        "success": False,
        "started_at_unix": time.time(),
    }
    try:
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() not in {"NVIDIA L4", "L4"}
        ):
            raise RuntimeError("本輪微型工作只接受一張 L4")
        result["experiments"] = development(config["job"], output)
        result["success"] = all(row["complete"] for row in result["experiments"].values())
    except Exception:
        result["exception"] = traceback.format_exc()
    finally:
        signal.alarm(0)
        result["finished_at_unix"] = time.time()
        trainer.json_file(output / "worker-result.json", result)
        members = sorted(path for path in output.rglob("*") if path.is_file())
        trainer.json_file(
            output / "file-manifest.json",
            {
                str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": trainer.file_sha(path)}
                for path in members
            },
        )
        destination = Path("/tmp/omni-result.zip")
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(output))
        if destination.stat().st_size > 128 * 1024**2:
            raise RuntimeError("結果超出事前預留的匯出上限")
        trainer.json_file(
            "/tmp/omni-result.json",
            {
                "job": config["job"],
                "success": result["success"],
                "bytes": destination.stat().st_size,
                "sha256": trainer.file_sha(destination),
                "final_opened": False,
                "config_sha256": trainer.file_sha("job.json"),
            },
        )
        print("OMNI_JOB_READY", flush=True)
        while True:
            time.sleep(10)


if __name__ == "__main__":
    main()
