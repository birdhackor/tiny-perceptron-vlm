"""既有映像內的有界成熟模型研究；套件隔離，不更換base Torch或建立新映像。"""

import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from scripts.omni_revision.worker_outputs import export_and_wait, json_file, sha

MAX_LOG_BYTES = 512 * 1024


def bounded_command(arguments, log, seconds, environment=None):
    """任一安裝／推論都在同一工作期限內；先kill並wait才可匯出。"""
    with Path(log).open("wb") as stream:
        child = subprocess.Popen(arguments, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        try:
            deadline = time.monotonic() + seconds
            while child.poll() is None:
                if Path(log).stat().st_size > MAX_LOG_BYTES:
                    raise RuntimeError("成熟研究子程式log超界")
                if time.monotonic() > deadline:
                    raise TimeoutError("成熟研究子程式deadline")
                time.sleep(0.2)
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=20)
        if Path(log).stat().st_size > MAX_LOG_BYTES or child.returncode:
            raise RuntimeError("成熟研究子程式失敗；保留原始log，不自動重跑")


def requirements(name):
    if name not in ("kev", "qwen25"):
        raise ValueError("只允許兩個已規劃的成熟研究工作")
    path = Path(f"scripts/omni_revision/research_requirements/{name}.txt")
    lines = path.read_text().splitlines()
    names = []
    for line in lines:
        if re.fullmatch(r"[a-zA-Z0-9_-]+==[a-zA-Z0-9.+_-]+", line) is None:
            raise ValueError("研究套件只允許確切版本的binary wheel")
        package = line.split("==")[0].lower().replace("_", "-")
        if package in {"torch", "triton", "pip"} or package.startswith("nvidia-"):
            raise ValueError("不得用研究套件更換既有Torch／CUDA runtime")
        names.append(package)
    if len(names) != len(set(names)):
        raise ValueError("重複研究套件")
    return path


def runtime(job):
    import torch

    if (
        platform.python_version() != "3.12.10"
        or str(torch.__version__) != "2.8.0+cu128"
        or torch.version.cuda != "12.8"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("既有Image的Python／Torch／CUDA／單GPU與事前規格不同")
    device = torch.cuda.get_device_properties(0)
    minimum = 44 if job["gpu"] == "L40S" else 22
    if device.name != "NVIDIA " + job["gpu"] or device.total_memory < minimum * 1024**3:
        raise ValueError("實際GPU型號／VRAM與固定形狀不同；不得自動升級")
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": device.name,
        "gpu_bytes": device.total_memory,
        "base_torch_location": str(torch.__file__),
    }


def interrupted(signum, frame):
    raise TimeoutError("worker總期限；還須保留有界匯出／停止時間")


def main():
    job = json.loads(Path("job.json").read_bytes())
    output = Path("result")
    output.mkdir(exist_ok=False)
    summary = {
        "schema": "omni-mature-worker-summary-v1",
        "job": job["job"],
        "success": False,
        "final_opened": False,
        "own_training_permitted": False,
        "core_selection_sha256": job["core_selection_sha256"],
    }
    started = time.monotonic()
    alarm_installed = False
    try:
        for filename, binding in job["input_files"].items():
            path = Path(filename)
            if path.is_symlink() or path.stat().st_size != binding["bytes"] or sha(path) != binding["sha256"]:
                raise ValueError("worker輸入與事前固定的bytes不同")
        base = runtime(job)
        json_file(output / "runtime-before.json", base)
        signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(job["worker_seconds"])
        alarm_installed = True
        lock = requirements(job["job"])
        python = Path("research-venv/bin/python").resolve()
        bounded_command(
            [sys.executable, "-m", "venv", "--system-site-packages", "research-venv"], output / "venv.log", 45
        )
        bounded_command(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--only-binary=:all:",
                "--no-input",
                "--disable-pip-version-check",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu128",
                "-r",
                str(lock),
            ],
            output / "install.log",
            300,
        )
        probe = "import importlib.metadata,json,torch;print(json.dumps({'torch':str(torch.__version__),'cuda':torch.version.cuda,'packages':{k:importlib.metadata.version(k) for k in ['transformers','huggingface-hub','tokenizers','numpy','accelerate','safetensors']}}))"
        bounded_command([str(python), "-c", probe], output / "isolated-runtime.json", 45)
        isolated = json.loads((output / "isolated-runtime.json").read_bytes())
        if isolated["torch"] != base["torch"] or isolated["cuda"] != base["cuda"]:
            raise ValueError("研究隔離環境改動了Torch／CUDA")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "GH_TOKEN",
                "GITHUB_TOKEN",
                "MODAL_TOKEN_ID",
                "MODAL_TOKEN_SECRET",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
            }
        }
        environment.update(
            HF_HUB_DISABLE_TELEMETRY="1",
            HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
            HF_HUB_OFFLINE="0",
            TRANSFORMERS_OFFLINE="0",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="4",
            HF_HUB_ETAG_TIMEOUT="30",
            HF_HUB_DOWNLOAD_TIMEOUT="60",
            PYTHONPATH=str(Path.cwd()),
            PYTHONUNBUFFERED="1",
        )
        if job["job"] == "kev":
            from scripts.omni_revision.research_upstream import verify

            json_file(output / "external-source-verification.json", verify("thirdparty/kev"))
            environment["PYTHONPATH"] += os.pathsep + str(Path("thirdparty/kev").resolve())
        # 保守記為已開研究材料；失敗不會被偽裝成從未看過final後重跑。
        summary["final_opened"] = True
        json_file(
            output / "research-opening.json",
            {
                "job": job["job"],
                "core_selection_sha256": job["core_selection_sha256"],
                "core_final_opening_sha256": job["core_final_opening_sha256"],
                "job_config_sha256": sha("job.json"),
                "data_definition_sha256": sha("data-definition.json"),
                "input_files": job["input_files"],
                "core_models_immutable": True,
            },
        )
        remaining = job["worker_seconds"] - (time.monotonic() - started) - 30
        if remaining < 60:
            raise TimeoutError("安裝已用盡研究期限，不能另加費用")
        module = "scripts.omni_revision.kev_research" if job["job"] == "kev" else "scripts.omni_revision.qwen_research"
        bounded_command([str(python), "-u", "-m", module], output / "inference.log", remaining, environment)
        research = json.loads((output / "research-result.json").read_bytes())
        if research.get("completed") is not True or research.get("final_opened") is not True:
            raise ValueError("研究沒有完整產生預定結果")
        summary.update(
            success=True,
            runtime=base,
            isolated_runtime=isolated,
            research_result_sha256=sha(output / "research-result.json"),
        )
    except BaseException as error:
        summary.update(error_type=type(error).__name__, error=str(error)[:500])
    finally:
        if alarm_installed:
            signal.alarm(0)
        summary["elapsed_seconds"] = time.monotonic() - started
        json_file(output / "worker-summary.json", summary)
    export_and_wait(output, summary, job)


if __name__ == "__main__":
    main()
