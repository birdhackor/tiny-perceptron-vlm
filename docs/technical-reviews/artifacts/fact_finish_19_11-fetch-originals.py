"""Persist anonymous public downloads and original runtime sources for lesson 19.11."""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import huggingface_hub
import requests
import torch
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.fetch_capstone import fetch_capstone  # noqa: E402
from tiny_perceptron.capstone import load_capstone  # noqa: E402
from tiny_perceptron.capstone_quantization import load_quantized_capstone  # noqa: E402

ART = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_finish_19_11"
LOCAL = Path("/tmp/fact_finish_19_11-public")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(name, value):
    (ART / f"{PREFIX}-{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    torch.set_num_threads(2)
    manifest_path = ROOT / "docs/course-experiments/capstone-public.json"
    manifest = json.loads(manifest_path.read_text())
    shutil.copyfile(manifest_path, ART / f"{PREFIX}-public-manifest.json")
    calls = []
    original = huggingface_hub.hf_hub_download

    def traced_download(repo_id, filename, **kwargs):
        path = original(repo_id, filename, **kwargs)
        calls.append({"repo_id": repo_id, "filename": filename, "kwargs": kwargs, "sha256": sha(path)})
        return path

    huggingface_hub.hf_hub_download = traced_download
    records = []
    for spec in manifest["models"]:
        directory = fetch_capstone(manifest, spec["id"], LOCAL)
        path = directory / "model.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        loader = load_capstone if spec["format_version"] == "capstone-v1" else load_quantized_capstone
        model, _ = loader(path)
        tensors = []
        for name, value in model.state_dict().items():
            tensors.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "finite": bool(torch.isfinite(value).all()),
                    "numel": value.numel(),
                    "sha256": hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest(),
                }
            )
        records.append(
            {
                "id": spec["id"],
                "bytes": path.stat().st_size,
                "sha256": sha(path),
                "keys": sorted(payload),
                "format": payload["format_version"],
                "inference_only": payload["inference_only"],
                "config": payload["config"],
                "tokenizer": payload["tokenizer"],
                "data_version": payload["data_version"],
                "stage": payload["stage"],
                "step": payload["step"],
                "metadata": payload["metadata"],
                "quantization": payload.get("quantization"),
                "description": model.description(),
                "tensors": tensors,
                "files": [
                    {
                        "output": item["output"],
                        "bytes": (directory / item["output"]).stat().st_size,
                        "sha256": sha(directory / item["output"]),
                    }
                    for item in spec["files"]
                ],
            }
        )
        for item in spec["files"]:
            if item["output"] != "model.pt":
                category = "student" if spec["id"].startswith("student-") else "main"
                shutil.copyfile(directory / item["output"], ART / f"{PREFIX}-public-{category}-{item['output']}")
        print(spec["id"], path.stat().st_size, sha(path), "loaded", flush=True)
    save_json("public-downloads", {"manifest_sha256": sha(manifest_path), "calls": calls, "models": records})
    tree = list(
        HfApi(token=False).list_repo_tree(
            manifest["repo"],
            path_in_repo="course/course-integration-v2",
            recursive=True,
            revision=manifest["revision"],
            token=False,
        )
    )
    save_json(
        "public-tree",
        {
            "repo": manifest["repo"],
            "revision": manifest["revision"],
            "token": False,
            "files": [{"path": item.path, "size": getattr(item, "size", None)} for item in tree],
        },
    )
    source_paths = {
        "torch-adam": Path(torch.__file__).parent / "optim/adam.py",
        "torch-adamw": Path(torch.__file__).parent / "optim/adamw.py",
        "torch-optimizer": Path(torch.__file__).parent / "optim/optimizer.py",
        "torch-random": Path(torch.__file__).parent / "random.py",
        "torch-serialization": Path(torch.__file__).parent / "serialization.py",
        "hf-file-download": Path(huggingface_hub.__file__).parent / "file_download.py",
    }
    source_records = []
    for name, path in source_paths.items():
        dst = ART / f"{PREFIX}-{name}.py.txt"
        shutil.copyfile(path, dst)
        source_records.append({"original": str(path), "snapshot": str(dst.relative_to(ROOT)), "sha256": sha(dst)})
    for name in [
        "tiny_perceptron/capstone.py",
        "tiny_perceptron/capstone_quantization.py",
        "tiny_perceptron/capstone_ui.py",
        "scripts/course_experiments/capstone.py",
        "scripts/capstone.py",
        "scripts/fetch_capstone.py",
        "scripts/capstone_release.py",
        "tests/test_capstone.py",
        "tests/test_capstone_release.py",
        "tiny_perceptron/training.py",
    ]:
        dst = ART / f"{PREFIX}-repo-{name.replace('/', '_')}.txt"
        shutil.copyfile(ROOT / name, dst)
        source_records.append({"original": name, "snapshot": str(dst.relative_to(ROOT)), "sha256": sha(dst)})
    save_json(
        "source-copies",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "hf": huggingface_hub.__version__,
            "sources": source_records,
        },
    )
    urls = {
        "torch-randomness": "https://docs.pytorch.org/docs/2.14/notes/randomness.html",
        "torch-saving": "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html",
        "hf-download-guide": "https://huggingface.co/docs/huggingface_hub/v1.33.0/en/guides/download",
    }
    downloads = []
    for name, url in urls.items():
        response = requests.get(url, timeout=45)
        dst = ART / f"{PREFIX}-{name}.html"
        dst.write_bytes(response.content)
        downloads.append(
            {
                "url": url,
                "final_url": response.url,
                "status": response.status_code,
                "snapshot": str(dst.relative_to(ROOT)),
                "sha256": sha(dst),
            }
        )
        print("official", name, response.status_code, response.url, flush=True)
    save_json("official-downloads", downloads)
    command = [sys.executable, "scripts/fetch_capstone.py", "--list"]
    listed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in listed.stdout.splitlines()]
    assert len(rows) == 11 and {row["stage"] for row in rows} == {spec["id"] for spec in manifest["models"]}
    save_json(
        "list-command",
        {
            "command": command,
            "exit_code": listed.returncode,
            "stdout": listed.stdout,
            "stderr": listed.stderr,
            "row_count": len(rows),
        },
    )


if __name__ == "__main__":
    main()
