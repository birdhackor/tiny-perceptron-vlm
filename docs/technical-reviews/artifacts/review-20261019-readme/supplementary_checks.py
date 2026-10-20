"""保存原始結果定位、參數重算與匿名 CPU 下載檢查。"""

import hashlib
import io
import json
import os
import platform
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    output = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
              "checks": {}, "source_files": {}, "network": []}
    checks = output["checks"]
    checks["raw_training"] = {}
    for model, directory in (("moe", "moe-native"), ("dense", "dense-weighted")):
        path = f"docs/selftrained/results/training-raw/{directory}/raw/train-receipt.json"
        x = read(path)
        checks["raw_training"][model] = {"path": path, "sha256": sha(ROOT / path),
            **{k:x[k] for k in ("steps", "tokens", "target_tokens", "train_records", "validation_records", "test_used_for_selection", "selection", "origin", "completed_requested_steps")}}
    selection = read("docs/selftrained/results/v2-final-training-source-selection.json")
    checks["selected_steps"] = {name: {k:v for k,v in value.items() if k in ("selected_step", "completed_requested_steps", "source_training_run", "sha256")}
                                for name,value in selection.items() if isinstance(value,dict) and "selected_step" in value}
    checks["raw_final_tests"] = {}
    for model in ("moe", "dense"):
        prefix = f"docs/selftrained/results/public-raw/{model}/test/"
        x = read(prefix + "metrics.json")
        checks["raw_final_tests"][model] = {"count": x["count"], "thresholds": x["thresholds"],
            "ocr_exact": x["per_task_final_reply"]["ocr"]["exact"],
            "tool_roundtrip": x["per_task_final_reply"]["tool_call"]["tool_roundtrip"],
            "journal": read(prefix + "evaluation-receipt.json")}
    from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
    results = read("docs/selftrained/results/v2-final-public-results.json")
    checks["recomputed_parameters"] = {}
    for name, item in results["architectures"].items():
        model = LimitedAssistant(SelftrainedConfig(**item["capacity"]["config"]))
        checks["recomputed_parameters"][name] = sum(p.numel() for p in model.parameters())
    archive = ROOT / "assets/training/selftrained-v2.tar.gz"
    checks["archive"] = {"working_tree_bytes": archive.stat().st_size}
    if archive.stat().st_size > 1024:
        counts = {"train": 0, "validation": 0, "test": 0}
        with tarfile.open(archive) as tar:
            members = [m for m in tar if m.isfile()]
            for member in members:
                if member.name.endswith(".jsonl"):
                    split = member.name.rsplit("-", 1)[-1].split(".")[0]
                    if split in counts:
                        counts[split] += sum(1 for row in tar.extractfile(member) if row.strip())
        checks["archive"].update(sha256=sha(archive), files=len(members), rows=counts)
    else:
        checks["archive"]["lfs_pointer"] = archive.read_text()
    from scripts.course_release import inference_payload
    exported = inference_payload({"format_version": 1, "config": {}, "model": {}, "optimizer": {"test": 1}, "rng": {"test": 1}, "training_state": {"test": 1}}, {})
    checks["public_export_keys"] = list(exported)
    from tiny_perceptron.lesson_products.inference import auto_solve
    class Stub:
        auto_policy = {"version": 1, "sum_depth": 3, "low_max_events": 1, "medium_max_events": 3}
        def synchronize(self):
            pass
        def judge(self, state, question):
            return {"selected": "boxes" if question == "choice" else "yes", "seconds": 0}
        def solve(self, state, loops):
            return {"loops": loops}
    checks["auto_order"] = auto_solve(Stub(), Stub(), "兩盒追蹤;A=2;B=6;事件=交換;查詢=B")
    with tempfile.TemporaryDirectory(prefix="public-probe-", dir=HERE) as temp:
        temp = Path(temp)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT),
               "HF_HOME": str(temp / "hf"), "HF_HUB_CACHE": str(temp / "hf/hub"),
               "HF_XET_CACHE": str(temp / "hf/xet")}
        command = ["uv", "run", "--no-sync", "python", "scripts/check_course_models.py", "--model", "text_foundation",
                   "--output", str(temp / "weights"), "--evidence", str(temp / "evidence")]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        checks["anonymous_cpu_probe"] = {"command": command, "returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr, "records": [json.loads(p.read_text()) for p in sorted((temp / "evidence").glob("*.json"))]}
    for name,url,needle in [
        ("mathjax", "https://docs.mathjax.org/en/v3.2/web/hosting.html", "local"),
        ("ipykernel", "https://raw.githubusercontent.com/ipython/ipykernel/v7.4.0/ipykernel/kernelspec.py", "sys-prefix"),
        ("colab_entry", "https://colab.research.google.com/", "Sign in"),
    ]:
        command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "40", url]
        r = subprocess.run(command, capture_output=True, timeout=45)
        text = re.sub("<[^>]+>", " ", r.stdout.decode(errors="replace"))
        text = re.sub(r"\s+", " ", text)
        pos = text.lower().find(needle.lower())
        output["network"].append({"id":name, "url":url, "command":command, "returncode":r.returncode,
            "raw_sha256":hashlib.sha256(r.stdout).hexdigest(), "accessed_on":"2026-10-20", "locator":needle,
            "found":pos>=0, "excerpt":" ".join(text[pos:pos+600].split()[:23]) if pos>=0 else ""})
    names = [
        "README.md", "course/figures/rewrite-01-character-ids.svg", "pyproject.toml", "uv.lock", "LICENSE", ".gitignore",
        "scripts/prepare_data.py", "scripts/train.py", "tiny_perceptron/selftrained/model.py", "scripts/selftrained/train.py",
        "tiny_perceptron/selftrained/tokenizer.py", "tiny_perceptron/lesson_products/inference.py", "tiny_perceptron/natural_assistant.py",
        "scripts/build_course.py", "scripts/build_visuals.py", "scripts/format_course_code.py", "scripts/check_notebooks.py",
        "scripts/export_course.py", "scripts/stage_mathjax.py", "scripts/course_release.py", "scripts/fetch_training_assets.py",
        "scripts/fetch_course_models.py", "scripts/check_course_models.py", "scripts/check_course_reviews.py", "scripts/check_technical_reviews.py",
        "scripts/check_env.py", "tests/test_env.py", ".github/workflows/pages.yml", ".github/workflows/ci.yml", ".github/workflows/gpu-smoke.yml",
        "course/web/course.css", "course/web/course.js", "course/web/mathjax.js", "course/web/mathjax-assets.json", "zensical.toml",
        "requirements-natural.txt", "docs/selftrained/v2-manifest.json", "docs/selftrained/results/v2-final-public-results.json",
        "docs/selftrained/results/v2-final-training-source-selection.json", "docs/selftrained/TRAINING.md", "docs/selftrained/v2-public-cpu-commands.md",
        "docs/lesson-products/results/final-report-v1.json", "docs/lesson-products/results/settings-freeze-v1.json",
        "docs/lesson-products/results/pretest-freeze-record-v1.json", "docs/lesson-products/results/settings-freeze-v1.test-started.json",
        "docs/lesson-products/results/initial-b-execution.json", "docs/lesson-products/results/initial-c-execution.json",
        "docs/course-experiments/public-models.json", "docs/course-experiments/capstone-public.json",
        "docs/validation-artifacts/release-compatibility-ci.json", "docs/natural-assistant/v4/STUDENT.md", "course/lesson-index.json",
    ]
    for name in names:
        path = ROOT / name
        output["source_files"][name] = {"sha256":sha(path), "bytes":path.stat().st_size}
    target = HERE / "supplementary-output.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output":str(target.relative_to(ROOT)), "sha256":sha(target),
        "recomputed_parameters":checks["recomputed_parameters"], "archive":checks["archive"],
        "public_probe_returncode":checks["anonymous_cpu_probe"]["returncode"], "network":output["network"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
