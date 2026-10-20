"""核對 README 的數量、設定與可安全執行的短流程，所有輸出寫在本目錄。"""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((ROOT / name).read_text())


def run(args, cwd=ROOT):
    process = subprocess.run(args, cwd=cwd, env=ENV, text=True, capture_output=True, timeout=120)
    return {"command": args, "cwd": str(cwd.relative_to(ROOT)), "returncode": process.returncode,
            "stdout": process.stdout, "stderr": process.stderr}


def main():
    import torch

    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "platform": platform.platform(), "device": "cpu"}
    data = {"environment": environment, "commands": [], "checks": {}}
    checks = data["checks"]
    chapters = sorted((ROOT / "course/chapters").glob("*.md"))
    headings = {}
    for path in chapters:
        headings[path.relative_to(ROOT).as_posix()] = re.findall(r"^## ([\dABC]+\.\d+) (.+)$", path.read_text(), re.M)
    auxiliary = {}
    for name in ("course/README.md", "course/first-steps.md", "course/training.md", "course/glossary.md"):
        auxiliary[name] = re.findall(r"^## (\S+\.\d+) (.+)$", (ROOT / name).read_text(), re.M)
    notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    checks["catalogue"] = {"chapters": headings, "auxiliary": auxiliary,
                           "chapter_sections": sum(map(len, headings.values())),
                           "auxiliary_sections": sum(map(len, auxiliary.values())), "notebooks": len(notebooks)}
    broken = []
    for link in re.findall(r"\]\(([^)]+)\)", (ROOT / "README.md").read_text()):
        if "://" not in link and not (ROOT / link.split("#")[0]).exists():
            broken.append(link)
    checks["missing_local_links"] = broken
    bootstraps = []
    for path in notebooks:
        nb = json.loads(path.read_text())
        cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        if cells:
            bootstraps.append((path.relative_to(ROOT).as_posix(), cells[0].get("metadata", {}).get("course_setup") is True))
    checks["notebook_bootstrap"] = {"with_code": len(bootstraps), "first_cell_setup": sum(x[1] for x in bootstraps),
                                     "exceptions": [x[0] for x in bootstraps if not x[1]]}
    public = load("docs/course-experiments/public-models.json")
    capstone = load("docs/course-experiments/capstone-public.json")
    checks["public_counts"] = {
        "experiments": len(public["models"]),
        "weight_files": sum(f["output"].endswith(".pt") for m in public["models"] for f in m["files"]),
        "capstone_models": len(capstone["models"]),
        "capstone_weights": sum(f["output"].endswith(".pt") for m in capstone["models"] for f in m["files"]),
        "text_foundation": next(m for m in public["models"] if m["id"] == "text_foundation")}
    manifest = load("docs/selftrained/v2-manifest.json")
    checks["v2_package"] = {"package": manifest["package"], "files": len(manifest["records"]) + len(manifest["assets"]),
                             "initialization": manifest["initialization"], "scope": manifest["scope"]}
    results = load("docs/selftrained/results/v2-final-public-results.json")
    checks["v2_results"] = {"data_scope": results["data_scope"], "architectures": {}}
    for name, item in results["architectures"].items():
        final = item["final_test"]
        checks["v2_results"]["architectures"][name] = {
            "capacity": item["capacity"], "selected_source": item["selected_source"],
            "count": final["count"], "evaluation_complete": final["evaluation_complete"],
            "per_task_counts": {k: v["count"] for k, v in final["per_task_final_reply"].items()},
            "criteria": {k: v for k,v in final.items() if re.search(r"criteria|gate|pass", k)}}
    stages = load("docs/selftrained/v2-training-stage-index.json")
    def stage_records(x):
        if isinstance(x, dict):
            if "completed_steps" in x:
                yield {k:v for k,v in x.items() if k in ("stage_key", "attempt", "completed_steps", "selected_step")}
            for v in x.values():
                yield from stage_records(v)
        elif isinstance(x, list):
            for v in x:
                yield from stage_records(v)
    checks["v2_stages"] = list(stage_records(stages))
    final = load("docs/lesson-products/results/final-report-v1.json")
    checks["lesson_products"] = {"b_effort5": final["b"]["5"], "c": final["c"],
        "c_wording": final["shifts"]["c_wording"], "gates": final["gates"],
        "freeze_binding": final["settings_freeze_sha256"],
        "actual_freeze_hash": digest(ROOT / "docs/lesson-products/results/settings-freeze-v1.json"),
        "recorded_at": final["recorded_at"],
        "freeze_recorded_at": load("docs/lesson-products/results/settings-freeze-v1.json")["recorded_at"]}
    checks["lesson_training"] = {}
    for name in ("b", "c"):
        x = load(f"docs/lesson-products/results/initial-{name}-execution.json")
        checks["lesson_training"][name] = {"gpu": x["gpu_name"], "status": x["status"],
            "training": {k:v for k,v in x["training"].items() if k in ("random_initialization", "last_step", "selected_step", "parameters", "environment", "weights_changed")}}
    experiments = []
    for path in sorted((ROOT / "docs/course-experiments/results").glob("*.json")):
        x = json.loads(path.read_text())
        experiments.append({k:x.get(k) for k in ("experiment_id", "device", "gpu", "torch_version", "status", "step_scale")})
    checks["experiments"] = experiments
    import tomllib
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    checks["installation"] = {"extras": project["project"]["optional-dependencies"],
        "groups": project["dependency-groups"], "torch_sources": project["tool"]["uv"]["sources"]["torch"],
        "torch_locked": [{"version": p["version"], "source": p["source"],
                          "wheel_names": [w["url"].split("/")[-1] for w in p.get("wheels", [])]}
                         for p in lock["package"] if p["name"] == "torch"]}
    png = ROOT / "outputs/course-review-20261019/figure-renders-04/61d45a7fdaef4ce990a51a9cf924ebc661cf41117f7964bd12d3b1c6c4fb02ad.png"
    checks["figure"] = {"svg_sha256": digest(ROOT / "course/figures/rewrite-01-character-ids.svg"),
        "viewed_png": png.relative_to(ROOT).as_posix(), "png_sha256": digest(png),
        "text": "貓看狗，狗看貓。", "ids": [3,2,1,4,1,2,3,0],
        "mapping": {"貓": 3, "看": 2, "狗": 1, "，": 4, "。": 0}}
    uv = ["uv", "run", "--no-sync", "python"]
    commands = [
        ["uv", "--version"],
        ["git", "--version"],
        uv + ["scripts/prepare_data.py", "--kind", "toy-text", "--output", str(HERE / "generated")],
        uv + ["scripts/train.py", "--task", "text", "--data", str(HERE / "generated/toy-text/train.jsonl"), "--device", "cpu", "--output", str(HERE / "unexpected-checkpoint.pt")],
        uv + ["scripts/fetch_training_assets.py", "--list"],
        uv + ["scripts/fetch_course_models.py", "--list"],
        uv + ["scripts/build_course.py", "--check"],
        uv + ["scripts/check_notebooks.py", "--mode", "python", "--lesson", "1.1"],
        uv + ["scripts/check_course_models.py", "--help"],
        ["uv", "sync", "--frozen", "--extra", "cpu", "--group", "notebook", "--dry-run"],
    ]
    for cmd in commands:
        data["commands"].append(run(cmd))
    checks["dry_run_saved_checkpoint"] = (HERE / "unexpected-checkpoint.pt").exists()
    path = HERE / "audit-output.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
        "commands": [{"command": c["command"], "returncode": c["returncode"]} for c in data["commands"]],
        "counts": {k: checks["catalogue"][k] for k in ("chapter_sections", "auxiliary_sections", "notebooks")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
