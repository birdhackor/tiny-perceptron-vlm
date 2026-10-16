"""Bounded CLI, original result, documentation and figure inspection receipts."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, UTC
from urllib.request import Request, urlopen
from collections import Counter

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).parent
os.chdir(ROOT)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=60)
    return {"command": args, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

r = {"reviewer_task": "/root/tech_transport_readme_20261016", "at": datetime.now(UTC).isoformat(), "environment": {"python": platform.python_version(), "torch": importlib.metadata.version("torch"), "device": "cpu"}}
r["commands"] = []
for args in (["uv", "--version"], ["uv", "sync", "--frozen", "--extra", "cpu", "--group", "notebook", "--dry-run"], [sys.executable, "-m", "ipykernel", "install", "--help"], [sys.executable, "-m", "jupyterlab", "--version"], [sys.executable, "scripts/selftrained/train_local_stage.py", "--help"], [sys.executable, "scripts/check_notebooks.py", "--help"]):
    r["commands"].append(command(args))
r["v2_final_metrics"] = {}
for arch in ("moe", "dense"):
    path = ROOT / f"docs/selftrained/results/public-raw/{arch}/test/metrics.json"
    m = json.loads(path.read_bytes())
    task = m["per_task_final_reply"]
    r["v2_final_metrics"][arch] = {
        "path": str(path.relative_to(ROOT)), "sha256": sha(path),
        "declared_count": m["count"], "summed_task_counts": sum(v["count"] for v in task.values()),
        "evaluation_complete": m["evaluation_complete"], "split": m["split"], "limited_smoke": m["limited_smoke"],
        "teacher_forcing_used_for_generation": m["teacher_forcing_used_for_generation"],
        "thresholds": m["thresholds"],
        "task_counts": {k: v["count"] for k, v in task.items()},
        "tool_roundtrip": task["tool_call"]["tool_roundtrip"],
        "tool_roundtrip_pass": task["tool_call"]["tool_roundtrip"]["rate"] >= m["thresholds"]["tool_roundtrip"],
        "voice_qa_semantic": task["voice_qa"]["semantic"],
        "voice_continuation_semantic": task["voice_topic_continuation"]["semantic"],
        "notes": m["notes"],
    }
r["model_inventory"] = {}
for label in ("public-models", "capstone-public"):
    p = ROOT / f"docs/course-experiments/{label}.json"
    m = json.loads(p.read_bytes())
    counts = Counter(Path(f["path"]).suffix for row in m["models"] for f in row["files"])
    r["model_inventory"][label] = {"path": str(p.relative_to(ROOT)), "sha256": sha(p), "models": len(m["models"]), "files_by_suffix": dict(counts)}
plan_path = ROOT / "docs/course-experiments/plan.json"
plan = json.loads(plan_path.read_bytes())
r["formal_experiment_environment"] = []
for item in plan["sequence"]:
    name = item if isinstance(item, str) else item["id"]
    p = ROOT / f"docs/course-experiments/results/{name}.json"
    m = json.loads(p.read_bytes())
    r["formal_experiment_environment"].append({"id": name, "path": str(p.relative_to(ROOT)), "sha256": sha(p), "python": m.get("python_version"), "torch": m.get("torch_version"), "gpu": m.get("gpu")})
original_paths = [
    "docs/validation-artifacts/release-compatibility-ci.json", "tests/test_env.py", "requirements-natural.txt", "tiny_perceptron/natural_assistant.py", "scripts/selftrained/train.py", "scripts/selftrained/train_local_stage.py", "tiny_perceptron/selftrained/tokenizer.py", "docs/selftrained/v2-manifest.json", "assets/training/selftrained-v2.tar.gz", "scripts/selftrained/export.py", "scripts/course_release.py", "course/web/course.css", "course/web/course.js", "course/web/mathjax.js", "scripts/stage_mathjax.py", "zensical.toml", "scripts/export_course.py", "LICENSE",
]
r["inspected_source_hashes"] = {p: sha(ROOT / p) for p in original_paths if (ROOT / p).exists()}
r["original_docs"] = []
for name, url, terms in (
    ("python-venv", "https://docs.python.org/3.13/library/venv.html", ("Activate.ps1", "bin/activate")),
    ("pytorch-checkpoint", "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html", ("optimizer", "epoch")),
):
    entry = {"name": name, "url": url, "accessed_on": datetime.now(UTC).date().isoformat()}
    try:
        with urlopen(Request(url, headers={"User-Agent": "independent-readme-review"}), timeout=30) as response:
            raw = response.read()
        (ROOT / f"work/transport-readme/sources/{name}.html").write_bytes(raw)
        plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw.decode()))
        entry.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), excerpts={term: [plain[max(0, match.start()-140):match.end()+380] for match in list(re.finditer(re.escape(term), plain, re.I))[-5:]] for term in terms})
    except Exception as error:
        entry["error"] = f"{type(error).__name__}: {error}"
    r["original_docs"].append(entry)
r["figure"] = {"source_path": "course/figures/rewrite-01-character-ids.svg", "source_sha256": sha(ROOT / "course/figures/rewrite-01-character-ids.svg"), "render_path": "work/transport-readme/character-ids.png", "render_sha256": sha(ROOT / "work/transport-readme/character-ids.png"), "renderer": "CairoSVG " + importlib.metadata.version("CairoSVG"), "render_command": "cairosvg.svg2png(url='course/figures/rewrite-01-character-ids.svg',write_to='work/transport-readme/character-ids.png')", "viewed_with": "view_image", "inspection": "Personally viewed the rendered 640x710 image. Read order is top row then bottom row; pairs are 貓3, 看2, 狗1, ，4, 狗1, 看2, 貓3, 。0. Repeated characters have identical IDs. Vertical bidirectional arrows join each visible character to its number. Footer correctly says IDs do not order meaning. No factual or direction mismatch found.", "browser_attempts": "Chromium CLI screenshot attempted twice and timed out without a PNG; ImageMagick delegate missing rsvg-convert. Used installed CairoSVG successfully; this is an SVG visual inspection, not website/mobile validation."}
(OUT / "supplement-results.json").write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"commands": [(x["command"], x["returncode"]) for x in r["commands"]], "formal_environment_count": len(r["formal_experiment_environment"]), "figure": r["figure"], "original_docs": [{"name": x["name"], "error": x.get("error")} for x in r["original_docs"]]}, ensure_ascii=False))
