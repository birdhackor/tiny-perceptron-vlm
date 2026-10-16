"""Independent README primary-evidence probe; no training, paid jobs or remote writes."""
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tomllib
from datetime import datetime, UTC

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent
SCRATCH = ROOT / "work/transport-readme"
SCRATCH.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

def run(args):
    result = subprocess.run([str(x) for x in args], cwd=ROOT, capture_output=True, text=True, timeout=180)
    return {"command": [str(x) for x in args], "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

report = {
    "reviewer_task": "/root/tech_transport_readme_20261016",
    "at": datetime.now(UTC).isoformat(),
    "source_sha256": sha(ROOT / "README.md"),
    "git_head": run(["git", "rev-parse", "HEAD"])["stdout"].strip(),
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "packages": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "ipykernel", "jupyterlab", "zensical")}},
}
index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
composite = module("transport_composite", ROOT / "docs/review-tools/composite_review.py")
inventory = composite.current_inventory(ROOT)
report["inventory"] = {
    "lesson_index_count": len(index),
    "actual_notebooks": len(list((ROOT / "notebooks").glob("**/*.ipynb"))),
    "canonical_pages": len(inventory["pages"]),
    "numbered_sections": sum(len(re.findall(r"^## [A-Z\d]+\.\d+ ", p.read_text(), re.M)) for p in list((ROOT / "course/chapters").glob("*.md")) + [ROOT / "course" / n for n in ("README.md", "first-steps.md", "training.md", "glossary.md")]),
    "chapter_sources": sorted({i["source"] for i in index}),
    "notebook_1_1_exists": (ROOT / "notebooks/01/1.1.ipynb").is_file(),
    "all_index_notebooks_exist": all((ROOT / i["notebook"]).is_file() for i in index),
}
bootstrap_missing = []
for item in index:
    nb = json.loads((ROOT / item["notebook"]).read_bytes())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if cells and "import google.colab" not in "".join(cells[0]["source"]):
        bootstrap_missing.append(item["id"])
report["notebook_bootstrap_missing"] = bootstrap_missing
report["build_course_check"] = run([sys.executable, "scripts/build_course.py", "--check"])
report["check_env"] = run([sys.executable, "scripts/check_env.py"])
report["training_assets_list"] = run([sys.executable, "scripts/fetch_training_assets.py", "--list"])
report["models_list"] = run([sys.executable, "scripts/fetch_course_models.py", "--list"])
report["prepare_toy"] = run([sys.executable, "scripts/prepare_data.py", "--kind", "toy-text", "--output", SCRATCH / "generated"])
trainer = module("transport_train", ROOT / "scripts/train.py")
import torch
states, optimizer_steps, saves = [], [], []
original_model = trainer.TinyLM
original_step = torch.optim.AdamW.step
original_save = trainer.save_checkpoint

def capture_model(*args, **kwargs):
    value = original_model(*args, **kwargs)
    states.append((value, {k: t.detach().clone() for k, t in value.state_dict().items()}))
    return value

def capture_step(*args, **kwargs):
    optimizer_steps.append(True)
    return original_step(*args, **kwargs)

def capture_save(*args, **kwargs):
    saves.append(True)
    return original_save(*args, **kwargs)

trainer.TinyLM = capture_model
torch.optim.AdamW.step = capture_step
trainer.save_checkpoint = capture_save
train_output = SCRATCH / "must-not-exist.pt"
assert not train_output.exists()
old_argv = sys.argv
sys.argv = ["scripts/train.py", "--task", "text", "--data", str(SCRATCH / "generated/toy-text/train.jsonl"), "--output", str(train_output)]
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    trainer.main()
sys.argv = old_argv
torch.optim.AdamW.step = original_step
trainer.save_checkpoint = original_save
raw = capture.getvalue()
final = json.loads(raw.splitlines()[-1])
report["dry_run"] = {
    "command": "python scripts/train.py --task text --data work/transport-readme/generated/toy-text/train.jsonl --output work/transport-readme/must-not-exist.pt (in-process hooks record calls; original computational functions retained)",
    "stdout": raw,
    "mode": final["mode"],
    "history": final["history"],
    "all_model_state_tensors_unchanged": all(torch.equal(before[k], model.state_dict()[k]) for model, before in states for k in before),
    "optimizer_step_calls": len(optimizer_steps),
    "save_checkpoint_calls": len(saves),
    "output_file_exists": train_output.exists(),
    "finite_loss_positive_finite_grad": all(math.isfinite(h["loss"]) and math.isfinite(h["grad_norm"]) and h["grad_norm"] > 0 for h in final["history"]),
}
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
report["parameter_counts"] = {}
for architecture in ("moe", "dense"):
    model = LimitedAssistant(SelftrainedConfig(vocab_size=550, architecture=architecture, width=256, ffn_hidden=512, top_k=2))
    report["parameter_counts"][architecture] = sum(p.numel() for p in model.parameters())
manifest = json.loads((ROOT / "docs/selftrained/v2-manifest.json").read_bytes())
report["v2_manifest"] = {"sha256": sha(ROOT / "docs/selftrained/v2-manifest.json"), "record_files": len(manifest["records"]), "assets": len(manifest["assets"]), "files_total": len(manifest["records"]) + len(manifest["assets"]), "package": manifest["package"]}
report["v2_raw"] = {}
for architecture, stage in (("moe", "moe-native"), ("dense", "dense-weighted")):
    receipt_path = ROOT / f"docs/selftrained/results/training-raw/{stage}/raw/train-receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    metrics_path = ROOT / f"docs/selftrained/results/public-raw/{architecture}/test/metrics.json"
    metrics = json.loads(metrics_path.read_bytes())
    eval_path = ROOT / f"docs/selftrained/results/public-raw/{architecture}/test/evaluation-receipt.json"
    evaluation = json.loads(eval_path.read_bytes())
    report["v2_raw"][architecture] = {
        "train_receipt_path": receipt_path.relative_to(ROOT).as_posix(), "train_receipt_sha256": sha(receipt_path),
        "train_receipt_fields": {k: receipt[k] for k in receipt if k not in ("asset_sha256", "data_sha256", "source_sha256", "history", "options", "effective_options", "config", "software", "environment")},
        "metrics_path": metrics_path.relative_to(ROOT).as_posix(), "metrics_sha256": sha(metrics_path),
        "metrics_top_level_fields": list(metrics), "evaluation_receipt": evaluation, "evaluation_receipt_path": eval_path.relative_to(ROOT).as_posix(), "evaluation_receipt_sha256": sha(eval_path),
    }
public = json.loads((ROOT / "docs/course-experiments/public-models.json").read_bytes())
capstone = json.loads((ROOT / "docs/course-experiments/capstone-public.json").read_bytes())
report["public_manifest_counts"] = {"models": len(public["models"]), "files": sum(len(x["files"]) for x in public["models"]), "capstone_top_level_keys": list(capstone)}
report["foundation_execution"] = run([sys.executable, "scripts/check_course_models.py", "--model", "text_foundation", "--output", SCRATCH / "models", "--evidence", SCRATCH / "foundation-evidence"])
foundation = sorted((SCRATCH / "foundation-evidence").glob("*.json"))
report["foundation_reports"] = [{"path": str(p.relative_to(ROOT)), "sha256": sha(p), "report": json.loads(p.read_bytes())} for p in foundation]
report["source_hashes"] = {p: sha(ROOT / p) for p in ("README.md", "pyproject.toml", "uv.lock", "scripts/train.py", "scripts/prepare_data.py", "scripts/check_env.py", "scripts/build_course.py", "scripts/fetch_training_assets.py", "scripts/fetch_course_models.py", "scripts/check_course_models.py", "tiny_perceptron/selftrained/model.py", "course/lesson-index.json", "docs/course-experiments/public-models.json", "docs/course-experiments/capstone-public.json", "requirements-natural.txt", ".github/workflows/ci.yml", "course/figures/rewrite-01-character-ids.svg")}
(OUT / "probe-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"inventory": report["inventory"], "bootstrap_missing": bootstrap_missing, "dry_run": report["dry_run"], "parameter_counts": report["parameter_counts"], "foundation_returncode": report["foundation_execution"]["returncode"]}, ensure_ascii=False))
