"""核對教材清單、固定結果與模型規模；不執行訓練或發布。"""

import datetime
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
inputs = {}


def read(path):
    raw = (ROOT / path).read_bytes()
    inputs[path] = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8")


def load(path):
    return json.loads(read(path))


def leaves(value, prefix="", match=None):
    result = {}
    if isinstance(value, dict):
        for k, v in value.items():
            result.update(leaves(v, f"{prefix}.{k}".strip("."), match))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            result.update(leaves(v, f"{prefix}[{i}]", match))
    elif match is None or re.search(match, prefix):
        result[prefix] = value
    return result


def excerpts(path, patterns, radius=2, limit=80):
    lines = read(path).splitlines()
    selected = set()
    for pattern in patterns:
        for i, line in enumerate(lines):
            if pattern in line:
                selected.update(range(max(0, i-radius), min(len(lines), i+radius+1)))
    return [{"line": i+1, "text": lines[i]} for i in sorted(selected)[:limit]]


out = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-curriculum/inspect_local.py",
       "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu"},
       "created_at": datetime.datetime.now(datetime.UTC).isoformat()}
source = "docs/course-review-20261019/freeze-04/sources/curriculum.md"
read(source)
read("docs/course-review-20261019/freeze-04.json")
read("docs/curriculum.md")
read("course/figures/curriculum_learning_flow.svg")
png = "outputs/course-review-20261019/figure-renders-04/55c76a1ce7d7482025064283219b7449ab13ebd7c4c518979f5a55fa16d41da4.png"
out["figure_viewed"] = {"path": png, "sha256": hashlib.sha256((ROOT/png).read_bytes()).hexdigest(), "inspection": "已用 view_image 看整張凍結算繪；X/Y、可讀前文、計分位置與 zero_grad→backward→step 方向和正文一致；明示本次不累積與累積時另訂清零。"}
chapters = sorted((ROOT / "course/chapters").glob("*.md"))
inventory = {}
for p in chapters + [ROOT / "course" / x for x in ("README.md", "first-steps.md", "training.md", "glossary.md")]:
    rel = p.relative_to(ROOT).as_posix()
    inventory[rel] = re.findall(r"^## ([A-Z\d]+\.\d+) (.+)$", read(rel), re.M)
out["inventory"] = {"chapters": len(chapters), "numeric_chapters": sum(p.stem.isdigit() for p in chapters), "extension_chapters": [p.stem for p in chapters if not p.stem.isdigit()], "chapter_sections": sum(len(inventory[p.relative_to(ROOT).as_posix()]) for p in chapters), "all_numbered_sections": sum(map(len, inventory.values())), "notebook_files": len(list((ROOT / "notebooks").glob("**/*.ipynb"))), "sections": inventory}
spec = importlib.util.spec_from_file_location("curriculum_notebook_build", ROOT / "scripts/build_course.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
out["notebook_sync"] = {"checked": 0, "stale": []}
for path, sections in inventory.items():
    if not path.startswith("course/chapters/"):
        continue
    p = ROOT / path
    parts = re.split(r"^## ([\dABC]+\.\d+) (.+)$", p.read_text(), flags=re.M)
    for i in range(1, len(parts), 3):
        sid, title, body = parts[i:i+3]
        target = ROOT / "notebooks" / p.stem / (sid+".ipynb")
        expected = json.dumps(module.notebook(sid, title, body, p), ensure_ascii=False, indent=1)+"\n"
        out["notebook_sync"]["checked"] += 1
        if not target.is_file() or target.read_text() != expected:
            out["notebook_sync"]["stale"].append(target.relative_to(ROOT).as_posix())
out["notebook_group"] = tomllib.loads(read("pyproject.toml"))["dependency-groups"]["notebook"]
site = tomllib.loads(read("zensical.toml"))["project"]
out["site_navigation"] = leaves(site["nav"])
out["historical_index"] = excerpts("docs/course-experiments/README.md", ["Flash", "PPO", "30", "capstone", "CUDA"], limit=65)
result = load("docs/selftrained/results/v2-final-public-results.json")
out["selftrained"] = {"data_scope": result["data_scope"], "architectures": {}}
for architecture, values in result["architectures"].items():
    final = values["final_test"]
    out["selftrained"]["architectures"][architecture] = {
        "capacity": values["capacity"], "count": final["count"],
        "task_count_sum": sum(v["count"] for v in final["per_task_final_reply"].values()),
        "evaluation_complete": final["evaluation_complete"], "run_id": final["run_id"],
        "capability_acceptance_passed": values["capability_acceptance_passed"],
        "ocr_exact": final["per_task_final_reply"]["ocr"]["exact"],
        "voice": final["per_task_final_reply"]["voice_qa"]["semantic"],
        "thresholds": final["thresholds"], "safe_weights": values["safe_weights"]}
import torch
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig

out["environment"]["torch"] = torch.__version__
out["recounted_models"] = {}
for architecture, values in result["architectures"].items():
    model = LimitedAssistant(SelftrainedConfig(**values["capacity"]["config"]))
    out["recounted_models"][architecture] = model.description()
    del model
manifest = load("docs/selftrained/v2-manifest.json")
out["data_manifest"] = {"model_config": manifest["model_config"], "initialization": manifest["initialization"], "scope": manifest["scope"], "records": manifest["records"]}
package = ROOT / manifest["package"]["path"]
out["data_package"] = {"path": manifest["package"]["path"], "present": package.exists()}
if package.exists() and tarfile.is_tarfile(package):
    out["data_package"]["sha256"] = hashlib.sha256(package.read_bytes()).hexdigest()
    counts = {}
    with tarfile.open(package) as archive:
        for member in archive:
            if member.name.endswith(".jsonl"):
                stream = archive.extractfile(member)
                raw = stream.read()
                counts[member.name] = {"count": len(raw.splitlines()), "sha256": hashlib.sha256(raw).hexdigest()}
    out["data_package"]["records"] = counts
out["training_selection"] = load("docs/selftrained/results/v2-final-training-source-selection.json")
out["four_export_release_job"] = load("docs/selftrained/v2-jobs/release-final-four-exports.json")
out["training_stages"] = {}
for name in ["v2-moe-native-balanced-4000-training-stage-index.json", "v2-dense-weighted-10000-training-stage-index.json"]:
    out["training_stages"][name] = load("docs/selftrained/results/"+name)
out["old_experiments"] = {}
for name in ["capstone_deployment", "capstone_student"]:
    j = load(f"docs/course-experiments/results/{name}.json")
    out["old_experiments"][name] = leaves(j["results"], match=r"(stages|evaluations).*(count|end_to_end_correct|protocol)$")
models = load("docs/course-experiments/public-models.json")["models"]
out["public_experiments"] = {"count": len(models), "first_model": models[0], "leaf_counts": {k: len(v) for k,v in models[0].items() if isinstance(v,list)}}
out["public_experiments"]["weight_count"] = sum(1 for m in models for f in m["files"] if f["path"].endswith((".pt", ".safetensors")))
out["public_capstone"] = load("docs/course-experiments/capstone-public.json")
out["natural_release"] = load("docs/natural-assistant/v4/public-release.json")
products = load("docs/lesson-products/results/final-report-v1.json")
out["products_final"] = {k:v for k,v in products.items() if k not in ("latency", "b")}
out["products_b"] = leaves(products.get("b", {}), match=r"(boxes|sum)\.(count|correct|accuracy)$")
out["products_failure_examples"] = load("docs/lesson-products/results/observed-failure-examples-v1.json")
out["products_data_audit"] = load("docs/lesson-products/data-v2-manifest.json")["audit"]
out["products_duplicate_example"] = load("docs/lesson-products/results/duplicate-observation-example-v1.json")
out["product_training"] = {}
for name in ("initial-b-execution.json", "initial-c-execution.json", "settings-freeze-v1.json", "pretest-freeze-record-v1.json"):
    value = load("docs/lesson-products/results/"+name)
    out["product_training"][name] = leaves(value, match=r"(device|gpu|steps|selected_step|completed_steps|status|checkpoint|recorded_at|created_at|initialization|frozen|sha256|seed|job|version|purpose|command|run_id)")
out["code_excerpts"] = {}
for path, patterns in [
    ("scripts/build_course.py", ["def build", "def notebook", '"course_setup"', "BOOTSTRAP"]),
    ("scripts/export_course.py", ["colab", "notebook", "curriculum", "read_text"]),
    ("tiny_perceptron/selftrained/model.py", ["class LimitedAssistant", "for index, expert", "topk", "class MaskedBlock"]),
    ("tiny_perceptron/selftrained/dataset.py", ["class RecordEncoder", "def encode", '"messages"', "input_ids", "labels"]),
    ("scripts/selftrained/hf_transport.py", ["save_file", "safetensors", "optimizer", "safe_exports"]),
    ("scripts/selftrained/train_local_stage.py", ["def local_source", "local-subprocess", "scripts/selftrained/train.py", "subprocess.Popen", "exact_resume"]),
    ("tiny_perceptron/lesson_products/inference.py", ["def rule_depth", "def auto_solve", 'c.judge(state', "requested =", "actual ="]),
    ("tiny_perceptron/lesson_products/data.py", ["family", "卡片進度", "family_split"]),
]:
    out["code_excerpts"][path] = excerpts(path, patterns)
policy = load("docs/lesson-products/results/b-auto-policy.json")
from tiny_perceptron.lesson_products.inference import parse_state, rule_depth
out["executed_auto_rules"] = {"policy": policy, "box_depths": {str(n): rule_depth(policy, "boxes", n) for n in range(1,7)}, "sum_depth": rule_depth(policy,"sum"), "parsed_state": parse_state("兩盒追蹤;A=2;B=6;事件=交換;查詢=B")}
out["inputs_sha256"] = inputs
(HERE / "local-evidence.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({"inventory": {k:v for k,v in out["inventory"].items() if k!="sections"}, "notebook_sync": out["notebook_sync"], "parameters": {k:v["parameters"] for k,v in out["recounted_models"].items()}, "data_package": out["data_package"], "inputs": len(inputs)}, ensure_ascii=False))
