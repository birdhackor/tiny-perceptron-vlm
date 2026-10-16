"""Read-only, proportionate curriculum checks; no training or formal test inference."""

import hashlib
import json
import platform
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices

source_pins = {}

def read(path):
    content = (ROOT / path).read_bytes()
    source_pins[path] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


out = {"environment": {"python": platform.python_version(), "device": "CPU, JSON/source inspection only"},
       "boundary": "Reviewer ran source/receipt/raw-score arithmetic and tiny public CPU calls only; no GPU, training, or formal test restart."}
freeze = read("docs/course-revision-20261016/freeze-14.json")
page = next(x for x in freeze["pages"] if x["page_id"] == "curriculum")
out["freeze"] = {"path": page["snapshot"], "actual_sha256": sha(ROOT / page["snapshot"]), "declared_sha256": page["source_sha256"]}
chapter_counts = {p.name: len(lesson_slices(p.read_text())) for p in sorted((ROOT / "course/chapters").glob("*.md"))}
front = {p: len(lesson_slices((ROOT / "course" / p).read_text())) for p in ("README.md", "first-steps.md", "training.md", "glossary.md")}
out["inventory"] = {"chapters": chapter_counts, "front": front, "chapter_lessons": sum(chapter_counts.values()), "all_numbered": sum(chapter_counts.values()) + sum(front.values()), "ipynb_count": len(list((ROOT / "notebooks").glob("*/*.ipynb")))}
old_root = ROOT / "docs/course-publish-20261008/freeze/current/source-files/course/chapters"
old_ids = set(k for p in old_root.glob("*.md") for k in lesson_slices(p.read_text()))
new_ids = set(k for p in (ROOT / "course/chapters").glob("*.md") for k in lesson_slices(p.read_text()))
out["inventory"]["prior_numbered_ids"] = len(old_ids)
out["inventory"]["added_ids"] = sorted(new_ids - old_ids)
out["inventory"]["removed_ids"] = sorted(old_ids - new_ids)
out["topic_locations"] = {}
for name, ranges in {"07.md": [(17, 25)], "09.md": [(11, 16)], "14.md": [(7, 18)], "19.md": [(13, 20)], "0C.md": [(1, 7)]}.items():
    path = ROOT / "course/chapters" / name
    if not path.exists():
        continue
    out["topic_locations"][name] = [line for line in path.read_text().splitlines() if re.match(r"^## ", line)]
out["v2"] = {}
summary = read("docs/selftrained/results/v2-final-public-results.json")
for arch, training in (("moe", "moe-native"), ("dense", "dense-weighted")):
    receipt = read(f"docs/selftrained/results/training-raw/{training}/raw/train-receipt.json")
    final = read(f"docs/selftrained/results/public-raw/{arch}/test/metrics.json")
    journal = read(f"docs/selftrained/results/public-raw/{arch}/test/evaluation-receipt.json")
    protocol = read(f"docs/selftrained/results/public-raw/{arch}/freeze/frozen.json")
    out["v2"][arch] = {"parameters": summary["architectures"][arch]["capacity"]["parameters"],
        "training": {k: receipt[k] for k in ("steps", "train_records", "validation_records", "completed_requested_steps", "test_used_for_selection", "seed", "tool_loss_weight", "numeric_run_loss_weight")},
        "origin_kind": receipt["origin"]["kind"], "test_count": final["count"], "completed_count": journal["completed_count"],
        "journal_complete": journal["status"], "test_once": protocol["test_once"], "freeze_created_at": protocol["created_at"],
        "test_updated_at": journal["updated_at"], "capability_acceptance_passed": summary["architectures"][arch]["capability_acceptance_passed"],
        "weight_sha_match": final["safe_weights_sha256"] == protocol["safe_weights_sha256"] == summary["architectures"][arch]["safe_weights"]["sha256"],
        "voice_scope": final["voice_scope"]}
out["v2_data_scope"] = summary["data_scope"]
dep = read("docs/course-experiments/results/capstone_deployment.json")
stu = read("docs/course-experiments/results/capstone_student.json")
out["historical_scores"] = {"deployment": {k: {q: dep["results"]["stages"][k][q] for q in ("count", "end_to_end_correct")} for k in ("joint", "dpo")}, "student": {k: {q: stu["results"]["evaluations"][k][q] for q in ("count", "end_to_end_correct")} for k in ("test_ce", "test_kd")}}
pub = read("docs/course-experiments/public-models.json")
cap = read("docs/course-experiments/capstone-public.json")
out["old_public_counts"] = {"groups": len(pub["models"]), "weights": sum(f["path"].endswith(".pt") for g in pub["models"] for f in g["files"]), "capstone_models": [g["id"] for g in cap["models"]]}
products = read("docs/lesson-products/results/final-report-v1.json")
settings = read("docs/lesson-products/results/settings-freeze-v1.json")
pref = read("docs/lesson-products/results/pretest-freeze-record-v1.json")
started = read("docs/lesson-products/results/final-report-v1.started.json")
out["new_products"] = {"freeze_sha_matches": sha(ROOT / "docs/lesson-products/results/settings-freeze-v1.json") == pref["settings_freeze_sha256"] == started["settings_freeze_sha256"] == products["settings_freeze_sha256"], "times": {"frozen": pref["recorded_at"], "started": started["recorded_at"], "reported": products["recorded_at"]}, "gates": products["gates"], "scope": products["scope"], "b_keys": list(products["b"]), "c_keys": list(products["c"]), "shift_keys": list(products["shifts"])}
for product in ("b", "c"):
    ex = read(f"docs/lesson-products/results/initial-{product}-execution.json")
    out["new_products"][product + "_training"] = {"status": ex["status"], "gpu": ex["gpu_name"], "random_initialization": ex["training"]["random_initialization"], "last_step": ex["training"]["last_step"], "selected_step": ex["training"]["selected_step"], "weights_changed": ex["training"]["weights_changed"]}
data = ROOT / "outputs/lesson-products/data-v2/c.jsonl"
duplicate = defaultdict(lambda: defaultdict(set))
families = defaultdict(set)
for line in data.open():
    row = json.loads(line)
    if row["split"] in ("train", "calibration", "policy", "test"):
        families[row["family_id"]].add(row["split"])
    if row["state"] == "卡片進度;申請編號=?":
        duplicate[row["question"]][row["split"]].add(row["gold"])
out["new_products"]["duplicate_observation"] = {q: {s: sorted(v) for s,v in splits.items()} for q,splits in duplicate.items()}
out["new_products"]["cross_core_split_c_families"] = sum(len(s)>1 for s in families.values())
for line in (ROOT / "outputs/lesson-products/data-v2/b.jsonl").open():
    row = json.loads(line)
    if row["split"] in ("train", "calibration", "policy", "test"):
        families[row["family_id"]].add(row["split"])
out["new_products"]["cross_core_split_b_or_c_families"] = sum(len(s)>1 for s in families.values())
predpath = ROOT / "outputs/lesson-products/final-report-v1.predictions.jsonl"
out["new_products"]["predictions_sha_matches"] = sha(predpath) == products["predictions"]["sha256"]
score_counts = defaultdict(Counter)
errors = []
for line in predpath.open():
    row = json.loads(line)
    product, split = row.get("product"), row.get("split")
    if product == "b" and row.get("task") == "sum" and split == "test":
        for depth, answer in row["answers_by_depth"].items():
            key = "b_sum_test_loops_" + depth
            score_counts[key]["count"] += 1
            score_counts[key]["correct"] += answer == row["gold"]
    if product == "c" and split == "wording_shift" and row.get("question") == "choice":
        predicted = row.get("selected")
        if predicted is None:
            candidates = row["candidates"]
            predicted = candidates[max(range(len(candidates)), key=lambda i: row["raw_scores"][i])]
        score_counts["c_wording_choice"]["count"] += 1
        score_counts["c_wording_choice"]["correct"] += predicted == row["gold"]
        if predicted != row["gold"] and len(errors)<3:
            errors.append({"id": row["id"], "gold": row["gold"], "predicted": predicted})
error_index = {r["id"]: r for r in errors}
for line in data.open():
    row = json.loads(line)
    if row["id"] in error_index:
        error_index[row["id"]]["state"] = row["state"]
out["new_products"]["raw_score_recount"] = dict(score_counts)
out["new_products"]["sample_errors"] = errors
out["new_products"]["raw_record_keys"] = list(row)
release = read("docs/lesson-products/results/publication-v1.json")
out["new_products"]["publication"] = {"status": release["status"], "public": release["execution"]["public"], "anonymous_download_count": len(release["execution"]["anonymous_downloads"])}
out["natural_v4"] = {k: v for k,v in read("docs/natural-assistant/v4/public-release.json").items() if k in ("selected_variant","base_model","asr_model","adapter_parameters")}
out["source_sha256"] = source_pins
out["source_sha256"].update({path: sha(ROOT / path) for path in [
    "tiny_perceptron/lesson_products/inference.py", "tiny_perceptron/lesson_products/data.py",
    "scripts/build_course.py", "scripts/export_course.py", "scripts/reading_time.py", "pyproject.toml",
    "outputs/lesson-products/data-v2/b.jsonl", "outputs/lesson-products/data-v2/c.jsonl",
    "outputs/lesson-products/final-report-v1.predictions.jsonl",
]})
out["heading_inventory_source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in list((ROOT / "course/chapters").glob("*.md")) + list(old_root.glob("*.md"))}
out["heading_inventory_scope"] = "Only section identifiers/order/topic headings inspected in these chapter files; no full-chapter technical pass is implied."
output = Path(__file__).with_name("curriculum-audit-output.json")
output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
