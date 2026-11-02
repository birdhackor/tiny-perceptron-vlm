"""Reviewer-owned read-only provenance/contract audit and saved-score recomputation.

No torch import, neural forward, training, selection, calibration, network, or publication.
"""
import ast
import gzip
import hashlib
import json
import math
import platform
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROOF = Path(__file__).resolve().parent
BASE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual"
RECEIPT = {"created_at": datetime.now(timezone.utc).isoformat(),
           "reviewer_task": "/root/omni_p5_technical_chapter_19", "python": platform.python_version(),
           "device": "CPU; standard-library inspection and saved-logit arithmetic only", "inputs": []}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    data = path.read_bytes()
    RECEIPT["inputs"].append({"path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
                              "bytes": len(data), "sha256": sha(data)})
    return data


def load(path):
    return json.loads(read(path))


freeze_path = BASE / "source-freeze-v10.json"
freeze = load(freeze_path)
packet = load(BASE / "technical-inputs-packet-v8.json")
assert packet["canonical_source_freeze"]["sha256"] == sha(freeze_path.read_bytes())
pages = {p["page_id"]: p for p in freeze["pages"]}
intro = read(pages["chapter-19"]["snapshot"])
assert sha(intro) == pages["chapter-19"]["source_sha256"]
chapter = read("course/chapters/19.md")
assert chapter[:re.search(rb"^## ", chapter, re.M).start()] == intro
RECEIPT["source_identity"] = {"canonical_source_sha256": sha(intro), "bytes": len(intro),
                               "selector": "before-first-section", "current_exact_slice": True,
                               "svg_references": re.findall(r"!\[[^]]*\]\(([^)]+\.svg)\)", intro.decode())}

# Navigation claims: verify exact frozen topics and explicit prerequisite destinations.
ids = [f"19.{i}" for i in range(1, 25)] + ["chapter-10", "chapter-11", "chapter-12", "chapter-20",
       "7.1", "7.2", "7.3", "15.4", "15.5", "14.11", "14.12", "14.13", "14.14", "14.15",
       "9.11", "9.12", "9.13", "9.14", "9.15", "glossary"]
RECEIPT["navigation_inventory"] = []
for identifier in ids:
    p = pages[identifier]
    body = read(p["snapshot"])
    assert sha(body) == p["source_sha256"]
    RECEIPT["navigation_inventory"].append({"page_id": identifier, "title": p["title"],
                                             "snapshot": p["snapshot"], "sha256": sha(body)})
RECEIPT["extension_code"] = {}
for identifier in ["19.21", "19.22", "19.23", "19.24"]:
    body = (ROOT / pages[identifier]["snapshot"]).read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", body, re.S)
    for block in blocks:
        ast.parse(block)
    RECEIPT["extension_code"][identifier] = {"python_blocks": len(blocks),
                                             "python_lines": sum(len(b.splitlines()) for b in blocks),
                                             "syntax_checked_without_execution": True,
                                             "cpu_demonstration_explicit": "CPU" in body,
                                             "training_and_acceptance_discussed_separately": "訓練" in body and "驗收" in body}

# Direct implementation inspection; record actual AST definitions and call paths.
model_source = read("tiny_perceptron/selftrained/model.py").decode()
tree = ast.parse(model_source)
RECEIPT["v2_module_definitions"] = {node.name: {"line": node.lineno, "end_line": node.end_lineno}
                                     for node in tree.body if isinstance(node, ast.ClassDef)}
assistant = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LimitedAssistant")
init = ast.get_source_segment(model_source, next(n for n in assistant.body if getattr(n, "name", None) == "__init__"))
for name in ["SelftrainedLanguageModel", "VisionEncoder", "OCREncoder", "AudioEncoder"]:
    assert name + "(" in init
assert "embeddings[row, slots] = tokens" in model_source
assert all(s in model_source for s in ["nn.Linear(64 * 2 * 2, width)", "nn.Linear(96, width)"])
RECEIPT["v2_contract"] = {"neural_modules": ["lm", "vision_encoder", "ocr_encoder", "audio_encoder"],
                           "three_perception_inputs": ["image", "ocr", "audio"],
                           "projection_and_reserved_embeddings_checked": True,
                           "method": "AST and actual source inspection; no forward executed"}
trainer = read("scripts/selftrained/train.py").decode()
assert 'model = LimitedAssistant(config).to(args.device)' in trainer
assert 'origin = source.get("origin") if source else {"kind": "all-neural-weights-random", "seed": args.seed}' in trainer
assert 'raise ValueError("只接受本從零管線 checkpoint")' in trainer

# Historical selected V2 training receipts (existing run only).
native = Path("/workspace/selftrained-v2/outputs/selftrained-v2/operations/moe-joint-native-balanced-4000-attempt-1/monitor/artifact/selftrained-run-3f9fbb2a04f08034c0b630e837e7afba1f584402-37440329212-1/gha-37440329212-1")
execution = load(native / "receipt.json")
training = load(native / "raw/train-receipt.json")
inference = load(native / "raw/inference-manifest.json")
public = load("docs/selftrained/results/v2-final-public-results.json")
assert execution["status"] == "completed" and execution["returncode"] == 0
assert training["origin"]["kind"] == "all-neural-weights-random"
assert training["steps"] == 4000 and training["completed_requested_steps"]
assert inference["selected_checkpoint_sha256"] == public["architectures"]["moe"]["selected_source"]["sha256"]
assert inference["files"]["model.safetensors"] == public["architectures"]["moe"]["safe_weights"]["sha256"]
RECEIPT["v2_training"] = {"run_id": execution["run_id"], "revision": execution["revision"],
                           "started_at": execution["started_at"], "finished_at": execution["finished_at"],
                           "completed_steps": training["steps"], "selected_step": inference["selected_step"],
                           "origin_kind": training["origin"]["kind"], "origin_seed": training["origin"]["seed"],
                           "stage_history": [{k: row[k] for k in ["stage", "step", "checkpoint_sha256", "selection"]} for row in training["stage_history"]],
                           "selected_checkpoint_sha256": inference["selected_checkpoint_sha256"],
                           "public_safe_weights_sha256": inference["files"]["model.safetensors"],
                           "binding_to_public_results": True}

# Material provenance is recorded explicitly; this is not a licensing re-audit.
RECEIPT["material_provenance"] = {}
for name in ["vision-sources", "ocr-sources", "voice"]:
    d = load(f"docs/selftrained/provenance/{name}.json")
    if name == "voice":
        selection = {k: d[k] for k in ["dataset", "config", "revision", "license", "inference_input_fields", "never_model_inputs", "speaker_disjoint"]}
    else:
        selection = {k: d[k] for k in ["source_revision", "license", "attribution", "downloads"]}
    RECEIPT["material_provenance"][name] = selection

# V2 once-only final aggregate and journal; preserve the incomplete capability boundary.
metrics = load("docs/selftrained/results/public-raw/moe/test/metrics.json")
journal = load("docs/selftrained/results/public-raw/moe/test/evaluation-receipt.json")
assert metrics["evaluation_complete"] and metrics["count"] == journal["completed_count"] == 3734
assert journal["checkpoint_sha256"] == inference["files"]["model.safetensors"]
assert sum(row["count"] for row in metrics["per_task_final_reply"].values()) == 3734
RECEIPT["v2_test"] = {"count": metrics["count"], "teacher_forcing_used_for_generation": metrics["teacher_forcing_used_for_generation"],
                       "per_task": {task: {"count": row["count"], "semantic": row["semantic"]} for task, row in metrics["per_task_final_reply"].items()},
                       "thresholds": metrics["thresholds"], "complete_journal": journal["status"],
                       "capability_acceptance_passed": public["architectures"]["moe"]["capability_acceptance_passed"],
                       "scope": "Aggregate/journal checked, not a new final NN measurement or independent replay of every generated reply."}

# B/C actual GPU runs and hashes of the retained original files.
settings_bytes = read("docs/lesson-products/results/settings-freeze-v1.json")
settings = json.loads(settings_bytes)
pretest = load("docs/lesson-products/results/pretest-freeze-record-v1.json")
started = load("docs/lesson-products/results/final-report-v1.started.json")
report = load("docs/lesson-products/results/final-report-v1.json")
assert sha(settings_bytes) == pretest["settings_freeze_sha256"] == started["settings_freeze_sha256"] == report["settings_freeze_sha256"]
assert settings["recorded_at"] < pretest["recorded_at"] < started["recorded_at"] < report["recorded_at"]
git_source = subprocess.run(["git", "show", "821a2c717f05ad7b57e08e88cd965e92ff0c1200:docs/lesson-products/results/settings-freeze-v1.json"], cwd=ROOT, capture_output=True, check=True).stdout
assert sha(git_source) == sha(settings_bytes)
RECEIPT["bc_pretest_freeze"] = {"settings_sha256": sha(settings_bytes), "protocol": settings["protocol"],
                                "freeze_recorded_at": settings["recorded_at"], "test_started_at": started["recorded_at"],
                                "test_report_recorded_at": report["recorded_at"],
                                "pretest_git_commit": "821a2c717f05ad7b57e08e88cd965e92ff0c1200", "git_object_matches": True}
RECEIPT["bc_training"] = {}
for product in ["b", "c"]:
    d = load(f"docs/lesson-products/results/initial-{product}-execution.json")
    local = Path("/workspace/tiny-perceptron-vlm/outputs/lesson-products") / product
    original = load(local / "training-receipt.json")
    assert sha((local / "training-receipt.json").read_bytes()) == settings["products"][product]["training-receipt.json"]
    assert original == d["training"]
    assert d["status"] == "completed" and d["gpu_name"] == "NVIDIA L4"
    assert original["weights_changed"] and original["random_initialization"]
    for name in ["model.pt", "config.json", "tokenizer.json"]:
        assert sha(read(local / name)) == settings["products"][product][name]
    RECEIPT["bc_training"][product] = {"run_id": d["run_id"], "revision": d["code_revision"], "gpu": d["gpu_name"],
                                      **{k: original[k] for k in ["random_initialization", "resume_source", "steps_this_attempt", "selected_step", "seed", "weights_changed"]},
                                      "retained_model_and_receipt_match_pretest_freeze": True}

# Execute only the public parser extracted from the inspected source; no neural module import.
source = read("tiny_perceptron/lesson_products/inference.py").decode()
tree = ast.parse(source)
parser = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_state")
namespace = {"re": re}
exec(compile(ast.Module(body=[parser], type_ignores=[]), "actual parse_state", "exec"), namespace)
RECEIPT["parser_probe"] = {}
for state in ["兩盒追蹤;A=2;B=6;事件=A設5,A設1;查詢=A", "三數加法;x=1;y=2;z=3", "卡片進度;申請編號=?", "任意照片與一般聊天"]:
    try:
        answer = namespace["parse_state"](state)
    except ValueError as e:
        answer = {"rejected": str(e)}
    RECEIPT["parser_probe"][state] = answer
read("tiny_perceptron/lesson_products/model.py")

# Pure recomputation of previously stored raw score rows, never a model forward.
raw_path = Path("/workspace/work/reference/omni-technical-19-17-v1/raw/final-report-v1.predictions.jsonl.gz")
archive = read(raw_path)
package = load("docs/lesson-products/results/publication-package-v1.json")
assert sha(archive) == package["files"]["raw/final-report-v1.predictions.jsonl.gz"]["sha256"]
raw_hash = hashlib.sha256()
counts = Counter()
correct = Counter()
families = defaultdict(set)
selected_disagreements = 0
reverse_disagreements = 0
probability_error = 0.0
rows_total = 0
with gzip.open(raw_path, "rb") as handle:
    for line in handle:
        rows_total += 1
        raw_hash.update(line)
        row = json.loads(line)
        if row.get("split") != "test":
            continue
        if row["product"] == "b":
            key = "b5/" + row["task"]
            scores = row["scores_by_depth"]["5"]
            predicted = max(range(len(scores)), key=lambda i: scores[i])
            selected_disagreements += predicted != row["answers_by_depth"]["5"]
            correct[key] += predicted == row["gold"]
        elif row["product"] == "c":
            key = "c/" + row["question"]
            scores = row["raw_scores"]
            candidates = row["candidates"]
            index = min(range(len(scores)), key=lambda i: (-scores[i], candidates[i]))
            predicted = candidates[index]
            selected_disagreements += predicted != row["selected"]
            correct[key] += predicted == row["gold"]
            if row["question"] == "choice":
                reverse_disagreements += predicted != row["reverse_selected"]
            # Positive preselected temperature preserves the argmax. Verify finite normalized probabilities.
            temperature = settings["calibration"]["temperatures"][row["question"]]
            exponentials = [math.exp((s - max(scores)) / temperature) for s in scores]
            probabilities = [value / sum(exponentials) for value in exponentials]
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities)
            probability_error = max(probability_error, abs(sum(probabilities) - 1.0))
        else:
            continue
        counts[key] += 1
        families[key].add(row["family_id"])
assert raw_hash.hexdigest() == report["predictions"]["sha256"]
assert selected_disagreements == 0
assert (correct["b5/boxes"], counts["b5/boxes"]) == (2928, 2956)
assert (correct["b5/sum"], counts["b5/sum"]) == (2, 42)
assert all(correct[f"c/{q}"] == counts[f"c/{q}"] for q in ["choice", "info", "rubric"])
RECEIPT["saved_score_recomputation"] = {"compressed_sha256": sha(archive), "raw_sha256": raw_hash.hexdigest(),
                                         "rows_in_archive": rows_total,
                                         "results": {k: {"correct": correct[k], "count": counts[k], "families": len(families[k]), "accuracy": correct[k] / counts[k]} for k in counts},
                                         "saved_selected_vs_argmax_disagreements": selected_disagreements,
                                         "choice_reverse_disagreements": reverse_disagreements,
                                         "max_probability_sum_error_float64": probability_error,
                                         "event_threshold_0_90_pass": correct["b5/boxes"] / counts["b5/boxes"] >= 0.90,
                                         "addition_target_0_85_pass": correct["b5/sum"] / counts["b5/sum"] >= 0.85,
                                         "scope": "Once-saved 2026-10-16 test rows only; preserves shared-family dependence. No new final run, weights, thresholds or policy."}

RECEIPT["finished_at"] = datetime.now(timezone.utc).isoformat()
with (PROOF / "inspection-recomputation-v1.json").open("x", encoding="utf-8") as f:
    json.dump(RECEIPT, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({k: RECEIPT[k] for k in ["created_at", "finished_at", "source_identity", "bc_training", "saved_score_recomputation", "extension_code"]}, ensure_ascii=False, indent=2))
