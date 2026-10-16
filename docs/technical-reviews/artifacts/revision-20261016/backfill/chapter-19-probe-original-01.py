"""Read-only checks for the frozen chapter-19 introduction; no training/test rerun."""

import collections
import hashlib
import json
import platform
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch

from tiny_perceptron.lesson_products.inference import Product


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


result = {
    "reviewer_task": "/root/backfill_tech_chapter_19",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/backfill/chapter-19-probe-original-01.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "scope": "Read existing original receipts and raw prediction rows, check SHA identities, make one CPU call per product on one training input, anonymously read the already published A weights. No optimizer, training, formal test or publication action.",
}
source_paths = [
    "docs/course-revision-20261016/freeze-15.json",
    "docs/course-revision-20261016/freeze-10/sources/chapter-19.md",
    "tiny_perceptron/selftrained/model.py",
    "tiny_perceptron/model.py",
    "tiny_perceptron/modern.py",
    "scripts/selftrained/train.py",
    "tiny_perceptron/lesson_products/model.py",
    "tiny_perceptron/lesson_products/inference.py",
    "tiny_perceptron/lesson_products/evaluation.py",
    "docs/selftrained/results/v2-final-public-results.json",
    "docs/selftrained/results/training-raw/moe-native/receipt.json",
    "docs/selftrained/provenance/vision-sources.json",
    "docs/selftrained/provenance/ocr-sources.json",
    "docs/selftrained/provenance/voice.json",
    "docs/lesson-products/results/initial-b-execution.json",
    "docs/lesson-products/results/initial-c-execution.json",
    "docs/lesson-products/results/settings-freeze-v1.json",
    "docs/lesson-products/results/pretest-freeze-record-v1.json",
    "docs/lesson-products/results/settings-freeze-v1.test-started.json",
    "docs/lesson-products/results/final-report-v1.started.json",
    "docs/lesson-products/results/final-report-v1.json",
]
freeze = read(source_paths[0])
page = next(x for x in freeze["pages"] if x["page_id"] == "chapter-19")
assert sha(ROOT / page["snapshot"]) == page["source_sha256"]
result["frozen_page"] = page
result["navigation"] = []
# Only inspect target headings, never other reviews or author plans.
for file, sections in [("07.md", ["7.1", "7.2", "7.3"]), ("11.md", []), ("12.md", []), ("15.md", ["15.4", "15.5"]), ("19.md", [f"19.{i}" for i in range(1, 21)]), ("20.md", [])]:
    path = ROOT / "course/chapters" / file
    headings = [line for line in path.read_text().splitlines() if line.startswith("## ")]
    for section in sections:
        target = next(line for line in headings if line.startswith("## " + section + " "))
        result["navigation"].append({"file": file, "section": section, "heading": target})
    assert path.exists()

result["a_history"] = []
for stage in ["pretrain", "sft", "vision", "ocr", "audio", "joint", "weighted", "native"]:
    receipt_path = f"docs/selftrained/results/training-raw/moe-{stage}/raw/train-receipt.json"
    execution_path = f"docs/selftrained/results/training-raw/moe-{stage}/raw/execution.json"
    source_paths += [receipt_path, execution_path]
    r, e = read(receipt_path), read(execution_path)
    assert r["origin"]["kind"] == "all-neural-weights-random"
    assert r["completed_requested_steps"] and not r["interrupted"]
    assert e["returncode"] == 0 and e["status"] == "completed"
    if stage == "pretrain":
        assert "--init-checkpoint" not in e["command"] and "--resume" not in e["command"]
    result["a_history"].append({
        "receipt_path": receipt_path, "execution_path": execution_path,
        "run_id": e["run_id"], "revision": e["revision"], "steps": r["steps"],
        "trainable_parameters": r["trainable_parameters"], "origin_kind": r["origin"]["kind"],
        "stage_history": [{k: s[k] for k in ["stage", "step", "checkpoint_sha256"]} for s in r["stage_history"]],
        "completion": e["status"], "initial_checkpoint": e["command"][e["command"].index("--init-checkpoint") + 1] if "--init-checkpoint" in e["command"] else None,
    })
result["a_historical_code_identity"] = []
for rev in [result["a_history"][0]["revision"], result["a_history"][-1]["revision"]]:
    for path in ["tiny_perceptron/selftrained/model.py", "tiny_perceptron/model.py", "tiny_perceptron/modern.py", "scripts/selftrained/train.py"]:
        raw = subprocess.check_output(["git", "show", rev + ":" + path], cwd=ROOT)
        result["a_historical_code_identity"].append({"revision": rev, "path": path, "sha256": hashlib.sha256(raw).hexdigest(), "same_current": raw == (ROOT / path).read_bytes()})
        if "model.py" in path or "modern.py" in path:
            assert raw == (ROOT / path).read_bytes()

public = read("docs/selftrained/results/v2-final-public-results.json")
native = read("docs/selftrained/results/training-raw/moe-native/receipt.json")
moe = public["architectures"]["moe"]
assert moe["selected_source"]["run_id"] == native["run_id"]
assert moe["selected_source"]["sha256"] == next(x["sha256"] for x in native["files"] if x["path"] == "best.pt")
assert moe["safe_weights"]["sha256"] == next(x["sha256"] for x in native["files"] if x["path"] == "model.safetensors")
url = f"https://huggingface.co/{public['hf_repo']}/resolve/{public['hf_revision']}/selftrained/v2/moe-joint/model.safetensors"
digest = hashlib.sha256()
size = 0
with urllib.request.urlopen(url, timeout=45) as response:
    status = response.status
    while chunk := response.read(1024 * 1024):
        size += len(chunk)
        digest.update(chunk)
assert digest.hexdigest() == moe["safe_weights"]["sha256"] and size == moe["safe_weights"]["bytes"]
result["a_public_read"] = {"url": url, "anonymous": True, "http_status": status, "bytes": size, "sha256": digest.hexdigest(), "expected_sha256": moe["safe_weights"]["sha256"], "selected_checkpoint_sha256": moe["selected_source"]["sha256"]}
result["a_data_sources"] = {}
for key, file in [("images", "vision-sources.json"), ("fonts", "ocr-sources.json"), ("recordings", "voice.json")]:
    x = read("docs/selftrained/provenance/" + file)
    result["a_data_sources"][key] = {k: x[k] for k in ["source_revision", "attribution", "dataset", "config", "revision", "author", "inference_contract", "audio_processing"] if k in x}
    if "downloads" in x:
        result["a_data_sources"][key]["download_sources"] = [{"url": d["url"], "sha256": d["sha256"]} for d in x["downloads"]]
    elif "parquet" in x:
        result["a_data_sources"][key]["parquet"] = x["parquet"]

settings_path = "docs/lesson-products/results/settings-freeze-v1.json"
settings, final = read(settings_path), read("docs/lesson-products/results/final-report-v1.json")
assert final["settings_freeze_sha256"] == sha(ROOT / settings_path)
assert settings["test_outcomes_observed"] is False
assert final["data_manifest_sha256"] == settings["data_manifest_sha256"]
assert sha(ROOT / "outputs/lesson-products/data-v2/manifest.json") == settings["data_manifest_sha256"]
result["bc_gpu_training"] = []
for product in ["b", "c"]:
    execution = read(f"docs/lesson-products/results/initial-{product}-execution.json")
    training = execution["training"]
    assert execution["status"] == "completed" and training["weights_changed"]
    assert training["random_initialization"] is True and training["environment"]["device"] == "cuda"
    result["bc_gpu_training"].append({"run_id": execution["run_id"], "gpu_name": execution["gpu_name"], **{k: training[k] for k in ["product", "random_initialization", "last_step", "selected_step", "seed", "batch_size", "initial_weights_sha256", "selected_weights_sha256", "weights_changed", "environment"]}})
    for name, expected in settings["products"][product].items():
        assert sha(ROOT / f"outputs/lesson-products/products/{product}/{name}") == expected
    assert sha(ROOT / f"outputs/lesson-products/products/{product}/model.pt") == execution["client_artifacts"]["model.pt"]["sha256"]

result["final_test_identity"] = {"settings_sha256": sha(ROOT / settings_path), "data_manifest_sha256": settings["data_manifest_sha256"], "pretest_recorded_at": read("docs/lesson-products/results/pretest-freeze-record-v1.json")["recorded_at"], "started_at": read("docs/lesson-products/results/final-report-v1.started.json")["recorded_at"], "completed_at": final["recorded_at"], "scope": final["scope"]}
predictions_path = ROOT / "outputs/lesson-products" / final["predictions"]["file"]
assert sha(predictions_path) == final["predictions"]["sha256"]
assert predictions_path.stat().st_size == final["predictions"]["bytes"]
counts = collections.Counter()
for line in predictions_path.open():
    x = json.loads(line)
    counts[x["product"]] += 1
result["final_test_identity"]["raw_predictions"] = {"path": str(predictions_path.relative_to(ROOT)), "sha256": sha(predictions_path), "bytes": predictions_path.stat().st_size, "row_counts_by_product": dict(counts)}
assert counts["b"] > 0 and counts["c"] > 0

torch.set_num_threads(1)
state = next(json.loads(line)["state"] for line in (ROOT / "outputs/lesson-products/data-v2/b.jsonl").open() if json.loads(line)["split"] == "train")
b = Product(ROOT / "outputs/lesson-products/products/b", "b")
c = Product(ROOT / "outputs/lesson-products/products/c", "c")
b_output, c_output = b.solve(state, 3), c.judge(state)
assert set(c_output["probabilities"]) == set(c_output["meanings"])
assert abs(sum(c_output["probabilities"].values()) - 1) < 1e-6
result["bc_single_cpu_input"] = {"input_origin": "one training row, no test rerun", "state": state, "b_class": type(b.model).__name__, "c_class": type(c.model).__name__, "b_output": b_output, "c_output": c_output, "scope": "API/data-path availability only; no new accuracy or speed inference."}
result["source_sha256"] = {p: sha(ROOT / p) for p in source_paths}
out = Path(__file__).with_name("chapter-19-probe-original-01.json")
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": "passed", "output": str(out.relative_to(ROOT)), "public_weight_bytes": size, "prediction_row_counts": dict(counts), "gpu_training_steps": [x["last_step"] for x in result["bc_gpu_training"]]}, ensure_ascii=False))
