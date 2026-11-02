"""核19.16 immutable來源與固定分數身份；不做任何模型推論或訓練。"""

import gzip
import hashlib
import json
import math
import platform
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = "docs/technical-reviews/artifacts/c-v3-v21-canonical-19-16-fresh"
TASK = "/root/c_v3_technical_coord/canonical_19_16_fresh"
COMMAND = (
    "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv "
    "PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 "
    "uv run --extra cpu --no-sync python " + ART + "/retention_check.py"
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def checked(path, digest):
    raw = (ROOT / path).read_bytes()
    assert sha(raw) == digest, path
    return raw


class PageText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, text):
        if not self.skip:
            self.parts.append(text)


def main():
    original_path = "docs/course-revision-20261101-c-v3/technical-v21/canonical_19_16_fresh/original-v1.json"
    original_sha = "076bb6168d9e27a174b15328e13576830346f25ce4b773c502f8a5d2f6956a26"
    original = json.loads(checked(original_path, original_sha))
    receipt_path = "docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/canonical-19-16-fresh-original-v1.json"
    receipt_sha = "fc95f69c8daab577468c0ef3a55c1c1aaf707157fe37d85b37649cdb3b266b74"
    receipt = json.loads(checked(receipt_path, receipt_sha))
    assert receipt["report"] == {"path": original_path, "sha256": original_sha}
    assert receipt["reviewer_task"] == TASK
    assert datetime.fromisoformat(receipt["received_at"]) < datetime.now(UTC)

    old_path = "docs/course-revision-20261029-omni/reviews/P5-actual/technical-19.16/technical-original-v1.json"
    old = json.loads(checked(old_path, "a80bae3a248a6440d291d9bb1bae1807761657b10180e87bef71158c024f65b2"))
    # The prior report supplies paths/lineage only. Its verdict/status is unused.
    old_artifacts = {x["id"]: x for x in old["artifacts"]}
    for item in old_artifacts.values():
        checked(item["path"], item["sha256"])
    saved = read(old_artifacts["a-execution"]["path"])
    inspected = read(old_artifacts["a-inspection"]["path"])
    identities = []
    for item in old["sources"]:
        if item["kind"] == "repository_code":
            raw = checked(item["path"], item["sha256"])
            identities.append({"path": item["path"], "sha256": item["sha256"], "bytes": len(raw)})

    body = (ROOT / "docs/course-revision-20261101-c-v3/source-freeze-v19/sources/19.16.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", body, re.S)
    assert [sha(block.encode()) for block in blocks] == saved["inline_code_sha256"]
    assert saved["source_sha256"] == original["retention_identity"]["source18_snapshot"]["sha256"]
    assert saved["inline_class_ast_equal"] and saved["inline_library_max_score_difference"] == 0
    assert saved["candidate_reorder_probabilities"][0] == saved["candidate_reorder_probabilities"][1]
    assert saved["boundary_checks"]["padded_candidate_probability"] == 0

    manifest = read("docs/lesson-products/results/c-artifact-manifest.json")
    settings = read("docs/lesson-products/results/settings-freeze-v1.json")
    product_dir = Path("/workspace/work/reference/omni-technical-19-16-v1/product-c")
    product_receipts = {}
    originals = {}
    for name, ref in saved["product_original_receipts"].items():
        raw = (product_dir / name).read_bytes()
        assert sha(raw) == ref["sha256"] == manifest["files"][name]["sha256"] == settings["products"]["c"][name]
        assert len(raw) == ref["bytes"] == manifest["files"][name]["bytes"]
        originals[name] = json.loads(raw)
        product_receipts[name] = {"path": str(product_dir / name), **ref}
    train = originals["training-receipt.json"]
    run = read("docs/lesson-products/results/initial-c-execution.json")
    for key, value in saved["training_receipt_excerpt"].items():
        assert train[key] == value == run["training"][key]
    assert train["seed"] == 9041 and train["last_step"] == train["selected_step"] == train["steps_this_attempt"] == 800
    assert train["environment"]["gpu"] == run["gpu_name"] == "NVIDIA L4"
    points = [x for x in train["history"] if "policy" in x]
    selected = max(points, key=lambda x: (x["policy"]["choice"]["accuracy"], -x["policy"]["choice"]["cross_entropy"]))
    assert selected["step"] == 800
    cal = originals["calibration.json"]
    assert cal == read("docs/lesson-products/results/c-calibration.json")
    assert cal == settings["calibration"]
    assert cal["temperatures"] == {"choice": 0.25, "info": 0.25, "rubric": 0.25}
    for filename, digest in settings["source_sha256s"].items():
        assert sha((ROOT / "tiny_perceptron/lesson_products" / filename).read_bytes()) == digest

    final = read("docs/lesson-products/results/final-report-v1.json")
    retained_metrics = {}
    for key, metrics in saved["recomputed_metrics"].items():
        split, kind = key.split("/")
        target = final["c"][kind] if split == "test" else final["shifts"]["c_wording"][kind]
        assert metrics == target
        retained_metrics[key] = metrics
    assert saved["reverse_selected_record_consistency"]["count"] == saved["reverse_selected_record_consistency"]["consistent"] == 11797
    assert saved["missing_required_candidate_subset"] == final["c_missing_best"] == {"count": 849, "correct": 849}
    assert final["c_reorder"] == {"count": 11797, "consistent": 11797, "consistency": 1.0}

    raw_path = Path("/workspace/work/reference/omni-technical-19-17-v1/raw/final-report-v1.predictions.jsonl.gz")
    compressed = raw_path.read_bytes()
    assert len(compressed) == saved["raw_predictions"]["gzip_bytes"]
    assert sha(compressed) == saved["raw_predictions"]["gzip_sha256"]
    del compressed
    raw_hash = hashlib.sha256()
    raw_bytes = 0
    failure = read("docs/lesson-products/results/observed-failure-examples-v1.json")["examples"][0]
    wanted = failure["id"].encode()
    weather = None
    with gzip.open(raw_path, "rb") as stream:
        for line in stream:
            raw_hash.update(line)
            raw_bytes += len(line)
            if wanted in line:
                candidate = json.loads(line)
                if candidate["product"] == "c" and candidate["split"] == "wording_shift":
                    weather = candidate
    assert raw_hash.hexdigest() == saved["raw_predictions"]["uncompressed_sha256"] == final["predictions"]["sha256"]
    assert raw_bytes == saved["raw_predictions"]["uncompressed_bytes"] == final["predictions"]["bytes"]
    for field in ("id", "candidates", "gold", "raw_scores"):
        assert weather[field] == failure[field]
    logits = [x / 0.25 for x in weather["raw_scores"]]
    weights = [math.exp(x - max(logits)) for x in logits]
    probabilities = [x / sum(weights) for x in weights]
    index = max(range(len(logits)), key=logits.__getitem__)
    concentration = (len(logits) * probabilities[index] - 1) / (len(logits) - 1)
    assert weather["gold"] == "none" and weather["candidates"][index] == "boxes"
    assert math.isclose(concentration, failure["concentration"], abs_tol=1e-14)

    external = []
    for item in inspected["external_originals"]:
        raw = Path(item["full_text_outside_git"]).read_bytes()
        assert len(raw) == item["bytes"] and sha(raw) == item["sha256"]
        if item["url"].startswith("https://docs.typesafe.ai/"):
            parser = PageText()
            parser.feed(raw.decode())
            text = re.sub(r"\s+", " ", " ".join(parser.parts))
            if item["url"].endswith("introduction"):
                start = text.index("Choose an option")
                excerpt = text[start:start + 640]
            else:
                start = text.index("A Noul answer is a single probability")
                excerpt = text[start:start + 1800]
        else:
            lines = raw.decode().splitlines()
            excerpt = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines) if 2182 <= i + 1 <= 2189 or 6381 <= i + 1 <= 6403 or 6487 <= i + 1 <= 6493)
        external.append({"url": item["url"], "version": item["version"], "sha256": item["sha256"], "bytes": len(raw), "archived_full_path": item["full_text_outside_git"], "read_by_current_reviewer_on": "2026-11-01", "original_acquired_on": item["read_date"], "locator": item["locators"], "small_original_excerpt": excerpt})

    result = {
        "created_at": datetime.now(UTC).isoformat(), "reviewer_task": TASK,
        "command": COMMAND, "result": "All immutable identity, source-block, selected-receipt and fixed-record arithmetic assertions passed.",
        "environment": {"python": platform.python_version(), "device": "cpu; file I/O and scalar arithmetic only", "model_execution": "none", "network": "none"},
        "original_intake": {"path": receipt_path, "sha256": receipt_sha, "received_at": receipt["received_at"]},
        "retained_execution": {"path": old_artifacts["a-execution"]["path"], "sha256": old_artifacts["a-execution"]["sha256"], "owner": old["reviewer_task"], "executed_at": saved["created_at"], "command": old_artifacts["a-execution"]["command"], "environment": saved["environment"], "scope": "2026-10-30 original owner's CPU inline demonstration and arithmetic on preserved logits; retained unchanged, not rerun by the current reviewer."},
        "repository_identities": identities,
        "source_code_blocks": {"current_sha256": [sha(block.encode()) for block in blocks], "retained_sha256": saved["inline_code_sha256"], "equal": True},
        "product_original_receipts": product_receipts,
        "training": {"fields": saved["training_receipt_excerpt"], "selected_step_by_policy_rule": selected["step"], "selection_rule": "maximum policy Choice accuracy; tie by lower policy mean CE", "policy_denominators": {kind: selected["policy"][kind]["count"] for kind in ("choice", "info", "rubric")}},
        "calibration": {"temperatures": cal["temperatures"], "grid": cal["grid"], "selected_on": cal["selected_on"], "denominators": {kind: cal["measurements"][kind]["before"]["count"] for kind in ("choice", "info", "rubric")}, "limitation": "Original calibration record and original selection procedure authenticated; complete grid logits are not retained here, so the grid minimization is not newly recomputed."},
        "retained_metrics": retained_metrics,
        "raw_predictions": {"local_path": str(raw_path), **saved["raw_predictions"], "actual_uncompressed_sha256": raw_hash.hexdigest(), "actual_uncompressed_bytes": raw_bytes, "scope": "Only byte identity and one weather record checked in this run; six full aggregate recomputations remain the immutable 2026-10-30 execution."},
        "weather_scalar_arithmetic": {"id": weather["id"], "raw_scores": weather["raw_scores"], "gold": weather["gold"], "selected": weather["candidates"][index], "selected_probability": probabilities[index], "concentration": concentration, "formula": "stable softmax(score/.25); (n*pmax-1)/(n-1)", "neural_forward": False},
        "original_authorities_read": external,
        "heldout_scope": read("docs/lesson-products/data-v2-manifest.json")["audit"]["c"],
        "token_audit": read("docs/lesson-products/data-v2-token-audit.json")["c"],
        "acceptance_contract": settings["protocol"]["c_primary"],
        "limitations": ["No trained weights loaded; no model construction, model inference, training, final opening, full old research or NN17 acceptance rerun.", "Historical hardware and updates are authenticated original observations, not new physical observation.", "Reverse-selected entries and six aggregate metrics are retained old fixed-score calculations; no reverse neural forward or all-dataset audit repeated.", "Official originals are the exact archived 2026-10-30 bytes read now, not a new network fetch or a claim about Jev internals.", "New C-v3 transport, GPU/MPS quality and product acceptance are outside this retention check."]
    }
    output = ROOT / ART / "retention-check.json"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": output.relative_to(ROOT).as_posix(), "sha256": sha(output.read_bytes()), "result": result["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
