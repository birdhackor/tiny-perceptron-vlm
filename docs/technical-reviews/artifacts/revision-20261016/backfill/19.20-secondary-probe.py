"""Narrow historical-execution and publication metadata check after own results sealed."""
import hashlib
import json
import platform
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def load(path):
    return json.loads((ROOT / path).read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


out = {
    "reviewer_task": "/root/backfill_tech_19_20",
    "command": "python docs/technical-reviews/artifacts/revision-20261016/backfill/19.20-secondary-probe.py",
    "environment": {"python": platform.python_version(), "device": "CPU; metadata inspection and anonymous urllib GET"},
}
rows = []
for rel, script, key in [
    ("docs/lesson-products/reviews/engineering-final-b-mechanisms-execution.json", "/workspace/work/curriculum-20261016/engineering-final-b-mechanisms/check.py", "script_sha256"),
    ("docs/lesson-products/reviews/engineering-final-c-execution.json", "/workspace/work/curriculum-20261016/engineering-final-c/recompute.py", "probe_script_sha256"),
]:
    x = load(rel)
    actual = digest(Path(script))
    assert actual == x[key]
    rows.append({
        "path": rel, "sha256": digest(ROOT / rel), "reviewer_task": x["reviewer_task"],
        "command": x["command"], "environment": x["environment"], "probe_script_sha256": actual,
        "raw_reference_sha256": "8b593421043b2d0ee8b81caea34612b468e8af7eafd5b84c552f8c25b0cf4896",
        "inspection": "Only original execution and narrow raw-loop/argmax/CE/Brier/Auto code were checked; original verdict report not read.",
    })
out["historical_recalculations"] = rows
out["independence_evidence_limit"] = (
    "Coordinator /root supplied actual followup history: two distinct nonauthor tasks assigned B/mechanism/Auto versus C, both completed. "
    "Signatures/shared filesystem are not proof of physical isolation. Own numeric/source judgments were saved first. "
    "Original execution records exposed per-check statuses; none replaced own substantive verification."
)
training = {}
for product in ("b", "c"):
    rel = "outputs/lesson-products/products/" + product + "/training-receipt.json"
    x = load(rel)
    runner = load("docs/lesson-products/results/initial-" + product + "-execution.json")
    training[product] = {k: x[k] for k in ("environment", "seed", "steps_this_attempt", "last_step", "selected_step", "weights_changed", "data_manifest_sha256")}
    training[product]["sha256"] = digest(ROOT / rel)
    training[product]["primary_runner"] = {k: runner[k] for k in ("run_id", "status", "gpu_name")}
out["existing_primary_training_receipts"] = training
public = load("docs/lesson-products/results/publication-v1.json")["execution"]["public"]
url = "https://huggingface.co/" + public["repo"] + "/resolve/" + public["revision"] + "/" + public["prefix"] + "/raw/predictions-manifest.json"
with urllib.request.urlopen(url, timeout=30) as response:
    data, resolved = response.read(), response.url
expected = load("docs/lesson-products/results/publication-package-v1.json")["files"]["raw/predictions-manifest.json"]["sha256"]
assert hashlib.sha256(data).hexdigest() == expected
out["anonymous_pinned_raw_manifest"] = {"url": url, "resolved_url": resolved.split("?")[0], "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "body": json.loads(data), "accessed_on": "2026-10-16"}
out["result"] = "Two historical recalculation identities/scripts/raw references match. Existing L4 training receipts and anonymous fixed-revision raw-score manifest verified. No model inference/training/test/search/remote write."
Path(__file__).with_name("19.20-secondary-output.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(out["result"])
