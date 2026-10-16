"""5.16 freeze03 有提示複查：只核版本、原始配置與未變證據，不訓練模型。"""

import contextlib
import datetime
import hashlib
import io
import json
import platform
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TASK = "/root/repair_tech_5_16"
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
original_path = ROOT / "docs/course-repair-20261017/reviews/technical-5-16-original-01.json"
assert digest(original_path) == "f7ed4e30b620d1f57d7493b64278448e1e2588bb4b9147e33aa11f2023cc041b"
original = json.loads(original_path.read_text())
freeze_path = ROOT / "docs/course-repair-20261017/freeze-03.json"
assert digest(freeze_path) == "c2651ec9c2b63fed22d68733451a2418282bd8ef3d35733b9690619b17627c89"
freeze = json.loads(freeze_path.read_text())
page = next(page for page in freeze["pages"] if page["page_id"] == "5.16")
old = (ROOT / "docs/course-repair-20261017/freeze-02/sources/5.16.md").read_text()
body = re.search(r"^## 5\.16 .*?(?=^## |\Z)", (ROOT / "course/chapters/05.md").read_text(), re.M | re.S)[0]
snapshot = ROOT / page["snapshot"]
assert body.encode() == snapshot.read_bytes()
assert hashlib.sha256(body.encode()).hexdigest() == page["source_sha256"]
old_sentence = "配方的筆數與步數相同，監督總量仍差近一倍。"
new_sentence = "每批筆數與步數相同，監督總量仍差近一倍。"
assert old.count(old_sentence) == 1
assert body == old.replace(old_sentence, new_sentence)
old_fences = re.findall(r"```.*?```", old, re.S)
new_fences = re.findall(r"```.*?```", body, re.S)
assert new_fences == old_fences
assert not re.search(r"!\[|<img\b|\.svg", body)

artifact_checks = []
for artifact in original["artifacts"]:
    observed = digest(ROOT / artifact["path"])
    assert observed == artifact["sha256"]
    artifact_checks.append({"id": artifact["id"], "path": artifact["path"], "sha256": observed, "same_bytes": True})
source_checks = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        observed = digest(ROOT / source["path"])
        assert observed == source["sha256"]
        source_checks.append({"id": source["id"], "path": source["path"], "sha256": observed, "same_bytes": True})
receipt = json.loads((ROOT / "docs/technical-reviews/artifacts/repair-20261017-5-16/pytorch-source-receipt.json").read_text())
full_doc = Path(receipt["full_source_outside_git"])
assert digest(full_doc) == receipt["full_source_sha256"]

namespace, out = {}, io.StringIO()
with contextlib.redirect_stdout(out):
    exec(re.search(r"```python\n(.*?)```", body, re.S)[1], namespace)
assert namespace["tokens"] == {"短答": 90, "長答": 200}
assert namespace["ratios"]["長答"] == 200 / 290

history_path = ROOT / "docs/course-experiments/results/sft_ablation.json"
assert digest(history_path) == original["original_experiment_artifact"]["sha256"]
history = json.loads(history_path.read_text())
branches = {}
for name in ["b-only", "replay"]:
    branch = history["results"]["runs"][name]
    training = branch["training"]
    assert training["steps"] == 500
    branches[name] = {
        "updates_recorded": training["steps"],
        "batch_size_reinspected_in_fit_lm_default": 16,
        "batch_size_override_in_run_sft_ablation": False,
        "sampled_records_implied_by_updates_times_batch": 500 * 16,
        "unique_pool_records": training["records"],
        "unique_pool_by_source": branch["source_records"],
        "recorded_effective_tokens": training["effective_tokens"],
        "A_eval": {split: {key: values[key] for key in ["matches", "records", "eos_rate"]} for split, values in branch["A_attributes"].items()},
        "B_eval": {split: {key: values[key] for key in ["matches", "records", "eos_rate"]} for split, values in branch["B_arithmetic"].items()},
    }
assert branches["b-only"]["unique_pool_records"] == 49
assert branches["replay"]["unique_pool_records"] == 94
assert branches["b-only"]["recorded_effective_tokens"] == 18453
assert branches["replay"]["recorded_effective_tokens"] == 36384
print(json.dumps({
    "reviewer_task": TASK,
    "current_context": "informed_original_owner_source_revision_callback",
    "completed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "environment": {"python": platform.python_version(), "device": "cpu", "platform": platform.platform()},
    "original_report": {"path": str(original_path.relative_to(ROOT)), "sha256": digest(original_path), "unchanged": True},
    "freeze03": {"path": str(freeze_path.relative_to(ROOT)), "sha256": digest(freeze_path), "own_page": page},
    "current_source_sha256": hashlib.sha256(body.encode()).hexdigest(),
    "source_matches_freeze03_bytes": True,
    "only_change": {"old": old_sentence, "new": new_sentence, "exact_single_replacement": True},
    "code_fences_unchanged": True,
    "current_code_stdout": out.getvalue(),
    "reused_artifact_hash_checks": artifact_checks,
    "reused_repository_source_hash_checks": source_checks,
    "reused_official_doc_full_snapshot_sha256": digest(full_doc),
    "original_experiment_sha256": digest(history_path),
    "original_experiment_branch_reinspection": branches,
    "original_limit": history["results"]["matched_budget"],
    "historical_model_or_training_rerun": False,
    "prior_microchecks_rerun": False,
    "reuse_scope": "Old a-run remains a freeze02 execution artifact. Its numerical, mask/loss and sampling evidence remains applicable because all substantive math/code, implementation hashes and original result bytes are unchanged; only the quoted wording changed. Original source identity inside old artifacts is retained, not rewritten.",
}, ensure_ascii=False, indent=2))
