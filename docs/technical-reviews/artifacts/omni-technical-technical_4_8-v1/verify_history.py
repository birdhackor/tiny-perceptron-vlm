"""只讀歷史公開權重、原始紀錄及來源鏈；不推論、不訓練。"""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


base = ROOT / "checkpoints/course/text_foundation"
record_path = ROOT / "docs/course-experiments/results/text_foundation.json"
release_path = ROOT / "docs/course-experiments/public-releases/text_foundation.json"
record = json.loads(record_path.read_text())
release = json.loads(release_path.read_text())["release"]
export = json.loads((base / "export-manifest.json").read_text())
download = json.loads((base / "download-manifest.json").read_text())
public = {x["output"]: x for x in release["public_manifest"]["files"]}
source_artifacts = {x["path"]: x for x in record["artifacts"]}
exports = {x["output"]: x for x in export["files"]}
assert export["provenance"]["revision"] == record["revision"]
assert download["revision"] == release["weights_revision"]
verified = []
for name in ["model.pt", "start.pt", "export-manifest.json", "download-manifest.json"]:
    path = base / name
    local_digest = digest(path)
    assert local_digest == public[name]["sha256"]
    row = {"name": name, "local_path": str(path.relative_to(ROOT)), "sha256": local_digest,
           "bytes": path.stat().st_size, "public_manifest_matches": True}
    if name in exports:
        assert exports[name]["sha256"] == local_digest
        assert exports[name]["source_sha256"] == source_artifacts[name]["sha256"]
        row["private_source_sha256"] = exports[name]["source_sha256"]
    url = "https://huggingface.co/" + release["repo"] + "/resolve/" + release["revision"] + "/" + public[name]["path"]
    data = urlopen(url, timeout=45).read()
    assert hashlib.sha256(data).hexdigest() == local_digest
    row.update({"url": url, "remote_digest_matches": True, "accessed_at": datetime.now(UTC).isoformat()})
    verified.append(row)

start = torch.load(base / "start.pt", weights_only=True, map_location="cpu")
model = torch.load(base / "model.pt", weights_only=True, map_location="cpu")
assert start["config"] == model["config"]
assert model["config"]["width"] == 64 and model["config"]["layers"] == 2
assert model["config"]["vocab_size"] == 264 and model["config"]["max_length"] == 128
assert model["metadata"]["revision"] == record["revision"]
assert start["model"].keys() == model["model"].keys()
count = sum(t.numel() for t in model["model"].values())
changed = sum(int((t != start["model"][name]).sum()) for name, t in model["model"].items())
assert count == 141568 and changed > 0
historical_files = []
cache = Path("/workspace/work/reference/omni-technical-technical_4_8-v1")
for filename, repo_path in [("historical-text.py", "scripts/course_experiments/text.py"),
                            ("historical-common.py", "scripts/course_experiments/common.py")]:
    path = cache / filename
    assert digest(path) == record["code_sha256"][repo_path]
    historical_files.append({"path": str(path), "sha256": digest(path), "repository_path": repo_path,
                             "git_revision": record["revision"]})
for repo_path in ["tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py", "tiny_perceptron/data.py"]:
    assert digest(ROOT / repo_path) == record["code_sha256"][repo_path]

print(json.dumps({
    "executed_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_4_8",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "verified_public_files": verified,
    "source_record": {"path": str(record_path.relative_to(ROOT)), "sha256": digest(record_path)},
    "public_release_record": {"path": str(release_path.relative_to(ROOT)), "sha256": digest(release_path)},
    "historical_code": historical_files,
    "config": model["config"],
    "tensor_count": len(model["model"]), "parameter_count": count,
    "changed_parameters_against_start": changed,
    "max_abs_parameter_change": max(float((t - start["model"][name]).abs().max()) for name, t in model["model"].items()),
    "recorded_denominators": {"seed": record["seed"], **{k: record["results"]["training"][k] for k in ["records", "steps", "effective_tokens"]}},
    "new_training_updates": 0, "new_model_inferences": 0,
    "limitation": "私有原訓練 checkpoint URL 回應 401。此處驗證公開匯出 bytes、來源 SHA 鏈、設定、參數數及起點差異；公開檔無 optimizer/step，未獨立重放 600 次更新、未驗品質。",
    "result": "公開 pinned 權重／manifest 摘要與歷史設定、141568 參數及權重變更相符。",
}, ensure_ascii=False, indent=2))
