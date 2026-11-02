"""核對已取得的官方原件／固定公開 metadata；完整外部原件只留 Git 外。"""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path("/workspace/work/reference/omni-technical-t-11-v1")
OUT = Path(__file__).with_name("source-receipts-v1.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


downloaded = []
for version in [1, 2, 3, 4]:
    receipt = json.loads((CACHE / f"download-receipts-v{version}.json").read_text())
    for item in receipt if isinstance(receipt, list) else [receipt]:
        assert "error" not in item
        path = Path(item.get("file", item.get("path")))
        assert digest(path) == item["sha256"] and path.stat().st_size == item["bytes"]
        downloaded.append({"url": item["url"], "cache": record(path),
                           "accessed_at": item["accessed_at"],
                           "inspection_method": "urllib.request.urlopen GET; source sections read independently",
                           "complete_original": path.name != "moe-inference-manifest.json"})

release = json.loads((ROOT / "tiny_perceptron/lesson_products/release.json").read_text())
package = json.loads((ROOT / "docs/lesson-products/results/publication-package-v1.json").read_text())
for name in ["b", "c"]:
    manifest_path = CACHE / f"{name}-artifact-manifest.json"
    assert digest(manifest_path) == release["products"][name]["manifest_sha256"]
    expected = package["files"][f"{name}/training-receipt.json"]
    receipt = CACHE / f"{name}-training-receipt.json"
    assert digest(receipt) == expected["sha256"] and receipt.stat().st_size == expected["bytes"]

moe_manifest_path = CACHE / "moe-inference-manifest-complete-v1.json"
assert digest(moe_manifest_path) == "f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e"
moe_manifest = json.loads(moe_manifest_path.read_text())
assert moe_manifest["selected_step"] == 1000
assert moe_manifest["origin"]["kind"] == "all-neural-weights-random"
heads = json.loads((CACHE / "weights-head-v1.json").read_text())
assert all(r["status"] == 200 for r in heads)
assert {r["name"]: r["content_length"] for r in heads} == {"moe": "22366908", "b": "685437", "c": "346625"}

training_path = ROOT / "docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json"
execution_path = ROOT / "docs/selftrained/results/training-raw/moe-native/raw/execution.json"
training = json.loads(training_path.read_text())
execution = json.loads(execution_path.read_text())
assert training["steps"] == 4000 and training["completed_requested_steps"] is True
assert training["interrupted"] is False and execution["status"] == "completed" and execution["returncode"] == 0
assert training["origin"]["kind"] == "all-neural-weights-random"
manifest_path = ROOT / "docs/selftrained/v2-manifest.json"
manifest = json.loads(manifest_path.read_text())
pointer = subprocess.check_output(["git", "show", "HEAD:assets/training/selftrained-v2.tar.gz"], cwd=ROOT, text=True)
assert f"oid sha256:{manifest['package']['sha256']}\n" in pointer
assert f"size {manifest['package']['bytes']}\n" in pointer

historical_code = CACHE / "historical-applications-910aebc.py"
assert digest(historical_code) == "8fbb7e1c1a5ee33c50884a3cb6658352aacd80d2131ac6561c272e8c75b2ade0"
assert '"messages": messages,' in historical_code.read_text()
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-t-11-v1/source-checks-v1.py",
    "environment": {"python": sys.version.split()[0], "device": "cpu; metadata/source only"},
    "external_originals": downloaded,
    "weights_head": heads,
    "fixed_release": {"release_config": record(ROOT / "tiny_perceptron/lesson_products/release.json"),
        "b_c_revision": release["revision"], "manifests_and_training_receipt_hashes_equal_public_package": True,
        "moe_manifest": record(moe_manifest_path), "moe_selected_step": 1000,
        "moe_weight_sha256": moe_manifest["files"]["model.safetensors"]},
    "moe_historical_training": {"receipt": record(training_path), "execution": record(execution_path),
        "steps": training["steps"], "seed": training["seed"], "architecture": training["architecture"],
        "origin_kind": training["origin"]["kind"], "completed_requested_steps": True,
        "selection": training["selection"], "config": training["config"],
        "test_used_for_selection": training["test_used_for_selection"], "source_revision": execution["revision"]},
    "fixed_lfs_package": {"manifest": record(manifest_path), "package": manifest["package"],
        "pointer": pointer, "records": len(manifest["records"]), "assets": len(manifest["assets"]),
        "scope": "只核 manifest 與 Git LFS pointer，未下載／解包67MB資料，也未重做完整訓練"},
    "historical_tool_source": {"cache": record(historical_code),
        "url": "https://github.com/birdhackor/tiny-perceptron-vlm/blob/910aebc6419c9fc6217279a27fde9851c5cfad30/scripts/course_experiments/applications.py",
        "locator": "_sample lines 42–103; _tool_episode lines 527–576",
        "quote": "\"messages\": messages,\n\"input_ids\": prompt,",
        "inspection_note": "舊版直接存同一messages列表，_tool_episode稍後用+=回填；prompt則是先建立的獨立IDs列表。"},
    "limitations": "HEAD僅驗權重存在及大小；沒有下載權重、重跑歷史模型或測任何新模型能力。初次1MiB限長MoE metadata原件保留但標記不完整，只採用另存完整且SHA吻合的原件。"
}
with OUT.open("x", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"result": str(OUT.relative_to(ROOT)), "sha256": digest(OUT),
                  "official_or_pinned_originals": len(downloaded), "weight_heads_ok": len(heads)}, ensure_ascii=False))
