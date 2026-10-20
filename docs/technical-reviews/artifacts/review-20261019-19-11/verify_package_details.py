"""以公開檔頭與固定資料包核對完整權重和已知驗證示例，不保存權重或資料全文。"""

import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tarfile
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parent
ROOT = ARTIFACT.parents[3]
sys.path.insert(0, str(ROOT))
REVISION = "979cdfacc588ad0536f1c64fff96f264571cf054"
REPO = "birdhackor/tiny-perceptron-course-models"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url, byte_range=None):
    args = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "60"]
    if byte_range:
        args += ["--range", byte_range]
    args.append(url)
    return subprocess.run(args, capture_output=True, check=True).stdout, args


def main():
    import torch
    from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig

    result = {"environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu"},
              "public_headers": {}}
    for variant in ("moe-pretrain", "moe-sft", "moe-joint", "dense-joint"):
        prefix = f"https://huggingface.co/{REPO}/resolve/{REVISION}/selftrained/v2/{variant}/"
        config_raw, _ = fetch(prefix + "model-config.json")
        weights, command = fetch(prefix + "model.safetensors", "0-131071")
        length = struct.unpack("<Q", weights[:8])[0]
        if length + 8 > len(weights):
            raise AssertionError("公開 tensor 檔頭長於已取得範圍")
        header = json.loads(weights[8:8+length])
        metadata = header.pop("__metadata__", {})
        model = LimitedAssistant(SelftrainedConfig(**json.loads(config_raw)))
        expected = model.state_dict()
        result["public_headers"][variant] = {
            "command": command, "received_bytes": len(weights), "received_sha256": sha(weights),
            "header_bytes": length, "tensor_count": len(header), "metadata": metadata,
            "same_keys_as_complete_model": set(header) == set(expected),
            "same_shapes": all(header[name]["shape"] == list(value.shape) for name, value in expected.items()),
            "all_dtypes_f32": all(value["dtype"] == "F32" for value in header.values()),
            "expected_model_dtypes_f32": all(value.dtype == torch.float32 for value in expected.values()),
            "tensor_keys": sorted(header),
        }
        del model, expected, weights

    manifest = json.loads((ROOT / "docs/selftrained/v2-manifest.json").read_bytes())
    package = manifest["package"]
    url = f"https://media.githubusercontent.com/media/{package['repo_id']}/{package['revision']}/{package['path']}"
    raw, command = fetch(url)
    if sha(raw) != package["sha256"] or len(raw) != package["bytes"]:
        raise AssertionError("資料包指紋不符")
    records = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for item in manifest["records"]:
            content = archive.extractfile(item["path"]).read()
            if sha(content) != item["sha256"] or len(content) != item["bytes"]:
                raise AssertionError("固定記錄檔指紋不符")
            records.extend(json.loads(line) for line in content.decode().splitlines() if line)
    del raw
    matches = {}
    example_assets = {
        "ocr": "images/ocr/3a8d1184bb954b6f30c6fc09b31c678e.png",
        "vision_clothing": "images/vision/c98cb897d639c880fde261eafc6d953d.png",
        "vision_relation": "images/vision/13913fb756b87b5c3e0f52e4ba194fec.png",
        "voice_qa": "audio/ef25d9a7a3a6ce3790a040ba.wav",
        "voice_topic_continuation": "audio/ef25d9a7a3a6ce3790a040ba.wav",
    }
    strip_metadata = lambda messages: [{"role": m["role"], "content": m["content"]} for m in messages]
    followup = json.loads((ROOT / "docs/selftrained/examples/v2/voice-continuation-user.json").read_bytes())
    for path in sorted((ROOT / "docs/selftrained/examples/v2").glob("*.messages.json")):
        messages = json.loads(path.read_bytes())
        task = path.name.removesuffix(".messages.json")
        found = [{"id": row["id"], "split": row["split"], "task": row["task"]}
                 for row in records if row["task"] == task
                 and strip_metadata(row["messages"][:len(messages)]) == strip_metadata(messages)
                 and (task != "voice_topic_continuation" or row["messages"][-2] == followup)
                 and (task not in example_assets or any(
                     source.get("image") == example_assets[task] or source.get("audio") == example_assets[task]
                     for source in [row, *row["messages"]]))]
        matches[path.name] = found
    result["known_examples"] = {"package_command": command, "package_sha256": package["sha256"],
                                "verified_record_files": len(manifest["records"]), "matches": matches}

    result["dense_origin"] = {}
    for variant in ("dense-pretrain", "dense-joint"):
        path = ROOT / f"docs/selftrained/results/training-raw/{variant}/raw/train-receipt.json"
        data = json.loads(path.read_bytes())
        result["dense_origin"][variant] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path.read_bytes()),
                                           **{key: data[key] for key in ("architecture", "stage", "origin", "stage_history", "steps")}}
    cache = ARTIFACT / "install-dry-run-cache"
    args = ["uv", "sync", "--frozen", "--extra", "cpu", "--extra", "selftrained", "--dry-run"]
    completed = subprocess.run(args, cwd=ROOT, env={**os.environ, "UV_CACHE_DIR": str(cache)}, capture_output=True, text=True)
    result["install_dry_run"] = {"command": args, "returncode": completed.returncode,
                                 "stdout": completed.stdout, "stderr": completed.stderr}
    if cache.exists():
        shutil.rmtree(cache)
    target = ARTIFACT / "package-details-output.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"公開完整權重鍵一致": {key: value["same_keys_as_complete_model"] for key, value in result["public_headers"].items()},
                      "範例資料切分": {key: sorted(set(row["split"] for row in values)) for key, values in matches.items()},
                      "安裝預演退出碼": completed.returncode}, ensure_ascii=False))


if __name__ == "__main__":
    main()
