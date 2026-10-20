"""核對公開推論包、版本一致性與本機訓練狀態；只在本證據目錄寫入。"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parent
ROOT = ARTIFACT.parents[3]
sys.path.insert(0, str(ROOT))
SCRATCH = ARTIFACT / "transient-work"
REVISION = "979cdfacc588ad0536f1c64fff96f264571cf054"
REPO = "birdhackor/tiny-perceptron-course-models"
MANIFEST_SHA = "f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e"
RESULT = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    (ARTIFACT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def curl(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "60", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    return completed.stdout, command


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def external_sources():
    specs = [
        ("safetensors", "https://raw.githubusercontent.com/huggingface/safetensors/v0.8.0/README.md",
         "v0.8.0", ["N bytes: a JSON UTF-8 string representing the header.", "Rest of the file: byte-buffer."],
         "Format 與 Safe 的定義：JSON 檔頭以 tensor 名稱為鍵，含 dtype、shape、data_offsets；metadata 限字串；資料區為連續位元組。安全指沒有反序列化任意程式的格式機制，不代表內容可信或排除解析器漏洞。"),
        ("adamw", "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/optim/adamw.py",
         "v2.14.1", ["class AdamW(Adam):", "exp_avgs: list[Tensor]", "exp_avg_sqs: list[Tensor]"],
         "AdamW.__doc__ 的更新式保存一階 m 與二階 v，兩者依前一步狀態與梯度更新；函式也接受 state_steps，權重本身不能決定這些歷史狀態。"),
        ("json", "https://raw.githubusercontent.com/python/cpython/v3.13.16/Lib/json/decoder.py",
         "v3.13.16", ["class JSONDecoder(object):", "def raw_decode(self, s, idx=0):"],
         "JSONDecoder 與 raw_decode／scan_once 將 JSON 字串解析為資料型別；本案的 object_pairs_hook 只檢查重複鍵，JSON 內容不能指定執行函式；字表使用預設 json.loads。"),
        ("hub", "https://raw.githubusercontent.com/huggingface/huggingface_hub/v1.33.0/src/huggingface_hub/file_download.py",
         "v1.33.0", ["def hf_hub_download("],
         "hf_hub_download 的 revision 參數接受 branch、tag 或 commit；commit 固定 snapshot，token=False 關閉驗證。本文使用完整 commit；無法由 commit 保證服務永遠可用。"),
    ]
    snapshots = []
    for identifier, url, version, excerpts, note in specs:
        raw, command = curl(url)
        text = raw.decode()
        for excerpt in excerpts:
            if excerpt not in text:
                raise AssertionError("來源摘錄未出現在原文")
        snapshots.append({"id": identifier, "url": url, "version": version,
                          "accessed_on": datetime.now(timezone.utc).date().isoformat(),
                          "full_document_sha256": sha(raw), "full_document_bytes": len(raw),
                          "command": command, "excerpts": excerpts, "inspection_note": note})
    save("original-source-excerpts.json", snapshots)
    return snapshots


def repository_checks():
    paths = ["scripts/selftrained/chat.py", "uv.lock"]
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "tiny_perceptron/selftrained").glob("*.py"))]
    paths += ["tiny_perceptron/" + name + ".py" for name in ("model", "modern", "multimodal", "attention", "data")]
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "docs/selftrained/examples/v2").rglob("*")) if p.is_file()]
    comparison = []
    for path in paths:
        tag = git("show", f"selftrained-v2:{path}")
        head = git("show", f"HEAD:{path}")
        comparison.append({"path": path, "tag_sha256": sha(tag), "head_sha256": sha(head),
                           "working_sha256": sha((ROOT / path).read_bytes()),
                           "same": tag == head == (ROOT / path).read_bytes()})
    relevant = [row for row in comparison if row["path"].startswith("tiny_perceptron/selftrained/")
                or row["path"] in ("tiny_perceptron/model.py", "tiny_perceptron/modern.py", "tiny_perceptron/multimodal.py",
                                    "tiny_perceptron/attention.py", "tiny_perceptron/data.py")
                or not row["path"].startswith("tiny_perceptron/")]
    bootstrap = (ROOT / "scripts/build_course.py").read_text().split('BOOTSTRAP = """', 1)[1].split('"""', 1)[0]
    return {"head": git("rev-parse", "HEAD").decode().strip(),
            "tag": git("rev-parse", "selftrained-v2").decode().strip(),
            "command": "git show selftrained-v2:<path>; git show HEAD:<path>; 比對工作樹 bytes",
            "relevant_comparison": relevant,
            "all_inference_examples_lock_same": all(row["same"] for row in relevant),
            "colab_bootstrap": bootstrap,
            "section_python_code_blocks": (ROOT / "docs/course-review-20261019/freeze-01/sources/19.11.md").read_text().count("```python")}


def public_checks():
    tree_url = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}/selftrained/v2?recursive=true&expand=false"
    raw, command = curl(tree_url)
    tree = json.loads(raw)
    result = {"tree_url": tree_url, "tree_command": command, "tree_sha256": sha(raw), "exports": {}}
    names = ("inference-manifest.json", "model.safetensors", "model-config.json", "tokenizer.json")
    for variant in ("moe-pretrain", "moe-sft", "moe-joint", "dense-joint"):
        prefix = f"selftrained/v2/{variant}"
        entries = [item for item in tree if item["path"].startswith(prefix + "/") and item["type"] == "file"]
        actual_names = sorted(Path(item["path"]).name for item in entries)
        if actual_names != sorted(names):
            raise AssertionError("公開目錄的檔案清單不符")
        content = {}
        for name in ("inference-manifest.json", "model-config.json", "tokenizer.json"):
            payload, _ = curl(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{prefix}/{name}")
            content[name] = {"sha256": sha(payload), "bytes": len(payload), "json": json.loads(payload)}
        full_manifest = content["inference-manifest.json"]["json"]
        for key in ("asset_sha256", "data_sha256"):
            mapping = full_manifest.pop(key, {})
            full_manifest[key + "_summary"] = {"count": len(mapping),
                                                "mapping_sha256": sha(json.dumps(mapping, sort_keys=True).encode())}
        manifest = content["inference-manifest.json"]["json"]
        weights = next(item for item in entries if item["path"].endswith("model.safetensors"))
        result["exports"][variant] = {"files": actual_names, "payloads": content,
                                       "weights_bytes": weights["size"],
                                       "weights_lfs_sha256": weights.get("lfs", {}).get("oid"),
                                       "optimizer_rng_sampler_files_present": any("optimizer" in name or name.endswith(".pt") for name in actual_names),
                                       "manifest_payload_hashes_match": all(content[name]["sha256"] == manifest["files"][name] for name in ("model-config.json", "tokenizer.json"))}
    return result


def run_cli():
    model_dir = SCRATCH / "public-model"
    env = dict(os.environ, HF_HOME=str(SCRATCH / "hf-cache"), XDG_CACHE_HOME=str(SCRATCH / "cache"),
               HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    args = [str(ROOT / ".venv/bin/python"), "scripts/selftrained/chat.py", "--model-dir", str(model_dir),
            "--asset-dir", str(SCRATCH / "nonexistent-assets"), "--repo", REPO, "--revision", REVISION,
            "--prefix", "selftrained/v2/moe-joint", "--manifest-sha256", MANIFEST_SHA,
            "--messages", "docs/selftrained/examples/v2/text.messages.json", "--task", "text", "--device", "cpu",
            "--max-new-tokens", "128", "--threads", "2"]
    first = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
    output = {"command": args, "environment_overrides": {key: env[key] for key in ("HF_HOME", "XDG_CACHE_HOME", "HF_HUB_DISABLE_XET")},
              "returncode": first.returncode, "stderr": first.stderr}
    if first.returncode:
        output["stdout"] = first.stdout
        return output
    output["first"] = json.loads(first.stdout)
    repeated = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
    output["repeat"] = {"returncode": repeated.returncode, "same_answer": json.loads(repeated.stdout)["answer"] == output["first"]["answer"]}
    target = model_dir / "tokenizer.json"
    previous = target.read_bytes()
    target.write_bytes(previous + b" ")
    conflict = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
    output["conflict"] = {"returncode": conflict.returncode, "stderr": conflict.stderr,
                          "local_file_preserved": target.read_bytes() == previous + b" "}
    target.write_bytes(previous)
    from safetensors import safe_open
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as archive:
        output["safetensors"] = {"keys": list(archive.keys()), "metadata": archive.metadata()}
    return output


def equal_state(a, b):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def local_checks():
    import torch
    from scripts.selftrained import train, train_local_stage

    data = SCRATCH / "synthetic-data"
    data.mkdir()
    rows = [{"id": f"{split}-{index}", "group_id": f"{split}-{index}", "split": split,
             "task": "text", "messages": [{"role": "user", "content": "問題"},
                                                {"role": "assistant", "content": answer}]}
            for split in ("train", "validation") for index, answer in enumerate(("回答甲", "回答乙"))]
    records = data / "records.jsonl"
    records.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    config = {"width": 16, "layers": 1, "heads": 2, "kv_heads": 1, "ffn_hidden": 32, "experts": 2, "max_length": 64}
    manifest = SCRATCH / "synthetic-manifest.json"
    save_manifest = {"schema_version": 1, "initialization": "random", "model_config": config,
                     "records": [{"path": records.name, "bytes": records.stat().st_size, "sha256": sha(records.read_bytes())}], "assets": []}
    manifest.write_text(json.dumps(save_manifest))
    common = ["--manifest", str(manifest), "--manifest-sha256", sha(manifest.read_bytes()), "--data-root", str(data), "--"]
    options = ["--batch-size", "2", "--context", "64", "--seed", "1729", "--eval-every", "10", "--save-every", "1", "--device", "cpu", "--threads", "2"]
    executions = []

    def run(name, stage, steps, parent=()):
        output = SCRATCH / name
        args = common + ["--output-dir", str(output), "--stage", stage, "--steps", str(steps)] + options + list(parent)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = train_local_stage.main(args)
        executions.append({"command": ["scripts/selftrained/train_local_stage.py", *args], "returncode": code, "stdout": stdout.getvalue()})
        if code != 0:
            raise AssertionError("合成 CPU 訓練未完成")
        return output

    whole_dir = run("whole", "pretrain", 2)
    part_dir = run("part", "pretrain", 1)
    resumed_dir = run("resumed", "pretrain", 2, ["--resume", str(part_dir / "latest.pt")])
    fresh_dir = run("fresh", "sft", 1, ["--init-checkpoint", str(part_dir / "best.pt")])
    whole = torch.load(whole_dir / "latest.pt", weights_only=False)
    part = torch.load(part_dir / "latest.pt", weights_only=False)
    resumed = torch.load(resumed_dir / "latest.pt", weights_only=False)
    fresh = torch.load(fresh_dir / "latest.pt", weights_only=False)
    state_comparison = {key: equal_state(whole[key], resumed[key]) for key in ("model", "optimizer", "rng", "sampler", "step", "tokens", "target_tokens")}
    invalid = {}
    for name, path, resume in (("safe_init", part_dir / "model.safetensors", False),
                               ("best_as_resume", part_dir / "best.pt", True),
                               ("latest_as_init", part_dir / "latest.pt", False)):
        try:
            train_local_stage.local_source(path, sha(manifest.read_bytes()), resume)
        except ValueError as error:
            invalid[name] = str(error)
        else:
            invalid[name] = "未拒絕"
    try:
        run("part", "pretrain", 1)
    except FileExistsError as error:
        invalid["existing_output"] = str(error)
    data_entry = save_manifest["records"][0]
    records.write_text(records.read_text() + "\n")
    try:
        train_local_stage.verified_file(data, data_entry)
    except ValueError as error:
        invalid["changed_data"] = str(error)
    return {"executions": executions, "state_comparison": state_comparison,
            "latest_state_keys": sorted(part.keys()),
            "optimizer_state_keys": sorted(next(iter(part["optimizer"]["state"].values())).keys()),
            "best_checkpoint_kind": torch.load(part_dir / "best.pt", weights_only=False)["checkpoint_kind"],
            "part_steps": part["step"], "resumed_steps": resumed["step"], "fresh_steps": fresh["step"],
            "fresh_optimizer_update_steps": sorted(set(float(value["step"]) for value in fresh["optimizer"]["state"].values())),
            "fresh_sampler_draws": fresh["sampler"]["draws"], "fresh_stage_history": fresh["stage_history"],
            "wrapper_rejections": invalid,
            "scope": "width16、單層、context64、batch2、兩個 train 與兩個 validation 文字樣本；驗證流程與狀態，沒有測量模型品質，沒有全量 native 訓練。"}


def main():
    if SCRATCH.exists():
        raise FileExistsError("驗證暫存目錄已存在，保留原始資料")
    SCRATCH.mkdir()
    RESULT["started_at"] = datetime.now(timezone.utc).isoformat()
    RESULT["environment"] = {"python": sys.version, "platform": platform.platform(), "device": "cpu",
                             **{name: importlib.metadata.version(name) for name in ("torch", "numpy", "safetensors", "huggingface_hub")}}
    try:
        RESULT["repository"] = repository_checks()
        external_sources()
        RESULT["external_sources"] = {"path": str((ARTIFACT / "original-source-excerpts.json").relative_to(ROOT)),
                                       "sha256": sha((ARTIFACT / "original-source-excerpts.json").read_bytes())}
        RESULT["public"] = public_checks()
        RESULT["cpu_sample"] = run_cli()
        RESULT["local_training"] = local_checks()
    except Exception as error:
        RESULT["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        RESULT["finished_at"] = datetime.now(timezone.utc).isoformat()
        save("verification-output-final.json", RESULT)
        shutil.rmtree(SCRATCH)
    print(json.dumps({"公開包": len(RESULT["public"]["exports"]),
                      "CPU命令退出碼": RESULT["cpu_sample"]["returncode"],
                      "續訓狀態逐項相等": RESULT["local_training"]["state_comparison"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
