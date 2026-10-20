"""核對變更後的安裝契約與公開 CPU 示例，保存新的獨立證據。"""

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parent
ROOT = ARTIFACT.parents[3]
SCRATCH = ARTIFACT / "transient-current-01"
OUTPUT = ARTIFACT / "recheck-output-current-01.json"
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_19-11/technical-19-11-original-01.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def execute(args, env):
    run = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
    return {"command": args, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}


def main():
    if OUTPUT.exists() or SCRATCH.exists():
        raise FileExistsError("新的複查輸出位置已存在，保留既有檔案")
    SCRATCH.mkdir()
    result = {"started_at": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu",
                              **{name: importlib.metadata.version(name) for name in ("torch", "numpy", "safetensors", "huggingface_hub")}}}
    original = json.loads(ORIGINAL.read_bytes())
    old_paths = [ORIGINAL, *[ROOT / a["path"] for a in original["artifacts"]], ARTIFACT / "self-check.txt"]
    old_hashes = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in old_paths}
    try:
        result["original_report"] = {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": sha(ORIGINAL.read_bytes()), "issues": original["issues"]}
        result["head"] = git("rev-parse", "HEAD").decode().strip()
        raw = (ROOT / "pyproject.toml").read_bytes()
        old_raw = git("show", "66950413baf1:pyproject.toml")
        after_raw = git("show", "3af7f13ef959:pyproject.toml")
        old_config, current_config = tomllib.loads(old_raw.decode()), tomllib.loads(raw.decode())
        old_nonruff, current_nonruff = copy.deepcopy(old_config), copy.deepcopy(current_config)
        old_ruff = old_nonruff["tool"].pop("ruff")
        current_ruff = current_nonruff["tool"].pop("ruff")
        old_excludes, current_excludes = old_ruff.pop("extend-exclude"), current_ruff.pop("extend-exclude")
        result["pyproject_change"] = {
            "command": ["git", "diff", "66950413baf1", "3af7f13ef959", "--", "pyproject.toml"],
            "diff": git("diff", "66950413baf1", "3af7f13ef959", "--", "pyproject.toml").decode(),
            "old_sha256": sha(old_raw), "after_commit_sha256": sha(after_raw), "current_sha256": sha(raw),
            "current_equals_after_commit": raw == after_raw,
            "non_ruff_configuration_unchanged": old_nonruff == current_nonruff,
            "other_ruff_settings_unchanged": old_ruff == current_ruff,
            "old_excludes_preserved": current_excludes[:len(old_excludes)] == old_excludes,
            "added_excludes": current_excludes[len(old_excludes):],
            "optional_dependencies": current_config["project"]["optional-dependencies"],
            "uv_configuration": current_config["tool"]["uv"],
        }
        source_checks = []
        for source in original["sources"]:
            if source["kind"] == "repository_code":
                current = sha((ROOT / source["path"]).read_bytes())
                source_checks.append({"id": source["id"], "path": source["path"], "original_sha256": source["sha256"],
                                      "current_sha256": current, "same": current == source["sha256"]})
        result["repository_sources"] = source_checks
        result["original_artifacts"] = [{"id": a["id"], "path": a["path"], "sha256": sha((ROOT / a["path"]).read_bytes()),
                                         "same": sha((ROOT / a["path"]).read_bytes()) == a["sha256"]} for a in original["artifacts"]]
        chapter = (ROOT / "course/chapters/19.md").read_bytes()
        start = chapter.index("## 19.11 ".encode())
        end = chapter.find(b"\n## ", start + 1)
        section = chapter[start:] if end < 0 else chapter[start:end+1]
        snapshot = ROOT / "docs/course-review-20261019/freeze-01/sources/19.11.md"
        freeze_path = ROOT / "docs/course-review-20261019/freeze-05.json"
        frozen_page = next(p for p in json.loads(freeze_path.read_bytes())["pages"] if p["page_id"] == "19.11")
        result["section"] = {"snapshot_sha256": sha(snapshot.read_bytes()), "current_sha256": sha(section),
                             "current_equals_snapshot": section == snapshot.read_bytes(),
                             "python_code_blocks": section.count(b"```python"),
                             "freeze_sha256": sha(freeze_path.read_bytes()), "freeze_page": frozen_page,
                             "figure_sha256": sha((ROOT / "course/figures/p6-19-delivery-package-roles.svg").read_bytes())}
        previous = json.loads((ARTIFACT / "verification-output-final.json").read_bytes())
        comparison = []
        for row in previous["repository"]["relevant_comparison"]:
            path = row["path"]
            tag = git("show", f"selftrained-v2:{path}")
            head = git("show", f"HEAD:{path}")
            working = (ROOT / path).read_bytes()
            comparison.append({"path": path, "tag_sha256": sha(tag), "head_sha256": sha(head),
                               "working_sha256": sha(working), "same": tag == head == working})
        result["tag_main_comparison"] = comparison
        result["original_environment_unchanged"] = all(result["environment"][key] == previous["environment"][key]
                                                       for key in ("python", "torch", "numpy", "safetensors", "huggingface_hub", "device"))
        env = dict(os.environ, UV_CACHE_DIR=str(SCRATCH / "uv-cache"), HF_HOME=str(SCRATCH / "hf-cache"),
                   XDG_CACHE_HOME=str(SCRATCH / "cache"), HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1",
                   PYTHONDONTWRITEBYTECODE="1")
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env.pop(name, None)
        result["install_dry_run"] = execute(["uv", "sync", "--frozen", "--extra", "cpu", "--extra", "selftrained", "--dry-run"], env)
        command = [str(ROOT / ".venv/bin/python"), "scripts/selftrained/chat.py", "--model-dir", str(SCRATCH / "public-model"),
                   "--asset-dir", str(SCRATCH / "nonexistent-assets"), "--repo", "birdhackor/tiny-perceptron-course-models",
                   "--revision", "979cdfacc588ad0536f1c64fff96f264571cf054", "--prefix", "selftrained/v2/moe-joint",
                   "--manifest-sha256", "f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e",
                   "--messages", "docs/selftrained/examples/v2/text.messages.json", "--task", "text", "--device", "cpu",
                   "--max-new-tokens", "128", "--threads", "2"]
        result["cpu_example"] = execute(command, env)
        if result["cpu_example"]["returncode"] == 0:
            answer = json.loads(result["cpu_example"]["stdout"])
            result["cpu_comparison"] = {"answer_matches_original": answer["answer"] == previous["cpu_sample"]["first"]["answer"],
                                        "model_receipt_matches_original": answer["model"] == previous["cpu_sample"]["first"]["model"],
                                        "public_source_matches_original": answer["public_source"] == previous["cpu_sample"]["first"]["public_source"],
                                        "generations_match_original": answer["generations"] == previous["cpu_sample"]["first"]["generations"]}
        result["claim_recheck"] = [{"claim_id": c["id"], "section_unchanged": result["section"]["current_equals_snapshot"],
                                    "referenced_artifacts_unchanged": all(next(a for a in result["original_artifacts"] if a["id"] == identifier)["same"] for identifier in c["artifact_ids"]),
                                    "changed_repository_sources": [ref["source_id"] for ref in c["evidence"] if any(s["id"] == ref["source_id"] and not s["same"] for s in source_checks)]}
                                   for c in original["claims"]]
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["old_files_preserved"] = all(sha((ROOT / path).read_bytes()) == expected for path, expected in old_hashes.items())
        result["old_file_sha256"] = old_hashes
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        with OUTPUT.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        shutil.rmtree(SCRATCH)
    print(json.dumps({"CPU安裝預演退出碼": result["install_dry_run"]["returncode"],
                      "公開CPU命令退出碼": result["cpu_example"]["returncode"],
                      "CPU輸出比較": result.get("cpu_comparison"), "舊檔案完整保留": result["old_files_preserved"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
