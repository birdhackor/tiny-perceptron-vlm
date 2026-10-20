"""複查凍結正文、證據版本與公開 CPU 示範；全部持久輸出使用排他建立。"""

import contextlib
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
ORIGINAL = "docs/course-review-20261019/reviews/rv1019_technical_19-1/technical-19-1-original-01.json"
FREEZE_SOURCE = "docs/course-review-20261019/freeze-05/sources/19.1.md"
FREEZE = "docs/course-review-20261019/freeze-05.json"
MODEL_PIN = "f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e"


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return sha_bytes((ROOT / path).read_bytes())


def load(path):
    return json.loads((ROOT / path).read_bytes())


def run(command):
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def git_bytes(version, path):
    return subprocess.check_output(["git", "show", f"{version}:{path}"], cwd=ROOT)


def main():
    import torch
    import safetensors
    from scripts.selftrained.prepare_text_tools import text_records

    original = load(ORIGINAL)
    assert sha(ORIGINAL) == "503f7ab0702878eb6788b0faf87b1d3745ba853d066b0fd6549f7e99fcf43a18"
    assert sha(FREEZE_SOURCE) == "4b62d9c985f81ddb108f79976ea21db5c621a33d6401fcb8dff9892a35fdb0df"
    assert sha(FREEZE) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-19-1/recheck-current-01.py",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {"python": sys.version, "torch": torch.__version__, "safetensors": safetensors.__version__,
                        "platform": platform.platform(), "device": "cpu", "threads": "2"},
        "original_report": {"path": ORIGINAL, "sha256": sha(ORIGINAL)},
        "source_sha256": sha(FREEZE_SOURCE), "freeze_sha256": sha(FREEZE),
        "figure_sha256": sha("course/figures/p6-19-start-core.svg"),
    }
    source = (ROOT / FREEZE_SOURCE).read_bytes()
    chapter = (ROOT / "course/chapters/19.md").read_bytes()
    start = chapter.index(b"## 19.1 ")
    end = chapter.index(b"\n## ", start + 1) + 1
    result["current_section_matches_freeze"] = chapter[start:end] == source
    assert result["current_section_matches_freeze"]
    result["diffs"] = {
        "source": run(["git", "diff", "--no-index", "docs/course-review-20261019/freeze-01/sources/19.1.md", FREEZE_SOURCE]),
        "notebook": run(["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "notebooks/19/19.1.ipynb"]),
        "project": run(["git", "diff", "66950413baf1", "3af7f13ef959", "--", "pyproject.toml"]),
    }
    result["repository_sources"] = []
    for item in original["sources"]:
        if item["kind"] == "repository_code":
            current = sha(item["path"])
            result["repository_sources"].append({"id": item["id"], "path": item["path"], "old_sha256": item["sha256"],
                "current_sha256": current, "unchanged": current == item["sha256"]})
    changed = {item["path"] for item in result["repository_sources"] if not item["unchanged"]}
    assert changed == {"notebooks/19/19.1.ipynb", "pyproject.toml"}
    result["original_artifacts"] = []
    for item in original["artifacts"]:
        current = sha(item["path"])
        assert current == item["sha256"]
        result["original_artifacts"].append({"id": item["id"], "path": item["path"], "sha256": current, "unchanged": True})
    result["original_self_check_sha256"] = sha("docs/technical-reviews/artifacts/review-20261019-19-1/self-check.txt")
    notebook_path = "notebooks/19/19.1.ipynb"
    notebook = load(notebook_path)
    old_notebook = json.loads(git_bytes("5ef56bc756f3", notebook_path))
    get_code = lambda n: ["".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code"]
    codes = get_code(notebook)
    result["notebook"] = {"path": notebook_path, "sha256": sha(notebook_path),
        "code_cells_equal_previous": codes == get_code(old_notebook),
        "pip_base_install": '"--quiet", "-e", str(root)' in codes[0],
        "selftrained_extra_in_setup": "selftrained" in codes[0], "uv_sync_in_setup": "uv sync" in codes[0],
        "code_cell_count": len(codes)}
    assert result["notebook"]["code_cells_equal_previous"]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    old_project = tomllib.loads(git_bytes("66950413baf1", "pyproject.toml").decode())
    without_format = lambda p: {k: v if k != "tool" else {x: y for x, y in v.items() if x != "ruff"} for k, v in p.items()}
    result["project"] = {"path": "pyproject.toml", "sha256": sha("pyproject.toml"),
        "all_settings_except_ruff_equal_previous": without_format(project) == without_format(old_project),
        "selftrained_extra": project["project"]["optional-dependencies"]["selftrained"],
        "cpu_extra": project["project"]["optional-dependencies"]["cpu"], "lock_sha256": sha("uv.lock")}
    assert result["project"]["all_settings_except_ruff_equal_previous"]
    rows = text_records()
    preferences = [r for r in rows if r["supervision"]["intent"] == "preference"]
    first_users = sorted({next(m["content"] for m in r["messages"] if m["role"] == "user") for r in preferences})
    result["text_materials"] = {"row_count": len(rows), "preference_count": len(preferences),
        "intents": dict(Counter(r["supervision"]["intent"] for r in rows)),
        "selection_kinds": dict(Counter(r["supervision"]["selection_kind"] for r in preferences)),
        "first_user_messages": first_users, "example": preferences[0]["messages"]}
    probe = load("docs/technical-reviews/artifacts/review-20261019-19-1/data-probe.json")
    manifest = load("docs/selftrained/v2-manifest.json")
    declared = {item["path"]: item["sha256"] for item in manifest["records"]}
    measured = {name: item["sha256"] for name, item in probe["records"].items()}
    assert declared == measured
    assert probe["download"]["sha256"] == manifest["package"]["sha256"]
    result["retained_data_evidence"] = {"artifact_path": "docs/technical-reviews/artifacts/review-20261019-19-1/data-probe.json",
        "artifact_sha256": sha("docs/technical-reviews/artifacts/review-20261019-19-1/data-probe.json"),
        "record_hashes_match_current_manifest": declared == measured,
        "package_sha256": manifest["package"]["sha256"], "ocr": probe["ocr"]}

    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="runtime-current-", dir=ART) as temp:
        runtime = Path(temp)
        os.environ["HF_HOME"] = str(runtime / "hf")
        os.environ["HF_HUB_CACHE"] = str(runtime / "hf" / "hub")
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "45"
        from scripts.selftrained import chat
        arguments = ["--model-dir", str(runtime / "export"), "--asset-dir", str(runtime / "no-assets"),
            "--repo", "birdhackor/tiny-perceptron-course-models", "--revision", "979cdfacc588ad0536f1c64fff96f264571cf054",
            "--prefix", "selftrained/v2/moe-joint", "--manifest-sha256", MODEL_PIN, "--messages",
            "docs/selftrained/examples/v2/text.messages.json", "--task", "text", "--device", "cpu", "--max-new-tokens", "128", "--threads", "2"]
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            answer = chat.main(arguments)
        expected = "1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。"
        result["cli"] = {"arguments": [s.replace(str(runtime), "{runtime}") for s in arguments],
            "path_note": "只有模型、快取與空素材根目錄改到指定 artifact 內的暫存子目錄；下載與生成使用原 chat.main。",
            "stdout_json_matches_return": json.loads(capture.getvalue()) == answer,
            "result": answer, "expected_answer_match": answer["answer"] == expected,
            "eos_match": answer["generations"][0]["eos"] is True}
        before = {p.name: sha_bytes(p.read_bytes()) for p in (runtime / "export").iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            repeated = chat.main(arguments)
        after = {p.name: sha_bytes(p.read_bytes()) for p in (runtime / "export").iterdir()}
        result["repeat"] = {"same_answer": repeated["answer"] == answer["answer"], "payloads_unchanged": before == after,
            "eos": repeated["generations"][0]["eos"], "asset_directory_created": (runtime / "no-assets").exists(), "files": after}
        assert result["cli"]["expected_answer_match"] and result["cli"]["eos_match"]
        assert result["repeat"]["payloads_unchanged"] and result["repeat"]["same_answer"]
        assistant = chat.InferenceAssistant(runtime / "export", runtime / "no-assets", device="cpu", manifest_sha256=MODEL_PIN)
        messages = load("docs/selftrained/examples/v2/text.messages.json")
        format_messages = [dict(m) for m in messages]
        for message in format_messages:
            message["content"] = message["content"].replace("兩點", "一句")
        content_messages = [dict(m) for m in messages]
        content_messages[-1]["content"] = "我想更新地址。"
        result["counterfactuals"] = {}
        for name, inputs in [("format_one_sentence", format_messages), ("content_address", content_messages)]:
            changed_answer = assistant.reply(inputs, task="text", max_new_tokens=128)
            result["counterfactuals"][name] = {"messages": inputs, "answer": changed_answer["answer"],
                "eos": changed_answer["generations"][0]["eos"], "generated_ids": changed_answer["generations"][0]["generated_ids"]}
    assert sha(ORIGINAL) == original_report_sha
    for item in result["original_artifacts"]:
        assert sha(item["path"]) == item["sha256"]
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    with (ART / "execution-current-01.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"source_matches_freeze": result["current_section_matches_freeze"],
        "notebook_code_unchanged": result["notebook"]["code_cells_equal_previous"],
        "project_runtime_settings_unchanged": result["project"]["all_settings_except_ruff_equal_previous"],
        "answer_match": result["cli"]["expected_answer_match"], "eos": result["cli"]["eos_match"],
        "preference_count": result["text_materials"]["preference_count"], "ocr_rows": result["retained_data_evidence"]["ocr"]["rows"]}, ensure_ascii=False))


original_report_sha = "503f7ab0702878eb6788b0faf87b1d3745ba853d066b0fd6549f7e99fcf43a18"
if __name__ == "__main__":
    main()
