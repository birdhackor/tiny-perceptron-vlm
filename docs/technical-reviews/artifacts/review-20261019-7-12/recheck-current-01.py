"""保存來源差異、切分重建與安全命令派送的複查收據。"""

import ast
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from scripts.course_experiments.common import split_records
from scripts.prepare_data import generate_records


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    receipt = {"command_argv": argv, "exit_code": result.returncode,
               "stdout": result.stdout, "stderr": result.stderr}
    assert result.returncode == 0, receipt
    return receipt


def fixed_recipe(raw):
    start = raw.index(b'<details id="fixed-sft-recipe" markdown="1">')
    end = raw.index(b"</details>", start) + len(b"</details>")
    return raw[start:end]


def main():
    original_path = "docs/course-review-20261019/reviews/rv1019_technical_7-12/technical-7-12-original-01.json"
    original_raw = (ROOT / original_path).read_bytes()
    assert sha(original_raw) == "146c64714e00c5d170da277bc30ad7daa927a9233b71d988c5d3228c281b585e"
    original = json.loads(original_raw)
    source_audit, artifact_audit = [], []
    for source in original["sources"]:
        if source["kind"] != "repository_code":
            continue
        current = sha((ROOT / source["path"]).read_bytes())
        source_audit.append({"id": source["id"], "path": source["path"],
                             "original_sha256": source["sha256"], "current_sha256": current,
                             "unchanged": current == source["sha256"]})
    assert [s["path"] for s in source_audit if not s["unchanged"]] == ["course/training.md"]
    for artifact in original["artifacts"]:
        current = sha((ROOT / artifact["path"]).read_bytes())
        assert current == artifact["sha256"]
        artifact_audit.append({"id": artifact["id"], "path": artifact["path"],
                               "sha256": current, "unchanged": True})
    diff = run(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
    old_doc = subprocess.check_output(["git", "show", "5ef56bc756f3:course/training.md"], cwd=ROOT)
    new_doc = subprocess.check_output(["git", "show", "6a7aa1de5227:course/training.md"], cwd=ROOT)
    current_doc = (ROOT / "course/training.md").read_bytes()
    assert sha(old_doc) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21"
    assert sha(current_doc) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
    assert current_doc == new_doc
    assert fixed_recipe(old_doc) == fixed_recipe(new_doc) == fixed_recipe(current_doc)
    invocation = run([".venv/bin/python", "docs/review-tools/section_facts.py", "course/chapters/07.md#7.12", "--execute"])
    helper = json.loads(invocation["stdout"])
    tool_dir = Path(helper["artifact_directory"])
    assert tool_dir.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
    copied = {}
    for name in ("section.md", "fence-1.py", "bootstrap.py", "extraction.json",
                 "execution.json", "environment.json", "stdout.txt", "stderr.txt"):
        raw = (tool_dir / name).read_bytes()
        copied[name] = {"sha256": sha(raw), "text": raw.decode()}
    frozen_path = "docs/course-review-20261019/freeze-01/sources/7.12.md"
    frozen = (ROOT / frozen_path).read_bytes()
    assert copied["section.md"]["text"].encode() == frozen
    assert sha(frozen) == original["source_sha256"] == helper["source_sha256"]
    assert json.loads(copied["execution.json"]["text"])["exit_code"] == 0
    assert copied["stderr.txt"]["text"] == ""
    assert helper["svg_references"] == 0
    code = copied["fence-1.py"]["text"]
    namespace = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(code, "凍結的切分程式", "exec"), namespace)
    assert namespace["train"] == [("紅", "圓"), ("藍", "圓"), ("藍", "方")]
    assert namespace["all_pairs"] == [("紅", "圓"), ("紅", "方"), ("藍", "圓"), ("藍", "方")]
    assert set(namespace["train"]) & namespace["held_out"] == set()
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call)]
    assert calls == ["print", "print", "set"]
    colors, shapes = Counter(c for c, _ in namespace["train"]), Counter(s for _, s in namespace["train"])
    assert colors == {"紅": 1, "藍": 2} and shapes == {"圓": 2, "方": 1}
    exercise = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(code.replace('{("紅", "方")}', '{("藍", "圓"), ("藍", "方")}'),
                     "留出整個顏色的練習", "exec"), exercise)
    assert exercise["train"] == [("紅", "圓"), ("紅", "方")]
    result = json.loads((ROOT / "docs/course-experiments/results/sft.json").read_bytes())
    rows = generate_records("attributes-sft")
    parts = split_records(rows, seed=result["seed"])
    assert len(rows) == 60 and len({r["family"] for r in rows}) == 12
    rebuilt = {}
    for split, values in parts.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values).encode()
        recorded = result["results"]["data"][split]
        assert sha(raw) == recorded["sha256"]
        assert len(values) == recorded["records"]
        assert len({r["family"] for r in values}) == recorded["families"]
        rebuilt[split] = {"sha256": sha(raw), "records": len(values), "families": recorded["families"]}
    cli = [run([".venv/bin/python", "-m", "scripts.course_experiments.run", "--help"]),
           run([".venv/bin/python", "-m", "scripts.course_experiments.run", "--list-assets", "sft"]),
           run([".venv/bin/python", "scripts/fetch_training_assets.py", "--help"])]
    runner = importlib.import_module("scripts.course_experiments.run")
    dispatches = []

    def capture(experiment_id, device, output, dependencies, assets, **kwargs):
        dispatches.append({"experiment_id": experiment_id, "device": device, "output": str(output),
                           "dependencies": str(dependencies), "assets": str(assets), **kwargs})
        return {"experiment_id": experiment_id, "elapsed_seconds": 0.0,
                "evidence_status": "僅核對參數，未執行訓練", "artifacts": []}

    for arguments in (["--experiment", "sft", "--device", "cuda"], ["--experiment", "sft"]):
        with patch.object(sys, "argv", ["run", *arguments]), patch.object(runner, "execute", capture):
            with redirect_stdout(io.StringIO()):
                runner.main()
    assert [d["device"] for d in dispatches] == ["cuda", "cpu"]
    assert all(Path(d["output"]) == ROOT / "outputs/course-experiments/course-v1/sft" for d in dispatches)
    assert "--asset" in cli[2]["stdout"] and "--device" not in cli[2]["stdout"]
    assert cli[1]["stdout"].strip() == "assets/training/ultrachat-sft-v1.tar.gz"
    freeze_path = "docs/course-review-20261019/freeze-06.json"
    freeze_raw = (ROOT / freeze_path).read_bytes()
    assert sha(freeze_raw) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
    page = next(p for p in json.loads(freeze_raw)["pages"] if p["page_id"] == "7.12")
    assert page["source_sha256"] == sha(frozen) and page["figures_sha256"] == {}
    receipt = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-12/recheck-current-01.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "device": "cpu", "python_optimization": str(sys.flags.optimize)},
        "result": "來源差異、既有證據完整性、小節實跑、練習、資料切分重建與命令派送核對均通過。",
        "original_report": {"path": original_path, "sha256": sha(original_raw)},
        "original_issues": original["issues"], "source_audit": source_audit, "artifact_audit": artifact_audit,
        "training_diff": diff,
        "fixed_recipe": {"anchor": "fixed-sft-recipe", "unchanged_bytes": True,
                         "sha256": sha(fixed_recipe(current_doc)), "current_text": fixed_recipe(current_doc).decode(),
                         "old_document_sha256": sha(old_doc), "new_document_sha256": sha(new_doc),
                         "current_document_sha256": sha(current_doc)},
        "section_execution": {"invocation": invocation, "copied_files": copied},
        "freeze": {"path": freeze_path, "sha256": sha(freeze_raw)},
        "toy_verification": {"all_pairs": namespace["all_pairs"], "train": namespace["train"],
                             "held_out": sorted(namespace["held_out"]), "colors": dict(colors), "shapes": dict(shapes),
                             "exercise_train": exercise["train"], "call_names": calls},
        "rebuilt_splits": rebuilt, "cli_checks": cli, "execute_mocked_dispatches": dispatches,
        "scope": "本節與其固定對話配方引用；其他訓練頁面的多模態與量化新增敘述不屬於本節主張，沒有據此擴大本節結論。",
        "limitations": ["沒有重新量測 GPU 訓練或時間。", "沒有執行下載或解包。",
                        "命令派送使用 mock，僅核對參數與路徑。", "未重抓外部原文，沿用原審閱的原始來源閱讀紀錄與未變 artifact。"],
    }
    destination = ARTIFACT / "recheck-current-01.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"path": destination.relative_to(ROOT).as_posix(), "sha256": sha(destination.read_bytes()),
                      "result": receipt["result"], "section_stdout": copied["stdout.txt"]["text"],
                      "fixed_recipe_unchanged": True, "dispatch_devices": [d["device"] for d in dispatches]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
