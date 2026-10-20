"""核對切分、實驗紀錄與命令介面，不執行訓練或取得資料。"""

import ast
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
from pathlib import Path
import platform
import re
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


def main():
    command_output = (ARTIFACT / "section-command-output.txt").read_text()
    helper = json.loads(command_output)
    tool_dir = Path(helper["artifact_directory"])
    names = [
        "section.md", "fence-1.py", "bootstrap.py", "extraction.json",
        "execution.json", "environment.json", "stdout.txt", "stderr.txt",
    ]
    copied = {
        "invocation": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/07.md#7.12" --execute',
        "command_output": helper,
        "copied_files": {
            name: {"sha256": sha((tool_dir / name).read_bytes()), "text": (tool_dir / name).read_text()}
            for name in names
        },
    }
    (ARTIFACT / "section-execution.json").write_text(
        json.dumps(copied, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/7.12.md").read_bytes()
    assert sha(frozen) == helper["source_sha256"]
    assert (tool_dir / "section.md").read_bytes() == frozen
    table_lines = [line for line in frozen.decode().splitlines() if line.startswith("|")][2:]
    table_rows = [[cell.strip() for cell in line.split("|")[1:-1]] for line in table_lines]
    assert len(table_rows) == 2 * 2 == 4
    assert [row[0] for row in table_rows] == ["紅、圓", "藍、圓", "藍、方", "紅、方"]
    assert [row[2] for row in table_rows] == ["圓形", "圓形", "方塊", "方塊"]
    assert all(row[2] in row[1] for row in table_rows)
    assert json.loads(copied["copied_files"]["execution.json"]["text"])["exit_code"] == 0
    assert not copied["copied_files"]["stderr.txt"]["text"]
    code = copied["copied_files"]["fence-1.py"]["text"]
    calls = [ast.unparse(node.func) for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)]
    namespace = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(code, "凍結的切分程式", "exec"), namespace)
    assert namespace["all_pairs"] == [("紅", "圓"), ("紅", "方"), ("藍", "圓"), ("藍", "方")]
    assert namespace["train"] == [("紅", "圓"), ("藍", "圓"), ("藍", "方")]
    assert set(namespace["train"]) & namespace["held_out"] == set()
    assert set(calls) == {"print", "set"}
    standard = {
        "all_pairs": namespace["all_pairs"], "train": namespace["train"],
        "held_out": sorted(namespace["held_out"]),
        "attribute_counts": {"colors": dict(Counter(c for c, _ in namespace["train"])),
                             "shapes": dict(Counter(s for _, s in namespace["train"]))},
        "call_names": calls,
    }
    altered = code.replace('{("紅", "方")}', '{("藍", "圓"), ("藍", "方")}')
    exercise_namespace = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(altered, "留出整個顏色的練習", "exec"), exercise_namespace)
    assert exercise_namespace["train"] == [("紅", "圓"), ("紅", "方")]
    exercise = {
        "train": exercise_namespace["train"], "held_out": sorted(exercise_namespace["held_out"]),
        "training_colors": sorted({c for c, _ in exercise_namespace["train"]}),
    }
    result_path = ROOT / "docs/course-experiments/results/sft.json"
    recorded = json.loads(result_path.read_bytes())
    results = recorded["results"]
    rows = generate_records("attributes-sft")
    parts = split_records(rows, seed=recorded["seed"])
    assert len(rows) == 60
    assert len({row["family"] for row in rows}) == 12
    groups = {}
    for row in rows:
        groups.setdefault(row["family"], []).append(row)
    assert all(len(values) == 5 for values in groups.values())
    split_receipts = {}
    for name, values in parts.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values).encode()
        entry = results["data"][name]
        assert sha(raw) == entry["sha256"]
        assert len(values) == entry["records"]
        assert len({row["family"] for row in values}) == entry["families"]
        split_receipts[name] = {
            "records": len(values), "families": sorted({row["family"] for row in values}),
            "sha256": sha(raw), "matches_recorded_sha256": True,
        }
    family_sets = {name: {row["family"] for row in values} for name, values in parts.items()}
    assert not (family_sets["train"] & family_sets["validation"])
    assert not (family_sets["train"] & family_sets["test"])
    assert not (family_sets["validation"] & family_sets["test"])
    held_out_samples = {}
    for split in ("validation", "test"):
        expected = [(r["messages"][:-1], r["messages"][-1]["content"]) for r in parts[split]]
        reports = [results["before"][split], results["after"][split],
                   results["pretrain_then_sft"]["before_sft"][split],
                   results["pretrain_then_sft"]["after_sft"][split]]
        assert all([(s["messages"], s["expected"]) for s in report["samples"]] == expected for report in reports)
        assert all(report["records"] == len(expected) == len(report["samples"]) for report in reports)
        held_out_samples[split] = {"samples_per_evaluation": len(expected), "evaluations_checked": len(reports),
                                   "all_prompts_and_expected_answers_match": True}
    direct = results["training"]["steps"]
    pretrain = results["pretrain_then_sft"]["pretraining"]["steps"]
    continuation = results["pretrain_then_sft"]["sft"]["steps"]
    assert (direct, pretrain, continuation) == (900, 250, 900)
    elapsed = recorded["elapsed_seconds"]
    assert f"{elapsed:.1f}" == "31.7"
    commands = []
    for argv in (
        [str(ROOT / ".venv/bin/python"), "scripts/fetch_training_assets.py", "--help"],
        [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"],
        [str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--list-assets", "sft"],
    ):
        completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True)
        commands.append({"argv": argv, "exit_code": completed.returncode,
                         "stdout": completed.stdout, "stderr": completed.stderr})
    asset_manifest = json.loads((ROOT / "assets/training/manifest.json").read_bytes())
    asset = next(a for a in asset_manifest["assets"] if a["id"] == "ultrachat-sft")
    assert commands[-1]["stdout"].strip() == asset["archive"]
    runner = importlib.import_module("scripts.course_experiments.run")
    dispatch = {}

    def capture_dispatch(experiment_id, device, output, dependencies, assets, **kwargs):
        dispatch.update(experiment_id=experiment_id, device=device, output=str(output),
                        dependencies=str(dependencies), assets=str(assets), **kwargs)
        return {"experiment_id": experiment_id, "elapsed_seconds": 0.0,
                "evidence_status": "僅檢查命令派送，未執行實驗", "artifacts": []}

    with patch.object(sys, "argv", ["run", "--experiment", "sft", "--device", "cuda"]):
        with patch.object(runner, "execute", capture_dispatch), redirect_stdout(io.StringIO()):
            runner.main()
    assert Path(dispatch["output"]) == ROOT / "outputs/course-experiments/course-v1/sft"
    assert dispatch["device"] == "cuda" and dispatch["step_scale"] == 1.0
    spec = runner.experiment_spec("sft")
    assert spec["module"] == "text" and spec["function"] == "run_sft"
    assert spec["assets"] == ["ultrachat-sft"]
    assert spec["dependencies"] == []
    source_paths = [
        "course/chapters/07.md", "course/chapters/05.md", "course/chapters/01.md", "course/training.md",
        "scripts/prepare_data.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py",
        "scripts/course_experiments/run.py", "scripts/fetch_training_assets.py", "docs/course-experiments/plan.json",
        "docs/course-experiments/results/sft.json", "assets/training/manifest.json", "tiny_perceptron/data.py",
        "docs/course-review-20261019/freeze-04.json", "docs/course-review-20261019/freeze-01/sources/7.12.md",
    ]
    source_hashes = {path: sha((ROOT / path).read_bytes()) for path in source_paths}
    freeze = json.loads((ROOT / "docs/course-review-20261019/freeze-04.json").read_bytes())
    page = next(p for p in freeze["pages"] if p["page_id"] == "7.12")
    assert page["source_sha256"] == sha(frozen)
    assert page["figures_sha256"] == {}
    receipt = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-12/verify.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "device": "cpu", "python_optimization": str(sys.flags.optimize)},
        "result": "所有核對通過；未執行訓練、資料下載或 GPU 量測。",
        "source_hashes": source_hashes, "source_matches_frozen": True, "figure_references": 0,
        "toy_table": table_rows, "standard_split": standard, "exercise": exercise,
        "independent_experiment": {
            "recorded_revision": recorded["revision"], "seed": recorded["seed"],
            "source_result_sha256": sha(result_path.read_bytes()), "total_families": len(groups),
            "total_records": len(rows), "records_per_family": sorted({len(v) for v in groups.values()}),
            "splits": split_receipts, "all_family_intersections_empty": True,
            "held_out_sample_audit": held_out_samples,
            "branch_updates": {"direct_sft": direct, "pretrain": pretrain, "continued_sft": continuation,
                               "pretrain_plus_sft": pretrain + continuation},
            "recorded_device": recorded["device"], "recorded_gpu": recorded["gpu"],
            "recorded_python": recorded["python_version"], "recorded_torch": recorded["torch_version"],
            "recorded_elapsed_seconds": elapsed, "rounded_elapsed_seconds": f"{elapsed:.1f}",
            "recorded_timing_scope": recorded["timing_scope"],
            "scope": results["scope"],
        },
        "cli_checks": commands, "dispatch_probe_with_execute_mocked": dispatch,
        "experiment_spec": spec,
        "limitations": ["未重跑獨立實驗或量測其 GPU 時間；只核對原始紀錄與計時程式。",
                        "未下載資料包；資料取得命令僅核對介面與原始碼。",
                        "命令派送 probe 以 mock 代替 execute，僅支持參數與預設路徑。",
                        "沒有訓練玩具切分的模型，不能提出答題正確率。"],
    }
    output = ARTIFACT / "verification.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
