"""核對日期評估時序的修訂、證據延用條件與本次小節執行。"""

import ast
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_8-6/technical-8-6-original-01.json"
    original_raw = original_path.read_bytes()
    assert sha(original_raw) == "4e533504c00c78d96616df9adea6dd5c3f003c64397ff2aca9da790c4cfee084", "原報告必須維持原指紋"
    original = json.loads(original_raw)
    fingerprints = []
    for item in original["sources"] + original["artifacts"]:
        if "path" in item and "sha256" in item:
            observed = sha((ROOT / item["path"]).read_bytes())
            assert observed == item["sha256"], "沿用證據必須維持原指紋"
            fingerprints.append({"path": item["path"], "expected_sha256": item["sha256"], "observed_sha256": observed})
    old_selfcheck = DEST / "self-check.txt"
    assert sha(old_selfcheck.read_bytes()) == "0d9a988b27110e17d1568a1cbffbbca2c1d222ee94c686a63545bda45af37596", "原自我檢查檔不可改動"
    old_path = ROOT / "docs/course-review-20261019/freeze-01/sources/8.6.md"
    new_path = ROOT / "docs/course-review-20261019/freeze-05/sources/8.6.md"
    old = old_path.read_bytes()
    new = new_path.read_bytes()
    assert sha(new) == "d2a9b74deac645f43b22d09dbe66c1a377f0c7498b2f4221c8c46fd80df057d9", "新版凍結原文指紋必須相符"
    removed = "不拿來更新模型，只在最後檢查。"
    replacement = "不拿來更新模型；訓練前後都用它們評估，下面列的是訓練後的結果。"
    assert old.count(removed.encode()) == 1, "舊文應只有一處指定的時序敘述"
    assert old.replace(removed.encode(), replacement.encode()) == new, "所有改動都應涵蓋於本次時序核對"
    diff_argv = ["git", "diff", "--no-index", str(old_path.relative_to(ROOT)), str(new_path.relative_to(ROOT))]
    difference = subprocess.run(diff_argv, cwd=ROOT, capture_output=True, text=True, check=False)
    assert difference.returncode == 1, "原文差異應存在且可讀"
    run = ROOT / sys.argv[1]
    copied = {
        name: {"sha256": sha((run / name).read_bytes()), "text": (run / name).read_text()}
        for name in ("stdout.txt", "stderr.txt", "execution.json", "environment.json", "extraction.json", "fence-1.py")
    }
    assert (run / "section.md").read_bytes() == new, "現行章檔抽出的小節必須等於新版凍結原文"
    execution = json.loads(copied["execution.json"]["text"])
    extraction = json.loads(copied["extraction.json"]["text"])
    environment = json.loads(copied["environment.json"]["text"])
    assert execution["exit_code"] == 0, "小節程式應成功執行"
    assert extraction["figure_sha256"] == {}, "本頁沒有圖"
    previous_execution = json.loads((DEST / "section-execution.json").read_text())
    assert copied["stdout.txt"]["text"] == previous_execution["copied_files"]["stdout.txt"]["text"], "未改的小節程式輸出應與原核對一致"
    assert copied["fence-1.py"]["sha256"] == previous_execution["copied_files"]["fence-1.py"]["sha256"], "小節程式原碼應保持相同"
    result_path = ROOT / "docs/course-experiments/results/style.json"
    result = json.loads(result_path.read_bytes())
    tok = ByteTokenizer()
    stages = {}
    for stage in ("before", "after"):
        selected = []
        for index, sample in enumerate(result["results"][stage]["test"]["samples"]):
            if sample.get("kind") != "clarification":
                continue
            ids = sample["generated_ids"]
            raw_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            assert tok.decode(raw_ids) == sample["generated"], "保存的原始 token 應與可見文字相同"
            assert (raw_ids == tok.encode(sample["expected"])) == sample["exact"], "逐題精確計分應符合原始 token"
            selected.append({
                "locator": f"results.{stage}.test.samples[{index}]",
                "prompt": sample["messages"][0]["content"],
                "expected": sample["expected"],
                "generated": sample["generated"],
                "exact": sample["exact"],
            })
        assert len(selected) == 6, "兩階段都應有完整六筆日期測試紀錄"
        missing = [row for row in selected if "date=?" in row["prompt"]]
        provided = [row for row in selected if "date=?" not in row["prompt"]]
        stages[stage] = {
            "samples": selected,
            "records": len(selected),
            "correct": sum(row["exact"] for row in selected),
            "missing_correct": sum(row["exact"] for row in missing),
            "missing_records": len(missing),
            "provided_correct": sum(row["exact"] for row in provided),
            "provided_records": len(provided),
        }
    assert [r["prompt"] for r in stages["before"]["samples"]] == [r["prompt"] for r in stages["after"]["samples"]], "訓練前後應評估同一批日期題"
    assert stages["before"]["correct"] == 0 and stages["after"]["correct"] == 3, "正文數字應對應訓練後而不是訓練前"
    assert stages["after"]["missing_correct"] == 3 and stages["after"]["provided_correct"] == 0, "正文分項結果應對應訓練後紀錄"
    assert stages["after"]["samples"][0]["generated"] == "已確認2026-10-10。", "正文日期反例應出自訓練後紀錄"
    behavior_path = ROOT / "scripts/course_experiments/behavior.py"
    behavior = behavior_path.read_text()
    tree = ast.parse(behavior)
    run_style = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_style")
    sequence = [ast.get_source_segment(behavior, n) for n in run_style.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in {"before", "training", "after"}]
    expected_sequence = [
        "before = _style_evaluations(model, conditional)",
        'training = fit_lm(model, conditional["train"], ctx, mode="sft", steps=_steps(ctx, 1_000), name="model")',
        "after = _style_evaluations(model, conditional)",
    ]
    assert sequence == expected_sequence, "評估時序應為訓練前、只用訓練集更新、訓練後"
    evaluation = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_style_evaluations")
    evaluation_source = ast.get_source_segment(behavior, evaluation)
    assert '("validation", "test")' in evaluation_source, "前後評估都應包含測試集"
    payload = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " " + sys.argv[1],
        "environment": {"python": platform.python_version(), "torch": environment["torch"], "device": "cpu"},
        "original_report": {"path": str(original_path.relative_to(ROOT)), "sha256": sha(original_raw)},
        "old_source": {"path": str(old_path.relative_to(ROOT)), "sha256": sha(old)},
        "current_source": {"path": str(new_path.relative_to(ROOT)), "sha256": sha(new)},
        "reuse_fingerprints": fingerprints,
        "diff": {"command_argv": diff_argv, "exit_code": difference.returncode, "stdout": difference.stdout, "stderr": difference.stderr},
        "changed_text": {"removed": removed, "replacement": replacement, "only_change": True},
        "current_section_execution": {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/08.md#8.6" --execute', "run": sys.argv[1], "copied_files": copied},
        "same_python_fence": True,
        "same_stdout": True,
        "stages": stages,
        "run_style_sequence": sequence,
        "evaluation_source": evaluation_source,
        "findings": {"test_evaluated_before_and_after": True, "reported_numbers_match_after": True, "test_not_passed_to_training": True, "all_reused_source_and_artifact_hashes_match": True},
        "scope": "本次執行新版小節，核對修訂句、保存的兩階段原始 token 與現行評估呼叫。未改內容沿用原審閱已核實且 hash 未變的證據；未重訓或載入 checkpoint 重新生成。",
    }
    output = DEST / "recheck-current-01.json"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": "修訂句、前後評估與訓練後分項結果皆核實；沿用證據指紋全部相符", "artifact": str(output.relative_to(ROOT)), "sha256": sha(output.read_bytes())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
