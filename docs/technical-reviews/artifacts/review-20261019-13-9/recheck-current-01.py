"""複查來源變動、引用版本與既有數值；只讀檔案，不訓練模型。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import runpy
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ORIGINAL = "docs/course-review-20261019/reviews/rv1019_technical_13-9/technical-13-9-original-02.json"
ARTIFACTS = "docs/technical-reviews/artifacts/review-20261019-13-9"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    selected = [i for i, match in enumerate(headers) if match[0].startswith(b"## " + identifier.encode() + b" ")]
    assert len(selected) == 1
    index = selected[0]
    end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
    return raw[headers[index].start():end]


def git_read(arguments):
    command = ["git", *arguments]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    return completed.stdout, {
        "command_argv": command,
        "exit_code": completed.returncode,
        "stderr": completed.stderr.decode(),
        "stdout_sha256": digest(completed.stdout),
        "stdout_bytes": len(completed.stdout),
    }


def main():
    raw = (ROOT / ORIGINAL).read_bytes()
    assert digest(raw) == "01712708b29b0602b9204fe9e5e9a4d777df6227b20b04c76e86630ee886dd4e"
    original = json.loads(raw)
    assert original["issues"] == []
    source_audit = []
    artifact_audit = []
    for collection, target in (("sources", source_audit), ("artifacts", artifact_audit)):
        for item in original[collection]:
            if "path" not in item:
                continue
            current = digest((ROOT / item["path"]).read_bytes())
            same = current == item["sha256"]
            target.append({"id": item["id"], "path": item["path"], "previous_sha256": item["sha256"], "current_sha256": current, "unchanged": same})
            if collection == "artifacts" or item["path"] != "course/training.md":
                assert same, item["path"]
    assert [item["path"] for item in source_audit if not item["unchanged"]] == ["course/training.md"]
    before, before_receipt = git_read(["show", "5ef56bc756f3:course/training.md"])
    after, after_receipt = git_read(["show", "6a7aa1de5227:course/training.md"])
    diff, diff_receipt = git_read(["diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
    current = (ROOT / "course/training.md").read_bytes()
    assert after == current
    assert digest(before) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21"
    assert digest(current) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
    old_target, target = section(before, "T.7"), section(current, "T.7")
    assert old_target == target
    target_text = target.decode()
    needed = [
        "先完成T.5的固定`style`",
        "scripts/fetch_training_assets.py --asset ultrafeedback-dpo",
        "scripts.course_experiments.run --experiment dpo --device cuda",
        "checkpoints/preferred.pt",
    ]
    assert all(text in target_text for text in needed)
    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/13.9.md").read_bytes()
    actual = section((ROOT / "course/chapters/13.md").read_bytes(), "13.9")
    assert actual == frozen
    assert digest(actual) == original["source_sha256"]
    freeze = (ROOT / "docs/course-review-20261019/freeze-06.json").read_bytes()
    assert digest(freeze) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"

    support = runpy.run_path(str(ROOT / ARTIFACTS / "verify_facts.py"), run_name="technical_recheck_support")
    torch = support["torch"]
    torch.set_num_threads(1)
    scope = {}
    capture = io.StringIO()
    code = re.search(r"```python\n(.*?)```", frozen.decode(), re.S).group(1)
    with contextlib.redirect_stdout(capture):
        exec(compile(code, "frozen-section", "exec"), scope)
    assert sum(scope["recipe"].values()) == 100
    assert len(scope["metrics"]) == 6 and all(value is None for value in scope["metrics"].values())
    assert capture.getvalue().count("尚未量測") == 6
    moved = dict(scope["recipe"])
    moved["應拒絕題"] += moved["正常題完成"]
    moved["正常題完成"] = 0
    assert moved["應拒絕題"] == 40 and moved["正常題完成"] == 0 and sum(moved.values()) == 100
    zero = 0
    assert ("尚未量測" if zero is None else zero) == 0

    render = support["render_chat"]
    examples = [render([{"role": "user", "content": "題目"}, {"role": "assistant", "content": answer}]) for answer in ("A", "BCD")]
    inputs, labels, valid = support["pad_batch"](examples)
    counts = [int((row != support["IGNORE"]).sum()) for row in labels]
    assert counts == [2, 4]
    assert (labels[~valid] == support["IGNORE"]).all()
    assert all(int(row[row != support["IGNORE"]][-1]) == 2 for row in labels)
    logits = torch.zeros((*labels.shape, 264), dtype=torch.float64)
    observed = support["sequence_log_probability"](logits, labels)
    expected = -torch.tensor(counts, dtype=torch.float64) * support["math"].log(264)
    assert torch.allclose(observed, expected, rtol=0, atol=1e-12)

    dpo = json.loads((ROOT / "docs/course-experiments/results/dpo.json").read_bytes())["results"]
    style = json.loads((ROOT / "docs/course-experiments/results/style.json").read_bytes())["results"]
    gen, pref = support["generation_summary"], support["preference_summary"]
    baseline = gen(style["content_evaluation"]["test"])
    runs = {name: gen(run["arithmetic"]["test"]) for name, run in dpo["runs"].items()}
    assert baseline["matches"] == 0 and baseline["records"] == 7
    prompts = [sample["messages"] for sample in baseline["samples"]]
    for run in runs.values():
        assert run["matches"] == 0 and run["records"] == 7
        assert [sample["messages"] for sample in run["samples"]] == prompts
    format_preference = pref(dpo["format_only"]["preference"]["test"])
    format_generation = gen(dpo["format_only"]["arithmetic"]["test"])
    assert format_preference["relative_positive"] == 7 and format_preference["records"] == 7
    assert format_generation["matches"] == 0 and format_generation["records"] == 7
    format_targets = dpo["format_only"]["training"]["effective_answer_tokens_both_sides"]
    assert format_targets == 34554
    natural = dpo["ultrafeedback_pilot"]
    natural_evaluation = {split: pref(values) for split, values in natural["evaluation"].items()}
    assert (natural_evaluation["validation"]["relative_positive"], natural_evaluation["validation"]["chosen_first"]) == (6, 7)
    assert (natural_evaluation["test"]["relative_positive"], natural_evaluation["test"]["chosen_first"]) == (8, 4)
    assert all(report["records"] == 10 for report in natural_evaluation.values())
    assert natural["sft_training"]["steps"] == natural["training"]["steps"] == 80
    model = support["TinyLM"](support["ModelConfig"](width=32, layers=1, max_length=256))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == natural["sft_training"]["parameters"] == 37728

    def without_samples(report):
        return {key: value for key, value in report.items() if key != "samples"}

    print(json.dumps({
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "original_report": {"path": ORIGINAL, "sha256": digest(raw), "issues": original["issues"]},
        "repository_source_audit": source_audit,
        "reused_artifact_audit": artifact_audit,
        "source_comparison": {"frozen_sha256": digest(frozen), "current_sha256": digest(actual), "identical": True, "changed_sentences": []},
        "training_dependency": {
            "before": before_receipt, "after": after_receipt,
            "diff": {**diff_receipt, "stdout": diff.decode()},
            "current_file_sha256": digest(current),
            "cited_section": "先檢查回答能力，再學較偏好的回答",
            "old_section_sha256": digest(old_target), "current_section_sha256": digest(target),
            "section_bytes_identical": True, "current_cited_section": target_text,
            "referenced_command_strings_present": needed,
            "judgment": "修改的三處均不在本節引用的訓練小節；固定style依賴、外部資料及dpo重做命令保持原樣，導引主張仍成立。",
        },
        "recipe": {"stdout": capture.getvalue(), "sum": sum(scope["recipe"].values()), "metrics_count": len(scope["metrics"]), "all_none": True, "moved": moved, "effective_target_examples": [30 * 2 * 100, 30 * 2 * 10]},
        "mask": {"labels": labels.tolist(), "valid": valid.tolist(), "counts": counts, "observed": observed.tolist(), "expected": expected.tolist(), "absolute_tolerance": 1e-12},
        "result_recounts": {"baseline": without_samples(baseline), "runs": {name: without_samples(run) for name, run in runs.items()}, "format_preference": without_samples(format_preference), "format_generation": without_samples(format_generation), "format_training_effective_targets": format_targets, "natural": {name: without_samples(report) for name, report in natural_evaluation.items()}, "sft_steps": natural["sft_training"]["steps"], "dpo_steps": natural["training"]["steps"], "parameters": parameters},
        "reviewed_claims": [{"id": claim["id"], "statement": claim["statement"], "status": "verified", "basis": "原文及原有依據未變，原證據SHA均已重查；數值與程式另做本次執行核對，訓練導引另比對原始位元組。"} for claim in original["claims"]],
        "scope": "只核對本頁全部主張、引用訓練小節及證據版本。沒有重審訓練頁其他未引用小節；沒有重跑GPU訓練。外部原始來源沿用原審閱的固定版本與查閱收據，沒有重新下載。",
        "result": "全部斷言通過。",
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
