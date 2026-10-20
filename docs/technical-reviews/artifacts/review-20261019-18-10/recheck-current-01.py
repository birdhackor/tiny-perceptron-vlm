import contextlib
import difflib
import hashlib
import io
import json
import math
import platform
import re
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from tiny_perceptron.attention import manual_attention
from tiny_perceptron.model import ModelConfig, TinyLM


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    match = re.search(rb"(?m)^## " + re.escape(identifier.encode()) + rb" .+?(?=^## |\Z)", raw, re.S)
    assert match is not None
    return match[0]


def main():
    torch.set_num_threads(1)
    artifact_root = Path(__file__).resolve().parent
    report_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_18-10/technical-18-10-original-01.json"
    report_raw = report_path.read_bytes()
    assert digest(report_raw) == "de5e048008f72b05751ac07b4e75eba34f3d08a5f3a0f18ef425fa1890e38e26"
    original = json.loads(report_raw)
    old = (ROOT / "docs/course-review-20261019/freeze-01/sources/18.10.md").read_bytes()
    current = (ROOT / "docs/course-review-20261019/freeze-05/sources/18.10.md").read_bytes()
    assert digest(old) == original["source_sha256"]
    assert digest(current) == "3a860d0cc1b9c5dcfc9ab593fb42843030294664ef08c3a53e60d3967a0c811d"
    chapter_path = ROOT / "course/chapters/18.md"
    chapter = chapter_path.read_bytes()
    assert section(chapter, "18.10") == current
    old_wording = "固定更新次數時蒸餾看似較好，固定時間時SFT卻多了五倍的更新，兩種比較回答的問題不同。"
    new_wording = "固定更新次數時蒸餾看似較好，固定時間時SFT的更新次數卻是蒸餾的五倍，兩種比較回答的問題不同。"
    assert old.count(old_wording.encode()) == 1
    assert old.replace(old_wording.encode(), new_wording.encode()) == current
    changed_diff = "".join(difflib.unified_diff(old.decode().splitlines(True), current.decode().splitlines(True), fromfile="freeze-01/sources/18.10.md", tofile="freeze-05/sources/18.10.md"))
    receipts = []
    for item in [s for s in original["sources"] if s["kind"] == "repository_code"] + original["artifacts"]:
        actual = digest((ROOT / item["path"]).read_bytes())
        unchanged = actual == item["sha256"]
        assert unchanged or item["id"] == "s_chapter"
        receipts.append({"id": item["id"], "path": item["path"], "previous_sha256": item["sha256"], "current_sha256": actual, "unchanged": unchanged})
    assert digest(chapter) == "4eaf60c696a65b9bf6ea05bf35367163a6ab23bb646aa97a97702924071e05be"
    history_command = ["git", "show", "5ef56bc756f3:course/chapters/18.md"]
    previous_chapter = subprocess.run(history_command, cwd=ROOT, capture_output=True, check=True).stdout
    previous_chapter_sha = next(s["sha256"] for s in original["sources"] if s["id"] == "s_chapter")
    assert digest(previous_chapter) == previous_chapter_sha
    chapter_diff_command = ["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/18.md"]
    chapter_diff = subprocess.run(chapter_diff_command, cwd=ROOT, capture_output=True, check=True).stdout.decode()
    dependency_sections = {}
    for identifier in ("18.2", "18.7", "18.12"):
        before = section(previous_chapter, identifier)
        after = section(chapter, identifier)
        dependency_sections[identifier] = {"previous_sha256": digest(before), "current_sha256": digest(after), "unchanged": before == after}
    assert dependency_sections["18.7"]["unchanged"] and dependency_sections["18.12"]["unchanged"]
    assert "蒸餾改變學習目標，不會直接刪掉模型的參數".encode() in section(chapter, "18.2")
    assert "兩個模型仍然是隨機初始化".encode() in section(chapter, "18.2")
    model_probe = []
    for width, layers in ((16, 2), (8, 1)):
        model = TinyLM(ModelConfig(width=width, layers=layers))
        q = torch.zeros((1, 1, 5, width))
        _, weights = manual_attention(q, q, q, torch.ones((1, 1, 5, 5), dtype=torch.bool))
        model_probe.append({"width": width, "layers": layers, "embedding_shape": list(model.embedding.weight.shape), "parameters": sum(p.numel() for p in model.parameters()), "attention_scores_shape": list(weights.shape)})
    assert [row["parameters"] for row in model_probe] == [16960, 6104]
    assert model_probe[0]["attention_scores_shape"] == model_probe[1]["attention_scores_shape"] == [1, 1, 5, 5]
    code = re.search(rb"```python\n(.*?)```", current, re.S)[1].decode()
    assert code.encode() == re.search(rb"```python\n(.*?)```", old, re.S)[1]
    executions = []
    for budget in (50, 100):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(code.replace("100, 50", f"100, {budget}"), "frozen section Python fence", "exec"), {})
        updates = [100 * budget // 10, 100 * budget // 50]
        expected = f"SFT 正確率 0.75 合計秒數 10 預算 {budget} 秒內可更新 {updates[0]} 次\n蒸餾 正確率 1.0 合計秒數 50 預算 {budget} 秒內可更新 {updates[1]} 次\n"
        assert stream.getvalue() == expected
        assert updates[0] == 5 * updates[1]
        executions.append({"budget": budget, "stdout": stream.getvalue(), "student_updates": updates, "ratio_sft_to_distillation": updates[0] / updates[1]})
    helpers = runpy.run_path(str(artifact_root / "verify.py"), run_name="recheck_helpers")
    tok = helpers["ByteTokenizer"]()
    result_path = ROOT / "docs/course-experiments/results/distillation.json"
    stored = json.loads(result_path.read_bytes())
    attrs = stored["results"]["tasks"]["attributes"]
    evaluations = {"teacher": attrs["teacher_test"]}
    evaluations.update({key: value["test"] for key, value in attrs["runs"].items() if "packed" not in key})
    test_results = {}
    for name, evaluation in evaluations.items():
        correct, targets = 0, 0
        for sample in evaluation["generated_samples"]:
            assert sample["expected"] == helpers["gold"](sample["question"])
            ids = sample["generated_ids"]
            raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            hit = raw == tok.encode(sample["expected"])
            assert hit == sample["exact"]
            correct += hit
            targets += len(tok.encode(sample["expected"])) + 1
        assert correct == evaluation["correct"]
        assert targets == evaluation["supervised_tokens"] == 69
        assert math.isclose(evaluation["nll_sum"] / targets, evaluation["answer_nll"], abs_tol=1e-14, rel_tol=0)
        test_results[name] = {"correct": correct, "examples": 10, "targets_with_eos": targets, "answer_nll": evaluation["answer_nll"]}
    tool_run = ROOT / "outputs/reviewer-tools/runs/20261020T085249354202Z"
    tool_metadata = json.loads((tool_run / "extraction.json").read_bytes())
    tool_execution = json.loads((tool_run / "execution.json").read_bytes())
    assert tool_metadata["source_sha256"] == digest(current)
    assert tool_execution["exit_code"] == 0
    tool_bundle = {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/18.md#18.10" --execute', "extraction": tool_metadata, "execution": tool_execution, "environment": json.loads((tool_run / "environment.json").read_bytes()), "code": (tool_run / "fence-1.py").read_text(), "stdout": (tool_run / "stdout.txt").read_text(), "stderr": (tool_run / "stderr.txt").read_text(), "copied_file_sha256": {name: digest((tool_run / name).read_bytes()) for name in ("extraction.json", "execution.json", "environment.json", "fence-1.py", "stdout.txt", "stderr.txt")}}
    assert tool_bundle["stdout"] == executions[0]["stdout"]
    assert tool_bundle["stderr"] == ""
    output = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-10/recheck-current-01.py", "result": "所有複查斷言通過；修正版總更新量比為5，所有沿用證據雜湊未變。", "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()}, "original_report_sha256": digest(report_raw), "source_sha256": digest(current), "chapter_sha256": digest(chapter), "integrity_receipts": receipts, "section_diff": changed_diff, "changed_sentence": {"previous": old_wording, "current": new_wording, "verified": True, "derivation": "相同50秒預算：500/100=5；100秒預算：1000/200=5。"}, "chapter_dependency_diff": {"command": chapter_diff_command, "stdout": chapter_diff, "history_read_command": history_command, "previous_chapter_sha256": digest(previous_chapter), "dependency_sections": dependency_sections, "inspection_note": "章檔的改動集中於學生尺寸限制、溫度梯度適用條件、混合目標分項作用與本節倍數措辭；同架構不因蒸餾自動縮小、隨機教師只作示範的內容仍成立。"}, "dependency_shape_probe": model_probe, "toy_executions": executions, "attribute_evaluations": test_results, "section_execution": tool_bundle, "claim_coverage": [{"id": claim["id"], "status": "verified", "basis": "新措辭與本回合輸出核對" if claim["id"] == "c21" else "已重新讀取完整新版小節；主張未改，依據內容與舊artifact雜湊確認沿用"} for claim in original["claims"]], "reused_evidence_scope": "本審閱者先前直接查核的原始論文、官方文件、完整GSM8K選題、hard targets與NLL計分器探針沿用原始artifact；本回合只讀指定原報告與其明確引用的證據。", "limitations": ["未重跑原始CUDA訓練、載入保存權重重測NLL或量測部署效能。", "保存實驗的compression.py SHA與現行程式不同；現行規則探針不等同逐byte重現當次訓練。", "單seed與小樣本的證據範圍未擴張；未執行初學讀者理解審閱。"]}
    target = artifact_root / "verification-current-01.json"
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"artifact": target.relative_to(ROOT).as_posix(), "sha256": digest(target.read_bytes()), "result": output["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
