"""保存 validation 頁面的局部重算、原始收據與命令介面核對。"""

import contextlib
import hashlib
import html
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import torch


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def command(argv, **options):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, **options)
    return {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def main():
    result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-validation/audit.py",
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "device": "cpu", "platform": platform.platform()},
              "revision": command(["git", "rev-parse", "HEAD"])["stdout"].strip()}
    paths = [
        "docs/validation.md", "docs/technical-review-guide.md", "scripts/check_notebooks.py",
        "scripts/build_course.py", "scripts/check_course_reviews.py", "scripts/check_technical_reviews.py",
        "docs/review-tools/check_review_round.py", "docs/editorial-guide.md",
        "scripts/selftrained/train.py", "tiny_perceptron/selftrained/dataset.py",
        "tiny_perceptron/selftrained/model.py", "tiny_perceptron/selftrained/inference.py",
        "tiny_perceptron/lesson_products/data.py", "tiny_perceptron/lesson_products/model.py",
        "tiny_perceptron/lesson_products/training.py", "tiny_perceptron/lesson_products/evaluation.py",
        "tiny_perceptron/lesson_products/inference.py", "scripts/gpu_smoke_check.py", "scripts/modal_smoke.py",
        "docs/gpu-smoke-result.json", "docs/selftrained/results/v2-final-public-results.json",
        "docs/selftrained/v2-manifest.json", "docs/selftrained/v2-public-cpu-commands.md",
        "docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-review.json",
        "docs/selftrained/infrastructure/v2-local-stage/independent-review/review-final.md",
        "docs/lesson-products/results/initial-b-execution.json", "docs/lesson-products/results/initial-c-execution.json",
        "docs/lesson-products/results/mechanism-fixed3-runner-result.json",
        "docs/lesson-products/results/mechanism-odd135-runner-result.json",
        "docs/lesson-products/results/final-report-v1.json", "docs/lesson-products/results/settings-freeze-v1.json",
        "docs/lesson-products/results/pretest-freeze-record-v1.json", "docs/lesson-products/results/final-report-v1.started.json",
        "docs/lesson-products/data-v2-manifest.json", "docs/lesson-products/results/settings-selection-v1.json",
        "docs/natural-assistant/v4/selection.json", "course/chapters/20.md",
        "docs/course-experiments/README.md", "docs/lesson-products/README.md",
        "scripts/reading_time.py", "pyproject.toml", "docs/natural-assistant/v4/TRAINING.md",
        "docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json",
        "scripts/selftrained/train_local_stage.py",
        "scripts/selftrained/chat.py", "scripts/selftrained/hf_transport.py",
        "docs/selftrained/infrastructure/v2-local-stage/independent-review/final-independent-actual-results.json",
    ]
    result["input_hashes"] = {p: sha((ROOT / p).read_bytes()) for p in paths}
    result["frozen_hashes"] = {p: sha((ROOT / p).read_bytes()) for p in [
        "docs/course-review-20261019/freeze-04/sources/validation.md", "docs/course-review-20261019/freeze-04.json"]}
    manifest = read("docs/lesson-products/data-v2-manifest.json")
    from tiny_perceptron.lesson_products.data import build_rows, audit_rows, canonical
    b, c, config = build_rows()
    audit = json.loads(json.dumps(audit_rows(b, c)))
    result["regenerated_data"] = {"audit": audit, "config_equal": config == manifest["config"],
                                   "audit_equal": audit == manifest["audit"], "files": {}}
    for name, rows in (("b.jsonl", b), ("c.jsonl", c)):
        digest = hashlib.sha256()
        size = 0
        for row in rows:
            raw = (canonical(row) + "\n").encode()
            digest.update(raw)
            size += len(raw)
        actual = {"sha256": digest.hexdigest(), "bytes": size, "rows": len(rows)}
        result["regenerated_data"]["files"][name] = {"actual": actual, "matches": actual == manifest["files"][name]}
    assert result["regenerated_data"]["audit_equal"]
    assert all(v["matches"] for v in result["regenerated_data"]["files"].values())
    del b, c
    from tiny_perceptron.lesson_products.model import Config, Tokenizer, LoopedSolver, CandidateScorer
    tokenizer = Tokenizer()
    result["parameters"] = {"tokenizer": len(tokenizer),
                            "b": sum(p.numel() for p in LoopedSolver(Config(vocab_size=len(tokenizer))).parameters()),
                            "c": sum(p.numel() for p in CandidateScorer(Config(vocab_size=len(tokenizer), width=64, ff=128)).parameters())}
    assert result["parameters"]["b"] == 170003 and result["parameters"]["c"] == 84545
    from tiny_perceptron.lesson_products.training import encoded_data, step, checkpoint, verify_resume
    torch.set_num_threads(1)
    torch.manual_seed(821)
    config = Config(vocab_size=len(tokenizer), width=16, ff=32)
    tiny = LoopedSolver(config)
    rows = [{"task": "boxes", "state": "兩盒追蹤;A=1;B=2;事件=交換;查詢=A", "gold": 2,
             "world": {"events": [("swap", 0)]}}]
    data = encoded_data(rows, "b", tokenizer, config)
    optimizer = torch.optim.AdamW(tiny.parameters(), lr=0.001, weight_decay=0.01, foreach=False)
    generator = torch.Generator().manual_seed(903)
    options = {"batch_size": 1, "mode": "fixed3", "depth_weights": [1, 1, 1, 1, 1]}
    result["resume_cpu_probe"] = {"first_update": step(tiny, optimizer, data, generator, **options)}
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        target = Path(temporary) / "checkpoint.pt"
        checkpoint(target, tiny, optimizer, generator, {"step": 1})
        result["resume_cpu_probe"]["restored_next_step"] = verify_resume(target, "b", "cpu", data, options)
    result["training_receipts"] = {}
    for name, path in {
        "b": "initial-b-execution.json", "c": "initial-c-execution.json",
        "fixed3": "mechanism-fixed3-runner-result.json", "odd": "mechanism-odd135-runner-result.json",
    }.items():
        record = read("docs/lesson-products/results/" + path)
        training = record.get("training", record.get("execution", {}).get("training", {}))
        result["training_receipts"][name] = {k: v for k, v in training.items() if k != "history"}
    final = read("docs/lesson-products/results/final-report-v1.json")
    result["product_scores"] = {
        "b_five_loops": final["b"]["5"], "c": final["c"], "reorder": final["c_reorder"],
        "entity_shift": final["shifts"]["entity_shift"]["boxes"], "c_wording": final["shifts"]["c_wording"],
        "gates": final["gates"], "predictions": final["predictions"]}
    result["calibration_worsened"] = {k: v["after"]["cross_entropy"] > v["before"]["cross_entropy"]
                                       for k, v in final["shifts"]["c_wording"].items()}
    logits = torch.tensor([[1., 2., 3.]])
    probabilities = torch.softmax(logits, -1)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([2]))
    result["cross_entropy_probe"] = {"probabilities": probabilities.tolist(), "loss": float(loss),
                                      "negative_log_gold": float(-probabilities[0, 2].log())}
    assert torch.allclose(loss, -probabilities[0, 2].log())
    v2 = read("docs/selftrained/results/v2-final-public-results.json")
    result["v2"] = {"data_scope": v2["data_scope"], "architectures": {}}
    for name, arch in v2["architectures"].items():
        test = arch["final_test"]
        per_task = test["per_task_final_reply"]
        result["v2"]["architectures"][name] = {
            "count": test["count"], "sum_task_counts": sum(v["count"] for v in per_task.values()),
            "capability_acceptance_passed": arch["capability_acceptance_passed"],
            "per_task": {k: {"count": v["count"], "semantic": v["semantic"],
                              "tool_roundtrip": v.get("tool_roundtrip")} for k, v in per_task.items()}}
    archive = ROOT / "assets/training/selftrained-v2.tar.gz"
    package = read("docs/selftrained/v2-manifest.json")
    result["v2_archive"] = {"path": str(archive.relative_to(ROOT))}
    if archive.read_bytes()[:2] == b"\x1f\x8b":
        observed = {}
        expected = {r["path"]: r for r in package["records"]}
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if Path(member.name).name in expected:
                    raw = tar.extractfile(member).read()
                    filename = Path(member.name).name
                    observed[filename] = {"rows": len(raw.splitlines()), "sha256": sha(raw),
                                          "matches": sha(raw) == expected[filename]["sha256"]}
        result["v2_archive"]["records"] = observed
        result["v2_archive"]["totals"] = {split: sum(v["rows"] for name, v in observed.items()
                                                      if name.endswith("-" + split + ".jsonl"))
                                                for split in ("train", "validation", "test")}
    else:
        result["v2_archive"]["status"] = "LFS pointer; no media or records read"
    result["freeze_history"] = command(["git", "log", "-1", "--format=%H %cI %s", "--",
                                         "docs/lesson-products/results/settings-freeze-v1.json"])
    result["pretest_record"] = read("docs/lesson-products/results/pretest-freeze-record-v1.json")
    result["test_started"] = read("docs/lesson-products/results/final-report-v1.started.json")
    natural = read("docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json")
    result["natural_validation"] = {k: natural[k] for k in ["variants", "denominators", "selected_variant"]}
    result["build_check"] = command([sys.executable, "scripts/build_course.py", "--check"])
    result["kernel_help"] = command([sys.executable, "-m", "ipykernel", "install", "--help"])
    result["notebook_help"] = command([sys.executable, "scripts/check_notebooks.py", "--help"])
    from scripts.reading_time import METHOD, chapter_introduction
    result["reading_time_method"] = METHOD
    result["introduction_slice"] = chapter_introduction("前言\n\n## 例子\n正文\n")
    assert result["introduction_slice"] == "前言\n\n"
    result["cpu_smoke_index"] = read("docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-review.json")
    local = read("docs/selftrained/infrastructure/v2-local-stage/independent-review/final-independent-actual-results.json")
    result["local_stage_cases"] = local
    assert len(local) == 9 and all(x["expected_exit"] == x["actual_exit"] for x in local)
    result["gpu_smoke_result"] = read("docs/gpu-smoke-result.json")
    result["external_sources"] = []
    selections = [
        ("https://scikit-learn.org/stable/modules/cross_validation.html", "scikit-learn 1.9.1",
         "Cross-validation: evaluating estimator performance", "A test set should still be held out for final evaluation",
         "核對訓練、選版與最後測試的角色，以及反覆用測試集調參造成的資訊外洩。"),
        ("https://scikit-learn.org/stable/modules/calibration.html", "scikit-learn 1.9.1",
         "Temperature Scaling", "temperature scaling does not alter the accuracy of the calibrating estimator",
         "閱讀 softmax(z/T)、T 由校準集 log loss 決定及 argmax 不變；不保證分布偏移時改善。"),
        ("https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html", "PyTorch Tutorials 2.14.0+cu130",
         "Training the Model", "using the real target outputs as each next input",
         "閱讀 teacher forcing 與用自身預測接續輸入的程式分支，以及訓練與生成不同。"),
    ]
    for url, version, locator, excerpt, note in selections:
        fetched = command(["curl", "--fail", "--location", "--silent", "--show-error", url])
        assert fetched["exit_code"] == 0
        parser = TextParser()
        parser.feed(fetched["stdout"])
        plain = re.sub(r"\s+", " ", html.unescape(" ".join(parser.parts)))
        assert excerpt in plain and version in plain
        result["external_sources"].append({"url": url, "version": version, "locator": locator,
            "accessed_on": "2026-10-20", "command": fetched["argv"], "exit_code": 0,
            "response_sha256": sha(fetched["stdout"].encode()), "excerpt": excerpt,
            "inspection_note": note, "checked_original": True})
    url = "https://raw.githubusercontent.com/Chia-Hsuan-Lee/Spoken-SQuAD/master/README.md"
    fetched = command(["curl", "--fail", "--location", "--silent", "--show-error", url])
    excerpt = "Then we utilized CMU Sphinx to generate the corresponding ASR transcriptions."
    assert fetched["exit_code"] == 0 and excerpt in fetched["stdout"]
    result["external_sources"].append({"url": url, "version": "master README；實際內容以 response_sha256 綁定",
        "locator": "Corpus Description", "accessed_on": "2026-10-20", "command": fetched["argv"],
        "exit_code": 0, "response_sha256": sha(fetched["stdout"].encode()), "excerpt": excerpt,
        "inspection_note": "原作者資料說明顯示來源文字經 TTS、ASR 後產生不同中間輸入，甚至可能失去答案片段；只支持兩站錯誤需區分的動機，沒有把這個英語抽取任務的分數移用到本課中文聊天。",
        "checked_original": True})
    result["notes"] = [
        "沒有執行遠端訓練、下載私人權重、讀取 outputs 或讀取其他教材審閱報告。",
        "歷史結果摘要的數字一致不等於已重算逐題預測或證明全部歷史操作。",
        "pytest 摘要另以短探針驗證；kernel 探針因本機缺少 nbformat 與 ipykernel 未能執行。",
    ]
    result["initial_attempts"] = [
        {"command": result["command"], "exit_code": 1,
         "stderr_tail": "assert result[\"regenerated_data\"][\"audit_equal\"]\nAssertionError\n",
         "resolution": "重算結構有整數鍵，JSON 紀錄用字串鍵；改在比較前 JSON round trip，檔案位元組雜湊另行精確核對。這是探針比較方式的修正。"},
        {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-validation/command-probes.py",
         "exit_code": 1, "stderr_tail": "import nbformat\nModuleNotFoundError: No module named 'nbformat'\n",
         "resolution": "記錄環境限制，不安裝套件或把未執行 kernel 寫成成功。"},
    ]
    result["independence_incident"] = {
        "command": "rg --files docs course scripts | rg '(reading-time|reading_time|reading-plan|read-time)'",
        "observed": "輸出包含 docs/course-revision-20261016/reading-time-merge-backfill-15.json、docs/course-publish-20261008/audit-original/freeze/implementation/scripts/reading_time.py 等禁止區域的檔名。",
        "content_opened": False,
        "details": "這次檔名搜尋漏加排除條件。沒有開啟這些歷史檔案；沒有讀取其他教材審閱者的主張或判定。"}
    (HERE / "audit-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"data_audit_equal": result["regenerated_data"]["audit_equal"],
                      "parameters": result["parameters"], "v2_archive": result["v2_archive"],
                      "build_check": result["build_check"], "calibration_worsened": result["calibration_worsened"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
