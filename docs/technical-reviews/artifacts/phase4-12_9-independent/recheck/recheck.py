"""Recheck revised wording against unchanged raw samples and retained contracts."""
import difflib
import hashlib
import json
import platform
import shutil
from pathlib import Path

ART = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ROOT = ART.parents[3]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected_new = "0930ed04d140c96d076a3ca8af5d3334f20d684293d2f6eaff8f04830d991149"
prior_report = ROOT / "docs/technical-reviews/12.9.json"
assert sha(prior_report) == "61c337daa4b6362ed7757481d7decac74fe625860af443c5931674650a77fa38"
prior = json.loads(prior_report.read_bytes())
assert prior["reviewer_task"] == "/root/phase4_factual_coordinator/factual_12_9"
shutil.copyfile(prior_report, OUT / "own-initial-revise.json")
extract = Path("/tmp/factual-12_9-independent-recheck-extract")
for name in ["section.md", "extraction.json", "fence-1.py", "bootstrap.py"]:
    shutil.copyfile(extract / name, OUT / name)
assert sha(OUT / "section.md") == expected_new
old = (ART / "extracted/section.md").read_text()
new = (OUT / "section.md").read_text()
old_paragraph = "`backward()` 後接頭梯度大於零，表示損失與聲音路徑相連；本程式沒有 optimizer step，沒有把這題教會模型。既有單音訓練另測 14 題，生成答案對 11/14，錯誤集中在邊界與時長變化；完整配方放補充，不與這段梯度示範混稱成果。"
new_paragraph = "`backward()` 後接頭梯度大於零，表示損失與聲音路徑相連；本程式沒有 optimizer step，沒有把這題教會模型。既有單音訓練另測 14 題，生成答案對 11/14；三個錯例的頻率都是 290 或 300 Hz。資料把振幅與時長一起改動，無法分辨兩者各自的影響。完整配方放補充，不與這段梯度示範混稱成果。"
assert old.replace(old_paragraph, new_paragraph) == new
diff = "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), fromfile="own original frozen section", tofile="personally read current section"))
(OUT / "source.diff.txt").write_text(diff)
identities = []
paths = ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/multimodal.py",
    "tiny_perceptron/attention.py", "scripts/course_experiments/modalities.py",
    "docs/course-experiments/results/audio.json", "course/figures/rewrite-12-frame-features.svg",
    "course/figures/rewrite-10-image-targets.svg"]
for name in paths:
    live, frozen = ROOT / name, ART / "inputs" / name
    assert sha(live) == sha(frozen)
    identities.append({"path": name, "sha256": sha(live), "same_as_personally_checked_original": True})
assert sha(OUT / "fence-1.py") == sha(ART / "extracted/fence-1.py")
identities.append({"path": "current section Python fence 1", "sha256": sha(OUT / "fence-1.py"), "same_as_original_executed_fence": True})
data = json.loads((ROOT / "docs/course-experiments/results/audio.json").read_bytes())
test = data["results"]["test"]
splits = data["results"]["data"]["splits"]
encode = lambda s: [b + 8 for b in s.encode("utf-8")]
samples = []
failed = []
for row, sample in zip(splits["test"]["records"], test["samples"], strict=True):
    ids = sample["generated_ids"]
    raw = ids[:ids.index(2)] if 2 in ids else ids
    exact = raw == encode(row["answer"])
    assert exact == sample["exact_match"]
    assert row["answer"] == ("high" if row["frequency"] > 300 else "low")
    observed = {"row": sample["row"], "frequency_hz": row["frequency"], "amplitude": row["amplitude"],
        "seconds": row["seconds"], "raw_generated_ids": ids, "target": row["answer"], "exact_match": exact}
    samples.append(observed)
    if not exact:
        failed.append(observed)
assert len(samples) == test["examples"] == 14
assert sum(s["exact_match"] for s in samples) == test["correct"] == 11
assert [r["row"] for r in failed] == [4, 6, 7]
assert [r["frequency_hz"] for r in failed] == [290, 300, 300]
paired = {}
for name, split in splits.items():
    pairs = sorted({(r["amplitude"], r["seconds"]) for r in split["records"]})
    assert pairs == [(0.25, 0.1), (0.5, 0.12)]
    by_frequency = {}
    for row in split["records"]:
        by_frequency.setdefault(str(row["frequency"]), []).append([row["amplitude"], row["seconds"]])
    assert all(sorted(x) == [[0.25, 0.1], [0.5, 0.12]] for x in by_frequency.values())
    paired[name] = {"unique_pairs": pairs, "by_frequency_pairs": by_frequency}
receipt = {
    "reviewer_task": prior["reviewer_task"], "environment": {"python": platform.python_version(), "device": "CPU, JSON-only"},
    "read_scope": "本人重新親讀新版12.9全部43行、fence、練習與details；再核自己的七個claim範圍及原始test樣本與成對設計。",
    "source": "course/chapters/12.md#12.9", "source_sha256": expected_new,
    "original_issue": {"id": "duration-confound", "original_statement": "錯誤集中在邊界與時長變化", "initial_verdict": "revise", "initial_report_sha256": sha(OUT / "own-initial-revise.json")},
    "new_paragraph": new_paragraph,
    "unchanged_identity_checks": identities,
    "json_pointers_personally_inspected": ["/results/test/examples", "/results/test/correct", "/results/test/samples", "/results/data/splits/train/records", "/results/data/splits/validation/records", "/results/data/splits/test/records"],
    "recomputed_correct": 11, "recomputed_examples": 14, "failed_rows": failed, "paired_design": paired,
    "independent_resolution": "原generated_ids重新核出三錯為row4/6/7，頻率290/300/300Hz。每個split、每個frequency都有且只有(振幅,秒數)=(0.25,0.1),(0.5,0.12)，兩因素完全共變。新版只陳述這些錯例與識別限制，沒有再歸因獨立時長效果，原問題已解決。",
    "unchanged_claim_scope_review": "其餘六claim逐一對照新版全文：入口、shape/264詞表、5有效targets、teacher forcing、backward不更新與200/low、歷史11/14均未改動。原fence/code/rawJSON及前置圖SHA相同，原親讀權威來源和短CPU執行證據支持範圍仍成立。",
    "execution_limit": "只重新彙總原JSON；沒有執行模型forward/backward、既有模型評測、訓練、GPU或新.pt。未copy runtime workspace或跟隨symlinks；只copy四個明確小型extract檔與本人初稿報告。",
    "render_scope": "新段落將以current source artifact渲染，明標審閱artifact；不宣稱8765或production頁面已更新。",
}
(OUT / "resolution-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
