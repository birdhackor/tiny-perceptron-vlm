"""核對凍結修文、沿用證據及數字片段邊界；只產生新的執行紀錄。"""

import difflib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch


from scripts.selftrained import train
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


old_report_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_19-4/technical-19-4-original-01.json"
old = load(old_report_path)
assert sha(old_report_path) == "e78127a45bac357f4c6fefb5079f3657b023c45c7432ee9c3700c1dc9dbcd58a", "原始報告指紋不同"
source_path = ROOT / "docs/course-review-20261019/freeze-05/sources/19.4.md"
source_raw = source_path.read_bytes()
assert sha(source_path) == "7e4c6d325cd2801fd569b5a27535c77741f152d61c949d8a0f86e342f311143e", "凍結原文指紋不同"
old_source = (ROOT / "docs/course-review-20261019/freeze-02/sources/19.4.md").read_text()
new_source = source_raw.decode()
chapter = (ROOT / "course/chapters/19.md").read_text()
start = chapter.index("## 19.4 ")
end = chapter.index("\n## ", start + 1) + 1
assert chapter[start:end].encode() == source_raw, "現行小節與凍結原文不同"

command = ["git", "diff", "--no-index", "docs/course-review-20261019/freeze-02/sources/19.4.md", "docs/course-review-20261019/freeze-05/sources/19.4.md"]
diff = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
assert diff.returncode == 1, "兩份原文應有可核對的差異"
removed = [line[1:] for line in diff.stdout.splitlines() if line.startswith("-") and not line.startswith("---")]
added = [line[1:] for line in diff.stdout.splitlines() if line.startswith("+") and not line.startswith("+++")]
assert len(removed) == len(added) == 2, "原文改動範圍不同"

source_audit = []
for source in old["sources"]:
    if source["kind"] != "repository_code":
        continue
    current = sha(ROOT / source["path"])
    source_audit.append({"id": source["id"], "path": source["path"], "expected_sha256": source["sha256"], "current_sha256": current, "unchanged": current == source["sha256"]})
assert all(s["unchanged"] for s in source_audit), "沿用的程式或資料已改變，需要重新核對"
artifact_audit = []
for artifact in old["artifacts"]:
    current = sha(ROOT / artifact["path"])
    artifact_audit.append({"id": artifact["id"], "path": artifact["path"], "expected_sha256": artifact["sha256"], "current_sha256": current, "unchanged": current == artifact["sha256"]})
assert all(a["unchanged"] for a in artifact_audit), "沿用的證據已改變"
previous_results = load(ARTIFACT / "behavior-results.json")
previous_commands = load(ARTIFACT / "command-checks.json")
raw_audit = []
for path, expected in {**previous_results["source_sha256"], **previous_commands["source_hashes"]}.items():
    current = sha(ROOT / path)
    raw_audit.append({"path": path, "expected_sha256": expected, "current_sha256": current, "unchanged": current == expected})
assert all(s["unchanged"] for s in raw_audit), "先前執行使用的原始來源檔已改變"

torch.set_num_threads(1)
tokenizer = CharacterTokenizer.build(["-0123456789。a"], required_chars="-0123456789。a")


def label_id(symbol):
    if symbol == "<eos>":
        return tokenizer.eos_id
    if symbol == "<ignored>":
        return -100
    return tokenizer.character_ids[symbol]


cases = [
    ("負號、數字、句號、EOS與忽略位置", ["-", "1", "2", "。", "a", "3", "<eos>", "<ignored>"], [4, 4, 4, 4, 1, 4, 4, 0]),
    ("只選數字後第一個非數字位置", ["1", "。", "a", "<eos>"], [4, 4, 1, 1]),
    ("忽略間隔不跨接後面的EOS", ["1", "<ignored>", "<eos>"], [4, 0, 1]),
    ("負號不跨忽略間隔連到數字", ["-", "<ignored>", "1", "。"], [1, 0, 4, 4]),
    ("不是數字前導的負號不加權", ["-", "a", "<eos>"], [1, 1, 1]),
    ("連續數字與最後的EOS", ["1", "2", "3", "<eos>"], [4, 4, 4, 4]),
    ("被忽略的位置保持零權重", ["<ignored>", "-", "1", "<ignored>"], [0, 4, 4, 0]),
]
observations = []
for name, symbols, expected in cases:
    labels = torch.tensor([[label_id(s) for s in symbols]], dtype=torch.int64)
    records = [{"task": "text", "split": "train"}]
    observed = train.language_loss_weights(labels, records, tokenizer, 1, 4)
    mask = train.numeric_run_loss_mask(labels, tokenizer)
    assert observed.tolist()[0] == expected, "數字片段權重與手列預期不同"
    observations.append({"name": name, "symbols": symbols, "labels": labels.tolist(), "labels_dtype": str(labels.dtype), "shape": list(labels.shape), "expected_weights": expected, "observed_weights": observed.tolist()[0], "numeric_mask": mask.tolist()[0], "matched": True})

# 額外核對表格所稱工具與數字片段同時加權：權重相乘，最後以總權重正規化。
labels = torch.tensor([[label_id(s) for s in cases[0][1]]], dtype=torch.int64)
records = [{"task": "tool_call", "split": "train"}]
weights = train.language_loss_weights(labels, records, tokenizer, 4, 4)
expected_tool_weights = [16, 16, 16, 16, 4, 16, 16, 0]
assert weights.tolist()[0] == expected_tool_weights, "工具與數字片段乘數不同"
logits = torch.arange(labels.numel() * tokenizer.vocab_size, dtype=torch.float32).reshape(1, labels.numel(), tokenizer.vocab_size) / 17
output = {"logits": logits}
observed_loss = train.weighted_language_loss(output, labels, records, tokenizer, 4, 4)
ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none").reshape_as(labels)
manual_weights = torch.tensor([expected_tool_weights], dtype=torch.float32)
expected_loss = (ce * manual_weights).sum() / manual_weights.sum()
assert torch.allclose(observed_loss, expected_loss, atol=1e-7, rtol=0), "加權CE與獨立手列權重不同"

helper = ROOT / "outputs/reviewer-tools/runs/20261020T085253937788Z"
execution = load(helper / "execution.json")
environment = load(helper / "environment.json")
assert execution["exit_code"] == 0, "新版小節程式執行失敗"
assert execution["source_sha256"] == sha(source_path), "小節程式執行來源與凍結版本不同"
assert (helper / "stdout.txt").read_text() == json.loads(previous_results["section_execution"]["execution.json"]).get("unused", previous_results["section_execution"]["stdout.txt"]), "新版短程式輸出與未變的索引不同"
figure = ROOT / "course/figures/p6-19-start-lineage.svg"
png = ROOT / "outputs/course-review-20261019/figure-renders-04/f705e60221752569ad9fcab6d342b51458faa3cdccc860005cbc9a5452774035.png"
assert sha(figure) == old["figure_sha256"]["course/figures/p6-19-start-lineage.svg"], "圖檔已改變"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-4/verify-current-01.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu"},
    "original_report": {"path": str(old_report_path.relative_to(ROOT)), "sha256": sha(old_report_path)},
    "source_freeze": {"path": "docs/course-review-20261019/freeze-05.json", "sha256": sha(ROOT / "docs/course-review-20261019/freeze-05.json")},
    "source_sha256": sha(source_path), "live_section_matches_freeze": True,
    "diff": {"command": command, "returncode": diff.returncode, "stdout": diff.stdout, "stderr": diff.stderr, "removed": removed, "added": added},
    "repository_source_audit": source_audit, "previous_artifact_audit": artifact_audit,
    "previous_execution_input_audit": raw_audit,
    "numeric_cases": observations,
    "tool_and_numeric": {"expected_weights": expected_tool_weights, "observed_weights": weights.tolist()[0], "valid_targets": int(labels.ne(-100).sum()), "weight_denominator": float(manual_weights.sum()), "expected_loss": float(expected_loss), "observed_loss": float(observed_loss), "absolute_tolerance": 1e-7, "matched": True},
    "weighted_job_coefficients": {name: {key: row["job"].get(key, 1) for key in ["tool_loss_weight", "numeric_run_loss_weight"]} for name, row in previous_results["raw_lineage"].items() if name.endswith("weighted")},
    "section_execution": {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/19.md#19.4\" --execute", "execution_json": execution, "environment_json": environment, "extraction_json": load(helper / "extraction.json"), "stdout": (helper / "stdout.txt").read_text(), "stderr": (helper / "stderr.txt").read_text(), "output_matches_previous": True},
    "figure": {"svg_path": str(figure.relative_to(ROOT)), "svg_sha256": sha(figure), "viewed_png": str(png.relative_to(ROOT)), "png_sha256": sha(png), "inspection_note": "本回合已用 view_image 看指定 PNG；八段箭頭方向、自己的 best.pt、重建 optimizer／抽樣狀態皆與未變的原碼及原始訓練收據一致，沒有把節點尺度當作步數或品質。PNG 不另複製成 artifact。"},
    "independence_note": "只讀自己的原始報告與其明確列出的證據，未讀其他角色報告。outputs 只讀指定PNG及本回合 section_facts 產生的執行紀錄；正式來源鏈沿用SHA不變的原始repo收據，不讀被禁止的舊審閱。",
    "result": "兩處修文與正式數字片段mask一致；七個邊界算例與工具乘數／CE算例均符合；舊報告所有引用來源及artifact雜湊不變，新版小節輸出仍為九行。",
}
path = ARTIFACT / "recheck-current-01.json"
with path.open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"result": "核對完成", "changed_lines": len(added), "boundary_cases": len(observations), "previous_sources_unchanged": len(source_audit), "previous_artifacts_unchanged": len(artifact_audit), "previous_execution_inputs_unchanged": len(raw_audit), "section_output_lines": len(result["section_execution"]["stdout"].splitlines())}, ensure_ascii=False))
