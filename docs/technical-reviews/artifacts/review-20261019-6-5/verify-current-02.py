"""核對凍結差異、既有證據完整性與目前小節的實跑輸出；只新增證據檔。"""

import difflib
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = Path.cwd()
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_6-5/technical-6-5-original-02.json"
OLD = ROOT / "docs/course-review-20261019/freeze-01/sources/6.5.md"
CURRENT = ROOT / "docs/course-review-20261019/freeze-05/sources/6.5.md"
RUN = ROOT / "outputs/reviewer-tools/runs/20261020T094357944684Z"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


old_report = load(ORIGINAL)
old_verification = load(ART / "verification.json")
old_text, current_text = OLD.read_text(), CURRENT.read_text()
assert sha(ORIGINAL) == "5f4c06c775723d69d72d9ff2df239455a18bec3a6565d170abab9233e36682db"
assert sha(CURRENT) == "9bff6436d575412635d1ea4bba877f131a08131222974e9023a513cfbe9a90d5"
freeze = ROOT / "docs/course-review-20261019/freeze-05.json"
assert sha(freeze) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"

source_receipts = []
for source in old_report["sources"]:
    if source["kind"] == "repository_code":
        actual = sha(ROOT / source["path"])
        assert actual == source["sha256"]
        source_receipts.append({"path": source["path"], "sha256": actual, "matches_original": True})
artifact_receipts = []
for artifact in old_report["artifacts"]:
    actual = sha(ROOT / artifact["path"])
    assert actual == artifact["sha256"]
    artifact_receipts.append({"path": artifact["path"], "sha256": actual, "matches_original": True})
input_receipts = []
for path, expected in old_verification["input_sha256"].items():
    if path.startswith(("tiny_perceptron/", "scripts/", "docs/course-experiments/", "course/")):
        actual = sha(ROOT / path)
        input_receipts.append({"path": path, "sha256": actual, "previous_sha256": expected, "matches_original": actual == expected})
        if path != "course/training.md":
            assert actual == expected

sentences = [
    "為什麼各位置的代價可以直接相加？tokenizer 切出的這一整串 token 的機率，是每個位置「在前文已知時」猜中下一個 token 的條件機率相乘（[1.5](01.md#1.5) 的 `p(看|貓)` 就是一個條件機率）。",
    "所以總 NLL 就是這串 token 機率的負對數（計入 EOS 時也包含 EOS）。",
    "同一段原文可能還有別種切法；本課只對 tokenizer 選定的那一種計分，不把所有切法的機率加總。",
]
assert all(sentence in current_text for sentence in sentences)
old_paragraphs = old_text.split("\n\n")
new_paragraphs = current_text.split("\n\n")
changed = [i for i, (old, new) in enumerate(zip(old_paragraphs, new_paragraphs, strict=True)) if old != new]
assert len(changed) == 1
old_sentences = old_paragraphs[changed[0]].split("。")
new_sentences = new_paragraphs[changed[0]].split("。")
operations = []
for tag, i, j, k, l in difflib.SequenceMatcher(a=old_sentences, b=new_sentences).get_opcodes():
    if tag != "equal":
        operations.append({"operation": tag, "old": old_sentences[i:j], "current": new_sentences[k:l]})

copied_names = ["stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json", "fence-1.py", "fence-2.py"]
copied = {name: (RUN / name).read_text() for name in copied_names}
extraction, execution, environment = load(RUN / "extraction.json"), load(RUN / "execution.json"), load(RUN / "environment.json")
assert extraction["source_sha256"] == sha(CURRENT)
assert execution["exit_code"] == 0 and not copied["stderr.txt"]
assert len(extraction["python_fences"]) == 2
for name in ["fence-1.py", "fence-2.py", "stdout.txt"]:
    assert copied[name] == load(ART / "section-execution.json")["copied_files"][name]

commands = []
for command in [
    [".venv/bin/python", "-m", "scripts.course_experiments.run", "--help"],
    [".venv/bin/python", "-m", "scripts.course_experiments.run", "--list-assets", "tokenizer"],
]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    commands.append({"command_argv": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
training = (ROOT / "course/training.md").read_text()
recipe = ".venv/bin/python -m scripts.course_experiments.run --experiment tokenizer --device cuda"
assert recipe in training
training_receipt = {"path": "course/training.md", "sha256": sha(ROOT / "course/training.md"), "locator": "固定課程配方", "excerpt": recipe, "inspection_note": "全文hash與初審不同；本次直接讀取引用的補充配方，tokenizer重做指令仍成立。"}

figures = []
for path, expected in old_report["figure_sha256"].items():
    assert sha(ROOT / path) == expected
    prior = old_verification["figure_checks"][path]
    png = prior["viewed_png"]
    assert sha(ROOT / png["path"]) == png["sha256"]
    figures.append({"path": path, "sha256": expected, "viewed_png": png, "inspection_note": "本次再次用view_image觀看指定PNG，並讀取SVG；數字、尺度、方向與限制文字相符。"})

record_path = ROOT / "docs/course-experiments/results/tokenizer.json"
record = load(record_path)
metrics = {}
for name, run in record["results"]["runs"].items():
    metrics[name] = {}
    for phase in ["before", "after"]:
        for split in ["test", "validation"]:
            entry = run[phase][split]
            total = entry["mean_token_nll"] * entry["effective_tokens"]
            bpb = total / (entry["raw_utf8_bytes"] * math.log(2))
            assert math.isclose(bpb, entry["bpb_including_eos_boundary_targets"], abs_tol=2e-15)
            metrics[name][f"{phase}.{split}"] = {"mean_nll": entry["mean_token_nll"], "targets": entry["effective_tokens"], "raw_bytes": entry["raw_utf8_bytes"], "total_nll": total, "recomputed_bpb": bpb}

chain = {"conditional_probabilities": [0.5, 0.25], "sequence_probability_given_prefix": 0.5 * 0.25, "total_nll": -math.log(0.5) - math.log(0.25), "negative_log_sequence_probability": -math.log(0.125)}
assert math.isclose(chain["total_nll"], chain["negative_log_sequence_probability"], abs_tol=1e-14)
boundary = old_verification["loss_and_boundaries"]
assert boundary["specials"][1:3] == ["<bos>", "<eos>"]
assert all(1 not in [target for target in row if target != -100] for row in boundary["target_ids"])
assert all(sum(target == 2 for target in row) == 1 for row in boundary["target_ids"])

receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-5/verify-current-02.py",
    "result": "核對完成",
    "environment": {"python": platform.python_version(), "torch": environment["torch"], "device": "cpu"},
    "original_report": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": sha(ORIGINAL)},
    "source_freeze": {"path": str(freeze.relative_to(ROOT)), "sha256": sha(freeze)},
    "current_source": {"path": str(CURRENT.relative_to(ROOT)), "sha256": sha(CURRENT)},
    "diff_command": "git diff --no-index docs/course-review-20261019/freeze-01/sources/6.5.md docs/course-review-20261019/freeze-05/sources/6.5.md",
    "unified_diff": "".join(difflib.unified_diff(old_text.splitlines(keepends=True), current_text.splitlines(keepends=True), fromfile=str(OLD.relative_to(ROOT)), tofile=str(CURRENT.relative_to(ROOT)))),
    "changed_sentences": sentences,
    "sentence_operations": operations,
    "changed_paragraph_count": len(changed),
    "chain_calculation": chain,
    "derivation": "給定前文h與選定序列T，P(T|h)=∏i P(t_i|h,t_<i)，所以-ln P(T|h)=Σi -ln P(t_i|h,t_<i)。EOS若列為最後有效label，其條件機率也在同一乘積內。對同一原文S可有不同T；P(S|h)=Σ[T:decode(T)=S]P(T|h)；本課的_evaluate_tokenizer每篇評分只用該篇原文的一份encode結果，再對那一串labels取loss_sum；生成範例另會編碼prompt，但不加入此總NLL，不枚舉或加總其它切分。本次改寫正確限定上述對象。",
    "primary_source_reused": {"url": "https://aclanthology.org/2023.acl-short.1.pdf", "version": "ACL 2023出版版", "locator": "Introduction與Preliminaries", "artifact": "docs/technical-reviews/artifacts/review-20261019-6-5/source-excerpts.json", "artifact_sha256": sha(ART / "source-excerpts.json"), "inspection_note": "沿用同一審閱者初審直接讀過的原始論文；本次重讀自己保存的短摘錄與公式，未讀其他角色報告。"},
    "repository_source_receipts": source_receipts,
    "reused_artifact_receipts": artifact_receipts,
    "input_receipts": input_receipts,
    "section_execution": {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/06.md#6.5\" --execute", "copied_files": copied, "copied_file_sha256": {name: sha(RUN / name) for name in copied_names}, "result": "exit_code=0；兩段原始程式與輸出和初審相同，原文hash符合新凍結版。"},
    "read_only_commands": commands,
    "training_recipe": training_receipt,
    "figure_inspection": figures,
    "record_metrics": metrics,
    "record_sha256": sha(record_path),
    "evidence_note": "此版將編碼敘述明確限於評分路徑；函式也會為生成範例編碼prompt，沒有把生成結果計入NLL。初版收據保留，數值與裁決不變。",
    "reuse_scope": "僅沿用hash相符的初審證據。資料重建、遮罩與曝光schedule不重跑，相關核心輸入hash全部相符；training.md變更的補充指令另讀另驗。",
    "unverified_scope": "不重訓、不重新評估作者權重、不估計多切分邊際化的效果大小；本次只核對修文、現行版本與既有證據能支持的結論。",
}
output = ART / "recheck-evidence-current-02.json"
with output.open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"result": "核對完成", "path": str(output.relative_to(ROOT)), "sha256": sha(output), "source_sha256": sha(CURRENT), "changed_sentence_count": len(sentences)}, ensure_ascii=False))
