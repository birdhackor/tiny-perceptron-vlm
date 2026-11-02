"""原 technical owner 對 fix14/12.2 的有界複核；不覆寫歷史證據。"""

import contextlib
import hashlib
import io
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
ART = Path("docs/technical-reviews/artifacts/omni-technical-technical_12_2-v1")
ORIGINAL = Path("docs/course-revision-20261029-omni/reviews/technical/technical_12_2/technical-original-v1.json")
FREEZE = Path("docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


original_sha = sha(ROOT / ORIGINAL)
assert original_sha == "7b9dde3b9be58b5966f9037297b67e7068983f7db677e6b9f87681f2236879dd"
original = json.loads((ROOT / ORIGINAL).read_text())
assert len(original["claims"]) == 23
for artifact in original["artifacts"]:
    assert sha(ROOT / artifact["path"]) == artifact["sha256"]
repository_sources = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        digest = sha(ROOT / source["path"])
        assert digest == source["sha256"]
        repository_sources.append({"id": source["id"], "path": source["path"], "sha256": digest, "unchanged": True})
    elif source["kind"] in ("official_source", "official_docs"):
        assert sha(Path(source["original_cache_path"])) == source["original_sha256"]

assert sha(ROOT / FREEZE) == "caaa502b629d60e77c1f6d06373419e007a47bb1a089501d7bf10f063a9b5e29"
freeze = json.loads((ROOT / FREEZE).read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "12.2")
snapshot = ROOT / page["snapshot"]
raw = snapshot.read_bytes()
assert sha(snapshot) == page["source_sha256"]
assert page["figures_sha256"] == {}
body = raw.decode()
current = re.search(r"^## 12\.2 .*?(?=^## |\Z)", (ROOT / page["source"]).read_text(), re.M | re.S).group()
assert current.encode() == raw
old_body = (ROOT / "docs/course-review-20261019/freeze-05/sources/12.2.md").read_text()
old_tail = "把 `7000.0` 改成 `6500.0`，核對最後一行仍印 True。"
new_tail = "把 `7000.0` 改成 `6500.0`，也把兩行print中的頻率名稱改成9500Hz、6500Hz，核對最後一行仍印 True。"
assert old_body.count(old_tail) == 1
assert old_body.replace(old_tail, new_tail) == body
old_blocks = re.findall(r"```python\n(.*?)```", old_body, re.S)
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 2 and blocks == old_blocks
old_execution = json.loads((ROOT / ART / "audit-output-v2.json").read_text())
fresh_outputs = []
for i, block in enumerate(blocks):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(block, f"fix14-12.2-block-{i + 1}", "exec"), namespace)
    assert output.getvalue() == old_execution["embedded_execution"][i]["stdout"]
    fresh_outputs.append({"block": i + 1, "code_sha256": hashlib.sha256(block.encode()).hexdigest(), "stdout": output.getvalue()})

# 嚴格依 fix14 的四項替換：兩個 tone 參數及兩個 print 中的標籤。
exercise = blocks[1].replace("9000.0", "9500.0").replace("7000.0", "6500.0")
exercise = exercise.replace("9000Hz", "9500Hz").replace("7000Hz", "6500Hz")
namespace = {}
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(exercise, "fix14-12.2-exercise", "exec"), namespace)
expected = (
    "樣本數 16\n"
    "9500Hz前四點 [0.0, -0.278, 0.462, -0.49]\n"
    "6500Hz前四點 [0.0, 0.278, -0.462, 0.49]\n"
    "每一點都只差正負號 True\n"
)
assert output.getvalue() == expected
high, low = namespace["high"], namespace["low"]
difference = (high + low).abs()
tolerance = 1e-5 + 1e-5 * low.abs()
assert high.shape == low.shape == (16,)
assert high.dtype == low.dtype == torch.float32
assert high.device.type == low.device.type == "cpu"
assert bool((difference <= tolerance).all())
notebook = ROOT / page["notebook"]
assert sha(notebook) == page["notebook_sha256"]
cells = json.loads(notebook.read_text())["cells"]
notebook_code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
notebook_markdown = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
assert all(any(b.strip() in cell for cell in notebook_code) for b in blocks)
assert "也把兩行print中的頻率名稱改成9500Hz、6500Hz" in notebook_markdown
assert sha(ROOT / ORIGINAL) == original_sha

print(json.dumps({
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_12_2",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_12_2-v1/fix14-recheck.py",
    "environment": {"python": sys.version, "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu"},
    "original_report": {"path": str(ORIGINAL), "sha256": original_sha},
    "original_issue": {"json_pointer": "/issues/0", "issue_id": "i1", "necessity": "optional"},
    "source_freeze": {"path": str(FREEZE), "sha256": sha(ROOT / FREEZE)},
    "snapshot": {"path": page["snapshot"], "sha256": sha(snapshot)},
    "source_matches_snapshot": True, "figure_map": {},
    "body_delta": {"exactly_one_instruction_insertion": True, "old_tail": old_tail, "new_tail": new_tail},
    "both_fenced_blocks_unchanged": True, "fresh_embedded_execution": fresh_outputs,
    "fresh_exercise": {"code_sha256": hashlib.sha256(exercise.encode()).hexdigest(), "stdout": output.getvalue(),
                       "samples_each": 16, "dtype": str(high.dtype), "max_abs_high_plus_low": float(difference.max()),
                       "min_tolerance_margin": float((tolerance - difference).min()), "allclose": True,
                       "labels_match_frequencies": True},
    "original_repository_sources": repository_sources,
    "original_artifact_digests_unchanged": True,
    "notebook": {"path": page["notebook"], "sha256": sha(notebook), "contains_exact_code_blocks": True, "contains_new_label_instruction": True},
    "claim_coverage": {"original_claim_ids": [c["id"] for c in original["claims"]],
                       "unchanged_claims_rechecked_against_full_current_body": True,
                       "changed_exercise_claim": "c21; labels now align after following the added instruction"},
    "prior_evidence_reuse": "本人先前已核實的23項原始來源及未變code/WAV證據沿用，指紋現場確認；此次親跑兩區塊及新標籤練習。沒有宣稱重跑331份Notebook，也沒有重新寫歷史WAV或重現其derived全檔SHA。",
    "result": "PASS: fix14 instruction resolves original optional i1; all current assertions completed",
}, ensure_ascii=False, indent=2))
