"""同一chapter-14 reviewer的source15有界回核；原執行證據不覆寫。"""

import contextlib
import hashlib
import io
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path.cwd()
ORIGINAL = Path("docs/course-revision-20261029-omni/reviews/technical/technical_chapter_14/technical-original-v1.json")
FREEZE = Path("docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json")


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def current_section(path, lesson):
    text = (ROOT / path).read_text()
    heading = re.search(r"^## " + re.escape(lesson) + r" ", text, re.M)
    start = heading.start()
    following = re.search(r"^## ", text[start + 1 :], re.M)
    end = start + 1 + following.start() if following else len(text)
    return text[start:end]


assert sha(ORIGINAL) == "2f63adb82b7390f7b68fd02805554eff69e9d27cfb7ac35dfa35f25905f82c10"
assert sha(FREEZE) == "1094ea8f343a414dc3cbe5401b5753641fdb467d17a4867ef87dc192fe412238"
original = json.loads((ROOT / ORIGINAL).read_bytes())
freeze = json.loads((ROOT / FREEZE).read_bytes())
pages = {page["page_id"]: page for page in freeze["pages"]}
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_chapter_14",
    "review_kind": "same_owner_source15_recheck",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    "freeze": {"path": str(FREEZE), "sha256": sha(FREEZE)},
    "original_report": {"path": str(ORIGINAL), "sha256": sha(ORIGINAL)},
}
page = pages["chapter-14"]
chapter = (ROOT / page["source"]).read_text()
intro = chapter[: re.search(r"^## ", chapter, re.M).start()]
assert intro.encode() == (ROOT / page["snapshot"]).read_bytes()
assert sha(page["snapshot"]) == page["source_sha256"]
assert page["figures_sha256"] == {} and "```" not in intro and "![" not in intro
assert "完成[一層文字模型](04.md#4.6)後" in intro
assert "接著在[14.11–14.18](14.md#14.11)學 Looped Transformer" in intro
assert "最後在[14.19](14.md#14.19)把位置旋轉延伸到時間與圖塊的高、寬座標。" in intro
result["current_intro"] = {"snapshot": page["snapshot"], "sha256": page["source_sha256"], "matches_current": True, "figure_map": {}, "I1_route_replacement_present": True, "I2_anchor_4_6_present": True}
result["unchanged_reuse_proof"] = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        assert sha(source["path"]) == source["sha256"]
        result["unchanged_reuse_proof"].append({"kind": "repository_code", "path": source["path"], "sha256": source["sha256"]})
for artifact in original["artifacts"]:
    assert sha(artifact["path"]) == artifact["sha256"]
    result["unchanged_reuse_proof"].append({"kind": "original_artifact", "path": artifact["path"], "sha256": artifact["sha256"]})
result["context_sections"] = []
for lesson in ["4.5", "4.6", *[f"14.{i}" for i in range(11, 20)]]:
    context = pages[lesson]
    current = current_section(context["source"], lesson)
    assert current.encode() == (ROOT / context["snapshot"]).read_bytes()
    assert sha(context["snapshot"]) == context["source_sha256"]
    result["context_sections"].append({"lesson_id": lesson, "snapshot": context["snapshot"], "sha256": context["source_sha256"], "matches_current": True, "heading": current.splitlines()[0]})
prerequisite45 = (ROOT / pages["4.5"]["snapshot"]).read_text()
prerequisite46 = (ROOT / pages["4.6"]["snapshot"]).read_text()
assert "class Block(nn.Module):" in prerequisite45
assert "class TinyLM(nn.Module):" in prerequisite46 and "self.output(self.final_norm(x))" in prerequisite46
result["I2_semantic_anchor_check"] = "4.5的Block輸出同寬表示；4.6的TinyLM包含embedding、blocks、最後norm與候選字output，符合一層文字模型的前提標籤。"
context19 = (ROOT / pages["14.19"]["snapshot"]).read_text()
blocks = re.findall(r"```python\n(.*?)\n```", context19, re.S)
assert len(blocks) == 1
namespace = {"__name__": "__chapter14_source15_context__"}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(blocks[0], pages["14.19"]["snapshot"] + ":python-block-1", "exec"), namespace)
axis_rope, coords, content = namespace["axis_rope"], namespace["coordinates"], namespace["content"]
base = axis_rope(content, coords(1300.0))
height = axis_rope(content, coords(1300.0, h=1.0))
width = axis_rope(content, coords(1300.0, w=1.0))
assert torch.equal(base[:2], height[:2]) and torch.equal(base[4:], height[4:]) and not torch.equal(base[2:4], height[2:4])
assert torch.equal(base[:4], width[:4]) and not torch.equal(base[4:], width[4:])
result["new_route_context_diagnostic"] = {"source": pages["14.19"]["snapshot"], "source_sha256": pages["14.19"]["source_sha256"], "executed_exact_python_blocks": 1, "stdout": stdout.getvalue(), "height_changes_only_height_pair": True, "width_changes_only_width_pair": True, "scope": "只支持導讀新增三軸旋轉路標；不簽核14.19整節或聲稱訓練品質。"}
result["reader_stage_note"] = "--reader-task沿用實際packet身分排除參數/root/rv1019_reader_ch14；未宣稱source15新reader或proper wave02 reader已完成，也未封stage。"
print(json.dumps(result, ensure_ascii=False, indent=2))
