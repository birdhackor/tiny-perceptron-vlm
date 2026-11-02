"""10.12有界callback：親核fix11來源、既有artifact及12.17共同時間軸承接。"""
import contextlib
import hashlib
import io
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
PREVIOUS_FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix08.json"
REPORTS = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_10_12"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pages(path):
    return {p["page_id"]: p for p in json.loads(path.read_bytes())["pages"]}


def inline(page):
    path = ROOT / page["snapshot"]
    assert sha(path) == page["source_sha256"]
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
    assert len(blocks) == 1
    return blocks[0]


pages, old_pages = load_pages(FREEZE), load_pages(PREVIOUS_FREEZE)
torch.set_num_threads(1)
namespaces, outputs, inputs = {}, {}, {}
for lesson in ["10.12", "12.17"]:
    page, old_page = pages[lesson], old_pages[lesson]
    code = inline(page)
    assert code == inline(old_page)
    chapter = (ROOT / page["source"]).read_text()
    section = re.search(r"^## " + re.escape(lesson) + r"\b[^\n]*\n.*?(?=^## |\Z)", chapter, re.M | re.S)
    assert section and section.group(0).encode() == (ROOT / page["snapshot"]).read_bytes()
    stdout, ns = io.StringIO(), {}
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, str(ROOT / page["snapshot"]), "exec"), ns)
    namespaces[lesson], outputs[lesson] = ns, stdout.getvalue().splitlines()
    inputs[lesson] = {
        "snapshot": page["snapshot"], "source_sha256": page["source_sha256"],
        "old_snapshot": old_page["snapshot"], "old_source_sha256": old_page["source_sha256"],
        "inline_python_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "inline_code_unchanged_since_fix08": True, "current_section_exact_match": True,
    }
assert outputs["10.12"] == json.loads((OUT / "execution-summary.json").read_bytes())["inline_stdout"]
context = namespaces["12.17"]
main = namespaces["10.12"]
assert outputs["12.17"] == [
    "影格／波形固定；局部 token 內容相同 True",
    "序列項數與發聲塊的 causal index 24 20",
    "A 的聲音錨點 [1100.0, 1300.0, 1500.0, 1700.0]",
    "B 的聲音錨點 [2100.0, 2300.0, 2500.0, 2700.0]",
    "移時兩題 ['圓', '方'] 整對正確 True",
    "刪圖／刪聲 不可判定 不可判定",
    "窗外擾動不改第 1 塊 True",
    "短塊補零後的支持／錨點 1000.0 1006.25 1003.125",
]
assert context["frames"].shape == (16, 1, 32, 32)
assert context["frame_ms"].tolist() == [float(i * 250) for i in range(16)]
assert torch.equal(main["frames"][0], context["circle"])
assert torch.equal(main["frames"][2], context["square"])
main["validate"](context["frames"], context["frame_ms"], 4000.0)
bridges = []
for offset, expected_time, expected_index, expected_answer in [(1000.0, 1300.0, 5, "圓"), (2000.0, 2300.0, 9, "方")]:
    tokens = context["local_tokens"](context["wave"], offset)
    active = [t for t in tokens if t["content"][-1] > 0.01]
    assert len(active) == 1
    anchor = active[0]["anchor"]
    index, image = main["at"](context["frames"], context["frame_ms"], 4000.0, anchor)
    observed_answer = "圓" if torch.equal(image, context["circle"]) else "方" if torch.equal(image, context["square"]) else "不可判定"
    assert anchor == expected_time and index == expected_index and observed_answer == expected_answer
    assert context["answer"](context["frames"], context["wave"], offset) == observed_answer
    bridges.append({"audio_start_ms": offset, "active_support_ms": [active[0]["start"], active[0]["end"]], "anchor_ms": anchor, "lookup_with_10_12_at": index, "shape_by_template": observed_answer, "agrees_with_12_17_rule": True})
previous_report = json.loads((REPORTS / "technical-current-v1.json").read_bytes())
preserved_artifacts = []
for artifact in previous_report["artifacts"]:
    actual = sha(ROOT / artifact["path"])
    assert actual == artifact["sha256"]
    preserved_artifacts.append({"path": artifact["path"], "sha256": actual})
model = next(s for s in previous_report["sources"] if s["id"] == "model")
assert sha(ROOT / model["path"]) == model["sha256"]
context_slices = []
for lesson in ["10.2", "10.4", "10.7", "12.17"]:
    page = pages[lesson]
    path = ROOT / page["snapshot"]
    assert sha(path) == page["source_sha256"]
    chapter = (ROOT / page["source"]).read_text()
    section = re.search(r"^## " + re.escape(lesson) + r"\b[^\n]*\n.*?(?=^## |\Z)", chapter, re.M | re.S)
    assert section and section.group(0).encode() == path.read_bytes()
    context_slices.append({"lesson_id": lesson, "path": page["snapshot"], "sha256": page["source_sha256"], "matches_current_section": True})
for path, expected in [(REPORTS / "technical-original-v1.json", "e4a6066eeae9a7327a5d74a418ff049264894702e5ecfefa3b577104a2141e9d"), (REPORTS / "technical-current-v1.json", "f4b4a0b423d2a5c383615c42c65dc493289b08efe6317bbdc571994dd9e5ad49")]:
    assert sha(path) == expected
for path, expected in pages["10.12"]["figures_sha256"].items():
    assert sha(ROOT / path) == expected
assert sha(ROOT / pages["10.12"]["notebook"]) == pages["10.12"]["notebook_sha256"]
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_10_12",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-10-12-v1/verify-callback-fix11.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu", "threads": "1"},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha(FREEZE)},
    "inputs": inputs, "context_slice_pins": context_slices,
    "inline_stdout": outputs, "common_timeline_bridge": bridges,
    "preserved_artifacts": preserved_artifacts,
    "implementation": {"path": model["path"], "sha256": model["sha256"], "unchanged": True},
    "result": "All assertions passed. Re-executed literal current 10.12 and 12.17 inline code on bounded CPU; 10.12 at/validate also correctly handle 12.17's sixteen frames at audio anchors 1300/2300 ms. All six previous artifacts and both previous reports remain unchanged. No model training or quality measurement.",
    "scope": "12.17 is checked only as the explicit continuation supporting 10.12 claim c20. Reading and executing its inline mechanism is not a substitute for a distinct full technical review of 12.17, nor evidence that local 200 ms features are mandatory for all audiovisual models.",
}
with (OUT / "callback-fix11-execution.json").open("x", encoding="utf-8") as f:
    f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"created_at": result["created_at"], "result": result["result"], "10.12_stdout": outputs["10.12"], "common_timeline_bridge": bridges}, ensure_ascii=False, indent=2))
