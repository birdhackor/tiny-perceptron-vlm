"""9.13 的短 CPU 診斷；只執行凍結內嵌程式、練習與 Auto 分支。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import torch

from tiny_perceptron.lesson_products.inference import auto_solve


ROOT = Path(__file__).resolve().parents[4]
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
TASK = "/root/omni_technical_coordinator/technical_9_13"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(code):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "<frozen-9.13-inline>", "exec"), namespace)
    return namespace, output.getvalue()


def current_section(path, section):
    raw = (ROOT / path).read_bytes()
    start = re.search(rb"^## " + re.escape(section.encode()) + rb" ", raw, re.M).start()
    following = re.search(rb"^## ", raw[start + 3 :], re.M)
    end = start + 3 + following.start() if following else len(raw)
    return raw[start:end]


print("created_at", datetime.now(UTC).isoformat())
print("reviewer_task", TASK)
print("python", sys.version.replace("\n", " "))
print("platform", platform.platform())
print("torch", torch.__version__)
print("device cpu; no model weights, training, GPU, or external API calls")
freeze = json.loads((ROOT / FREEZE).read_bytes())
for lesson in ["7.12", "9.9", "9.11", "9.13", "9.14", "18.1", "19.17"]:
    page = next(p for p in freeze["pages"] if p["page_id"] == lesson)
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert digest(raw) == page["source_sha256"]
    assert current_section(page["source"], lesson) == raw
    print("freeze_and_current_match", lesson, page["snapshot"], digest(raw))
    if lesson == "9.13":
        own_page, own_raw = page, raw

assert own_page["figures_sha256"] == {}
assert not re.search(rb"!\[[^\]]*\]\(", own_raw)
print("figures none: no image references and exact empty frozen figure map")
blocks = re.findall(r"```python\n(.*?)```", own_raw.decode(), re.S)
assert len(blocks) == 2
notebook_path = ROOT / own_page["notebook"]
notebook_raw = notebook_path.read_bytes()
assert digest(notebook_raw) == own_page["notebook_sha256"]
notebook = json.loads(notebook_raw)
code_cells = ["".join(c["source"]).strip() for c in notebook["cells"] if c["cell_type"] == "code"]
for i, block in enumerate(blocks):
    assert block.strip() in code_cells
    print("inline_block", i + 1, "sha256", digest(block.encode()), "matches_notebook True")

ns, output = run(blocks[0])
print("baseline_exact_inline_output")
print(output, end="")
expected_suffixes = [
    "資訊充分 yes 下一步 查卡片進度",
    "資訊充分 no 下一步 詢問申請編號",
    "資訊充分 yes 下一步 查地址變更規則",
    "資訊充分 不適用 下一步 轉給真人客服",
]
assert len(output.splitlines()) == 4
assert all(line.endswith(tail) for line, tail in zip(output.splitlines(), expected_suffixes))

exercise = blocks[0].replace(
    "for m in materials:", 'materials.append({"申請編號": "K17"})\nfor m in materials:', 1
)
_, exercise_output = run(exercise)
print("exercise_added_material_output")
print(exercise_output.splitlines()[-1])
assert exercise_output.splitlines()[-1].endswith("資訊充分 no 下一步 詢問任務")

print("dict_API_probes", json.dumps({
    "get_missing_returns_None": {}.get("任務") is None,
    "membership_checks_keys": "查卡片進度" in ns["needs"],
    "membership_does_not_check_values": "K17" not in {"申請編號": "K17"},
}, ensure_ascii=False))
try:
    {}["任務"]
except KeyError:
    print("subscript_missing_raises KeyError")
else:
    raise AssertionError("missing key did not raise")

loop = "for m in materials:" + blocks[0].split("for m in materials:", 1)[1]
probes = [
    {"任務": "查卡片進度", "申請編號": "OTHER_PERSON"},
    {"任務": "查卡片進度", "申請編號": None},
    {"任務": "查卡片進度", "無關文字": "K17"},
    {"任務": "查地址變更規則", "無關文字": "很多額外文字"},
]
probe_ns = {"materials": probes, "needs": ns["needs"]}
with contextlib.redirect_stdout(io.StringIO()) as output_buffer:
    exec(loop, probe_ns)
probe_output = output_buffer.getvalue()
print("presence_only_scope_probes")
print(probe_output, end="")
assert [" yes " in row for row in probe_output.splitlines()] == [True, True, False, True]

numeric_ns, numeric_output = run(blocks[1])
print("numeric_exact_inline_output")
print(numeric_output, end="")
assert numeric_output == "P(yes) 0.03 P(no) 0.97 信心 0.97\n"
assert Decimal(1) - Decimal("0.03") == Decimal("0.97")
print("decimal_hand_formula 1 - 0.03 = 0.97; max(0.03, 0.97) = 0.97")


class StubB:
    def synchronize(self):
        pass

    def solve(self, *args, **kwargs):
        raise AssertionError("unsupported or missing-info branch must not call solver")


class StubC:
    def __init__(self, choice, info):
        self.choice, self.info, self.calls = choice, info, []

    def synchronize(self):
        pass

    def judge(self, state, question):
        self.calls.append(question)
        return {"selected": self.choice if question == "choice" else self.info, "seconds": 0.0}


print("actual_auto_solve_branch_probes_with_stubs")
for choice, info, expected in [
    ("none", "yes", "unsupported"),
    ("none", "no", "unsupported"),
    ("boxes", "no", "clarification_needed"),
]:
    c = StubC(choice, info)
    result = auto_solve(StubB(), c, "兩盒追蹤;A=2;B=6;事件=交換;查詢=B")
    assert result["status"] == expected and result["b_loops"] == 0
    assert c.calls == ["choice", "info"]
    print(json.dumps({"choice": choice, "info": info, "status": result["status"],
                      "b_loops": result["b_loops"], "judge_calls": c.calls}, ensure_ascii=False))
print("auto_scope unsupported overrides info answer; both judgments are computed before branch")
print("inference_py_sha256", digest((ROOT / "tiny_perceptron/lesson_products/inference.py").read_bytes()))
print("PASS: all assertions; no empirical model-quality claim tested")
