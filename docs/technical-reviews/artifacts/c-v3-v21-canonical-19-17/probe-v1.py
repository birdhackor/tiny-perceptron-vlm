"""19.17 的有限 CPU 核對；不啟動訓練、final 或成品重新驗收。"""

import ast
import contextlib
import difflib
import hashlib
import inspect
import io
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.distributions import Bernoulli, Categorical

from tiny_perceptron.decision_product.data import Example
from tiny_perceptron.decision_product.model import Tokenizer as ByteTokenizer
from tiny_perceptron.decision_product.runtime import candidate_eligibility
from tiny_perceptron.decision_product.service import Device, DeviceSession
from tiny_perceptron.lesson_products.data import CANDIDATES, REQUIRED, answer, judgment
from tiny_perceptron.lesson_products.inference import EFFORT, Product, auto_solve, rule_depth

ROOT = Path(__file__).resolve().parents[4]
SNAPSHOT = "docs/course-revision-20261101-c-v3/source-freeze-v20/sources/19.17.md"
OLD_SNAPSHOT = "docs/course-revision-20261029-omni/reviews/wave-01/source-freeze/sources/19.17.md"
FILES = [
    SNAPSHOT,
    OLD_SNAPSHOT,
    "docs/course-revision-20261101-c-v3/source-freeze-v21.json",
    "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18.json",
    "tiny_perceptron/lesson_products/inference.py",
    "tiny_perceptron/lesson_products/data.py",
    "tiny_perceptron/lesson_products/model.py",
    "tiny_perceptron/lesson_products/training.py",
    "tiny_perceptron/lesson_products/evaluation.py",
    "tiny_perceptron/lesson_products/release.json",
    "tiny_perceptron/decision_product/model.py",
    "tiny_perceptron/decision_product/data.py",
    "tiny_perceptron/decision_product/runtime.py",
    "tiny_perceptron/decision_product/service.py",
    "docs/lesson-products/data-v2-manifest.json",
    "docs/lesson-products/results/settings-freeze-v1.json",
    "docs/lesson-products/results/final-report-v1.json",
    "docs/lesson-products/results/b-auto-policy.json",
    "docs/lesson-products/results/b-artifact-manifest.json",
    "docs/lesson-products/results/c-artifact-manifest.json",
    "docs/lesson-products/results/publication-package-v1.json",
    "docs/course-revision-20261101-c-v3/evidence/kernel-refresh-source21-v1.json",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def raises_value_error(function):
    try:
        function()
    except ValueError as error:
        return str(error)
    raise AssertionError("預期 ValueError")


torch.set_num_threads(2)
source = (ROOT / SNAPSHOT).read_text()
old = (ROOT / OLD_SNAPSHOT).read_text()
blocks = re.findall(r"```python\n(.*?)```", source, flags=re.S)
assert len(blocks) == 2
namespace = {}
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(blocks[0], SNAPSHOT + ":block1", "exec"), namespace)
    exec(compile(blocks[1], SNAPSHOT + ":block2", "exec"), namespace)
assert ast.dump(ast.parse(blocks[0])) == ast.dump(ast.parse(inspect.getsource(rule_depth)))
expected_lines = [
    f"兩盒追蹤 {n} 個事件 → auto 選 {3 if n <= 4 else 5} 輪" for n in range(1, 7)
] + [
    "三數加法 → auto 選 1 輪",
    "手動檔位 {'low': 1, 'medium': 3, 'high': 5}",
    "指定 medium、硬上限 2 → 可跑預算 2 輪",
]
assert output.getvalue().splitlines() == expected_lines
policy = read("docs/lesson-products/results/b-auto-policy.json")
exercise = {**policy, "medium_max_events": 2}
assert [rule_depth(exercise, "boxes", n) for n in (3, 4)] == [5, 5]

# 載入既有、已逐檔綁定的原版 B/C；只執行六個具體 smoke calls。
local_root = Path("/workspace/tiny-perceptron-vlm/outputs/lesson-products")
release = read("tiny_perceptron/lesson_products/release.json")
package = read("docs/lesson-products/results/publication-package-v1.json")
local_files = {}
for kind in ("b", "c"):
    manifest_path = local_root / kind / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert digest(manifest_path) == release["products"][kind]["manifest_sha256"]
    assert digest(manifest_path) == digest(ROOT / f"docs/lesson-products/results/{kind}-artifact-manifest.json")
    for name, record in manifest["files"].items():
        path = local_root / kind / name
        assert digest(path) == record["sha256"]
        assert path.stat().st_size == record["bytes"]
        assert package["files"][f"{kind}/{name}"]["sha256"] == digest(path)
        local_files[f"{kind}/{name}"] = {"sha256": digest(path), "bytes": path.stat().st_size}
assert digest(local_root / "b/auto-policy.json") == digest(ROOT / "docs/lesson-products/results/b-auto-policy.json")
b = Product(local_root / "b", "b", device="cpu")
c = Product(local_root / "c", "c", device="cpu")
complete = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
cases = [
    ("complete_auto", complete, "auto", 5, "answered", 3),
    ("complete_low", complete, "low", 5, "answered", 1),
    ("complete_high", complete, "high", 5, "answered", 5),
    ("manual_medium_cap2", complete, "medium", 2, "budget_limited", 2),
    ("missing_initial", "兩盒追蹤;A=?;B=6;事件=交換;查詢=B", "auto", 5, "clarification_needed", 0),
    ("unsupported_card", "卡片進度;申請編號=123456", "auto", 5, "unsupported", 0),
]
actual_calls = []
for name, state, effort, cap, status, loops in cases:
    result = auto_solve(b, c, state, effort=effort, max_loops=cap)
    assert result["status"] == status
    assert result["b_loops"] == loops
    assert result["c_seconds"] == result["choice"]["seconds"] + result["info"]["seconds"]
    assert result["seconds"] >= result["c_seconds"]
    if name == "manual_medium_cap2":
        assert result["requested_loops"] == 3 and result["solution"]["loops"] == 2
    actual_calls.append({"case": name, "state": state, "result": result})

version_error = raises_value_error(lambda: rule_depth({**policy, "version": 2}, "boxes", 1))
depth_error = raises_value_error(lambda: rule_depth({**policy, "sum_depth": 2}, "sum"))
world = {"task": "boxes", "a": 2, "b": 6, "events": [("swap", 0)], "query": "B"}
assert answer(world) == 2 and answer({**world, "a": 3}) == 3
assert judgment(world, ("a",), "info", CANDIDATES["info"]) == "no"
assert REQUIRED["boxes"] == ("a", "b", "events", "query")
assert REQUIRED["sum"] == ("x", "y", "z")

# C-v3 的 schema／tokenizer 邊界；不冒充新的神經品質試驗。
class CaptureEngine:
    def _predict(self, example, use_cache):
        return example.to_dict()


devices = [Device("d1", "record", True, None), Device("d2", "play", True, True), Device("d3", "save", True, True)]
session = DeviceSession(CaptureEngine(), devices, "zh")
device_info_example = session.information("d1")
assert "record需要許可與就緒" in device_info_example["question"]
unsupported_task_error = raises_value_error(
    lambda: DeviceSession(CaptureEngine(), [Device("d1", "sum", True, True), *devices[1:]], "zh")
)
byte_input = "三數加法x=2🙂"
byte_tokens = ByteTokenizer({}).encode(byte_input)
assert byte_tokens and 3 in byte_tokens
eligibility_example = Example(
    "裝置A：任務record；許可是；就緒是。", "選擇裝置A的動作。",
    (("x", "裝置A：record"), ("y", "裝置B：record")), "", "choice", "", "service", ""
)
assert candidate_eligibility(eligibility_example) == [True, False, False]

# 只重算 immutable 舊紀錄上的選擇及算術；不重開原 final。
settings = read("docs/lesson-products/results/settings-freeze-v1.json")
final = read("docs/lesson-products/results/final-report-v1.json")
data = read("docs/lesson-products/data-v2-manifest.json")
for name, expected in settings["source_sha256s"].items():
    assert digest(ROOT / "tiny_perceptron/lesson_products" / name) == expected
assert digest(ROOT / "docs/lesson-products/results/settings-freeze-v1.json") == final["settings_freeze_sha256"]
assert digest(ROOT / "docs/lesson-products/data-v2-manifest.json") == final["data_manifest_sha256"]
assert package["files"]["report.json"]["sha256"] == digest(ROOT / "docs/lesson-products/results/final-report-v1.json")
assert package["files"]["settings-freeze.json"]["sha256"] == final["settings_freeze_sha256"]
assert settings["test_outcomes_observed"] is False
grid = settings["auto_search"]
assert len(grid) == 45
feasible = [row for row in grid if row["quality"]["mean_actual_b_loops_all_complete"] <= 3]
selected = min(feasible, key=lambda row: (
    -row["quality"]["accuracy_end_to_end"], row["quality"]["mean_actual_b_loops_all_complete"],
    json.dumps(row["rule"], sort_keys=True)))
assert all(policy[key] == value for key, value in selected["rule"].items())
exercise_row = next(row for row in grid if row["rule"] == {"version": 1, "low_max_events": 0, "medium_max_events": 2, "sum_depth": 1})
assert exercise_row["quality"]["mean_actual_b_loops_all_complete"] > 3
counts = data["audit"]["b"]["counts"]
assert counts["test/boxes"] == 2956 and counts["test/sum"] == 42
assert counts["policy/boxes"] + counts["policy/sum"] == 3091
auto = final["auto"]["auto"]
medium = final["auto"]["medium"]
assert auto["count_complete_eligible"] == medium["count_complete_eligible"] == 2998
assert auto["covered"] == medium["covered"] == 2998
assert auto["correct_end_to_end"] == 2926 and medium["correct_end_to_end"] == 2927
auto_total = round(auto["mean_actual_b_loops_all_complete"] * 2998)
assert abs(auto["mean_actual_b_loops_all_complete"] * 2998 - auto_total) < 1e-9
assert auto_total == 2956 * 3 + 42
assert 2998 * 3 - auto_total == 42 * 2 == 84
assert final["b"]["1"]["sum"]["correct"] == 0
assert final["b"]["3"]["sum"]["correct"] == 1
assert sum(final["b"]["3"][f"boxes_length_{n}"]["count"] for n in range(1, 5)) == 2956
latency = final["latency"]
assert len(latency["sample_ids"]) == 24 and latency["warmup_per_effort"] == 3 and latency["repetitions"] == 3
assert all(row["requests"] == 72 for row in latency["measurements"].values())

official = {}
for cls, properties in [(Bernoulli, ("mean", "variance", "sample")), (Categorical, ("sample",))]:
    original = Path(inspect.getsourcefile(cls))
    excerpts = {name: inspect.getsource(getattr(cls, name).fget if isinstance(getattr(cls, name), property) else getattr(cls, name)) for name in properties}
    official[cls.__name__] = {"path": str(original), "sha256": digest(original), "class_docstring": inspect.getdoc(cls), "excerpts": excerpts}

receipt = {
    "created_at": datetime.now(UTC).isoformat(),
    "scope": "Fresh CPU snippet and six original B/C smoke calls; source/delta/metadata audit; retained fixed experiment arithmetic. No training, final, qualification or anonymous download.",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "threads": str(torch.get_num_threads())},
    "input_sha256": {name: digest(ROOT / name) for name in FILES},
    "source_delta": "".join(difflib.unified_diff(old.splitlines(keepends=True), source.splitlines(keepends=True), fromfile="source18", tofile="source21")),
    "snippet": {"verbatim_blocks": blocks, "stdout": output.getvalue(), "function_ast_matches_library": True, "manual_cap_example_is_budget_arithmetic_only": True},
    "exercise": {"boxes_3_4_depths": [rule_depth(exercise, "boxes", n) for n in (3, 4)], "retained_policy_grid_record": exercise_row, "denominator": "3091 policy complete eligible inputs; not a universal mean across arbitrary inputs"},
    "rule_errors": {"version": version_error, "depth": depth_error},
    "actual_original_bc_calls": actual_calls,
    "local_product_identity": {"root": str(local_root), "release_revision": release["revision"], "files": local_files, "policy_repository_local_package_sha_equal": True},
    "field_contract": {"boxes_required": list(REQUIRED["boxes"]), "sum_required": list(REQUIRED["sum"]), "missing_a_info_gold": "no", "indistinguishable_observed_input_two_possible_answers": [answer(world), answer({**world, "a": 3})]},
    "c_v3_boundary": {"byte_input": byte_input, "byte_tokens": byte_tokens, "captured_information_example": device_info_example, "unsupported_sum_task_error": unsupported_task_error, "id_only_candidate_eligibility": [True, False, False], "neural_quality_evaluation": False},
    "retention": {
        "selected_rule_from_45_raw_grid_rows": selected,
        "test_outcomes_observed_at_settings_freeze": settings["test_outcomes_observed"],
        "test_complete_counts": {"boxes": 2956, "sum": 42, "all": 2998, "families": data["audit"]["b"]["family_counts"]["test"]},
        "auto": auto, "medium": medium,
        "arithmetic": {"medium_total_loops": 8994, "auto_total_loops": auto_total, "loops_saved": 84, "fewer_correct": 1, "sum_correct_depth1": 0, "sum_correct_depth3": 1},
        "exact_changed_set_derivation": "Coverage=2998 means all complete inputs were accepted. Every sum has event_count=0 and receives1 for either supported C choice. Every1–4-event boxes input receives1 or3; observed total8910=42*1+2956*3 forces every boxes input to3. Thus only42 sums differ from fixedmedium, saving42*(3-1)=84 loops. Raw result counts give2927-2926=1 fewer correct.",
        "stress5_6": final["shifts"]["long_shift"]["boxes"],
        "latency": {key: value for key, value in latency.items() if key != "sample_ids"},
        "new_final_or_new_benchmark": False,
    },
    "official_installed_probability_source": official,
    "assertions": "all passed",
}
print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
