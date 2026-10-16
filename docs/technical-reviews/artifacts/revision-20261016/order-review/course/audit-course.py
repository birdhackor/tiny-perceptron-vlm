"""Independent whole-entry technical checks; never trains or evaluates formal tests."""
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import torch

ROOT = Path.cwd()
BASE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def run(args):
    p = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    assert p.returncode == 0, (args, p.stdout, p.stderr)
    return {"argv": [sys.executable, *args], "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


receipt = {"reviewer": "/root/order_tech_course_20261016", "environment": {
    "python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "device": "cpu"
}, "scope": "Fresh execution of deterministic entry/navigation/checksum/local CLI checks; read and extract existing frozen results only. No training, formal evaluation, broad HF download, or UI/render test."}
body = Path("course/README.md").read_text()
freeze = next(p for p in load("docs/course-revision-20261016/freeze-18.json")["pages"] if p["page_id"] == "course")
assert digest("course/README.md") == freeze["source_sha256"]
assert freeze["figures_sha256"] == {}
receipt["source"] = {"sha256": digest("course/README.md"), "freeze_page": freeze, "sections": re.findall(r"^## (.+)$", body, re.M)}

links = []
for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", body):
    if "://" in target:
        continue
    path, _, fragment = target.partition("#")
    p = (Path("course") / path).resolve()
    assert p.is_file(), target
    if fragment:
        assert re.search(r"^## " + re.escape(fragment) + r"(?: |$)", p.read_text(), re.M), target
    links.append({"label": label, "target": target, "path": str(p.relative_to(ROOT)), "fragment": fragment, "sha256": digest(p)})
receipt["local_link_checks"] = {"count": len(links), "all_files_and_numbered_fragments_resolve": True, "links": links}

receipt["notebook_sync"] = run(["scripts/build_course.py", "--check"])
index = load("course/lesson-index.json")
assert all(Path(i["notebook"]).is_file() and load(i["notebook"])["metadata"]["lesson_id"] == i["id"] for i in index)
receipt["index"] = {"lessons": len(index), "all_notebooks_present_with_matching_lesson_id": True}

nav = tomllib.loads(Path("zensical.toml").read_text())["project"]["nav"]
def flatten(entries):
    for entry in entries:
        for value in entry.values():
            yield from flatten(value) if isinstance(value, list) else [value]
paths = list(flatten(nav))
assert paths.index("7.25.md") < paths.index("C.1.md") < paths.index("C.5.md") < paths.index("8.1.md")
assert paths.index("13.17.md") < paths.index("C.6.md") < paths.index("C.7.md") < paths.index("14.1.md")
assert [p for p in paths if re.fullmatch(r"14\.\d+\.md", p)] == [f"14.{n}.md" for n in range(1, 19)]
assert [p for p in paths if re.fullmatch(r"19\.\d+\.md", p)] == [f"19.{n}.md" for n in range(1, 21)]
receipt["navigation_order"] = {"C_split_after_7_and_13": True, "14_numeric_order_1_to_18": True, "19_numeric_order_1_to_20": True}

chapter = Path("course/chapters/01.md").read_text()
code = re.search(r"```python\n(.*?)```", chapter, re.S)[1]
ns = {}
out = io.StringIO()
with contextlib.redirect_stdout(out):
    exec(compile(code, "course/chapters/01.md:1.1", "exec"), ns)
assert ns["ids"] == [3, 2, 1, 4, 1, 2, 3, 0]
assert ns["restored"] == "貓看狗，狗看貓。"
receipt["first_lesson_execution"] = {"code_sha256": hashlib.sha256(code.encode()).hexdigest(), "stdout": out.getvalue(), "restored": ns["restored"], "repeated_cat_ids": [ns["ids"][0], ns["ids"][6]]}

raw = {}
public = load("docs/selftrained/results/v2-final-public-results.json")
for name in ("moe", "dense"):
    path = f"docs/selftrained/results/public-raw/{name}/test/metrics.json"
    d = load(path)
    assert d["count"] == d["expected_count"] == 3734 and d["evaluation_complete"] and not d["limited_smoke"]
    assert sum(v["count"] for v in d["per_task_final_reply"].values()) == 3734
    rt = d["per_task_final_reply"]["tool_call"]["tool_roundtrip"]
    assert rt["rate"] < d["thresholds"]["tool_roundtrip"]
    assert public["architectures"][name]["capability_acceptance_passed"] is False
    raw[name] = {"path": path, "sha256": digest(path), "count": d["count"], "complete": d["evaluation_complete"], "tool_roundtrip": rt, "required_tool_roundtrip": d["thresholds"]["tool_roundtrip"], "task_counts": {k: v["count"] for k, v in d["per_task_final_reply"].items()}, "raw_voice_qa": d["per_task_final_reply"]["voice_qa"]["semantic"], "selected_source": public["architectures"][name]["selected_source"]}
receipt["existing_v2_raw_results_read_only"] = {"architectures": raw, "data_scope": public["data_scope"], "comparison_limit": public["comparison_limit"]}

products = load("docs/lesson-products/results/final-report-v1.json")
receipt["existing_bc_results_read_only"] = {"path": "docs/lesson-products/results/final-report-v1.json", "sha256": digest("docs/lesson-products/results/final-report-v1.json"), "gates": products["gates"], "b5_boxes": products["b"]["5"]["boxes"], "b5_sum": products["b"]["5"]["sum"], "c_choice": products["c"]["choice"]["after"], "scope": products["scope"]}

release = load("tiny_perceptron/lesson_products/release.json")
cache = Path("outputs/lesson-products/public-v1-complete")
from tiny_perceptron.lesson_products.inference import check_manifest
for name in ("b", "c"):
    check_manifest(cache / name, name)
    assert digest(cache / name / "artifact-manifest.json") == release["products"][name]["manifest_sha256"]
receipt["bc_cache"] = {"directory": str(cache), "matches_current_release": True, "release": release}
commands = [
    ["solve", "--b", str(cache / "b"), "--state", "兩盒追蹤;A=2;B=6;事件=交換;查詢=B", "--loops", "3"],
    ["judge", "--c", str(cache / "c"), "--state", "卡片進度;申請編號=?", "--question", "info"],
    ["auto", "--b", str(cache / "b"), "--c", str(cache / "c"), "--state", "兩盒追蹤;A=2;B=6;事件=交換;查詢=B", "--effort", "auto"],
    ["auto", "--b", str(cache / "b"), "--c", str(cache / "c"), "--state", "兩盒追蹤;A=2;B=6;事件=交換;查詢=B", "--effort", "medium", "--max-loops", "2"],
]
receipt["local_cli"] = [run(["-m", "tiny_perceptron.lesson_products", *args]) for args in commands]
outputs = [json.loads(x["stdout"]) for x in receipt["local_cli"]]
assert outputs[0]["answer"] == 2 and outputs[0]["loops"] == 3
assert outputs[1]["selected"] == "no" and abs(sum(outputs[1]["probabilities"].values()) - 1) < 1e-6
assert outputs[2]["solution"]["answer"] == 2 and outputs[2]["b_loops"] == 3
assert outputs[3]["status"] == "budget_limited" and outputs[3]["b_loops"] == 2
receipt["assertions"] = "All passed; four familiar CLI inputs check interface and pinned cached weights, not generalization or formal performance."
(BASE / "audit-course-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": "pass", "source_sha256": receipt["source"]["sha256"], "local_links": len(links), "notebooks": len(index), "cli_calls": len(commands), "receipt": str((BASE / "audit-course-receipt.json").relative_to(ROOT))}, ensure_ascii=False))
