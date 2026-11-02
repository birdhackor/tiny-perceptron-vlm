"""Independent, bounded CPU/source checks for the frozen course guide.

No training, model download, remote runtime, or publication is performed.
The historic result checks inspect archived producer records, not reviewer reports.
"""

from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

import nbformat
from nbclient import NotebookClient
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.export_course import checked_notebook, output_html
from tiny_perceptron.lesson_products.inference import auto_solve
from tiny_perceptron.lesson_products.model import Config, LoopedSolver, Tokenizer, padded

FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-reader-fix04.json"
CACHE = Path("/workspace/work/reference/technical-course-20261029")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    p = ROOT / path
    return {"path": path, "sha256": sha(p), "bytes": p.stat().st_size}


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    torch.set_num_threads(1)
    freeze = load(str(FREEZE.relative_to(ROOT)))
    page = next(p for p in freeze["pages"] if p["page_id"] == "course")
    body = (ROOT / page["snapshot"]).read_text()
    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewer_task": "/root/omni_technical_coordinator/technical_course",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "frozen_page": page,
        "freeze_sha256": sha(FREEZE),
        "snapshot_sha256": sha(ROOT / page["snapshot"]),
        "current_source_sha256": sha(ROOT / page["source"]),
        "course_code_fences": len(re.findall(r"^```", body, re.M)),
        "course_inline_figures": re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body),
    }
    assert out["snapshot_sha256"] == out["current_source_sha256"] == page["source_sha256"]
    assert page["figures_sha256"] == {} and out["course_inline_figures"] == []

    links = []
    for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", body):
        if "://" in target:
            continue
        rel, _, fragment = target.partition("#")
        p = ((ROOT / page["source"]).parent / rel).resolve() if rel else ROOT / page["source"]
        ok = p.is_file()
        if ok and fragment:
            headings = re.findall(r"^## ([A-Z\d]+\.\d+) .+$", p.read_text(), re.M)
            ok = fragment in headings
        links.append({"label": label, "target": target, "valid_file_and_numbered_anchor": ok})
    out["local_links"] = {"count": len(links), "invalid": [x for x in links if not x["valid_file_and_numbered_anchor"]]}

    first = next(p for p in freeze["pages"] if p["page_id"] == "1.1")
    first_body = (ROOT / first["snapshot"]).read_text()
    code = re.findall(r"```python\n(.*?)```", first_body, re.S)
    assert len(code) == 1
    namespace = {}
    stream = io.StringIO()
    with redirect_stdout(stream):
        exec(compile(code[0], first["snapshot"], "exec"), namespace)
    out["exact_first_lesson_code"] = {
        "snapshot": first["snapshot"], "source_sha256": first["source_sha256"],
        "code_sha256": hashlib.sha256(code[0].encode()).hexdigest(),
        "stdout": stream.getvalue(), "input": namespace["text"], "chars": namespace["chars"],
        "ids": namespace["ids"], "restored": namespace["restored"],
        "both_cat_ids": [namespace["ids"][i] for i,c in enumerate(namespace["text"]) if c == "貓"],
        "scope": "Exact embedded 1.1 code only; no semantic-understanding claim and no prerequisite figure judgment.",
    }
    assert namespace["ids"] == [3,2,1,4,1,2,3,0]
    assert namespace["restored"] == namespace["text"]

    sync = subprocess.run([sys.executable,"scripts/build_course.py","--check"], cwd=ROOT, capture_output=True, text=True)
    out["notebook_sync"] = {"argv": [sys.executable,"scripts/build_course.py","--check"], "returncode": sync.returncode, "stdout": sync.stdout, "stderr": sync.stderr}
    assert sync.returncode == 0
    index = load("course/lesson-index.json")
    executed_root = ROOT / "outputs/omni-notebooks-reader-fix04-current"
    matched, errors, site_cpu_labels = 0, [], 0
    for item in index:
        try:
            checked_notebook(load(item["notebook"]), executed_root / Path(item["notebook"]).relative_to("notebooks"))
            matched += 1
        except Exception as exc:
            errors.append({"id": item["id"], "error": str(exc)})
        html = ROOT / "outputs/omni-review-site-current" / (item["id"] + ".html")
        if html.is_file() and "實際執行結果 · CPU" in html.read_text():
            site_cpu_labels += 1
    out["all_stored_cpu_notebooks"] = {
        "index_count": len(index), "matched_executed_notebooks": matched, "errors": errors,
        "html_pages_with_nonempty_cpu_output": site_cpu_labels,
        "scope": "Stored local executed notebooks checked for identical cells, execution_count, no errors; no new all-book run and no semantic sufficiency judgment.",
        "build_info": load("outputs/omni-review-site-current/build-info.json"),
    }
    out["export_renderer_smoke"] = output_html([{"output_type":"stream","name":"stdout","text":"<3>\n"}])
    assert "&lt;3&gt;" in out["export_renderer_smoke"]

    nb = nbformat.read(ROOT / "notebooks/01/1.1.ipynb", as_version=4)
    nb = NotebookClient(nb, timeout=60, kernel_name="python3", resources={"metadata":{"path":str(ROOT)}}).execute()
    executed = CACHE / "executed-1.1.ipynb"
    nbformat.write(nb, executed)
    out["fresh_local_notebook_execution"] = {
        "source": record("notebooks/01/1.1.ipynb"),
        "execution_copy_outside_git": str(executed), "execution_copy_sha256": sha(executed),
        "code_cells": sum(c.cell_type == "code" for c in nb.cells),
        "all_code_cells_executed_without_error": all(c.execution_count is not None and not any(o.output_type=="error" for o in c.outputs) for c in nb.cells if c.cell_type=="code"),
        "stream_outputs": [o.text for c in nb.cells if c.cell_type=="code" for o in c.outputs if o.output_type=="stream"],
        "remote_colab_executed": False,
    }

    policy = load("docs/lesson-products/results/b-auto-policy.json")
    class ProbeB:
        auto_policy = policy
        def synchronize(self): pass
        def solve(self, state, loops): return {"probe_only":True,"state":state,"loops":loops}
    class ProbeC:
        def __init__(self, choice, info): self.choice,self.info=choice,info
        def synchronize(self): pass
        def judge(self, state, question): return {"selected":self.choice if question=="choice" else self.info,"seconds":0.0}
    states = [
        ("unsupported","天氣預測","none","yes",5),
        ("missing","三數加法;x=?;y=2;z=3","sum","no",5),
        ("complete","三數加法;x=1;y=2;z=3","sum","yes",5),
        ("budget","兩盒追蹤;A=1;B=2;事件=A設3;查詢=A","boxes","yes",2),
    ]
    out["actual_auto_control_flow_with_explicit_stubs"] = {
        "scope":"Real auto_solve and parse_state, deterministic declared judge/solver stubs; no C/B accuracy evidence.",
        "cases": {name:auto_solve(ProbeB(),ProbeC(choice,info),state,max_loops=limit) for name,state,choice,info,limit in states},
    }
    assert [v["status"] for v in out["actual_auto_control_flow_with_explicit_stubs"]["cases"].values()] == ["unsupported","clarification_needed","answered","budget_limited"]
    tokenizer = Tokenizer()
    torch.manual_seed(27)
    model = LoopedSolver(Config(len(tokenizer),width=16,heads=4,ff=32)).eval()
    tokens = padded([tokenizer.encode("三數加法;x=1;y=2;z=3")],128)
    with torch.no_grad(): scores = model(tokens,loops=3,all_depths=True)
    out["random_looped_model_forward"] = {"score_shape":list(scores.shape),"all_finite":bool(scores.isfinite().all()),"scope":"Fresh random width16 CPU forward; no update, trained accuracy or production performance claim."}
    assert scores.shape == (1,3,19) and bool(scores.isfinite().all())

    out["historic_bc_initialization_records"] = {}
    for product in ("b","c"):
        rel = f"docs/lesson-products/results/initial-{product}-execution.json"
        receipt = load(rel)
        training = receipt["training"]
        out["historic_bc_initialization_records"][product] = {
            **record(rel), "code_revision": receipt["code_revision"],
            "training": {k:training[k] for k in ("product","random_initialization","resume_source","steps_this_attempt","last_step","seed","initial_weights_sha256","selected_weights_sha256","weights_changed")},
        }
    a_index = load("docs/selftrained/v2-training-stage-index.json")
    first_a = next(s for s in a_index["stages"] if s["stage_key"] == "moe-pretrain")
    original_a = ROOT / first_a["receipt_path"]
    out["a_original_training_provenance"] = {
        "manifest":record("docs/selftrained/v2-manifest.json"),
        "initialization_recorded":load("docs/selftrained/v2-manifest.json")["initialization"],
        "stage_index":record("docs/selftrained/v2-training-stage-index.json"),
        "moe_pretrain_first_stage":first_a,
        "original_receipt_exists":original_a.is_file(),
        "limitation":"Manifest/index labels alone do not independently prove full historical A origin; first original execution receipt is unavailable at its recorded path.",
    }
    bc = load("docs/lesson-products/results/final-report-v1.json")
    a = load("docs/selftrained/results/v2-final-public-results.json")
    tests = {
        "A_tool_roundtrip":a["architectures"]["moe"]["final_test"]["per_task_final_reply"]["tool_call"]["tool_roundtrip"],
        "B_sum_depth5":bc["b"]["5"]["sum"],
        "C_wording_choice":bc["shifts"]["c_wording"]["choice"]["after"],
    }
    out["historic_nonperfect_aggregate_results"] = {"sources":[record("docs/selftrained/results/v2-final-public-results.json"),record("docs/lesson-products/results/final-report-v1.json")],"values":tests,"scope":"Recomputed nonperfect counts from archived producer aggregates; no new trained-model evaluation and no raw prediction authenticity audit."}
    out["new_module_status"] = {
        "production_paths_exist": {p:(ROOT/p).exists() for p in ["tiny_perceptron/omni_products/av.py","tiny_perceptron/omni_products/decision.py"]},
        "linked_frozen_scope": {p["page_id"]:{"path":p["snapshot"],"sha256":p["source_sha256"]} for p in freeze["pages"] if p["page_id"] in ["19.21","19.22","19.23","19.24"]},
        "scope":"Read directly linked overview/status prose; forward or manual control-flow examples do not establish formal new training or mature-model quality.",
    }
    out["own_license"] = {**record("LICENSE"),"first_line":(ROOT/"LICENSE").read_text().splitlines()[0]}
    out["torch_original_files"] = []
    for obj in [torch.nn.parameter.Parameter,torch.nn.Embedding,torch.optim.SGD]:
        path = Path(inspect.getfile(obj))
        out["torch_original_files"].append({"object":obj.__name__,"path":str(path),"sha256":sha(path),"first_line":inspect.getsourcelines(obj)[1]})
    out["torch_git_version"] = torch.version.git_version
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
