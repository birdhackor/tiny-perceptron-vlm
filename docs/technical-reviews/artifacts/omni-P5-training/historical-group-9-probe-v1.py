"""Prompted historical recheck only: immutable records and isolated stdlib parser/branch execution.

Does not import torch, instantiate/load/run a neural network, train, download, or create a seal.
"""
import argparse
import ast
import contextlib
import copy
import datetime
import hashlib
import io
import json
import platform
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent
TASK = "/root/omni_p5_technical_training_whole"
REQUEST = "docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json"
FULL = "docs/course-revision-20261029-omni/reviews/P5-actual/technical-training/technical-current-v1.json"
ORIGINAL = "docs/course-revision-20261029-omni/reviews/P5-actual/technical-training/technical-original-v1.json"
INTAKE = "docs/course-revision-20261029-omni/reviews/P5-actual/technical-original-collection-intake-v14.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(name):
    return {"path": name, "sha256": digest((ROOT / name).read_bytes())}


def read(reference):
    raw = (ROOT / reference["path"]).read_bytes()
    assert digest(raw) == reference["sha256"], reference["path"]
    return json.loads(raw)


def node(value, pointer):
    for token in pointer.split("/")[1:] if pointer else []:
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


request_ref = ref(REQUEST)
assert request_ref["sha256"] == "714f16f7d5c20924e20f91a8b48bca7a87bd77e2602d1a1ae501d06d93f819e5"
group = read(request_ref)["groups"][9]
assert group["page_id"] == "training" and group["old_role"] == "technical"
assert len(group["items"]) == 2
full_ref, original_ref, intake_ref = ref(FULL), ref(ORIGINAL), ref(INTAKE)
assert full_ref["sha256"] == "c987365ee519a9ea7dcbe3c7384883683323a8920916ded2adfdfcbd67d92163"
assert original_ref["sha256"] == "d31b396b2fd906cfdfd672edd65ac7621cfe218779fac8edb9553dfb03184c4e"
assert intake_ref["sha256"] == "2ae89eca0bef94b43de11a509e63b14617743320bd9d0d29c43ad22cdf1bcca8"
full, original, intake = read(full_ref), read(original_ref), read(intake_ref)
receipt_index, receipt = next((i, x) for i, x in enumerate(intake["collection_index"]) if x["path"] == FULL)
assert receipt["sha256"] == full_ref["sha256"] and receipt["reviewer_task"] == TASK
assert receipt["received_at"] == "2026-10-30T12:01:01.788159+00:00"
assert datetime.datetime.fromisoformat(full["created_at"]) <= datetime.datetime.fromisoformat(receipt["received_at"])
freeze = read(full["source_freeze"])
frozen = next(x for x in freeze["pages"] if x["page_id"] == "training")
body = (ROOT / frozen["snapshot"]).read_bytes()
assert digest(body) == frozen["source_sha256"] == full["source_sha256"] == original["source_sha256"]
assert (ROOT / "course/training.md").read_bytes() == body
assert full["figure_sha256"] == original["figure_sha256"] == frozen["figures_sha256"] == {}
current_quote = body.decode().splitlines()[8]
assert "這三個工具都沒有 `--device` 參數" in current_quote

pointer = read(group["prior_active_pointer"])
prior_manifest_ref = {"path": pointer["manifest"], "sha256": pointer["manifest_sha256"]}
prior_manifest = read(prior_manifest_ref)
prior_frozen = next(x for x in read(prior_manifest["source_freeze"])["pages"] if x["page_id"] == "training")
prior_body_ref = {"path": prior_frozen["snapshot"], "sha256": prior_frozen["source_sha256"]}
assert digest((ROOT / prior_body_ref["path"]).read_bytes()) == prior_body_ref["sha256"]
unavailable = read(group["unavailability"]["report"])
unavailable_record = node(unavailable, group["unavailability"]["json_pointer"])
assert unavailable_record["reviewer_task"] == group["old_owner"] and unavailable_record["unavailable"] is True

instances = []
for item in group["items"]:
    old_original = read(item["original_report"])
    issue = node(old_original, item["json_pointer"])
    disposition = node(read(item["disposition_ref"]), item["disposition_ref"]["json_pointer"])
    old_current = read(item["historical_fix_verification"]["report"])
    assert old_original["reviewer_task"] == old_current["reviewer_task"] == group["old_owner"]
    assert old_original["verdict"] == "revise" and issue["necessary"] is True
    assert disposition["issue_id"] == item["issue_id"]
    assert disposition["action"] == "fixed" and disposition["final_necessity"] == item["necessity"] == "necessary"
    assert disposition["report"] == item["original_report"] and disposition["json_pointer"] == item["json_pointer"]
    assert item["historical_fix_verification"] in disposition["verification"]
    assert old_current["verdict"] == "pass" and old_current["source_sha256"] == prior_frozen["source_sha256"]
    assert old_current["figure_sha256"] == prior_frozen["figures_sha256"] == {}
    old_freeze = read(old_original["source_freeze"])
    old_page = next(x for x in old_freeze["pages"] if x["page_id"] == "training")
    old_raw = (ROOT / old_page["snapshot"]).read_bytes()
    assert digest(old_raw) == old_page["source_sha256"] == old_original["source_sha256"]
    old_device_lines = [{"line": i + 1, "quote": line} for i, line in enumerate(old_raw.decode().splitlines()) if "沒寫 `--device`" in line]
    assert old_device_lines
    instances.append({
        "issue_id": item["issue_id"], "original_report": item["original_report"], "json_pointer": item["json_pointer"],
        "original_verdict_preserved": old_original["verdict"], "original_issue": issue,
        "original_source": {"path": old_page["snapshot"], "sha256": old_page["source_sha256"]},
        "original_device_lines": old_device_lines, "disposition_ref": item["disposition_ref"],
        "accepted_disposition": disposition, "historical_fix_verification": item["historical_fix_verification"],
        "old_closure": {"reviewer_task": old_current["reviewer_task"], "source_sha256": old_current["source_sha256"],
                        "figure_sha256": old_current["figure_sha256"], "verdict": old_current["verdict"],
                        "issue_resolutions": old_current["issue_resolutions"],
                        "claims": [x for x in old_current["claims"] if x["id"] in {"device_generalization", "experiment_default_cpu", "quantize_fixed_cpu", "data_tools_no_device", "three_tools_no_device_argument"}]}
    })
assert instances[0]["original_issue"] == instances[1]["original_issue"]

old_audit_ref = ref("docs/technical-reviews/artifacts/review-20261019-training/audit.json")
old_execution_ref = ref("docs/technical-reviews/artifacts/review-20261019-training/verification-current-01.json")
assert old_audit_ref["sha256"] == "99d84cf2af87ecf9e90b16dc716fbd4720b5b919b65f2b984f2079dc2452ab0c"
assert old_execution_ref["sha256"] == "e944182b172c8a6b3c7f4a832f430b05cd17d5762dd6ea16696f30b147250cec"
old_audit, old_execution = read(old_audit_ref), read(old_execution_ref)
assert all(x["returncode"] == 2 for x in old_audit["runs"][:3])
assert all(x["returncode"] == 2 for x in old_execution["runs"][-3:])
old_priority = old_execution["observations"]["device_priority"]
assert [x["auto"] for x in old_priority] == ["cuda", "mps", "cpu"]
assert old_execution["observations"]["experiment_default"]["device"] == "cpu"
assert old_execution["observations"]["quantization_counterexample"]["saved_devices"] == ["cpu"]

paths = ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py", "scripts/course_experiments/run.py", "scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py", "tiny_perceptron/training.py"]
current_sources = {x["path"]: x for x in full["sources"] if x.get("path") in paths}
trees = {}
for name in paths:
    raw = (ROOT / name).read_bytes()
    assert digest(raw) == current_sources[name]["sha256"]
    trees[name] = ast.parse(raw.decode(), filename=name)


def isolated_parser(name):
    """Execute unchanged parser-definition AST only, never main's parse/load/train/data work."""
    tree = trees[name]
    namespace = {"argparse": argparse, "Path": Path, "ROOT": ROOT, "__doc__": ast.get_docstring(tree)}
    if name == "scripts/train.py":
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "parser")
        program = ast.Module(body=[copy.deepcopy(function)], type_ignores=[])
        exec(compile(ast.fix_missing_locations(program), name + "::isolated_parser_definition", "exec"), namespace)
        return namespace["parser"](), {"function": "parser", "start_line": function.lineno, "end_line": function.end_lineno}
    if name == "scripts/evaluate.py":
        helper = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "parse_limit")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(helper)], type_ignores=[])), name, "exec"), namespace)
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "main")
    prefix = []
    for statement in function.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Attribute) and statement.value.func.attr == "parse_args":
            parser_name = statement.value.func.value.id
            break
        prefix.append(copy.deepcopy(statement))
    else:
        raise AssertionError("parse_args boundary absent: " + name)
    assert prefix and all(isinstance(x, (ast.Assign, ast.Expr)) for x in prefix)
    exec(compile(ast.fix_missing_locations(ast.Module(body=prefix, type_ignores=[])), name + "::isolated_parser_prefix", "exec"), namespace)
    return namespace[parser_name], {"function": "main parser prefix", "start_line": prefix[0].lineno, "end_line": prefix[-1].end_lineno, "excluded_from_line": statement.lineno}


parsers = []
for name in paths[:-1]:
    parser, location = isolated_parser(name)
    required = ["not-opened-fixture.pt"] if name in {"scripts/evaluate.py", "scripts/infer.py", "scripts/quantize.py"} else []
    defaults = vars(parser.parse_args(required))
    device_action = next((a for a in parser._actions if "--device" in a.option_strings), None)
    record = {"path": name, "sha256": current_sources[name]["sha256"], "parser_ast": location,
              "minimal_argv": required, "has_device": device_action is not None,
              "default_device": defaults.get("device"), "device_choices": None if device_action is None else device_action.choices}
    if name in {"scripts/train.py", "scripts/evaluate.py", "scripts/infer.py"}:
        assert record["default_device"] == "auto"
        assert vars(parser.parse_args(required + ["--device", "cpu"]))["device"] == "cpu"
        record["explicit_cpu"] = "cpu"
    elif name == "scripts/course_experiments/run.py":
        assert record["default_device"] == "cpu" and record["device_choices"] == ("cpu", "cuda")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            try:
                parser.parse_args(["--device", "mps"])
            except SystemExit as error:
                record["mps_rejection_exit"] = error.code
        assert record["mps_rejection_exit"] == 2
        record["mps_rejection_stderr"] = stderr.getvalue()
    else:
        assert device_action is None
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            try:
                parser.parse_args(required + ["--device", "cpu"])
            except SystemExit as error:
                record["unsupported_device_exit"] = error.code
        assert record["unsupported_device_exit"] == 2
        record["unsupported_device_stderr"] = stderr.getvalue()
    parsers.append(record)

training_tree = trees["tiny_perceptron/training.py"]
choose = next(x for x in training_tree.body if isinstance(x, ast.FunctionDef) and x.name == "choose_device")
priority = []
for cuda, mps in [(True, True), (False, True), (False, False)]:
    fake = SimpleNamespace(device=lambda name: name, cuda=SimpleNamespace(is_available=lambda: cuda), backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))
    namespace = {"torch": fake}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(choose)], type_ignores=[])), "tiny_perceptron/training.py::choose_device_only", "exec"), namespace)
    priority.append({"cuda_available_mock": cuda, "mps_available_mock": mps, "auto": namespace["choose_device"](), "explicit_cpu": namespace["choose_device"]("cpu")})
assert [x["auto"] for x in priority] == ["cuda", "mps", "cpu"]
assert all(x["explicit_cpu"] == "cpu" for x in priority)
load = next(x for x in training_tree.body if isinstance(x, ast.FunctionDef) and x.name == "load_checkpoint")
assert ast.literal_eval(load.args.defaults[0]) == "cpu"
loads = [x for x in ast.walk(load) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and isinstance(x.func.value, ast.Name) and x.func.value.id == "torch" and x.func.attr == "load"]
assert len(loads) == 1 and any(k.arg == "map_location" and ast.literal_eval(k.value) == "cpu" for k in loads[0].keywords)
calls = [x for x in ast.walk(trees["scripts/quantize.py"]) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "load_checkpoint"]
assert len(calls) == 1 and len(calls[0].args) == 1 and not calls[0].keywords
cpu_load = {"method": "static AST assertions only; loader and model construction never executed", "quantize_call": {"line": calls[0].lineno, "code": ast.unparse(calls[0])}, "loader_signature": {"line": load.lineno, "device_default": "cpu"}, "torch_load": {"line": loads[0].lineno, "code": ast.unparse(loads[0])}, "scope": "CPU inference follows the CLI's omitted device argument and loader default, supported also by original saved_devices=[cpu]; not a new quantization run."}

observed_at = datetime.datetime.now(datetime.UTC).isoformat()
assert datetime.datetime.fromisoformat(receipt["received_at"]) < datetime.datetime.fromisoformat(observed_at)
result = {
    "created_at": observed_at, "reviewer_task": TASK, "kind": "prompted historical issue recheck; not independent initial discovery or new full review",
    "request": {"report": request_ref, "json_pointer": "/groups/9"}, "full_review": full_ref, "independent_original": original_ref,
    "full_review_actual_receipt": {"report": intake_ref, "json_pointer": "/collection_index/" + str(receipt_index), "entry": receipt},
    "source_freeze": full["source_freeze"], "current_source": {"path": frozen["snapshot"], "sha256": digest(body)}, "current_figure_sha256": {},
    "current_device_quote": {"source": "course/training.md", "line": 9, "quote": current_quote},
    "full_review_claims_already_covering_fix": [x for x in full["claims"] if x["id"] in {"i02", "i03"}],
    "prior_active_pointer": group["prior_active_pointer"], "prior_manifest": prior_manifest_ref, "prior_source_freeze": prior_manifest["source_freeze"], "prior_frozen_training_source": prior_body_ref,
    "historical_instances": instances, "duplicate_relationship": {"same_original_issue_object": True, "same_owner": group["old_owner"], "same_original_claim_id": "device_generalization", "distinct_original_report_and_disposition_identities": True, "retained_instance_count": 2, "necessity_preserved_for_each": "necessary", "scope": "Two saved instances of one underlying correction; neither instance nor original revise judgment is omitted, downgraded, or counted as a new independent discovery."},
    "unavailability": {**group["unavailability"], "original_record": unavailable_record, "scope": "Root's preserved actual 2026-10-29 followup/listing events; not new tool events performed by this reviewer."},
    "original_execution_evidence": {"audit": old_audit_ref, "initial_rejection_runs": old_audit["runs"][:3], "original_fix_execution": old_execution_ref, "fixed_rejection_runs": old_execution["runs"][-3:], "device_priority_mock": old_priority, "fixed_runner_default_parser_intercept": old_execution["observations"]["experiment_default"], "quantization_saved_devices": old_execution["observations"]["quantization_counterexample"]["saved_devices"], "scope": "Read existing original receipts only. Historical model/quantization observations were not rerun or attributed to this task."},
    "current_repository_sources": [{"path": name, "sha256": current_sources[name]["sha256"]} for name in paths],
    "isolated_current_parser_execution": parsers, "isolated_choose_device_execution": {"source_lines": [choose.lineno, choose.end_lineno], "cases": priority, "scope": "Actual source function AST with a minimal availability/device stub; no torch import or hardware execution."}, "current_cpu_loader_static_proof": cpu_load,
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "none; standard library only", "torch_imported": "no"},
    "scope_limits": ["Not a fresh full-page re-review; full current1 is already genuinely received and its independent118 claims are preserved.", "No neural model construction/load/forward,training,quantization execution,NN final-family re-inference,CUDA/MPS/Windows/Colab execution,network,Modal,workflow or publication.", "Native Windows T.11 remains pending book PR CI exactly as in full current1.", "No current technical stage seal,appendix collection receipt or final historical/check-numbered gate is created or claimed."]
}
target = HERE / "historical-group-9-probe-results-v1.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"result": ref(target.relative_to(ROOT).as_posix()), "bytes": target.stat().st_size, "created_at": observed_at, "instances": len(instances), "parsers": len(parsers), "torch_imported": False}, ensure_ascii=False))
