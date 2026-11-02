"""Post-receipt identity audit and fixed-score arithmetic; never invokes training/final/model inference."""

import ast
import hashlib
import json
import math
import pathlib
import platform
import re
import subprocess
import sys
import zipfile
from datetime import UTC, datetime

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
OLD = ROOT / "docs/technical-reviews/artifacts/omni-P5-9.18"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def check(value, message):
    if not value:
        raise AssertionError(message)


original_path = ROOT / "docs/course-revision-20261101-c-v3/technical-v21/canonical_9_18/original-v1.json"
receipt_path = ROOT / "docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/canonical-9-18-original-v1.json"
check(sha(original_path.read_bytes()) == "de9bb4bf41e51d217952e7597b7c93ca102181f7365042acae9cb0be4d27db4e", "original changed")
check(sha(receipt_path.read_bytes()) == "8bc9ea76c3ed6018abd4480fa4ebc570565e61f051600ea0cbad2417c1690746", "receipt changed")
receipt = load(receipt_path)
check(datetime.fromisoformat(receipt["received_at"]) < datetime.now(UTC), "no prior original receipt")
prior_path = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-9.18/technical-original-v1.json"
check(sha(prior_path.read_bytes()) == "0c420c78c75cb2b6b72a635a51faeb25d08de7f96a928e7ccba504093ad2854a", "prior lineage changed")
prior = load(prior_path)  # Sources/artifacts index only; no inherited verdict is used.
inspection = load(OLD / "inspection-receipts-v1.json")
data = load(OLD / "preserved-inputs-v1.json")
execution = load(OLD / "execution-v1.json")
derivations = load(OLD / "hand-derivations-v1.json")
check(len(derivations["derivations"]) == 6, "derivation identities missing")
references = []
for entry in prior["sources"] + prior["artifacts"]:
    if "path" not in entry:
        continue
    path = ROOT / entry["path"]
    actual = sha(path.read_bytes())
    if entry["path"] == "notebooks/09/9.18.ipynb":
        check(actual == "aba3651b8fcf2f395417c947de16647d6c379e4215c9f44e1338f6ebd460166a", "current Notebook drift")
        state = "whole-file updated; current source21 identity independently checked below"
    else:
        check(actual == entry["sha256"], "retained reference drift " + entry["path"])
        state = "byte-identical"
    references.append({"id": entry["id"], "path": entry["path"], "prior_sha256": entry["sha256"],
                       "current_sha256": actual, "state": state})
for ref in [inspection["canonical_freeze"], inspection["snapshot"]]:
    path = ROOT / (ref.get("path") or ref["snapshot"])
    expected = ref.get("sha256") or ref["source_sha256"]
    check(sha(path.read_bytes()) == expected, "original freeze/snapshot identity")

external = []
for item in inspection["primary_external_sources"]:
    path = pathlib.Path(item["outside_git_path"])
    check(path.stat().st_size == item["bytes"] and sha(path.read_bytes()) == item["sha256"], "original authority bytes")
    external.append({"url": item["url"], "version": item["version"], "sha256": item["sha256"],
                     "original_accessed_on": item["accessed_on"], "current_reinspection_on": "2026-11-01",
                     "read_locators": item["read_locators"]})
paper = subprocess.run(["pdftotext", "-layout", inspection["primary_external_sources"][0]["outside_git_path"], "-"],
                       check=True, capture_output=True, text=True).stdout
check("T is optimized with respect to" in paper and "NLL on the validation set" in paper, "paper original locator")
check("class prediction" in paper and "test sets are drawn from the same distribution" in paper, "paper assumptions")
kev = pathlib.Path(inspection["primary_external_sources"][1]["outside_git_path"]).read_text()
check("are z / T, so z = logits * T" in kev and "risk\": errors / n if n else None" in kev, "Kev locator")

raw_receipts, originals = [], {}
for item in inspection["original_raw_inputs"]:
    path = pathlib.Path(item["original_local_path"])
    raw = path.read_bytes()
    check(len(raw) == item["bytes"] and sha(raw) == item["sha256"], "original raw input changed")
    originals[item["archive_locator"]] = json.loads(raw)
    raw_receipts.append({"archive_locator": item["archive_locator"], "local_path": str(path),
                         "bytes": len(raw), "sha256": sha(raw)})
archive_ref = inspection["final_archive"]
archive_path = pathlib.Path(archive_ref["local_path"])
check(archive_path.stat().st_size == archive_ref["bytes"] and sha(archive_path.read_bytes()) == archive_ref["sha256"],
      "old original archive changed")
member_identity = []
with zipfile.ZipFile(archive_path) as archive:
    names = set(archive.namelist())
    for item in inspection["original_raw_inputs"]:
        name = item["archive_locator"]
        if name not in names:
            continue
        raw = archive.read(name)
        check(sha(raw) == item["sha256"], "old archive original member changed " + name)
        member_identity.append({"member": name, "sha256": sha(raw)})


def project(row, kinds):
    task_fields = ("ids", "logits", "answer", "gold")
    return {"family": row["family"], "source": row["source"], "panel": row["panel"],
            "tasks": {kind: {key: row["tasks"][kind][key] for key in task_fields}
                      for kind in kinds},
            "choice_info": {key: row["choice_info"][key] for key in ("ids", "logits", "answer")},
            "world_supported": row["world_supported"],
            "choice_information_sufficient": row["choice_information_sufficient"]}


raw_rows = {"calibration": originals["selection-evidence/calibration.json"],
            "policy": originals["selection-evidence/policy-predictions.json"],
            "final": originals["measurements/choice/results.json"]["primary"]["rows"]}
for split, rows in raw_rows.items():
    kinds = ("choice", "info", "rubric") if split == "calibration" else ("choice",)
    check(all(set(row["tasks"]) == set(kinds) for row in data[split]), "retained projection task scope")
    check([project(row, kinds) for row in rows] == data[split], "portable projection differs from actual raw " + split)
    check({row["family"] for row in rows} == set(data["expected_family_ids"][split]), "family IDs differ")
definition = originals["frozen/data-definition.json"]
for split, rows in raw_rows.items():
    check({row["family"] for row in rows} == {row["family"] for row in definition["C"]["panels"][split]},
          "raw family IDs differ from frozen definition")
selection = originals["frozen/model-selection.json"]
opening = originals["frozen/final-opening.json"]
origin = originals["measurements/choice/part-origin.json"]
completion = originals["measurements/choice/part-completion.json"]
check(selection["temperature"] == data["fixed_temperature"] and selection["threshold"] == data["fixed_threshold"],
      "old fixed selection differs")
raw_hashes = {item["archive_locator"]: item["sha256"] for item in raw_receipts}
check(opening["model_selection_sha256"] == origin["model_selection_sha256"] == completion["model_selection_sha256"] ==
      raw_hashes["frozen/model-selection.json"], "selection not bound into old final")
check(origin["final_opening_sha256"] == completion["final_opening_sha256"] == raw_hashes["frozen/final-opening.json"],
      "old opening binding")
check(completion["results_sha256"] == raw_hashes["measurements/choice/results.json"], "old raw result binding")
for name, binding in selection["selection_evidence"].items():
    check(binding["sha256"] == raw_hashes["selection-evidence/" + name], "old selection-evidence binding")
times = [selection["created_at"], opening["created_at"], origin["created_at"], completion["created_at"]]
check(sorted(datetime.fromisoformat(t) for t in times) == [datetime.fromisoformat(t) for t in times], "old time sequence")


def choice_stats(rows, temperature):
    correct, losses, decisions = 0, [], []
    for row in rows:
        record = row["tasks"]["choice"]
        scores = record["logits"]
        index = min(range(len(scores)), key=lambda i: (-scores[i], record["ids"][i]))
        correct += record["ids"][index] == record["gold"]
        scaled = [x / temperature for x in scores]
        maximum = max(scaled)
        denominator = sum(math.exp(x - maximum) for x in scaled)
        gold = record["ids"].index(record["gold"])
        losses.append(math.log(denominator) + maximum - scaled[gold])
        info = row["choice_info"]
        info_index = min(range(len(info["logits"])), key=lambda i: (-info["logits"][i], info["ids"][i]))
        take = 1 / denominator >= data["fixed_threshold"] and record["ids"][index] != "none" and info["ids"][info_index] == "yes"
        wrong = (record["ids"][index] != record["gold"] or not row["world_supported"] or
                 not row["choice_information_sufficient"])
        decisions.append((take, take and wrong))
    accepted, errors = sum(x[0] for x in decisions), sum(x[1] for x in decisions)
    return {"correct": correct, "families": len(rows), "mean_nll": sum(losses) / len(rows),
            "accepted": accepted, "errors": errors, "coverage": accepted / len(rows),
            "risk": errors / accepted if accepted else None}


raw_stats = choice_stats(raw_rows["final"], 1)
fixed_stats = choice_stats(raw_rows["final"], data["fixed_temperature"])
check(raw_stats["correct"] == fixed_stats["correct"] == 39 and
      (fixed_stats["accepted"], fixed_stats["errors"]) == (53, 19), "old raw fixed-score counts")
check(abs(raw_stats["mean_nll"] - execution["final"]["raw_mean_NLL"]) < 1e-12 and
      abs(fixed_stats["mean_nll"] - execution["final"]["calibrated_mean_NLL"]) < 1e-12, "old raw NLL")

source21 = (ROOT / "docs/course-revision-20261101-c-v3/source-freeze-v19/sources/9.18.md").read_text()
inline = re.search(r"```python\n(.*?)\n```", source21, re.S).group(1)
expected_stdout = re.search(r"```text\n(.*?)\n```", source21, re.S).group(1)
check(inline.strip() == (OLD / "inline-v1.py").read_text().strip(), "source21 teaching code changed")
check(execution["inline"]["stdout"].strip() == expected_stdout.strip(), "source21 expected output changed")
notebook_path = ROOT / "notebooks/09/9.18.ipynb"
notebook = load(notebook_path)
executed_path = ROOT / "outputs/omni-c-v3-kernel-bundle-v19/09/9.18.ipynb"
executed = load(executed_path)
check(len(notebook["cells"]) == len(executed["cells"]) and
      all(x["cell_type"] == y["cell_type"] and x["source"] == y["source"]
          for x, y in zip(notebook["cells"], executed["cells"])), "actual executed Notebook source differs")
code_cells = [x for x in notebook["cells"] if x["cell_type"] == "code"]
check(len(code_cells) == 2 and "".join(code_cells[1]["source"]).strip() == inline.strip(), "current notebook teaching source")
notebook_stdout = "".join("".join(out.get("text", [])) for cell in executed["cells"]
                          for out in cell.get("outputs", []) if out.get("output_type") == "stream" and out.get("name") == "stdout")
check(notebook_stdout == execution["inline"]["stdout"], "retained source21 kernel output differs")
check(not any(out.get("output_type") == "error" for cell in executed["cells"] for out in cell.get("outputs", [])),
      "retained source21 kernel has errors")

report = originals["final-report.json"]
rules = data["fixed_metric_rules"]
check(fixed_stats["correct"] < rules["choice_minimum_correct"] and
      fixed_stats["risk"] > rules["maximum_empirical_accepted_risk"], "old C-v2 failure did not imply non-promotion")
check(report["promote_C_service"] is False and "retain historical product" in report["failed_module_default"],
      "old adoption record differs")
legacy_cli = ROOT / "tiny_perceptron/lesson_products/__main__.py"
check(opening["evaluation_sources_sha256"]["tiny_perceptron/lesson_products/__main__.py"] == sha(legacy_cli.read_bytes()),
      "historical CLI not bound to actual old final")
legacy_tree = ast.parse(legacy_cli.read_text())
legacy_imports = [node.module for node in ast.walk(legacy_tree) if isinstance(node, ast.ImportFrom)]
check(not any(module and "omni_products" in module for module in legacy_imports), "old default imports research candidate")
release = load(ROOT / "tiny_perceptron/lesson_products/release.json")
check(release["revision"] == "2c6a1062f3e66a602cf83eda6c703456ec18bca0" and
      release["products"]["c"]["prefix"] == "lesson-products/v1-20261016/c", "old default release pointer")

result = {
    "created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/c_v3_technical_coord/canonical_9_18",
    "environment": {"python": sys.version.split()[0], "device": "CPU", "platform": platform.platform()},
    "original_receipt": {"path": receipt_path.relative_to(ROOT).as_posix(), "sha256": sha(receipt_path.read_bytes()),
                         "received_at": receipt["received_at"]},
    "prior_index_only": {"path": prior_path.relative_to(ROOT).as_posix(), "sha256": sha(prior_path.read_bytes()),
                         "verdict_used_as_evidence": False},
    "retained_reference_identities": references,
    "authority_original_identity": external,
    "original_raw_input_identities": raw_receipts,
    "original_archive": {"sha256": archive_ref["sha256"], "bytes": archive_ref["bytes"],
                         "members_checked": member_identity},
    "raw_projection": {split: {"rows": len(rows), "all_projected_fields_equal": True,
                               "projected_task_types": ["choice", "info", "rubric"] if split == "calibration" else ["choice"],
                               "frozen_definition_family_ids_equal": True} for split, rows in raw_rows.items()},
    "C_v2_fixed_raw_score_arithmetic": {"temperature": data["fixed_temperature"], "threshold": data["fixed_threshold"],
                                       "raw": raw_stats, "fixed": fixed_stats,
                                       "pre_final_selection_opening_measurement_times": times},
    "retained_artificial_execution": {"original_created_at": execution["created_at"],
                                      "original_environment": execution["environment"],
                                      "original_command": next(a["command"] for a in prior["artifacts"] if a["id"] == "a_execution"),
                                      "inline_code_sha256": sha((OLD / "inline-v1.py").read_bytes()),
                                      "source21_code_equal": True, "source21_displayed_stdout_equal": True,
                                      "T": execution["inline"]["T"], "tau": execution["inline"]["tau"],
                                      "nll": execution["inline"]["nll_unrounded"],
                                      "exercise_new": execution["inline"]["exercise_changed_new"],
                                      "exercise_all": execution["inline"]["exercise_changed_all"],
                                      "reexecuted_this_round": False},
    "current_notebook_identity": {"path": notebook_path.relative_to(ROOT).as_posix(),
                                  "sha256": sha(notebook_path.read_bytes()),
                                  "code_cell_sha256": [sha("".join(c["source"]).encode()) for c in code_cells],
                                  "executed_path": str(executed_path), "executed_sha256": sha(executed_path.read_bytes()),
                                  "all_cell_sources_equal": True, "stdout_equals_original_execution": True,
                                  "kernel_refresh_receipt_sha256": sha((ROOT / "docs/course-revision-20261101-c-v3/evidence/kernel-refresh-source21-v1.json").read_bytes()),
                                  "claim_scope": "local CPU executed bytes retained; no Colab/cloud/network execution asserted"},
    "historical_adoption": {"promote_C_service": False, "failed_module_default": report["failed_module_default"],
                            "choice_correct": fixed_stats["correct"], "required_choice_correct": rules["choice_minimum_correct"],
                            "accepted_risk": fixed_stats["risk"], "max_accepted_risk": rules["maximum_empirical_accepted_risk"],
                            "CLI_same_as_original_final_binding": True, "release_revision": release["revision"],
                            "release_c_prefix": release["products"]["c"]["prefix"]},
    "limits": ["Retains source-identical math/research/artificial code and original 2026-10-30 execution; prior pass was not used.",
               "This command only checks provenance and at most60 retained C-v2 score rows; no new model inference, training, calibration search or final invocation.",
               "The old Notebook whole-file SHA is intentionally replaced by actual source21 SHA and cell/source comparison.",
               "The original template-heldout world is not an independently sourced corpus; finite risk estimates do not ensure future reliability.",
               "No anonymous C-v3 public-download success is asserted."],
    "all_assertions_passed": True,
}
(HERE / "retention-check-v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"all_assertions_passed": True, "prior_local_references_checked": len(references),
                  "original_raw_inputs_checked": len(raw_receipts), "raw_projection_rows": sum(map(len, raw_rows.values())),
                  "C_v2_correct": fixed_stats["correct"], "C_v2_accepted": fixed_stats["accepted"],
                  "C_v2_errors": fixed_stats["errors"], "current_notebook_sha256": sha(notebook_path.read_bytes())}))
