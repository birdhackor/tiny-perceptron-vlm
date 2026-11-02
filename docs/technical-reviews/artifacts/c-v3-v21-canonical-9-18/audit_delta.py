"""Read retained bytes and exercise only the new typed input boundary; no training/final invocation."""

import collections
import difflib
import hashlib
import json
import pathlib
import platform
import re
import sys
import zipfile
from datetime import UTC, datetime

import torch

from tiny_perceptron.decision_product.service import Device, DeviceSession

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
ARCHIVE = pathlib.Path(
    "/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/"
    "tiny-perceptron-c-v3-verified.zip"
)
EXPERIMENT = ARCHIVE.parent.parent / "indexed-hardened96-r59-v5"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(value, message):
    if not value:
        raise AssertionError(message)


old_name = "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v2/sources/9.18.md"
new_name = "docs/course-revision-20261101-c-v3/source-freeze-v19/sources/9.18.md"
old, new = (ROOT / old_name).read_bytes(), (ROOT / new_name).read_bytes()
old_lines, new_lines = old.decode().splitlines(True), new.decode().splitlines(True)
opcodes = difflib.SequenceMatcher(a=old_lines, b=new_lines).get_opcodes()
changes = [
    {"operation": op, "old_lines_1_based": [i + 1, j], "new_lines_1_based": [k + 1, l],
     "old": "".join(old_lines[i:j]), "new": "".join(new_lines[k:l])}
    for op, i, j, k, l in opcodes if op != "equal"
]
require(len(changes) == 1 and changes[0]["operation"] == "replace", "unexpected textual delta")
working = (ROOT / "course/chapters/09.md").read_text()
match = re.search(r"^## 9\.18 .*$", working, re.M)
next_match = re.search(r"^## ", working[match.end():], re.M)
end = match.end() + next_match.start() if next_match else len(working)
require(working[match.start():end].encode() == new, "working section differs from source21")
require(digest(new) == "0028f2e0c215cbe526a4caea6a6a0fd9b0fc7ff611861accb15ea36ef66fbcbb", "source21 mismatch")
require(digest(old) == "42beae81f8ab416713c452ab0f8ead4e2a6c465a269bfad482731011614601dd", "source18 mismatch")

acquisition = read(ROOT / "docs/decision-product/v3/metadata/acquisition.json")
archive_raw = ARCHIVE.read_bytes()
require(len(archive_raw) == acquisition["archive"]["bytes"] == 6168350, "archive length mismatch")
require(digest(archive_raw) == acquisition["archive"]["sha256"] ==
        "552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88", "archive SHA mismatch")

with zipfile.ZipFile(ARCHIVE) as archive:
    prefix = "tiny-perceptron-c-v3/"
    member_receipts = []

    def member(name):
        data = archive.read(prefix + name)
        member_receipts.append({"member": prefix + name, "bytes": len(data), "sha256": digest(data)})
        return data

    extracted = []
    for item in acquisition["files"]:
        archive_data = archive.read(item["archive_member"])
        local_data = (ROOT / item["path"]).read_bytes()
        require(archive_data == local_data, "small metadata is not byte identical: " + item["path"])
        require(len(local_data) == item["bytes"] and digest(local_data) == item["sha256"], "metadata receipt mismatch")
        extracted.append({"path": item["path"], "archive_member": item["archive_member"],
                          "sha256": digest(local_data), "byte_identical": True})
    plan = json.loads(member("evidence/plan.json"))
    completion_data = member("evidence/completion.json")
    completion = json.loads(completion_data)
    selection_data = member("evidence/selection.json")
    selection = json.loads(selection_data)
    opening_data = member("evidence/final-opening.json")
    opening = json.loads(opening_data)
    result = json.loads(member("evidence/final-results.json"))
    final_data = member("evidence/final-families.json")
    final_families = json.loads(final_data)
    registry_data = member("evidence/registry-before-final.json")
    registry = json.loads(registry_data)
    source_freeze_data = member("evidence/source-freeze.json")
    source_freeze = json.loads(source_freeze_data)
    training_data = member("evidence/training.jsonl")
    training = [json.loads(line) for line in training_data.splitlines()]
    initial_weights = member("evidence/initial.safetensors")
    trained_weights = member("bundle/weights.safetensors")
    policy = json.loads(member("bundle/policy.json"))
    bundle = json.loads(member("bundle/bundle.json"))

    require(digest(completion_data) == selection["completion_sha256"], "completion-selection binding")
    require(digest(selection_data) == opening["selection_sha256"], "selection-opening binding")
    require(digest(opening_data) == result["final_opening_sha256"], "opening-result binding")
    require(digest(final_data) == result["families_sha256"], "family-result binding")
    require(digest(registry_data) == opening["previous_registry_sha256"], "registry-opening binding")
    require(digest(source_freeze_data) == completion["source_freeze_sha256"] == selection["source_freeze_sha256"],
            "source freeze binding")
    require(digest(training_data) == completion["artifacts"]["training.jsonl"], "training log binding")
    require(digest(initial_weights) == source_freeze["artifacts"]["initial.safetensors"], "initial weights binding")
    require(digest(trained_weights) == completion["artifacts"]["best.safetensors"] == opening["weights_sha256"] ==
            selection["weights_sha256"] == bundle["files"]["weights.safetensors"], "trained weights binding")
    require(digest(initial_weights) != digest(trained_weights), "no weight change")
    require(policy == {"temperatures": selection["temperatures"], "threshold": selection["selected"]["threshold"]},
            "deployed policy differs from pre-final selection")
    times = [plan["created_at"], completion["created_at"], selection["created_at"],
             opening["created_at"], result["created_at"]]
    require([datetime.fromisoformat(t) for t in times] == sorted(datetime.fromisoformat(t) for t in times),
            "training/selection/final timestamps unordered")
    require(completion["completed_updates"] == 1500 and completion["selected_best_update"] == 1500, "training counts")
    stdout_path = EXPERIMENT / "stdout.log"
    stdout_data = stdout_path.read_bytes()
    require(digest(stdout_data) == completion["artifacts"]["stdout.log"], "training stdout binding")
    stdout_events = [json.loads(line) for line in stdout_data.splitlines() if line.startswith(b"{")]
    require(any(row.get("event") == "training_complete" and row.get("best_step") == 1500
                for row in stdout_events), "missing raw training completion in stdout")
    require(max(row.get("step", 0) for row in training) == 1500, "raw training log did not reach step1500")
    source_matches = {}
    for name, expected in opening["implementation"].items():
        path = ROOT / "tiny_perceptron/decision_product" / name
        require(digest(path.read_bytes()) == expected, "implementation drift " + name)
        source_matches[name] = expected

    final_keys = {row["key"] for row in final_families}
    calibration_keys = set(selection["families"]["calibration"])
    policy_keys = set(selection["families"]["policy"])
    excluded = {key for entry in registry["entries"] for key in entry["family_keys"]}
    require(len(final_keys) == len(final_families) == 1500, "final count/uniqueness")
    require(len(calibration_keys) == len(policy_keys) == 600, "selection family count")
    require(not final_keys & excluded and not final_keys & calibration_keys and not final_keys & policy_keys,
            "final family overlap")
    require(not calibration_keys & policy_keys, "calibration-policy overlap")
    require(len(excluded) == opening["previous_exposed_family_count"] == 608, "registry exclusion count")
    device_counts = dict(collections.Counter(len(row["facts"]) for row in final_families))
    require(set(device_counts) <= {4, 5, 6}, "final includes old three-device scope")
    local_data_receipts = []
    local_keys = {}
    for name in ("train-families.json", "curriculum-families.json", "dev-examples.json"):
        path = EXPERIMENT / name
        data = path.read_bytes()
        require(digest(data) == source_freeze["artifacts"][name], "retained development input drift: " + name)
        rows = json.loads(data)
        key_field = "family" if name == "dev-examples.json" else "key"
        require(all(key_field in row for row in rows), "missing family identity " + name)
        keys = {row[key_field] for row in rows}
        require(not final_keys & keys, "final overlaps development input " + name)
        local_keys[name] = {"rows": len(rows), "unique_families": len(keys), "final_overlap": 0}
        local_data_receipts.append({"local_path": str(path), "bytes": len(data), "sha256": digest(data),
                                    "source_freeze_binding": True})


class Capture:
    def _predict(self, example, use_cache=True):
        return {"kind": example.kind, "state": example.state, "question": example.question,
                "candidate_ids": [key for key, _ in example.candidates], "gold": example.gold}


devices = [Device("主錄音室", "record", True, True), Device("播放台", "play", True, False),
           Device("封存NAS", "save", False, True)]
session = DeviceSession(Capture(), devices, "zh")
boundary = [session.choose("主錄音室"), session.information("主錄音室"), session.grade("主錄音室")]
require([x["kind"] for x in boundary] == ["choice", "info", "rubric"], "typed question dispatch")
require(all(x["gold"] == "" for x in boundary), "gold passed through input boundary")
require(all(x["state"] == boundary[0]["state"] for x in boundary), "shared typed material")
require(session._slots == {"主錄音室": "A", "播放台": "B", "封存NAS": "C"}, "ID mapping")
try:
    DeviceSession(Capture(), [Device("A", "record", "true", True), *devices[1:]], "zh")
except ValueError as error:
    invalid_type = str(error)
else:
    raise AssertionError("typed interface accepted string boolean")

receipt = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/c_v3_technical_coord/canonical_9_18",
    "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "CPU",
                    "platform": platform.platform()},
    "source_identity": {"source18_snapshot": old_name, "source18_sha256": digest(old),
                        "source21_snapshot": new_name, "source21_sha256": digest(new),
                        "working_source21_equal": True, "changes": changes,
                        "unchanged_scope": "all preceding math, artificial code/results, split cautions, C-v2 table and exercise"},
    "archive": {"path": str(ARCHIVE), "bytes": len(archive_raw), "sha256": digest(archive_raw),
                "public_transport_verified": False},
    "metadata_archive_identity": extracted,
    "original_members_read": member_receipts,
    "training": {"plan_completion_selection_opening_result_times": times,
                 "completed_updates": completion["completed_updates"],
                 "initial_weights_sha256": digest(initial_weights), "trained_weights_sha256": digest(trained_weights),
                 "weight_bytes_changed": True, "training_complete_event_found": True,
                 "retained_log_rows": len(training), "training_log_max_step": 1500,
                 "stdout_receipt": {"local_path": str(stdout_path), "bytes": len(stdout_data),
                                    "sha256": digest(stdout_data), "completion_binding": True}},
    "frozen_policy": policy,
    "final_identity": {"families": len(final_families), "unique_families": len(final_keys),
                       "device_counts": device_counts, "calibration_families": len(calibration_keys),
                       "policy_families": len(policy_keys), "previous_exposed_unique_families": len(excluded),
                       "final_overlap_with_registry_calibration_policy": 0,
                       "exposure_entries": [{"kind": e["kind"], "unique_families": len(set(e["family_keys"]))}
                                            for e in registry["entries"]],
                       "development_inputs": local_keys},
    "development_input_receipts": local_data_receipts,
    "opening_implementation_matches_current": source_matches,
    "typed_boundary": {"id_mapping": session._slots, "questions": boundary,
                       "invalid_boolean_error": invalid_type,
                       "scope": "real DeviceSession input conversion with Capture; no model outputs or model accuracy measured"},
    "limits": ["No training, calibration fitting, policy fitting, final generation, final scoring, or weight loading was invoked.",
               "No prior review reports or review attachments were read; metadata author status/pass fields do not establish truth.",
               "Old source18 evidence is not yet released; unchanged research/math/artificial-output/C-v2 measurements remain pending retention.",
               "C-v3 final independence is checked for retained family identities, not broad natural-language or new-source reliability.",
               "Local ZIP byte check does not establish anonymous public transport."],
    "all_assertions_passed": True,
}
(HERE / "audit-delta-result.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"all_assertions_passed": True, "delta_hunks": len(changes), "metadata_members": len(extracted),
                  "completed_updates": completion["completed_updates"], "final_families": len(final_families),
                  "final_overlaps": 0, "typed_boundary_kinds": [x["kind"] for x in boundary]}, ensure_ascii=False))
