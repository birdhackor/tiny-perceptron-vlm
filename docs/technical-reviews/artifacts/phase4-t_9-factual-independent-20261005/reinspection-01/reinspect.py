import ast
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import shlex
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import torch
from scripts import evaluate as evaluation
from tiny_perceptron.model import TinyLM

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
task = "/root/phase4_factual_coordinator/factual_t_9"
started = datetime.now(UTC).isoformat()
old = (BASE / "section.md").read_bytes()
new = (OUT / "section.md").read_bytes()
before = "設定選完才把相同三條命令的資料路徑改為`test.jsonl`、另取輸出檔名"
after = "設定選完才把相同三條命令的資料路徑改為`test.jsonl`，加上`--split-label test`並另取輸出檔名"
assert old.count(before.encode()) == new.count(after.encode()) == 1
assert old.replace(before.encode(), after.encode()) == new
assert sha(OUT / "section.md") == "57a92209fb5ee6a25257caadb05618771750845ea006b66bd47873d2d9ad34c2"
fence_pattern = rb"(?ms)^```([^\r\n]*)\n(.*?)^```\s*$"
assert re.findall(fence_pattern, old) == re.findall(fence_pattern, new)
own = json.loads((ROOT / "docs/technical-reviews/T.9.json").read_bytes())
assert own["reviewer_task"] == task and own["verdict"] == "revise"
assert sha(ROOT / "docs/technical-reviews/T.9.json") == "e91a5dbdfa9fb660ead13d11427553b8f1abe756552a26df75238b61fb0c03a0"
history = ROOT / "docs/technical-reviews/history/phase4-T_9-initial-revise-20261005.opaque.json"
assert sha(history) == sha(ROOT / "docs/technical-reviews/T.9.json")
hash_checks = {}
for artifact in own["artifacts"]:
    assert sha(ROOT / artifact["path"]) == artifact["sha256"]
    hash_checks[artifact["path"]] = artifact["sha256"]
for source in own["sources"]:
    if source["kind"] == "repository_code":
        assert sha(ROOT / source["path"]) == source["sha256"]
    if "snapshot_path" in source:
        assert sha(ROOT / source["snapshot_path"]) == source["snapshot_sha256"]
copies = json.loads((BASE / "code-copy-manifest.json").read_bytes())
current_code_checks = {}
for original, item in copies.items():
    assert sha(ROOT / original) == sha(ROOT / item["copy"]) == item["sha256"]
    current_code_checks[original] = item["sha256"]
previous = json.loads((BASE / "cpu-probe.json").read_bytes())
assert previous["test_cli_as_in_text"]["declared_split"] == "validation"
assert previous["test_cli_with_explicit_label"]["declared_split"] == "test"
assert previous["test_cli_as_in_text"]["exit_code"] == previous["test_cli_with_explicit_label"]["exit_code"] == 0
assert previous["test_cli_with_explicit_label"]["argv"][-2:] == ["--split-label", "test"]
assert "--split-label" not in previous["test_cli_as_in_text"]["argv"]

# Personally read unchanged evidence again: these assertion checks concern the
# originally recorded execution, not new model scores or repeated training.
assert previous["packing_and_float_rounding"]["float_bytes_before_rounding"] == previous["packing_and_float_rounding"]["float_bytes_after_rounding"] == 20
assert previous["packing_and_float_rounding"]["packed_bytes"] == 3
assert [previous["synthetic_payloads"][k]["tensor_bytes"] for k in ["attributes.pt", "attributes-int4.pt", "attributes-int8.pt"]] == [11440, 8272, 8896]
assert all(v["non_tensor_float_count"] == 0 and v["value_types"] == ["Tensor"] for v in previous["synthetic_payloads"].values())
assert previous["scripted_evaluate_contract"]["skipped_rows"] == [1]
assert [s["row"] for s in previous["scripted_evaluate_contract"]["samples"]] == [0, 2, 3]
audit = json.loads((BASE / "original-audit.json").read_bytes())
assert audit["training_provenance"]["steps"] == audit["training_provenance"]["optimizer_updates"] == 120
assert all(audit["necessary_original_current_leaf_ast_equal"].values())
assert all(run["measurements"]["validation"]["sample_count"] == 5 and run["measurements"]["test"]["sample_count"] == 10 for run in audit["runs"].values())

# Render all three revised command variants directly from the original fence.
commands = []
for language, body in re.findall(fence_pattern, new):
    if language == b"bash":
        for line in body.decode().splitlines():
            if "scripts/evaluate.py" in line:
                args = shlex.split(line)
                args[args.index("--data") + 1] = "data/generated/attributes-sft/test.jsonl"
                args[args.index("--output") + 1] = args[args.index("--output") + 1].replace("-validation.json", "-test.json")
                args += ["--split-label", "test"]
                assert args[args.index("--tokens") + 1] == "24"
                assert args[args.index("--limit") + 1] == "all"
                commands.append(args)
assert len(commands) == 3

# Metadata-only fresh execution of the actual main function. The scoring helper
# is replaced so this reinspection produces no new neural quality measurements.
torch.set_num_threads(1)
assert torch.version.cuda is None
source = ROOT / "outputs/natural-r4-public-checkpoints/public-models/quantization/fp32.pt"
assert sha(source) == previous["readonly_public_exports"]["fp32.pt"]["sha256"]
scoring_calls = []
def metadata_only(model, selected, mode, tokens, tokenizer=None):
    assert isinstance(model, TinyLM)
    assert mode == "sft" and tokens == 24 and len(selected) == 2
    scoring_calls.append({"selected_rows": len(selected), "mode": mode, "tokens": tokens})
    return {"samples": [], "metadata_contract_probe": True}
with tempfile.TemporaryDirectory(prefix="t9-reinspect-metadata-") as tmp:
    tmp = Path(tmp)
    data = tmp / "test.jsonl"
    data.write_text("\n".join(json.dumps({"question": q, "answer": "blue"}) for q in ["a?", "b?"]) + "\n")
    output = tmp / "attributes-fp32-test.json"
    args = ["scripts/evaluate.py", str(source), "--data", str(data), "--mode", "sft", "--tokens", "24", "--limit", "all", "--device", "cpu", "--output", str(output), "--split-label", "test"]
    stdout = io.StringIO()
    with patch.object(sys, "argv", args), patch.object(evaluation, "evaluate", metadata_only), contextlib.redirect_stdout(stdout):
        evaluation.main()
    response = json.loads(output.read_bytes())
    assert response["declared_split"] == "test" and response["limit"] == "all"
    assert response["records_read"] == 2 and response["unselected_records"] == 0
    metadata_execution = {"actual_in_process_argv": args, "scope": "Original evaluate.main executed with only evaluation.evaluate replaced by an explicit metadata-only stub; no scoring, generation or training. Previous two full CPU CLI executions independently remain applicable because source hash is unchanged.", "read_only_checkpoint_sha256": sha(source), "stdout": stdout.getvalue(), "selected_report": {k: response[k] for k in ["declared_split", "limit", "records_read", "unselected_records", "metadata_contract_probe"]}, "scoring_stub_calls": scoring_calls}

result = {"reviewer_task": task, "source": "course/training.md#T.9", "source_sha256": sha(OUT / "section.md"), "prior_source_sha256": sha(BASE / "section.md"), "started_at": started, "completed_at": datetime.now(UTC).isoformat(), "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda)}, "actual_read_scope": ["Complete new T.9 412-474 raw UTF-8 section", "Own initial claim/issue/check records and original CPU/audit/provenance evidence in full", "Original and current scripts/evaluate.py main164-209; unchanged SHA and literal --split-label/default/declared_split contract", "All other formal artifact and original source copy hashes verified; unchanged claims rechecked against their previously personally read evidence", "Only own report history inspected; coordinator administrative revision artifacts not used as technical authority"], "only_section_edit": {"before": before, "after": after, "exact_single_replacement": True}, "fences_unchanged": True, "current_code_copy_sha256": current_code_checks, "formal_prior_artifact_sha256_verified": hash_checks, "old_cli_evidence": {"without_flag": previous["test_cli_as_in_text"], "with_flag": previous["test_cli_with_explicit_label"]}, "rendered_revised_test_commands": commands, "fresh_metadata_only_main_execution": metadata_execution, "existing_unaffected_claim_ids_rechecked": [c["id"] for c in own["claims"] if c["id"] != "evaluation-contract-and-label"], "decision": "T9-test-label resolved by the revised sentence. No other unresolved substantive claim found; eligible for pass after own canonical assertions and separate single-section checker.", "history": {"initial_report": str(history.relative_to(ROOT)), "initial_report_sha256": sha(history), "initial_own_report": str((BASE / "history/report-prechecker-revise.json").relative_to(ROOT)), "initial_scope": str((BASE / "scope-final.json").relative_to(ROOT)), "historical_whole_input": str((BASE / "inputs/course/training.md").relative_to(ROOT)), "historical_whole_input_sha256": sha(BASE / "inputs/course/training.md"), "historical_whole_input_status": "Retained frozen input at initial copy time, not a claim about the current full chapter. Bytes and hash not updated."}}
(OUT / "reinspection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
(OUT / "environment.json").write_text(json.dumps(result["environment"], indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
