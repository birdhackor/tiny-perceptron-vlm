"""T.4 substantive v9 callback: six short CPU evaluations and greedy call tracing."""

import ast
import copy
import difflib
import hashlib
import inspect
import json
import os
import platform
import re
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch

import scripts.evaluate as evaluator
from tiny_perceptron.data import load_jsonl
from tiny_perceptron.model import generate
from tiny_perceptron.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
OLD_RUN = Path("/workspace/work/omni-technical-t-4-v1/run")
REPLAY = OLD_RUN / "callback-v2"
REPLAY.mkdir(exist_ok=False)
FOLDER = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_t_4"
ORIGINAL = FOLDER / "technical-original-v2.json"
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v9.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": digest(path)}


original = json.loads(ORIGINAL.read_text())
old_cpu = json.loads((ART / "cpu-verification.json").read_text())
freeze = json.loads(FREEZE.read_text())
page = next(page for page in freeze["pages"] if page["page_id"] == "T.4")
snapshot = ROOT / page["snapshot"]
assert digest(snapshot) == page["source_sha256"]
raw = snapshot.read_text()
old_snapshot = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.4.md"
old_raw = old_snapshot.read_text()
new_paragraph = "以下的`evaluate.py`生成答案時固定採用**greedy（貪婪選取）**：每一步取目前機率最高的token，再接著生成下一步，不做隨機抽樣。這個工具目前沒有切換選字方式的命令列參數；前後比較使用同一份驗證題與這項選字規則。"
assert raw.replace("\n" + new_paragraph + "\n", "", 1) == old_raw
assert re.findall(r"```(?:bash|python|jsonl)\n(.*?)```", raw, re.S) == re.findall(r"```(?:bash|python|jsonl)\n(.*?)```", old_raw, re.S)
live = (ROOT / "course/training.md").read_text()
section = re.search(r"^## T\.4 .+?\n(?:(?!^## ).)*", live, re.M | re.S)[0]
assert section.encode() == snapshot.read_bytes()
canonical_ref = freeze["canonical_source_freeze"]
assert digest(ROOT / canonical_ref["path"]) == canonical_ref["sha256"]
canonical = json.loads((ROOT / canonical_ref["path"]).read_text())
training = next(page for page in canonical["pages"] if page["page_id"] == "training")
canonical_snapshot = ROOT / training["snapshot"]
assert digest(canonical_snapshot) == training["source_sha256"]
canonical_slice = re.search(r"^## T\.4 .+?\n(?:(?!^## ).)*", canonical_snapshot.read_text(), re.M | re.S)[0]
assert canonical_slice.encode() == snapshot.read_bytes()

report = {"started_at": datetime.now(UTC).isoformat(), "reviewer_task": original["reviewer_task"], "lesson_id": "T.4", "command": "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-t-4-v1/verify_callback_v2.py", "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git": torch.version.git_version, "device": "cpu", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, "source_freeze": ref(FREEZE), "canonical_source_freeze": canonical_ref, "snapshot": ref(snapshot), "source_diff": "".join(difflib.unified_diff(old_raw.splitlines(keepends=True), raw.splitlines(keepends=True), fromfile="T.4-v5", tofile="T.4-v9")), "all_embedded_code_blocks_identical_to_original": True, "canonical_slice_and_live_source_match": True, "figure_sha256": page["figures_sha256"], "original_report": ref(ORIGINAL), "commands": [], "reused_artifacts": [], "repository_sources": [], "primary_originals_rechecked": [], "prerequisites": []}
assert report["figure_sha256"] == {}
for artifact in original["artifacts"]:
    assert digest(ROOT / artifact["path"]) == artifact["sha256"]
    report["reused_artifacts"].append({"id": artifact["id"], "path": artifact["path"], "sha256": artifact["sha256"], "fingerprint_rechecked": True})
for source in original["sources"]:
    if source["kind"] == "repository_code":
        assert digest(ROOT / source["path"]) == source["sha256"]
        report["repository_sources"].append({"id": source["id"], "path": source["path"], "sha256": source["sha256"], "unchanged_since_original": True})
primary = json.loads((ART / "primary-source-inspection.json").read_text())
for source in primary["source_inspections"]:
    assert digest(source["original_path"]) == source["original_sha256"]
    assert digest(source["readable_text_path"]) == source["readable_text_sha256"]
    report["primary_originals_rechecked"].append({"id": source["id"], "original_sha256": source["original_sha256"], "text_sha256": source["readable_text_sha256"], "support_scope": source["inspection_note"], "use": "Original first-review source inspection remains applicable to identical conceptual passages; not represented as a fresh download."})
for prerequisite in primary["prerequisites"]:
    entry = copy.deepcopy(prerequisite)
    path = ROOT / prerequisite["path"]
    if "selector" in prerequisite:
        sid = prerequisite["selector"].split(":", 1)[1]
        text = re.search(r"^## " + re.escape(sid) + r" .+?\n(?:(?!^## ).)*", path.read_text(), re.M | re.S)[0]
        entry["current_section_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    else:
        entry["current_sha256"] = digest(path)
    entry["callback_read_scope"] = "Re-read the explicitly linked full text section/document for the T.4 continuation, mask, split, token-budget and tokenizer context. Did not review/render prerequisite figures or sign another lesson."
    report["prerequisites"].append(entry)
for source in json.loads((ART / "pytorch-source-receipts.json").read_text()):
    assert digest(source["file"]) == source["sha256"]
report["historical_source_pins_rechecked"] = []
for row in json.loads((ART / "author-results-audit.json").read_text())["results"]:
    pins = []
    for source in row["code"]:
        value = subprocess.run(["git", "show", row["revision"] + ":" + source["path"]], cwd=ROOT, check=True, capture_output=True).stdout
        assert hashlib.sha256(value).hexdigest() == source["report_sha256"]
        pins.append({"path": source["path"], "sha256": source["report_sha256"]})
    report["historical_source_pins_rechecked"].append({"experiment_id": row["experiment_id"], "revision": row["revision"], "pins": pins, "scope": "Original historical record and actual pinned Git source, not a newly executed fixed recipe."})

for kind, value in old_cpu["data"].items():
    for name, expected in value["repeat_exact_hashes"].items():
        assert digest(OLD_RUN / "data/generated" / kind / name) == expected
report["cached_checkpoint_fingerprints"] = []
expected_weights = {"start": old_cpu["baseline"]["checkpoint_sha256"], **{key: value["checkpoint_sha256"] for key, value in old_cpu["training_summaries"].items()}}
for name, expected in expected_weights.items():
    path = OLD_RUN / "checkpoints" / (name + ".pt")
    assert digest(path) == expected
    report["cached_checkpoint_fingerprints"].append({"name": name, "path": str(path), "sha256": expected})
report["reuse_semantics"] = ["All old artifact file hashes and all 17 repository/record source hashes match the formal original.", "Every embedded Python/bash/JSONL block is byte-identical; old training and data-generation evidence still describes the exact present commands.", "Original 200/500/500 update counts and hand-calculation/executed distinctions remain historical first-review evidence; no training was repeated on callback.", "Current generation rule is independently traced below and all six current evaluations are replayed with identical checkpoint/data/budget, changing only output destination to preserve old files."]

interpreter = ["uv", "run", "--project", str(ROOT), "--extra", "cpu", "python"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")


def run(arguments, label, expected_exit=0):
    command = interpreter + arguments
    started = datetime.now(UTC).isoformat()
    process = subprocess.run(command, cwd=OLD_RUN, env=env, text=True, capture_output=True, timeout=30)
    report["commands"].append({"label": label, "command": shlex.join(command), "cwd": str(OLD_RUN), "started_at": started, "completed_at": datetime.now(UTC).isoformat(), "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr})
    assert process.returncode == expected_exit, report["commands"][-1]
    return process


run(["scripts/evaluate.py", "--help"], "Current evaluation CLI help")
rejected = run(["scripts/evaluate.py", "checkpoints/start.pt", "--temperature", "0.7", "--device", "cpu"], "Current CLI rejects unimplemented temperature switch before evaluation", 2)
assert "unrecognized arguments: --temperature 0.7" in rejected.stderr
tree = ast.parse((ROOT / "scripts/evaluate.py").read_text())
options = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
        options.extend(arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
report["evaluation_cli_options"] = options
assert not set(options) & {"--temperature", "--top-k", "--top-p", "--do-sample", "--greedy", "--sampling"}
report["replayed_evaluations"] = {}
for line in raw.splitlines():
    if line.startswith(".venv/bin/python scripts/evaluate.py "):
        arguments = shlex.split(line)[1:]
        index = arguments.index("--output") + 1
        name = Path(arguments[index]).name
        arguments[index] = str(REPLAY / name)
        run(arguments, "Current exact evaluation inputs/budgets; output destination isolated")
        value = json.loads((REPLAY / name).read_text())
        assert value == old_cpu["evaluations"][name], name
        report["replayed_evaluations"][name] = {"identical_to_original_complete_json": True, "current_full_output": value}
assert len(report["replayed_evaluations"]) == 6

model, _ = load_checkpoint(OLD_RUN / "checkpoints/attributes.pt", "cpu")
record = load_jsonl(OLD_RUN / "data/generated/attributes-sft/validation.jsonl")[0]
calls = []
actual_generate = generate


def traced_generate(current_model, ids, *args, **kwargs):
    bound = inspect.signature(actual_generate).bind(current_model, ids, *args, **kwargs)
    bound.apply_defaults()
    trace = []

    def observe(module, inputs, output):
        logits = output["logits"][:, -1]
        probabilities = logits.softmax(-1)
        chosen = logits.argmax(-1)
        assert torch.equal(chosen, probabilities.argmax(-1))
        trace.append({"input_length": inputs[0].shape[1], "argmax_logit_id": chosen.tolist(), "argmax_probability_id": probabilities.argmax(-1).tolist(), "selected_logit": logits.gather(-1, chosen[:, None]).flatten().tolist(), "selected_probability": probabilities.gather(-1, chosen[:, None]).flatten().tolist()})

    hook = current_model.register_forward_hook(observe)
    before = torch.get_rng_state().clone()
    try:
        with patch.object(torch, "multinomial", side_effect=AssertionError("Unexpected random sampling during evaluator generation")):
            output = actual_generate(current_model, ids, *args, **kwargs)
    finally:
        hook.remove()
    generated = output[0, ids.shape[1]:].tolist()
    assert generated == [entry["argmax_logit_id"][0] for entry in trace]
    assert torch.equal(before, torch.get_rng_state())
    calls.append({"bound_defaults": {key: bound.arguments[key] for key in ("max_new_tokens", "temperature", "eos_id", "use_cache")}, "prompt_ids": ids.tolist(), "generated_ids": generated, "step_trace": trace, "torch_rng_unchanged": True, "sampling_call_guard": "torch.multinomial would raise; no call occurred"})
    return output


with patch.object(evaluator, "generate", traced_generate):
    instrumented = evaluator.evaluate(model, [record], mode="sft", max_new_tokens=24)
assert len(calls) == 1 and calls[0]["bound_defaults"]["temperature"] == 0.0
assert calls[0]["generated_ids"] == old_cpu["evaluations"]["attributes-after.json"]["samples"][0]["generated_ids"]
report["actual_evaluate_to_generate_trace"] = {"input": record, "calls": calls, "result": instrumented, "scope": "Actual loaded lesson model and real validation row, actual evaluate and generate; observers only, no updates. This verifies this evaluator's local greedy call contract, not model quality or global sequence optimality."}

torch_root = Path(torch.__file__).parent
argmax_path = torch_root / "_torch_docs.py"
argmax_text = argmax_path.read_text()
position = argmax_text.index("    torch.argmax,")
argmax_excerpt = argmax_text[position:position + 560]
softmax_path = Path(inspect.getsourcefile(torch.nn.functional.softmax))
softmax_lines, start = inspect.getsourcelines(torch.nn.functional.softmax)
report["new_official_source_inspection"] = [{"name": "torch.argmax", "version": str(torch.__version__), "git_version": torch.version.git_version, "path": str(argmax_path), "sha256": digest(argmax_path), "start_line": argmax_text[:position].count("\n") + 1, "short_excerpt": argmax_excerpt}, {"name": "torch.nn.functional.softmax", "version": str(torch.__version__), "git_version": torch.version.git_version, "path": str(softmax_path), "sha256": digest(softmax_path), "start_line": start, "short_excerpt": "".join(softmax_lines[:17])}]
report["softmax_order_derivation"] = "For finite logits z_i, p_i=exp(z_i)/sum_j exp(z_j). The denominator is positive and common to all i, and exp is strictly increasing, so an argmax logit is also a maximum-probability token. Ties remain valid maxima; installed argmax documents selecting the first maximal index. This is an algebraic derivation, not a training experiment. The separate executed trace checks logits/probability maxima on the actual row."
report["original_claim_reassessment"] = [{"claim_ids": ["c01", "c04", "c05", "c18", "c27", "c40"], "finding": "Conceptual passages, source identity and prior inspected original authorities unchanged; full current T.4 and explicit context re-read. Objective, split/group, mean-NLL and tokenizer identity limitations remain valid."}, {"claim_ids": ["c02", "c03", "c06", "c07", "c08", "c09", "c10", "c11"], "finding": "Unchanged generator, tokenizer/chat mask and literal embedded baseline code; cached data/checkpoint hashes and complete original execution evidence rechecked. Counts/seed/masks/optimizer-absence remain correctly scoped."}, {"claim_ids": ["c12", "c13", "c14", "c15", "c16", "c17"], "finding": "Training commands and relevant implementation hashes unchanged; historical short-update and resume proofs remain evidence of those exact configurations. All six evaluation commands replay exactly; no callback training or timing extrapolation."}, {"claim_ids": ["c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26"], "finding": "Full current evaluate/model read; replayed complete JSON equals old outputs, preserving effective targets 35/36, actual sample denominators 1/5, raw IDs, EOS, null text matches and skip/limit semantics. Existing deliberate mask/skip/illegal-token probes' unchanged software and evidence remain valid."}, {"claim_ids": ["c28", "c29", "c30", "c31", "c32", "c33", "c34", "c35", "c36", "c37", "c38", "c39", "c41", "c42"], "finding": "Fixed-route commands/text/records unchanged; current file hashes and all 30 original source pins rechecked. Historical times/configurations/denominators are retained as recorded single L4 runs, not callback executions. No fixed long/paid/GPU route run."}]
covered = [identifier for group in report["original_claim_reassessment"] for identifier in group["claim_ids"]]
assert len(covered) == len(set(covered)) == 42 and set(covered) == {claim["id"] for claim in original["claims"]}
report["limitations"] = ["Only T.4 is reviewed; no T.9 or other reviewer/reader conclusions are signed.", "Full current T.4 and explicit necessary text context were read; no prerequisite figure-viewing claim.", "No new training, GPU, paid work, download or publishing occurred; six CPU evaluations and one observed real-row generation only.", "Both after-SFT validations remain exact 0/5 with EOS 5/5; lower NLL is not answer-quality evidence.", "This is a substantive current-source callback, not merely a source-hash replacement or metadata-only correction. Raw original, formal original and prior current files are untouched."]
report["completed_at"] = datetime.now(UTC).isoformat()
with (ART / "callback-v2-verification.json").open("x") as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": "all callback assertions passed", "source_sha256": page["source_sha256"], "old_claims_reassessed": len(covered), "full_evaluation_outputs_replayed_identically": len(report["replayed_evaluations"]), "generation_default": calls[0]["bound_defaults"], "actual_generated_ids": calls[0]["generated_ids"]}, ensure_ascii=False))
