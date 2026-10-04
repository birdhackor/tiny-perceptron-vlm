"""Read-only CPU audit of the optional training recipe; no model load/forward/train."""
import ast
import collections
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "natural-final-fact-20.8-training-"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    path = OUT / (PREFIX + name)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return str(path.relative_to(ROOT))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guide_path = ROOT / "docs/natural-assistant/TRAINING.md"
guide = guide_path.read_text()
fences = re.findall(r"```bash\n(.*?)\n```", guide, re.S)
shell_path = OUT / (PREFIX + "commands.sh")
shell_path.write_text("\n\n".join(fences) + "\n")
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
commands = [
    ["bash", "-n", str(shell_path.relative_to(ROOT))],
    [".venv-natural/bin/python", "scripts/natural_assistant.py", "--help"],
    [".venv-natural/bin/python", "scripts/fetch_natural_data.py", "--help"],
    [".venv-natural/bin/python", "scripts/fetch_natural_data.py", "--list"],
    [".venv-natural/bin/python", "scripts/fetch_natural_data.py", "--manifest", "docs/natural-assistant/manifest.json", "--output", "outputs/natural-extension/data-fetch-student-smoke/data", "--verify"],
]
executions = []
for index, command in enumerate(commands):
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    stdout = OUT / (PREFIX + f"command-{index}.stdout.txt")
    stderr = OUT / (PREFIX + f"command-{index}.stderr.txt")
    stdout.write_text(result.stdout)
    stderr.write_text(result.stderr)
    executions.append({"argv": command, "cwd": str(ROOT), "exit_code": result.returncode, "elapsed_seconds": time.monotonic() - started, "stdout": str(stdout.relative_to(ROOT)), "stdout_sha256": sha(stdout), "stderr": str(stderr.relative_to(ROOT)), "stderr_sha256": sha(stderr)})
    assert result.returncode == 0, executions[-1]

cli = load_module("fact_training_cli", "scripts/natural_assistant.py")
helper = load_module("fact_training_data", "scripts/fetch_natural_data.py")
parsed = []
for fence in fences:
    for line in fence.replace("\\\n", " ").splitlines():
        if line.startswith(".venv-natural/bin/python scripts/natural_assistant.py "):
            tokens = shlex.split(line)
            if ">" in tokens:
                tokens = tokens[:tokens.index(">")]
            options = cli.parser().parse_args(tokens[2:])
            item = {key: str(value) if isinstance(value, Path) else value for key, value in vars(options).items()}
            item["source_argv"] = tokens
            parsed.append(item)
assert len(parsed) == 5
stages = [item["stage"] for item in parsed]
assert stages == ["prepare", "train", "train", "validation", "evaluate"], stages
fresh, resumed = parsed[1:3]
assert fresh["adapter"] is None
assert resumed["adapter"] == "outputs/natural-my-experiment/train/adapter"
assert resumed["output"] != fresh["output"]
for item in (fresh, resumed):
    assert item["steps"] == 180 and item["learning_rate"] == 1e-4
    assert item["dtype"] == "bfloat16" and item["device"] == "cuda"
    assert item["lora_rank"] == 8 and item["gradient_accumulation"] == 2 and item["seed"] == 42
    assert item["checkpoint_every"] == 25 and item["max_seconds"] == 3300 and item["local_files_only"]
assert parsed[3]["split"] == "validation" and parsed[4]["split"] == "test"
defaults = vars(cli.parser().parse_args(["train", "--output", "unused-parser-only"]))
assert defaults["learning_rate"] == 3e-5 and defaults["steps"] == 100

manifest_path = ROOT / "docs/natural-assistant/manifest.json"
manifest = helper.load_manifest(manifest_path, helper.DEFAULT_MANIFEST_SHA256)
rows = collections.Counter(row["split"] for row in manifest["rows"])
audio = collections.Counter(row["split"] for row in manifest["audio_rows"])
assert [rows[k] for k in ("train", "validation", "test")] == [272, 52, 66]
assert [audio[k] for k in ("train", "validation", "test")] == [24, 6, 12]
archive_total = sum(item["bytes"] for item in manifest["archives"])
assert archive_total == 70648031 and len(manifest["files"]) == 345
denominators = {split: rows[split] + 2 * audio[split] for split in ("validation", "test")}
assert denominators == {"validation": 64, "test": 90}

core_path = ROOT / "tiny_perceptron/natural_assistant.py"
core = core_path.read_text()
tree = ast.parse(core)
train_ast = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_train")
resume_ast = next(node for node in train_ast.body if isinstance(node, ast.If) and ast.unparse(node.test) == "options.adapter")
guard_ast = next(node for node in resume_ast.body if isinstance(node, ast.For))
guard_fields = ast.literal_eval(guard_ast.iter)
assert guard_fields == ("manifest_sha256", "asset_sha256", "model_revision", "lora_rank", "gradient_accumulation", "seed", "learning_rate")
# Execute precisely the extracted production guard, not run_train or any optimizer/model.
guard_code = compile(ast.fix_missing_locations(ast.Module(body=[guard_ast], type_ignores=[])), str(core_path) + "#resume_guard_only", "exec")
prior = {field: f"unchanged-{field}" for field in guard_fields}
exec(guard_code, {"prior": prior, "record": copy.deepcopy(prior)})
rejections = []
for field in guard_fields:
    changed = copy.deepcopy(prior)
    changed[field] = "changed"
    try:
        exec(guard_code, {"prior": prior, "record": changed})
    except ValueError as error:
        assert str(error) == f"Resume contract changed: {field}"
        rejections.append({"field": field, "actual_exception": str(error)})
    else:
        raise AssertionError(field)
total_guard = next(node for node in resume_ast.body if isinstance(node, ast.If) and "options.steps" in ast.unparse(node.test))
total_code = compile(ast.fix_missing_locations(ast.Module(body=[total_guard], type_ignores=[])), str(core_path) + "#total_steps_guard_only", "exec")
from types import SimpleNamespace
exec(total_code, {"options": SimpleNamespace(steps=180), "prior": {"completed_steps": 175}})
try:
    exec(total_code, {"options": SimpleNamespace(steps=180), "prior": {"completed_steps": 181}})
except ValueError as error:
    total_rejection = str(error)
else:
    raise AssertionError("Target below prior completed step was accepted")
save_ast = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "save_checkpoint")
save_dict = next(node for node in ast.walk(save_ast) if isinstance(node, ast.Dict) and any(isinstance(key, ast.Constant) and key.value == "optimizer" for key in node.keys))
state_keys = [key.value for key in save_dict.keys]
assert state_keys == ["optimizer", "torch_rng", "cuda_rng"]

original_path = ROOT / "docs/natural-assistant/evidence/train/result.json"
original = json.loads(original_path.read_text())
original_fields = ["model_revision", "dtype", "device", "gpu_name", "min_pixels", "max_pixels", "max_tokens", "seed", "lora_rank", "lora_targets", "requested_steps", "completed_steps", "learning_rate", "gradient_accumulation", "trained_rows", "trainable_parameters", "optimizer_only_lora", "changed_adapter_tensor_count", "frozen_check_scope", "frozen_parameter_samples_unchanged", "elapsed_seconds", "peak_cuda_memory_allocated_bytes"]
assert original["learning_rate"] == 1e-4 and original["completed_steps"] == 180 and original["trained_rows"] == 360
assert original["trainable_parameters"] == 1605632 and len(original["optimizer_parameter_names"]) == 112
assert all("lora_" in name for name in original["optimizer_parameter_names"])
assert len(original["initial_adapter_tensors"]) == 112 and original["changed_adapter_tensor_count"] == 112
adapter_params = sum(__import__("math").prod(item["shape"]) for item in original["initial_adapter_tensors"].values())
assert adapter_params == 1605632
metadata_path = ROOT / "outputs/natural-extension/runs/train-37190116187/natural-natural-v3-train-37190116187-1/review/adapter/training.json"
metadata = json.loads(metadata_path.read_text())
metadata_difference = {key: {"public_evidence": original.get(key), "retained_adapter_metadata": metadata.get(key)} for key in set(original) | set(metadata) if original.get(key) != metadata.get(key)}
assert all(metadata[field] == original[field] for field in guard_fields)
metadata_snapshot = OUT / (PREFIX + "original-adapter-metadata.json")
metadata_snapshot.write_bytes(metadata_path.read_bytes())
state_exists = (metadata_path.parent / "training_state.pt").exists()

release = json.loads((ROOT / "docs/natural-assistant/public-release.json").read_text())
modal = (ROOT / "scripts/modal_natural.py").read_text()
workflow = (ROOT / ".github/workflows/natural-assistant.yml").read_text()
assert '"--learning-rate", "0.00003"' in modal
assert "learning_rate:" not in workflow and "learning-rate:" not in workflow
assert "range(record[\"completed_steps\"], options.steps)" in core
assert "refusing silent multimodal truncation" in core
assert '"Finite phrase rubric, not a complete semantic judge"' in core
assert '"EOS reports decoder stopping, not semantic completeness' in core

source_paths = ["docs/natural-assistant/TRAINING.md", "course/chapters/20.md", "scripts/natural_assistant.py", "scripts/fetch_natural_data.py", "tiny_perceptron/natural_assistant.py", "scripts/modal_natural.py", ".github/workflows/natural-assistant.yml", "docs/natural-assistant/manifest.json", "docs/natural-assistant/public-release.json", "docs/natural-assistant/evidence/train/result.json", "docs/natural-assistant/STUDENT.md", "requirements-natural.txt"]
current_sources = {path: sha(ROOT / path) for path in source_paths}
old = json.loads((OUT / "natural-final-fact-20.8-pass-3733ef68-report.json").read_text())
preserved_artifacts = [{"id": item["id"], "path": item["path"], "recorded_sha256": item["sha256"], "actual_sha256": sha(ROOT / item["path"])} for item in old["artifacts"]]
assert all(item["recorded_sha256"] == item["actual_sha256"] for item in preserved_artifacts)
unchanged_live_sources = []
for item in old["sources"]:
    if item["kind"] == "repository_code":
        actual = sha(ROOT / item["path"])
        unchanged_live_sources.append({"id": item["id"], "path": item["path"], "recorded_sha256": item["sha256"], "actual_sha256": actual})
        assert actual == item["sha256"], item

fixed_git = []
for path in ("scripts/natural_assistant.py", "scripts/fetch_natural_data.py"):
    url = f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/1d5fccbc76a6ebbb3757f9b31df17a7c22ea171b/{path}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "independent-factual-review", "Accept-Encoding": "identity"}), timeout=30) as response:
        remote = response.read(200000)
        status = response.status
    local = (ROOT / path).read_bytes()
    assert remote == local and status == 200
    snapshot = OUT / (PREFIX + "student-" + path.replace("/", "__"))
    snapshot.write_bytes(remote)
    fixed_git.append({"url": url, "status": status, "bytes": len(remote), "sha256": sha(snapshot), "matches_current": True, "snapshot": str(snapshot.relative_to(ROOT))})

result = {"scope": "CPU help, shell syntax, parser, original source guard excerpt, data hash verification and fixed public code read only; no new training/model loading/forward/ASR/install", "current_source_hashes": current_sources, "executions": executions, "shell_fences": len(fences), "parsed_production_cli_commands": parsed, "cli_defaults": {"steps": defaults["steps"], "learning_rate": defaults["learning_rate"]}, "data": {"pin": helper.DEFAULT_REVISION, "manifest_sha256": sha(manifest_path), "files": len(manifest["files"]), "archive_total_bytes": archive_total, "archive_decimal_MB": archive_total / 1e6, "rows_by_split": dict(rows), "audio_by_split": dict(audio), "per_variant_generation_denominators": denominators}, "resume": {"state_keys": state_keys, "seven_guard_fields": list(guard_fields), "actual_guard_rejections": rejections, "total_steps_below_completed_rejection": total_rejection, "accepted_180_from_175_means_remaining_updates": list(range(175, 180)), "not_programmatically_guarded": ["dtype", "device", "min_pixels", "max_pixels", "max_tokens", "package versions", "code SHA", "model repo ID"], "guide_instruction_keeps_additional_settings_same": True, "actual_retained_metadata_source": str(metadata_path.relative_to(ROOT)), "metadata_snapshot": str(metadata_snapshot.relative_to(ROOT)), "metadata_sha256": sha(metadata_snapshot), "metadata_matches_guard_fields": True, "metadata_difference": metadata_difference, "training_state_pt_available_here": state_exists, "no_actual_resume_execution": True, "public_inference_bundle_has_optimizer_or_rng_state": False}, "original_training": {key: original[key] for key in original_fields}, "original_trainable_tensor_shape_sum": adapter_params, "original_peak_allocated_GiB": original["peak_cuda_memory_allocated_bytes"] / 2**30, "time_scope": {"train_starts": "after load_manifest/data verification and torch.manual_seed; before load_core", "guard": "before each update, not preemption", "possible_overrun": "one complete update plus synchronization, final tensor/sample hashes, final checkpoint and JSON writes", "record_elapsed_stops": "after loop and synchronize, before final hashes/frozen sample check/final save", "evaluation_deadline": "before manifest verification/load; checked before each ASR/visual generation/audio pair, not preemption"}, "metrics_scope": {"completed": "requested record cardinalities only, not successful answers", "EOS": "decoder stopping only; inspect truncated and completion_unknown and full answers", "automatic_score": "finite keyword/exact/NFKC proxy, not image semantic judgment", "ASR_routes": "one raw hypothesis per audio shared across base/adapter; separate actual speech_chat and typed official transcript", "selection": "own validation-only choice, own frozen weights before test, preserve failed test; no author selection reuse or test reselection"}, "prior_54_artifacts_hashes_preserved": preserved_artifacts, "prior_registered_live_sources_unchanged": unchanged_live_sources, "fixed_student_git_helpers": fixed_git}
save("audit.json", result)
print(json.dumps({"commands_exit_codes": [item["exit_code"] for item in executions], "parsed_stages": stages, "files_verified": 345, "per_variant": denominators, "seven_actual_resume_guard_rejections": len(rejections), "actual_training_state_available": state_exists, "preserved_old_artifacts": len(preserved_artifacts), "public_fixed_helpers": len(fixed_git)}, ensure_ascii=False))
