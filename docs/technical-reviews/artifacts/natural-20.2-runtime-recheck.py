from pathlib import Path
from datetime import datetime, timezone
import ast
import difflib
import hashlib
import json
import math
import platform
import re

ROOT = Path(__file__).resolve().parents[3]
PREFIX = Path("docs/technical-reviews/artifacts")
REPORT = ROOT / "docs/technical-reviews/20.2.json"
prior_path = ROOT / PREFIX / "natural-20.2-prior-report-before-lr-recheck.json"
if not prior_path.exists():
    prior_path.write_bytes(REPORT.read_bytes())
prior = json.loads(prior_path.read_bytes())
old_source = next(s for s in prior["sources"] if s["id"] == "s_runtime")
current_path = ROOT / "scripts/modal_natural.py"
current = current_path.read_bytes()
current_sha = hashlib.sha256(current).hexdigest()
added = b'args.extend(["--steps", str(options["steps"]), "--checkpoint-every", "25", "--learning-rate", "0.00003"])'
previous = b'args.extend(["--steps", str(options["steps"]), "--checkpoint-every", "25"])'
assert current.count(added) == 1
restored = current.replace(added, previous)
assert hashlib.sha256(restored).hexdigest() == old_source["sha256"]
old_path = ROOT / PREFIX / "natural-20.2-modal-prior.py.txt"
if old_path.exists():
    assert old_path.read_bytes() == restored
else:
    old_path.write_bytes(restored)
current_copy = ROOT / PREFIX / "natural-20.2-modal-current.py.txt"
current_copy.write_bytes(current)
diff = "".join(difflib.unified_diff(restored.decode().splitlines(keepends=True), current.decode().splitlines(keepends=True),
    fromfile="prior-modal-sha-" + old_source["sha256"], tofile="current-modal-sha-" + current_sha))
(ROOT / PREFIX / "natural-20.2-modal-lr-diff.txt").write_text(diff)
section_text = (ROOT / "course/chapters/20.md").read_text(encoding="utf-8")
section = re.search(r"^## 20\.2 .*?(?=^## )", section_text, re.M | re.S).group()
section_sha = hashlib.sha256(section.encode()).hexdigest()
assert section_sha == prior["source_sha256"]
assert section == (ROOT / PREFIX / "natural-20.2-section.md").read_text()
unchanged = []
for artifact in prior["artifacts"]:
    actual = hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest()
    assert actual == artifact["sha256"], artifact["id"]
    unchanged.append({"id": artifact["id"], "path": artifact["path"], "sha256": actual})
source_status = []
for source in prior["sources"]:
    if source["kind"] == "repository_code":
        actual = hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
        assert source["id"] == "s_runtime" or actual == source["sha256"]
        source_status.append({"id": source["id"], "old_sha256": source["sha256"], "current_sha256": actual,
            "unchanged": source["sha256"] == actual})
tree = ast.parse(current.decode(), filename="scripts/modal_natural.py")
execution = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute_stage")
train_branch = next(n for n in execution.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
    and isinstance(n.test.left, ast.Name) and n.test.left.id == "stage"
    and len(n.test.comparators) == 1 and isinstance(n.test.comparators[0], ast.Constant) and n.test.comparators[0].value == "train")
assert ast.get_source_segment(current.decode(), train_branch.body[0]) == added.decode()
image_start = current.decode().index('natural_image = (')
image_end = current.decode().index('\n\n\ndef execute_stage', image_start)
runtime_block = current.decode()[image_start:image_end]
for pin in ['python_version="3.12"', '"torch==2.8.0"', '"torchvision==0.23.0"', '"transformers==4.57.6"', '"peft==0.18.1"']:
    assert pin in runtime_block
assistant = (ROOT / "tiny_perceptron/natural_assistant.py").read_text()
assistant_tree = ast.parse(assistant)
constants = {}
for node in assistant_tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in {"MODEL_ID", "MODEL_REVISION", "ASR_ID", "ASR_REVISION", "LORA_TARGETS"}:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
assert constants["MODEL_REVISION"] == "89644892e4d85e24eaac8bacfd4f463576704203"
assert constants["ASR_REVISION"] == "973afd24965f72e36ca33b3055d56a652f456b4d"
assert constants["MODEL_ID"] == "Qwen/Qwen3-VL-2B-Instruct"
assert constants["ASR_ID"] == "openai/whisper-small"
source_folder = ROOT / PREFIX / "natural-20.2-sources"
counts = {}
for name, expected in [("qwen", 2127532032), ("whisper", 241734912), ("adapter", 1605632)]:
    header = json.loads((source_folder / (name + "-header.json")).read_bytes())
    value = sum(math.prod(t["shape"]) for key, t in header.items() if key != "__metadata__")
    assert value == expected
    counts[name] = value
assert sum(counts.values()) == 2370872576
baseline = json.loads((ROOT / "docs/natural-assistant/evidence/baseline/result.json").read_bytes())
train = json.loads((ROOT / "docs/natural-assistant/evidence/train/result.json").read_bytes())
cpu = json.loads((ROOT / "docs/natural-assistant/evidence/usage/cpu-base-input-routes.json").read_bytes())
assert baseline["model_revision"] == constants["MODEL_REVISION"] and baseline["asr_revision"] == constants["ASR_REVISION"]
assert baseline["status"] == "completed" and baseline["gpu_name"] == "NVIDIA L4"
assert baseline["total_parameters"] == counts["qwen"] and baseline["trainable_parameters"] == 0
assert cpu["provenance"]["device"] == "cpu" and cpu["provenance"]["dtype"] == "float32"
assert cpu["provenance"]["model_revision"] == constants["MODEL_REVISION"]
assert cpu["provenance"]["asr_revision"] == constants["ASR_REVISION"]
assert train["trainable_parameters"] == counts["adapter"] and train["optimizer_only_lora"] is True
assert len(train["optimizer_parameter_names"]) == 112
assert all("lora_" in name for name in train["optimizer_parameter_names"])
assert train["completed_steps"] == 180 and train["learning_rate"] == 0.0001
assert 'model.requires_grad_(False)' in assistant
assert '@torch.inference_mode()\ndef transcribe' in assistant
assert 'WhisperForConditionalGeneration.from_pretrained' in assistant
copy_index = json.loads((ROOT / "docs/natural-assistant/evidence/runtime/gentle-offline-copy-index.json").read_bytes())
gentle = []
for entry in copy_index["entries"]:
    raw = (ROOT / entry["destination"]).read_bytes()
    assert len(raw) == entry["bytes"]
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert raw == (ROOT / entry["source"]).read_bytes()
    data = json.loads(raw)
    assert data["program_sha256"] == current_sha
    if "cases" in data:
        assert len(data["cases"]) == 4
        assert all(case["all_entrypoints_present"] and case["remote_calls"] == 0 for case in data["cases"])
        assert all(case["modal_version"] == "1.6.0" for case in data["cases"])
        count = len(data["cases"])
    else:
        assert all(check["passed"] is True for check in data["checks"])
        count = len(data["checks"])
        assert count == (17 if "selection" in entry["destination"] else 26)
    gentle.append({"path": entry["destination"], "sha256": entry["sha256"], "program_sha256": data["program_sha256"],
        "scope": data["scope"], "passed_count": count})
affected = {"c18": "Its repository-code runtime source changed. Runtime image Python/torch/Transformers pins are byte-identical; this claim remains verified."}
impact = []
for c in prior["claims"]:
    assert c["status"] == "verified"
    impact.append({"claim_id": c["id"], "status": "verified", "assessment": affected.get(c["id"],
        "Re-read unchanged full section. Changed train-only learning-rate argument does not alter this claim; its registered original proof SHA is unchanged.")})
result = {
    "reviewer_task": "/root/natural_factual_20_2", "reviewed_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"python": platform.python_version(), "device": "cpu, local file/AST/header audit only"},
    "source_sha256": section_sha,
    "runtime_change": {"prior_sha256": old_source["sha256"], "current_sha256": current_sha,
        "method": "Recovered the prior source by reversing one exact added argument line, then authenticated all reconstructed raw bytes against the prior report SHA256; no Git operation. Saved both complete byte versions and exact unified diff.",
        "changed_line": train_branch.lineno + 1, "old": previous.decode(), "new": added.decode(),
        "runtime_image_block_sha256": hashlib.sha256(runtime_block.encode()).hexdigest(),
        "impact": "Only train argument assembly adds explicit0.00003. Baseline/ASR/model pins/Dense structure/LoRA target/counts/runtime versions are unchanged."},
    "registered_repository_sources": source_status,
    "reused_registered_artifacts": unchanged,
    "model_pins": constants, "recomputed_header_parameters": counts, "combined_parameters": sum(counts.values()),
    "baseline_rechecked": {key: baseline[key] for key in ("status", "model_revision", "asr_revision", "gpu_name", "device", "total_parameters", "trainable_parameters", "versions")},
    "historical_train_scope": {"recorded_learning_rate": train["learning_rate"], "completed_steps": train["completed_steps"],
        "trainable_parameters": train["trainable_parameters"], "scope": "This unchanged historical training record uses0.0001. Current explicit0.00003 recipe is independently inspected, not a new completed GPU training result or quality claim."},
    "gentle_primary_evidence": gentle, "claim_reverification": impact,
    "verdict": "pass", "scope": "Full20.2 re-read and registered proof rehash; no GPU, model inference/generation, Git or textbook change. Existing SDK4/offline26/selection17 records are audited at the current script SHA; they establish their recorded offline scope, not new model quality or remote execution.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
