"""Necessary-context callback only: explicit scene arguments versus defaults."""
import ast
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CALLBACK = Path(__file__).resolve().parent
PREVIOUS = CALLBACK.parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.multimodal import patchify, scene

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

prior = json.loads((CALLBACK / "prior-report.opaque.json").read_bytes())
assert prior["reviewer_task"] == "/root/phase4_factual_coordinator/factual_10_3"
spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
section_facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(section_facts)
primary, _, primary_line = section_facts.original_section(ROOT / "course/chapters/10.md", "10.3")
context, _, context_line = section_facts.original_section(ROOT / "course/chapters/10.md", "10.2")
assert hashlib.sha256(primary).hexdigest() == prior["source_sha256"]
assert primary == (PREVIOUS / "inputs/section.md").read_bytes()
assert context == (CALLBACK / "inputs/10.2.md").read_bytes()
primary_fences = section_facts.fences(primary, primary_line)
context_fences = section_facts.fences(context, context_line)
assert len(primary_fences) == len(context_fences) == 1
assert primary_fences[0]["raw"] == (PREVIOUS / "inputs/fence-1.py").read_bytes()
(CALLBACK / "inputs/context-fence.py").write_bytes(context_fences[0]["raw"])

# Exact hashes permit reusing the original owner's inspected, immutable authority
# snapshots and bounded execution, without rereading judgments or repeating probes.
reused_artifacts = []
for artifact in prior["artifacts"]:
    observed = digest(ROOT / artifact["path"])
    assert observed == artifact["sha256"], artifact["id"]
    reused_artifacts.append({"id": artifact["id"], "path": artifact["path"], "sha256": observed})
reused_repository_code = []
for source in prior["sources"]:
    if source["kind"] == "repository_code":
        observed = digest(ROOT / source["path"])
        assert observed == source["sha256"], source["id"]
        reused_repository_code.append({"source_id": source["id"], "path": source["path"], "sha256": observed})
for path, expected in prior["figure_sha256"].items():
    assert digest(ROOT / path) == expected

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
environment = {
    "python": platform.python_version(), "python_executable": sys.executable,
    "torch": torch.__version__, "torch_git_version": torch.version.git_version,
    "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
    "device": "cpu", "threads": torch.get_num_threads(), "cwd": str(Path.cwd()),
    "offline": {key: os.environ.get(key, "unset") for key in ["CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"]},
}
old_environment = json.loads((PREVIOUS / "execution/probe.stdout.json").read_bytes())["environment"]
assert environment["python"] == old_environment["python"]
assert environment["torch"] == old_environment["torch"]
assert environment["torch_git_version"] == old_environment["torch_git_version"]

default_image = scene()[None]
explicit_image = scene("red", "square")[None]
assert torch.equal(default_image, explicit_image)
assert list(explicit_image.shape) == [1, 3, 16, 16]
assert torch.equal(explicit_image[0, :, 0, 0], torch.zeros(3))
assert torch.equal(explicit_image[0, :, 8, 8], torch.tensor([1.0, 0.0, 0.0]))
assert torch.equal(patchify(default_image, 4), patchify(explicit_image, 4))
assert list(patchify(explicit_image, 4).shape) == [1, 16, 48]

# Execute the changed necessary-context fence verbatim, after personally reading it.
print("Exact current 10.2 fence stdout:")
namespace = {"__name__": "__main__"}
exec(compile(context_fences[0]["raw"], "course/chapters/10.md#10.2:current-original-fence", "exec"), namespace)
assert torch.equal(namespace["image"], explicit_image)
assert torch.equal(namespace["patches"], patchify(default_image, 4))
assert torch.equal(namespace["image"], namespace["restored"])

result = {
    "reviewer_task": prior["reviewer_task"], "environment": environment,
    "primary": {"source": "course/chapters/10.md#10.3", "sha256": hashlib.sha256(primary).hexdigest(),
                "first_line": primary_line, "same_original_bytes": True,
                "same_original_fence_bytes": True, "fence_sha256": hashlib.sha256(primary_fences[0]["raw"]).hexdigest()},
    "necessary_context": {"source": "course/chapters/10.md#10.2", "sha256": hashlib.sha256(context).hexdigest(),
                          "first_line": context_line, "fence_sha256": hashlib.sha256(context_fences[0]["raw"]).hexdigest()},
    "context_transition": {"default_scene_equals_explicit_red_square": True,
                           "all_pixel_values_equal": True, "all_patch_values_equal": True,
                           "image_shape": [1, 3, 16, 16], "patch_shape": [1, 16, 48],
                           "black_background_corner_rgb": [0, 0, 0], "red_square_center_rgb": [1, 0, 0],
                           "current_context_roundtrip_exact": True},
    "reuse": {"original_artifacts_checked": reused_artifacts, "repository_sources_checked": reused_repository_code,
              "scope": "Own 2026-10-05 original primary execution, axis/affine/seed/lookup/collision/gradient checks and authority inspection retained after exact hash checks; not rerun today."},
    "new_execution_scope": "Explicit/default scene equality and changed current 10.2 fence only; no parameter update, training, metrics, downloaded material or weights persistence.",
    "assertions": "all passed",
}
(CALLBACK / "verification-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"assertions": result["assertions"], "primary_sha256": result["primary"]["sha256"],
                  "context_sha256": result["necessary_context"]["sha256"],
                  "context_transition": result["context_transition"], "retained_artifacts": len(reused_artifacts),
                  "environment": environment}, ensure_ascii=False, indent=2))
