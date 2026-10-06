"""Same-owner narrow context verification; hash checks and rendering only, no model run."""
from pathlib import Path
import difflib
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parents[1]
OLD = ROOT / "docs/technical-reviews/artifacts/phase4-10_8-independent"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def section(path, identifier):
    raw = (ROOT / path).read_bytes()
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, h in enumerate(headers) if h[0].startswith(("## " + identifier + " ").encode()))
    start = headers[index].start()
    end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
    return raw[start:end], raw[:start].count(b"\n") + 1

prior_path = ART / "prior-opaque/10.8-prior-report.opaque.json"
prior_raw = prior_path.read_bytes()
assert digest(prior_raw) == "3c9cb644d4eacf7e389a6e5b29d0847a585760aa900df139502c939a3195ad5f"
prior = json.loads(prior_raw)
primary, first_primary_line = section("course/chapters/10.md", "10.8")
assert digest(primary) == prior["source_sha256"] == "d1148ca5e9fc08f72f8ee10d06081a2db1386336cdbab0d3faffdf8268cf5d12"
assert primary == (OLD / "inputs/10.8.md").read_bytes()
context, first_context_line = section("course/chapters/16.md", "16.8")
old_context = (OLD / "inputs/16.8.md").read_bytes()
# The actual current change is one orthographic replacement, not a guessed conclusion.
assert old_context.count("两条".encode()) == 0
assert old_context.count("兩条".encode()) == 1
assert old_context.replace("兩条".encode(), "兩條".encode()) == context
diff = "".join(difflib.unified_diff(old_context.decode().splitlines(True), context.decode().splitlines(True),
                                    fromfile="own-frozen-16.8", tofile="current-16.8"))
(ART / "context.diff.txt").write_text(diff)

# Retain the exact current method slice personally read, before empirical details.
method_end = context.index(b"<details>")
method = context[:method_end]
(ART / "inputs/16.8-current-method-slice.md").write_bytes(method)
for path in ("docs/review-tools/factual-reviewer-instructions.md",
             ".agents/skills/clear-tutorial/references/review-protocol.md"):
    dest = ART / "inputs" / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes((ROOT / path).read_bytes())

source_checks = []
for source in prior["sources"]:
    if source["kind"] == "repository_code":
        current = digest((ROOT / source["path"]).read_bytes())
        assert current == source["sha256"], source["path"]
        source_checks.append({"path": source["path"], "expected_sha256": source["sha256"],
                              "current_sha256": current, "match": True})
artifact_checks = []
for item in prior["artifacts"]:
    current = digest((ROOT / item["path"]).read_bytes())
    assert current == item["sha256"], item["path"]
    artifact_checks.append({"id": item["id"], "path": item["path"],
                            "expected_sha256": item["sha256"], "current_sha256": current, "match": True})

prior_variants = json.loads((OLD / "variants.json").read_bytes())
environment = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
               "torch_git_version": str(torch.version.git_version), "cuda_build": str(torch.version.cuda),
               "cuda_available": str(torch.cuda.is_available()), "device": "cpu",
               "platform": platform.platform(), "operation": "Hashes, context bytes comparison, and SVG rendering; no model forward or optimizer."}
assert torch.version.cuda is None and not torch.cuda.is_available()
assert environment["torch"] == prior_variants["environment"]["torch"]
assert environment["torch_git_version"] == prior_variants["environment"]["torch_git_version"]

figure_rel = "course/figures/rewrite-16-causal-mask.svg"
figure = (ROOT / figure_rel).read_bytes()
figure_dest = ART / "inputs/rewrite-16-causal-mask.svg"
figure_dest.write_bytes(figure)
render_dest = ART / "context-causal-mask.png"
command = ["/usr/bin/inkscape", str(figure_dest), "--export-type=png", "--export-width=640",
           "--export-filename=" + str(render_dest)]
render = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60, check=False)
(ART / "render.stdout.txt").write_bytes(render.stdout)
(ART / "render.stderr.txt").write_bytes(render.stderr)
assert render.returncode == 0 and render_dest.is_file()
version = subprocess.check_output(["/usr/bin/inkscape", "--version"], text=True).strip()

result = {
    "reviewer_task": "/root/phase4_factual_coordinator/factual_10_8",
    "callback_date": "2026-10-06", "primary_source": "course/chapters/10.md#10.8",
    "primary_source_sha256": digest(primary), "primary_first_line": first_primary_line,
    "prior_opaque_file": prior_path.relative_to(ROOT).as_posix(), "prior_opaque_sha256": digest(prior_raw),
    "preserved_original_read_scope": prior["read_scope"],
    "current_context_source": "course/chapters/16.md#16.8", "context_first_line": first_context_line,
    "old_frozen_context_sha256": digest(old_context), "current_context_sha256": digest(context),
    "current_context_method_slice": {"relative_lines": [1, method.count(b"\n")],
                                      "sha256": digest(method), "end_before": "<details> empirical supplementary block"},
    "actual_content_change": "Exactly one character replacement: 兩条路 → 兩條路. All mathematical/API/shape/mask/dropout/tolerance statements and empirical prose are byte-identical after that replacement.",
    "source_code_hash_checks": source_checks, "prior_artifact_hash_checks": artifact_checks,
    "environment": environment,
    "figure": {"source": figure_rel, "sha256": digest(figure),
               "snapshot": figure_dest.relative_to(ROOT).as_posix(), "render": render_dest.relative_to(ROOT).as_posix(),
               "render_sha256": digest(render_dest.read_bytes()), "renderer_version": version,
               "command": shlex.join(command), "exit_code": render.returncode,
               "visual_inspection": "Pending owner view_image; do not infer a visual verdict from this render command."},
    "reuse": {"cpu_proof_reexecuted": False, "official_sources_refetched": False,
              "reason": "10.8 and its five actual repository-code dependencies are unchanged; all 21 prior artifacts retain exact hashes and the same CPU wheel is installed. The context edit adds no substantive knowledge.",
              "scope": "Prior 4 forward pairs/3 missing-marker cases/17 gradient pairs still support only the original single unbatched text, same TinyLM CPU float32 path; no training or quality/skill-retention claim."},
    "context_role": "16.8 was personally read as forward-comparison background, not a necessary primary authority or empirical proof for 10.8. Its math/SDPA comparison cautions do not replace direct same-object path proof. Preserve the original full-read declaration without expanding this callback to verifying its L4/Flash/GPU scores.",
    "original_result_json_pointers_read": [],
}
(ART / "callback-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"primary_sha256": result["primary_source_sha256"],
                  "old_context_sha256": result["old_frozen_context_sha256"],
                  "current_context_sha256": result["current_context_sha256"],
                  "actual_content_change": result["actual_content_change"],
                  "unchanged_repository_sources": len(source_checks), "unchanged_prior_artifacts": len(artifact_checks),
                  "renderer_version": version, "render_exit": render.returncode, "new_model_runs": 0}, ensure_ascii=False))
