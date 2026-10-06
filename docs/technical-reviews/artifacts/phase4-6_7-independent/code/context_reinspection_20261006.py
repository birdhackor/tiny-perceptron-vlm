"""Snapshot and check the original owner's narrow prerequisite reinspection."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
OUT = BASE / "context-reinspection-20261006"
OUT.mkdir(exist_ok=True)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

prior_path = ROOT / "docs/technical-reviews/history/phase4-6_7-own-prior-context-reinspection-db63e4a38a78141c5e2ec06950925322c78c72a4821ce43d9bfef5d97db334af.json"
prior_bytes = prior_path.read_bytes()
assert sha(prior_bytes) == "db63e4a38a78141c5e2ec06950925322c78c72a4821ce43d9bfef5d97db334af"
# This is this reviewer's own report, not another owner's report.
prior = json.loads(prior_bytes)
verified_prior_artifacts = []
for artifact in prior["artifacts"]:
    p = ROOT / artifact["path"]
    actual = sha(p.read_bytes())
    assert actual == artifact["sha256"], artifact["id"]
    verified_prior_artifacts.append({"id": artifact["id"], "path": artifact["path"],
                                     "expected_sha256": artifact["sha256"], "actual_sha256": actual})

raw = (ROOT / "course/chapters/06.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
sections = []
for lesson in ("6.3", "6.5", "6.6", "6.7"):
    index = next(i for i, h in enumerate(headers) if h[0].startswith(("## " + lesson + " ").encode()))
    body = raw[headers[index].start():headers[index+1].start()]
    old_path = BASE / ("original/section.md" if lesson == "6.7" else f"inputs/prerequisite-{lesson}.md")
    old = old_path.read_bytes()
    p = OUT / f"current-{lesson}.md"
    p.write_bytes(body)
    diff = "".join(difflib.unified_diff(old.decode().splitlines(keepends=True),
                                      body.decode().splitlines(keepends=True),
                                      fromfile=f"own-frozen-{lesson}", tofile=f"current-{lesson}"))
    (OUT / f"diff-{lesson}.txt").write_text(diff, encoding="utf-8")
    sections.append({"lesson_id": lesson, "current_path": p.relative_to(ROOT).as_posix(),
                     "current_sha256": sha(body), "own_frozen_path": old_path.relative_to(ROOT).as_posix(),
                     "own_frozen_sha256": sha(old), "byte_identical": old == body,
                     "first_source_line": raw[:headers[index].start()].count(b"\n")+1,
                     "image_references": re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body.decode())})

live_inputs = []
for name in ("tiny_perceptron/data.py", "tiny_perceptron/tokenization.py", "scripts/course_experiments/text.py",
             "docs/course-experiments/results/tokenizer.json"):
    current = (ROOT / name).read_bytes()
    old = (BASE / "inputs/current" / name).read_bytes()
    assert current == old, name
    live_inputs.append({"path": name, "current_sha256": sha(current), "own_frozen_sha256": sha(old),
                        "byte_identical": True, "inspection": "Fingerprints only in this reinspection; prior personal contract/measurement checks retained."})

method = ROOT / "docs/review-tools/factual-reviewer-instructions.md"
(OUT / "factual-reviewer-instructions.md").write_bytes(method.read_bytes())
figures = []
for name in ("rewrite-06-05-shared-denominator.svg", "tokenizer_common_scale.svg"):
    p = ROOT / "course/figures" / name
    snapshot = OUT / name
    snapshot.write_bytes(p.read_bytes())
    png = OUT / (p.stem + ".png")
    command = ["inkscape", str(snapshot), "--export-type=png", "--export-width=1400", "--export-filename=" + str(png)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (OUT / (p.stem + "-render-stdout.txt")).write_text(result.stdout)
    (OUT / (p.stem + "-render-stderr.txt")).write_text(result.stderr)
    assert result.returncode == 0 and png.is_file(), name
    figures.append({"source_path": p.relative_to(ROOT).as_posix(), "source_sha256": sha(p.read_bytes()),
                    "snapshot_path": snapshot.relative_to(ROOT).as_posix(), "snapshot_sha256": sha(snapshot.read_bytes()),
                    "render_path": png.relative_to(ROOT).as_posix(), "render_sha256": sha(png.read_bytes()),
                    "command_argv": command, "exit_code": result.returncode,
                    "status": "rendered; pending owner's actual view_image inspection"})

receipt = {"kind": "same_owner_context_reinspection_facts", "owner": prior["reviewer_task"],
           "at": datetime.now(timezone.utc).isoformat(), "python": sys.version, "cwd": str(ROOT),
           "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-6_7-independent/code/context_reinspection_20261006.py",
           "priorhistory_path": prior_path.relative_to(ROOT).as_posix(), "priorhistory_sha256": sha(prior_bytes),
           "sections": sections, "live_inputs": live_inputs, "verified_prior_artifacts": verified_prior_artifacts,
           "figures": figures, "new_cpu_tokenizer_or_model_runs": 0,
           "external_source_fetches": 0, "gpu_or_training_runs": 0}
(OUT / "facts.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"sections": sections, "live_inputs": live_inputs, "figures": figures,
                  "verified_prior_artifact_count": len(verified_prior_artifacts)}, ensure_ascii=False, indent=2))
