"""Acquire official text sources, freeze own review inputs, run original fence once."""
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

inputs = [
    "docs/review-tools/factual-reviewer-instructions.md",
    "docs/review-tools/section_facts.py", "scripts/check_technical_reviews.py",
    ".agents/skills/clear-tutorial/references/review-protocol.md",
    "tiny_perceptron/data.py", "tiny_perceptron/model.py",
    "scripts/infer.py", "scripts/evaluate.py", "scripts/train.py",
    "scripts/build_course.py", "pyproject.toml",
]
for name in inputs:
    dest = BASE / "inputs" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, dest)
sys.path.insert(0, str(ROOT / "docs/review-tools"))
from section_facts import original_section
for name, lesson in [("07", "7.1"), ("06", "6.6")]:
    raw, whole, start = original_section(ROOT / f"course/chapters/{name}.md", lesson)
    (BASE / "inputs" / f"prerequisite-{lesson}.md").write_bytes(raw)

urls = {
    "hf-chat-v4.57.1.md": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/chat_templating.md",
    "hf-generation-v4.57.1.py": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/utils.py",
    "python-3.13-stdtypes.html": "https://docs.python.org/3.13/library/stdtypes.html",
    "pytorch-sparse-v2.9.0.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/modules/sparse.py",
}
(BASE / "sources").mkdir(exist_ok=True)
def fetch(pair):
    name, url = pair
    request = urllib.request.Request(url, headers={"User-Agent": "independent-course-factual-review/1"})
    with urllib.request.urlopen(request, timeout=25) as response:
        raw = response.read()
        result = {"url": url, "final_url": response.url, "status": response.status,
                  "accessed_at": datetime.now(timezone.utc).isoformat(), "response_bytes": len(raw)}
    path = BASE / "sources" / name
    path.write_bytes(raw)
    result.update(path=str(path.relative_to(ROOT)), sha256=sha(path))
    return name, result
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    receipts = dict(pool.map(fetch, urls.items()))
write(BASE / "source-acquisition.json", receipts)

out = ROOT / "outputs/reviewer-tools/runs/phase4-7_2-independent-original"
argv = [str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/07.md#7.2",
        "--output", str(out), "--execute", "--timeout", "45"]
proc = subprocess.run(argv, cwd=ROOT, capture_output=True, timeout=55)
(BASE / "original-launch.stdout.txt").write_bytes(proc.stdout)
(BASE / "original-launch.stderr.txt").write_bytes(proc.stderr)
for path in out.iterdir():
    if path.is_file():
        shutil.copyfile(path, BASE / ("original-" + path.name))
write(BASE / "original-launch.json", {"command_argv": argv, "cwd": str(ROOT), "exit_code": proc.returncode,
      "timeout_seconds": 55, "source_sha256": json.loads((out / "extraction.json").read_text())["source_sha256"]})
write(BASE / "input-provenance.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
      "snapshots": [{"original_path": name, "snapshot": str((BASE / "inputs" / name).relative_to(ROOT)),
                     "sha256": sha(BASE / "inputs" / name)} for name in inputs],
      "reader_reports_opened": False, "previous_technical_reports_opened": False,
      "chapter_intro": {"status": "not_applicable", "reason": "7.2 is not the chapter's first numbered section"},
      "existing_empirical_results": {"status": "not_applicable", "reason": "7.2 contains no historical model metric or result claim"}})
print(json.dumps({"original_exit": proc.returncode, "source_sha256": json.loads((out / "extraction.json").read_text())["source_sha256"], "official_sources": len(receipts)}))
