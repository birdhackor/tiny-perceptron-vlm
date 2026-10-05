"""Save fresh section 4.6 inputs and run its original fences in a bounded CPU worker."""
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = ART / "original"
    original.mkdir(exist_ok=True)
    inputs = [
        "docs/review-tools/section_facts.py", "scripts/check_technical_reviews.py",
        "docs/review-tools/factual-reviewer-instructions.md",
        ".agents/skills/clear-tutorial/references/review-protocol.md",
        "tiny_perceptron/model.py", "tiny_perceptron/attention.py",
        "tiny_perceptron/modern.py", "scripts/build_course.py", "pyproject.toml",
    ]
    for name in inputs:
        target = original / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    tmp = Path(tempfile.mkdtemp(prefix="phase4-4_6-")) / "run"
    command = [str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py",
               "course/chapters/04.md#4.6", "--output", str(tmp), "--execute", "--timeout", "30"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=40, check=False)
    (ART / "prepare.stdout.txt").write_bytes(result.stdout)
    (ART / "prepare.stderr.txt").write_bytes(result.stderr)
    for path in tmp.iterdir():
        if path.is_file():
            shutil.copyfile(path, original / path.name)
    receipt = {
        "command_argv": command, "cwd": str(ROOT), "exit_code": result.returncode,
        "timeout_seconds": 40,
        "permanent_copy_policy": "All regular section/execution/code inputs retained; temp workspace symlinks are not evidence.",
        "source_inputs": [{"path": name, "sha256": sha(ROOT / name)} for name in inputs],
        "training_or_checkpoint_inputs": [],
    }
    (ART / "prepare-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(result.stdout.decode(), end="")
    print(result.stderr.decode(), end="")
    print("Original fence worker exit:", result.returncode)


if __name__ == "__main__":
    main()
