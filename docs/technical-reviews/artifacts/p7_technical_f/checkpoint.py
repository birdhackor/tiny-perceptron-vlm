"""Write only this owner's supplied checkpoint, then unlock one current unit."""
import json
import subprocess
import sys
from pathlib import Path

SESSION = "6f2c72d53cc64e9d8396b432fb621e7d"
ROOT = Path(__file__).resolve().parents[4]
checkpoint = json.load(sys.stdin)
for attached in checkpoint.pop("attach_figures", []):
    receipt_path = attached["receipt"]
    receipt = json.loads((ROOT / receipt_path).read_text())
    checkpoint.setdefault("visual_checks", []).append({"figure": attached["figure"], "source_sha256": receipt["source_sha256"], "status": "verified", "details": attached["details"], "observation": receipt["observation"], "receipt": {"path": receipt_path, "sha256": __import__("hashlib").sha256((ROOT / receipt_path).read_bytes()).hexdigest()}, "artifacts": receipt["artifacts"]})
folder = ROOT / "outputs/reader-checkpoints/p7_technical_f"
folder.mkdir(parents=True, exist_ok=True)
target = folder / (checkpoint["page_id"] + "-" + str(checkpoint["unit_index"]) + ".json")
if target.exists():
    raise SystemExit("Refusing to overwrite checkpoint")
target.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n")
result = subprocess.run([str(ROOT / ".venv/bin/python"), "docs/review-tools/phase7_review.py", "next", SESSION, "--checkpoint", str(target)], cwd=ROOT, capture_output=True, text=True)
print(result.stdout, end="")
print(result.stderr, end="", file=sys.stderr)
sys.exit(result.returncode)
