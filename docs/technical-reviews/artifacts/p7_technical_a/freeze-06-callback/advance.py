"""Save this owner's manually written checkpoint and ask the reader CLI for next unit."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SESSION = "077d89a8d55544b08d4bbc341a6f44f3"
record = json.load(sys.stdin)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
dest = Path(__file__).parent / "checkpoints" / f"{stamp}-{record['page_id']}-{record['unit_index']}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
result = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "docs/review-tools/phase7_review.py"), "next", SESSION, "--checkpoint", str(dest)], cwd=ROOT)
sys.exit(result.returncode)
