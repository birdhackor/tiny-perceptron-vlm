"""Revalidate fixed HTML and installed Jupyter shortcut originals reused by W.1."""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).resolve().parent
old = json.loads((A / "entry-and-interface-inspection.json").read_bytes())
schema = ROOT / old["jupyterlab_shortcut"]["installed_official_schema"]
actual_hash = hashlib.sha256(schema.read_bytes()).hexdigest()
assert actual_hash == old["jupyterlab_shortcut"]["sha256"]
data = json.loads(schema.read_bytes())
shortcuts = [r for r in data["jupyter.lab.shortcuts"] if r.get("command") == "notebook:run-cell-and-select-next"]
assert any("Shift Enter" in r.get("keys", []) for r in shortcuts)
website = Path(old["public_1_1_html"]["raw_path"])
website_hash = hashlib.sha256(website.read_bytes()).hexdigest()
assert website_hash == old["public_1_1_html"]["sha256"]
result = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_w_1", "lesson_id": "W.1", "installed_jupyterlab_schema": {"path": str(schema), "sha256": actual_hash, "shortcut": shortcuts}, "fixed_public_html_original": {"path": str(website), "sha256": website_hash, "same_original_bytes": True}, "scope": "Revalidate the inspected original versions used in W.1; no live website update, browser visual test, or cloud runtime claim."}
p = A / "fix11-installed-context-v1.json"
with p.open("x") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "passed": True}))
