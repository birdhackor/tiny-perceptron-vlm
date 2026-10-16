"""Validate only this review; do not read peer reports."""

import hashlib
import importlib.util
import json
import platform
from pathlib import Path

root = Path(__file__).resolve().parents[4]
tool = root / "scripts/check_technical_reviews.py"
spec = importlib.util.spec_from_file_location("technical_checker", tool)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
path = root / "docs/course-repair-20261017/reviews/technical-5-10-original-01.json"
report = json.loads(path.read_text())
source = root / "course/chapters/05.md"
body = dict(module.sections(source))["5.10"]
owners = {"/root/repair_tech_5_10": ["5.10"]}
readers = ["/root/astra_repair_01_readability"]
errors = module._validate(root, source, "5.10", body, report, owners, readers)
print(json.dumps({
    "command": "uv run --frozen --extra cpu --no-sync python docs/technical-reviews/artifacts/repair-20261017-5-10/validate-report.py > docs/technical-reviews/artifacts/repair-20261017-5-10/validation-output.json",
    "reviewer_owners": owners,
    "reader_tasks": readers,
    "report_path": str(path.relative_to(root)),
    "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "checker_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
    "environment": {"python": platform.python_version(), "device": "cpu"},
    "errors": errors,
    "interpretation": "Format, artifact hashes, source hashes and declared task separation only; this is not automatic proof of technical correctness.",
}, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
