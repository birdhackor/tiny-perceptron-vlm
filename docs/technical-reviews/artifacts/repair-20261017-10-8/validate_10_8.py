"""Call the existing _validate on only the independently authored 10.8 report."""

import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import torch


root = Path(__file__).resolve().parents[4]
checker = root / "scripts/check_technical_reviews.py"
spec = importlib.util.spec_from_file_location("technical_checker_10_8", checker)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source = root / "course/chapters/10.md"
body = dict(module.sections(source))["10.8"]
report_path = root / "docs/course-repair-20261017/reviews/technical-10-8-original-01.json"
report = json.loads(report_path.read_text())
owners = {"/root/repair_tech_10_8": ["10.8"]}
readers = ["/root/astra_repair_01_readability"]
errors = module._validate(
    root, source, "10.8", body, report, reviewer_owners=owners, reader_tasks=readers
)
print(json.dumps({
    "command": "uv run --frozen --extra cpu --no-sync python docs/technical-reviews/artifacts/repair-20261017-10-8/validate_10_8.py > docs/technical-reviews/artifacts/repair-20261017-10-8/validation-final.json",
    "checker_sha256": hashlib.sha256(checker.read_bytes()).hexdigest(),
    "report": report_path.relative_to(root).as_posix(),
    "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "reviewer_owners": owners,
    "reader_tasks": readers,
    "errors": errors,
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "scope": "Format, source/evidence hash, and declared identity checks only; this does not automatically prove technical correctness.",
}, ensure_ascii=False, indent=2))
if errors:
    raise SystemExit(1)
