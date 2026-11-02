"""Exercise unchanged delivery code against the new source14 preview and metadata."""

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import export_course, reading_time

SITE = ROOT / "outputs/omni-review-site-technical-fix14"
INDEX = json.loads((ROOT / "course/lesson-index.json").read_text())
PIN = "80c5db17e8b077b2c0a5e4e9baa48e760ccb0532"
PREVIEW = ROOT / "outputs/omni-reading-times-technical-fix14-preview.json"
cases = []


def cli(label, arguments, expected_exit, expected_fragments=(), output=None):
    if output:
        assert not output.exists()
    process = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert (process.returncode == 0) == (expected_exit == 0)
    for text in expected_fragments:
        assert text in process.stdout + process.stderr
    if output:
        assert not output.exists()
    cases.append({"name": label, "command": arguments, "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr, "output_directory_created": output.exists() if output else None})


cli("actual_new_site_full_check", ["uv", "run", "--extra", "cpu", "python", "scripts/check_site.py", "--site", str(SITE.relative_to(ROOT))], 0, ["331 份下載與 Colab 入口"])
for item in INDEX:
    nb = json.loads((ROOT / item["notebook"]).read_text())
    footer = export_course.section_footer(item, nb, PIN, True, direct_colab=False)
    assert item["notebook"] in footer and 'href="https://colab.research.google.com/"' in footer and '/github/' not in footer
    assert all(s in footer for s in ["尚未發布的預覽版", "Colab 直連尚未提供", "Upload notebook", "File → Upload notebook"])
cases.append({"name": "current_section_footer_all331", "passed": True})
try:
    export_course.verify_colab_notebooks(INDEX, PIN)
except ValueError as exc:
    message = str(exc)
    assert "Colab入口尚未對上本次Notebook" in message
    assert message.count("固定版本與目前下載內容不同") == 18 and message.count("固定版本沒有此檔案") == 18
else:
    raise AssertionError("old full-course pin was accepted")
cases.append({"name": "actual_old_pin_full331_rejected", "observed_error": message, "different": 18, "missing": 18})
pin_audit = json.loads(Path(__file__).with_name("course-presence-callback-v4.json").read_text())["preview_colab_pin"]
same = {r["page_id"] for r in pin_audit["same_current_bytes"]}
subset = [item for item in INDEX if item["id"] in same]
assert len(subset) == 295
export_course.verify_colab_notebooks(subset, PIN)
cases.append({"name": "actual_commit295_matching_subset_accepted", "count": len(subset), "scope": "True commit and blobs, but only a subset; not an actual full331 release or publication."})

# Both actual CLI paths fail before mutation; no successful formal build is claimed.
for label, metadata, formal, fragments in [
    ("actual_default_stale_metadata_rejected", ROOT / "course/reading-times.json", False, ["12.2: 閱讀時間評審不是目前正文"]),
    ("actual_formal_preview_missing_two_rejected", PREVIEW, True, ["缺少 2 頁閱讀時間", "12.2", "first-steps"]),
]:
    destination = ROOT / "outputs" / ("omni-technical-r-1-v4-negative-" + label)
    command = ["uv", "run", "--extra", "cpu", "python", "scripts/export_course.py", "--output", str(destination.relative_to(ROOT)), "--executed", "outputs/omni-notebooks-technical-fix14-current", "--revision", PIN, "--reading-times", str(metadata)]
    if formal:
        command.append("--require-reading-times")
    cli(label, command, 1, fragments, destination)
inventory = reading_time.build_inventory(ROOT, INDEX, export_course.DOCUMENTS, export_course.home_introduction(INDEX, True))
assert len(reading_time.load_estimates(PREVIEW, inventory)) == 367
result = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/verify_colab_delivery_callback_v4.py",
    "environment": {"python": sys.version, "device": "CPU; bounded read-only CLI and real git-object checks; no training/cloud"},
    "implementation": [{"path": p, "sha256": sha256((ROOT / p).read_bytes()).hexdigest()} for p in ["scripts/export_course.py", "scripts/check_site.py", "scripts/reading_time.py"]],
    "cases": cases, "scope": "New site/source14 checked directly. Source13 positive/negative fixtures are retained only as historical evidence of identical software bytes, not rerun or publication. Current default and formal guards refuse stale/missing estimates. No output site was built by these negative cases, no commit/deployment/cloud runtime."}
with Path(__file__).with_name("colab-delivery-callback-v4.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2); handle.write("\n")
print(json.dumps({"cases": len(cases), "actual_site_check": cases[0], "subset": len(subset), "old_pin_different": 18, "old_pin_missing": 18, "negative_outputs_created": [c["output_directory_created"] for c in cases if c.get("output_directory_created") is not None]}, ensure_ascii=False, indent=2))
