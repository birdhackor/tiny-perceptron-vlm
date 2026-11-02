"""Exercise the new preview route and publication guards without publishing."""

from contextlib import redirect_stdout
from datetime import UTC, datetime
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import check_site, export_course

SITE = ROOT / "outputs/omni-review-site-technical-fix13"
INDEX = json.loads((ROOT / "course/lesson-index.json").read_text())
PIN = "80c5db17e8b077b2c0a5e4e9baa48e760ccb0532"
cases = []


def run_case(name, action, expected_error=None, scope=""):
    output = StringIO()
    error = None
    with redirect_stdout(output):
        try:
            action()
        except ValueError as exc:
            error = str(exc)
    if expected_error is None:
        assert error is None, (name, error)
    else:
        assert error is not None and expected_error in error, (name, error)
    cases.append({"name": name, "expected_error_substring": expected_error, "observed_error": error, "stdout": output.getvalue(), "passed": True, "scope": scope})


# The whole current preview is checked using the unmodified actual CLI.
command = ["uv", "run", "--extra", "cpu", "python", "scripts/check_site.py", "--site", str(SITE.relative_to(ROOT))]
checked = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
assert checked.returncode == 0
actual_site_check = {"command": command, "exit_code": checked.returncode, "stdout": checked.stdout, "stderr": checked.stderr}

for item in INDEX:
    notebook = json.loads((ROOT / item["notebook"]).read_text())
    footer = export_course.section_footer(item, notebook, PIN, True, direct_colab=False)
    assert 'href="https://colab.research.google.com/"' in footer
    assert '/github/' not in footer
    assert "尚未發布的預覽版" in footer and "Colab 直連尚未提供" in footer
    assert "Upload notebook" in footer and "File → Upload notebook" in footer
    assert item["notebook"] in footer
cases.append({"name": "actual_section_footer_upload_mode_all_331", "passed": True, "scope": "Calls the real footer function for every current Notebook; this is a preview, not publication."})

run_case("actual_full_331_old_commit_rejected", lambda: export_course.verify_colab_notebooks(INDEX, PIN), "Colab入口尚未對上本次Notebook", "Real git objects; the current full 331 set is not in the P1 base commit.")
run_case("symbolic_ref_rejected", lambda: export_course.verify_colab_notebooks(INDEX, "main"), "完整40位Git commit")
run_case("nonexistent_commit_rejected", lambda: export_course.verify_colab_notebooks(INDEX, "0" * 40), "不是已存在的Git commit")
blob = subprocess.run(["git", "rev-parse", PIN + ":notebooks/01/1.1.ipynb"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
run_case("real_blob_sha_is_not_commit", lambda: export_course.verify_colab_notebooks(INDEX[:1], blob), "不是已存在的Git commit")

# A real matching subset exercises successful genuine-commit checks. It is
# explicitly not a release of the full current 331-Notebook course.
prior = json.loads(Path(__file__).with_name("course-presence-callback-v2.json").read_text())["preview_colab_pin"]
bad = {r["page_id"] for r in prior["different_current_bytes"] + prior["absent_at_pin"]}
subset = [item for item in INDEX if item["id"] not in bad]
assert len(subset) == 296
run_case("actual_commit_296_matching_subset_accepted", lambda: export_course.verify_colab_notebooks(subset, PIN), scope="Real existing commit, 296 matching current files only; not a full-course release or publication.")


def protocol_fixture(drift=False):
    result = b"a" * 40 + b" commit 7\nfixture\n"
    for i, item in enumerate(INDEX):
        raw = (ROOT / item["notebook"]).read_bytes()
        if drift and i == 0:
            raw += b"\nfixture-drift"
        result += b"b" * 40 + b" blob " + str(len(raw)).encode() + b"\n" + raw + b"\n"
    return result


class FixtureProcess:
    returncode = 0

    def __init__(self, output):
        self.output = output

    def communicate(self, input):
        assert len(input.decode().splitlines()) == 332
        return self.output, b""


for drift in [False, True]:
    with patch.object(export_course.subprocess, "Popen", lambda *a, **kw: FixtureProcess(protocol_fixture(drift))):
        run_case("fixture_full_331_" + ("drift_rejected" if drift else "exact_bytes_accepted"), lambda: export_course.verify_colab_notebooks(INDEX, "a" * 40), "目前下載內容不同" if drift else None, "Simulated cat-file protocol only, not a real commit, publishable artifact, cloud run, or actual release.")

original_read = Path.read_text
info_path = (SITE / "build-info.json").resolve()
page_path = (SITE / "1.1.html").resolve()


def changed_info(path, *args, **kwargs):
    value = original_read(path, *args, **kwargs)
    if path.resolve() == info_path:
        data = json.loads(value)
        data["publication_ready"] = True
        return json.dumps(data)
    return value


with patch.object(Path, "read_text", changed_info):
    run_case("fixture_upload_mode_cannot_be_marked_publication_ready", lambda: check_site.check(SITE), "正式網站尚未核對Colab同版本Notebook", "Virtual read-only metadata fixture; no actual website files changed.")


def removed_instructions(path, *args, **kwargs):
    value = original_read(path, *args, **kwargs)
    if path.resolve() == page_path:
        return value.replace('id="colab-upload-instructions"', 'id="removed-in-test-fixture"')
    return value


with patch.object(Path, "read_text", removed_instructions):
    run_case("fixture_missing_upload_instruction_anchor_rejected", lambda: check_site.check(SITE), "Colab預覽缺上傳操作說明：1.1", "Virtual read-only HTML fixture; semantic instruction text was inspected separately for all 331 pages.")

# The actual formal CLI also refuses currently incomplete estimates. The missing
# three reader estimates are not filled with fabricated values to build a release.
destination = ROOT / "outputs/omni-technical-r-1-v3-negative-formal-export"
assert not destination.exists()
formal_command = ["uv", "run", "--extra", "cpu", "python", "scripts/export_course.py", "--output", str(destination.relative_to(ROOT)), "--executed", "outputs/omni-notebooks-technical-fix13-current", "--reading-times", "/workspace/work/omni-reading-times-preview-fix13.json", "--revision", PIN, "--require-reading-times"]
formal = subprocess.run(formal_command, cwd=ROOT, capture_output=True, text=True, timeout=60)
assert formal.returncode != 0 and not destination.exists()
assert any(x in formal.stderr for x in ["9.12", "9.15", "first-steps"])
result = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/verify_colab_delivery_callback_v3.py",
    "environment": {"python": sys.version, "device": "CPU; static/current CLI and bounded function cases; no model training or cloud runtime"},
    "implementation": [{"path": p, "sha256": sha256((ROOT / p).read_bytes()).hexdigest()} for p in ["scripts/export_course.py", "scripts/check_site.py"]],
    "actual_site_check": actual_site_check, "cases": cases,
    "actual_negative_formal_CLI": {"command": formal_command, "exit_code": formal.returncode, "stdout": formal.stdout, "stderr": formal.stderr, "output_directory_created": destination.exists(), "scope": "Actual CLI rejects incomplete current reader estimates before any publication; full 331 pin mismatch is independently tested by the real helper case."},
    "scope": "Actual preview and real committed-object behavior are distinguished from virtual negative/positive fixtures. No real full-331 matching release commit is claimed, no site deployment or commit was performed, no Colab login/cell/cloud execution occurred."}
with Path(__file__).with_name("colab-delivery-callback-v3.json").open("x") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"actual_site_check": actual_site_check, "cases_passed": len(cases), "actual_formal_exit": formal.returncode, "formal_output_created": destination.exists()}, ensure_ascii=False, indent=2))
