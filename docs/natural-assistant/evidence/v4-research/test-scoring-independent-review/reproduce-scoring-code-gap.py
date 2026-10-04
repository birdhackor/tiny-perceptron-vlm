"""Synthetic audit-gap reproduction; never loads a model or real generations."""

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts import score_natural_v4_test as scoring


def main():
    spec = importlib.util.spec_from_file_location(
        "original_author_test_fixtures", ROOT / "tests/test_natural_v4_test_scoring.py"
    )
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    output = Path(__file__).parent
    with tempfile.TemporaryDirectory(prefix="synthetic-only-", dir=output) as temporary:
        fixture = fixture_module.artifacts.__wrapped__(Path(temporary), SimpleNamespace(param="base"))
        fixture_module.mutate(
            fixture,
            "generations-base.json",
            lambda rows: next(row for row in rows if row["task"] == "ocr").update(prediction="ab文字"),
        )
        loaded, blind, grades = fixture_module.grade(fixture)
        before = scoring.score_blind(loaded, blind, grades)
        script = fixture.root / "scripts/score_natural_v4_test.py"
        old = script.read_text()
        needle = 'return "".join(text.split()) if strip_whitespace else text'
        assert old.count(needle) == 1
        script.write_text(old.replace(needle, 'return ("".join(text.split()) if strip_whitespace else text).casefold()'))
        command = [
            sys.executable, str(script), "score", "--manifest", str(fixture.manifest),
            "--protocol", str(fixture.protocol), "--data-root", str(fixture.data),
            "--test-dir", str(fixture.run), "--selection", str(fixture.selection),
            "--blind-dir", str(blind), "--grades", str(grades), "--output", str(fixture.tmp / "changed-scored"),
        ]
        run = subprocess.run(command, cwd=fixture.root, capture_output=True, text=True)
        if "--expect-rejected" in sys.argv:
            assert run.returncode != 0
            assert "Test artifacts or blind packet changed after export" in run.stderr
            assert not (fixture.tmp / "changed-scored").exists()
            print(json.dumps({
                "scope": "Constructed fixture only; no model inference, real grades or real final-test scores",
                "original_scorer_sha256": scoring.core.sha256(ROOT / "scripts/score_natural_v4_test.py"),
                "author_test_source_sha256": scoring.core.sha256(ROOT / "tests/test_natural_v4_test_scoring.py"),
                "copied_scorer_source_changed_after_export": old != script.read_text(),
                "changed_cli_returncode": run.returncode,
                "expected_rejection": "Test artifacts or blind packet changed after export",
                "new_scores_directory_created": False,
                "original_single_ocr_correct": before["correct_counts"]["single_ocr"],
                "bound_scoring_sources": before["artifact_binding"]["scoring_sources"],
            }, indent=2))
            return
        assert run.returncode == 0, run.stderr
        after = json.loads((fixture.tmp / "changed-scored/scores.json").read_text())
        result = {
            "scope": "Constructed fixture only; no model inference, real grades or real final-test scores",
            "original_scorer_sha256": scoring.core.sha256(ROOT / "scripts/score_natural_v4_test.py"),
            "author_test_source_sha256": scoring.core.sha256(ROOT / "tests/test_natural_v4_test_scoring.py"),
            "copied_scorer_source_changed_after_export": old != script.read_text(),
            "copied_source_change": "normalized now casefolds, contrary to frozen preserve-case normalization",
            "changed_cli_returncode": run.returncode,
            "unchanged_artifact_binding": before["artifact_binding"] == after["artifact_binding"],
            "unchanged_blind_packet_sha256": before["blind_packet_sha256"] == after["blind_packet_sha256"],
            "before_single_ocr_correct": before["correct_counts"]["single_ocr"],
            "after_single_ocr_correct": after["correct_counts"]["single_ocr"],
            "test_denominator": after["denominators"]["single_ocr"],
        }
        assert result["before_single_ocr_correct"] == 9
        assert result["after_single_ocr_correct"] == 10
        assert result["unchanged_artifact_binding"]
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
