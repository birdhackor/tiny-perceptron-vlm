"""Informed rerun of immutable original probes; only fixture SDK/schema expectations change."""
import hashlib
from pathlib import Path

original = Path(__file__).with_name("engineering-publication-probes-original.py")
assert hashlib.sha256(original.read_bytes()).hexdigest() == "5d178cd120e8fbbb256bf0e6d6b8bd2f69ac38ca142aedd2f32ef7065839486d"
source = original.read_text()

def replace(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)

replace('spec["transport_sha256"] = sha(archive)', 'spec["transport_sha256"] = sha(archive)\n                spec.update(release_id=456, release_commit="c" * 40, asset_name="fixture.tar.gz", asset_bytes=archive.stat().st_size)')
replace('if kind in ("existing", "existing-mismatch"):', 'if kind in ("existing", "existing-unrelated", "existing-extra", "existing-mismatch"):')
replace('storage_before = storage.copy()', 'if kind == "existing-unrelated":\n                storage["another-prefix/unrelated.txt"] = b"unrelated"\n            if kind == "existing-extra":\n                storage[prefix + "unlisted.txt"] = b"must retain extra prior bytes"\n            storage_before = storage.copy()')
replace('def file_exists(self, repo, name):\n                    return name in storage', 'def list_repo_files(self, repo, revision):\n                    assert revision == "a" * 40\n                    events.append("list-at-pinned-info-revision")\n                    return list(storage)')
replace('book = json.loads(budget.read_bytes())', 'execution = json.loads((case / "course/lesson-products-publications/audit/execution.json").read_bytes())\n            assert execution == receipt\n            if kind == "token-error":\n                assert token_marker not in json.dumps(execution)\n                assert "[REDACTED]" in execution["error"]\n            book = json.loads(budget.read_bytes())')
replace('if kind == "partial-prefix":\n                result.update', 'if kind in ("partial-prefix", "existing-extra"):\n                result.update')
replace('kind in ("new", "existing") else "failed"', 'kind in ("new", "existing", "existing-unrelated") else "failed"')
replace('for kind in ("new", "existing", "existing-mismatch", "private", "partial-prefix", "token-error"):', 'for kind in ("new", "existing", "existing-unrelated", "existing-extra", "existing-mismatch", "private", "partial-prefix", "token-error"):')
replace('assert observed == ["necessary_model_must_be_present_in_settings_freeze", "remote_partial-prefix", "remote_token-error"]', 'assert observed == []')
replace('"initial_context": "fresh"', '"initial_context": "fresh_original/current_informed"')
replace('"verdict": "revise"', '"verdict": "pass"')
replace('No genuine production spec exists; source/draft-release provenance not exercised here.', 'No genuine production spec exists; separate callback client/inputs probes exercise mocked source/draft metadata.')
replace('failed counterexamples are preserved as failures of contract, not claimed passes.', 'all three original counterexamples rerun from passing bases now reject/no-overwrite/redact as expected.')
replace('engineering-publication-probes-results-original.json', 'engineering-publication-probes-results-callback-01.json')
exec(compile(source, str(__file__), "exec"), globals())
