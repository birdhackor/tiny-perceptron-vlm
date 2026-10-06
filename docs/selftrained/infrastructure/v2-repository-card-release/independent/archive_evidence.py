"""Copy review evidence byte-for-byte; never mutate implementation or commit."""
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/workspace/selftrained-v2')
FROZEN = Path('/tmp/p5-repository-card-release-candidate')
REVIEW = Path('/tmp/p5-repository-card-engineering-review')
ARCHIVE = ROOT / 'docs/selftrained/infrastructure/v2-repository-card-release'

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

files = [
    (FROZEN / 'candidate.patch', 'frozen/candidate.patch'),
    (FROZEN / 'proof.json', 'frozen/author-proof.json'),
    (FROZEN / 'test_repository_card_release.py', 'synthetic/test_repository_card_release.py'),
    (REVIEW / 'actualproof.json', 'independent/actualproof.json'),
    (REVIEW / 'candidate_plugin.py', 'independent/candidate_plugin.py'),
    (REVIEW / 'test_independent_extra.py', 'independent/test_independent_extra.py'),
    (REVIEW / 'verify_frozen.py', 'independent/verify_frozen.py'),
    (REVIEW / 'archive_evidence.py', 'independent/archive_evidence.py'),
    (REVIEW / 'baseline-pytest.txt', 'logs/baseline-pytest.txt'),
    (REVIEW / 'author-tests-rerun.txt', 'logs/author-tests-rerun.txt'),
    (REVIEW / 'candidate-existing-regression.txt', 'logs/candidate-existing-regression.txt'),
    (REVIEW / 'independent-extra-tests.txt', 'logs/independent-extra-tests-initial.txt'),
    (REVIEW / 'independent-extra-tests-final.txt', 'logs/independent-extra-tests-final.txt'),
    (REVIEW / 'hub-sdk-create-commit-source.txt', 'dependency/hub-sdk-create-commit-source.txt'),
    (REVIEW / 'hub-sdk-send-commit-source.txt', 'dependency/hub-sdk-send-commit-source.txt'),
]
assert digest((REVIEW / 'actualproof.json').read_bytes()) == 'cdc73ab4dca2d2f9c20f716104ad9f37437178a2d6337f5f87e847ecbade7621'
assert digest((FROZEN / 'candidate.patch').read_bytes()) == '10d06273609487db5c5d40e75affbdc5017f492e9dd77e81387de9e665b36ab4'
ARCHIVE.mkdir(parents=True, exist_ok=True)
entries = []
for original, relative in files:
    raw = original.read_bytes()
    copy = ARCHIVE / relative
    copy.parent.mkdir(parents=True, exist_ok=True)
    if copy.exists():
        assert copy.read_bytes() == raw, f'Refuse different existing evidence: {copy}'
    else:
        copy.write_bytes(raw)
    restored = copy.read_bytes()
    assert restored == raw and len(restored) == len(raw) and digest(restored) == digest(raw)
    entries.append({
        'original': str(original),
        'copy': copy.relative_to(ROOT).as_posix(),
        'bytes': len(raw),
        'original_sha256': digest(raw),
        'copy_sha256': digest(restored),
        'byte_equal': True,
    })
proof = json.loads((REVIEW / 'actualproof.json').read_text())
observations = {}
for path in ('scripts/selftrained/modal_runner.py', 'scripts/selftrained/hf_transport.py'):
    raw = (ROOT / path).read_bytes()
    expected = proof['frozen_input_bindings'][path]
    assert digest(raw) == expected['sha256'] and len(raw) == expected['bytes']
    observations[path] = {'bytes': len(raw), 'sha256': digest(raw), 'matches_frozen_candidate': True}
index = {
    'status': 'ARCHIVED_ENGINEERING_APPROVAL_AND_ROOT_APPLIED_CODE_BYTE_OBSERVATION',
    'archived_at': datetime.now(UTC).isoformat(),
    'frozen_patch_sha256': '10d06273609487db5c5d40e75affbdc5017f492e9dd77e81387de9e665b36ab4',
    'independent_original_proof_sha256': 'cdc73ab4dca2d2f9c20f716104ad9f37437178a2d6337f5f87e847ecbade7621',
    'original_to_copy': entries,
    'archived_artifact_count': len(entries),
    'root_applied_code_observation': {
        'git_head_at_archiving': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'actual_files': observations,
        'meaning': 'Read-only byte/hash observation after root reported applying the frozen patch. Does not imply this reviewer applied it or ran new paid/remote work.'
    },
    'test_evidence_scope': {
        'author_test_source_rerun': '40 passed in 1.02s; independent rerun log retained; author did not claim an original standalone pytest log.',
        'candidate_existing_infra_and_gross': '85 passed in 4.11s; frozen candidate runner/transport substituted without changing existing assertions; actual candidate reserve AST used.',
        'independent_extra': '7 passed in 0.32s; pinned SDK singleton CAS / no-op race / ambiguous response checked locally.',
        'baseline': '85 passed in 2.88s, before reviewing the frozen patch.',
        'initial_harness_failure': 'Initial independent SDK fixture lacked _upload_mode=regular; initial failure log retained. Fixture corrected, candidate unchanged; final 7 passed.',
        'root_applied_regression': 'Root separately reported its 85-test run in progress when requesting this archive; this archive makes no claim about that run outcome.'
    },
    'limits': [
        'Engineering approval only; no production financial readiness or actual HF publish readiness claim.',
        'Review validation used local CPU synthetic fixtures, with external socket connections blocked in candidate runs; no HF/Modal API, paid/GPU/model/checkpoint/test-gold operation.',
        'Archive does not invoke any external writes, apply patches, modify core/finance or create a commit.',
        'Original independent proof remains byte-for-byte as recorded before root applied the patch; later archiving is recorded only in this index and README.',
        'Archived test sources retain original execution paths and are evidence, not an automatic production execution workflow.'
    ],
}
(ARCHIVE / 'integrity-index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
(ARCHIVE / 'README.md').write_text('''# Repository-card release engineering evidence

This archive records independent engineering approval of frozen patch
`10d06273609487db5c5d40e75affbdc5017f492e9dd77e81387de9e665b36ab4`.
The patch changes only the existing `modal_runner.py` and `hf_transport.py`.

The frozen author's proof, synthetic test source, independent actual proof,
test adapter, extra checks, raw pytest logs and pinned Hub SDK source excerpts
are copied without modification. [integrity-index.json](integrity-index.json)
records each original path, copied path, byte count and both SHA-256 values.

The independent candidate runs passed 40 author-test cases, 85 existing
infrastructure/gross-quota cases and 7 additional cases. The extra tests use
the actual pinned Hub 1.33.0 SDK control flow for a singleton README CAS,
SDK no-op race rejection and an ambiguous response without retry. The initial
extra-test fixture failure and its final passing log are both retained;
the reviewer corrected the fixture's missing regular upload mode, leaving
the frozen candidate unchanged.

The original [independent proof](independent/actualproof.json) records the
review before root applied the patch and remains byte-for-byte unchanged.
During archiving, the reviewer observed actual runner SHA
`ee7676871d1c24962905660775b869f50ac5c83818980934cef920f5ba1c2947`
and transport SHA
`500417c5f922cd1ce7ae0e6add8a31e2bc34bb7434a30bd845be93b47e831600`,
both matching the frozen candidate. This is a code-byte observation only;
the separate root regression run was still in progress at the archive request.

Approval covers the committed-source gates, single authorized root README
operation, pinned-parent CAS, immutable anonymous byte verification,
accurate no-op status, and preservation of the existing batch/resources,
Secret, serialization and gross-quota controls. It does not establish actual
HF publication or production financial readiness. No remote, paid, GPU,
model/checkpoint or held-out-gold operation was performed for this review.
The archived test sources retain their original execution paths; this archive
does not introduce a production execution workflow.
''')
print(json.dumps({'archive': str(ARCHIVE), 'original_to_copy_count': len(entries), 'all_byte_equal': True, 'index_sha256': digest((ARCHIVE / 'integrity-index.json').read_bytes()), 'root_applied_sources_match_frozen': True}))
