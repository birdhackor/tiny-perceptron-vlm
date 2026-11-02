import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/workspace/work/omni-book-integrated-v1')
WORK = Path('/workspace/work/omni-p8-history-owner-mediated-20261101-v1/owner-probes-v1')
ART = ROOT / 'docs/technical-reviews/artifacts/omni-technical-review-tools-history-v2-v1'
OUT = ART / 'root-mediated-original-probes-20261101-v1.json'
TASK = '/root/omni_technical_coordinator/technical_review_tools_history_v2'
RELAY = '/root/omni_p8_technical_chapter14'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ref(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}

started = datetime.now(UTC).isoformat()
expected = {
    'docs/review-tools/historical_fix_rechecks_v2.py': 'b551d9ef6ca0e5f47cd4dd7b27c7dda0361b47d4c099775b6382088c543a04a3',
    'tests/test_historical_rechecks_v2.py': '706e4264b261ed2fa7d1c520148f631026e1b91e98cd38e790a6a7e09081bfe8',
    'docs/review-tools/incremental_revision.py': '654b5796af2be03deed1407ce49d3537dc663208db27d43811896f59b624608d',
    'scripts/omni_revision/review_setup.py': '701a87eb97fe51d2449b88602c4814236bb73297dce1494964e6102ff6684ad0',
}
before = {path: digest(ROOT / path) for path in expected}
assert before == expected
original_path = ART / 'execution-v1.json'
assert digest(original_path) == '59bbf75f5341ef2a721834171489f8997e4f8dbe9d83321948b170076b356418'
original = json.loads(original_path.read_bytes())
prefix = original['mutation_reproduction']['script'].split('probes = [', 1)[0]
prefix_sha = hashlib.sha256(prefix.encode('utf-8')).hexdigest()
assert prefix_sha == 'b9bfbbb13ac342206872137a9142f01109902f0d46b125df40b8a32b0f724211'
old = "REPO = Path('/workspace/tiny-perceptron-vlm')"
new = "REPO = Path('/workspace/work/omni-book-integrated-v1')"
assert prefix.count(old) == 1
namespace = {}
exec(compile(prefix.replace(old, new), '<original-owner-prefix-relocated>', 'exec'), namespace)
namespace['WORK'] = WORK
fixture = namespace['fixture']
assert Path(fixture.__file__).resolve() == ROOT / 'tests/test_historical_rechecks_v2.py'
assert Path(namespace['api'].__file__).resolve() == ROOT / 'docs/review-tools/incremental_revision.py'
WORK.mkdir(exist_ok=False)
rows = []
cases = [
    ('valid_current_fixture', None, None),
    ('honest_prior_revise_original', None, 'technical_original_honest_revise'),
    ('continuity_current_created_before_original', 'chronology', None),
    ('technical_original_created_after_current', 'original_technical_time', None),
    ('technical_original_header_shell_wrong_source', 'original_technical_shell', None),
    ('continuity_original_unbound_source', 'original_continuity_source', None),
]
for name, mutation, variant in cases:
    case_root = WORK / name
    case_root.mkdir()
    built = fixture.fixture(case_root, variant)
    if mutation is not None:
        namespace[mutation](case_root, built)
    namespace['sync_seals'](case_root, built)
    began = datetime.now(UTC).isoformat()
    try:
        value = fixture.check(case_root, built)
        observed, error = 'accept', None
    except Exception as exc:
        value = None
        observed, error = 'reject', type(exc).__name__ + ': ' + str(exc)
    wanted = 'reject' if mutation else 'accept'
    row = {'name': name, 'expected': wanted, 'observed': observed, 'matched': wanted == observed,
           'result': value, 'error': error, 'started_at': began, 'completed_at': datetime.now(UTC).isoformat(),
           'scope': 'Synthetic fixture via real native check_dispositions only; no real stage or full-book pass.'}
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
after = {path: digest(ROOT / path) for path in expected}
assert after == before
result = {'schema_version': 1, 'kind': 'root_mediated_original_owner_probe_execution_not_review_verdict',
          'reviewer_task': TASK, 'executed_by': RELAY, 'probe_design_by': TASK,
          'started_at': started, 'created_at': datetime.now(UTC).isoformat(),
          'command': 'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 uv run --extra cpu --group notebook --no-sync python ' + Path(__file__).relative_to(ROOT).as_posix(),
          'cwd': str(ROOT), 'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'CPU; synthetic std-library fixtures'},
          'original_script_artifact': ref(original_path), 'original_prefix_sha256': prefix_sha,
          'harness_only_path_relocation': {'before': old, 'after': new},
          'fixture_file': ref(Path(fixture.__file__).resolve()), 'native_api_file': ref(Path(namespace['api'].__file__).resolve()),
          'checked_files_before': before, 'checked_files_after': after, 'cases': rows,
          'all_expected': all(row['matched'] for row in rows), 'actual_stage_or_whole_book_approval': False}
with OUT.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': ref(OUT), 'all_expected': result['all_expected']}, ensure_ascii=False), flush=True)
assert result['all_expected'], rows
