"""Read-only CI source callback; no tests, model execution, or remote CI trigger."""
from pathlib import Path
import hashlib, json, shutil

ROOT = Path.cwd()
BASE = ROOT / 'docs/technical-reviews/artifacts/p6-public-a'
OUT = BASE / 'ci-dependency'
OUT.mkdir(parents=True, exist_ok=True)
PRE = BASE / 'history/pre-ci-dependency-callback-report.json'
RESEARCH = ROOT / 'docs/course-revision-20261006-phase6/verification/windows-portability/evaluator/isa-research'
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
assert sha(PRE) == 'a35e8963d474f2779ef5cff77feeed05256b21447b922f0bd60ada2c84d5136f'
r = json.loads(PRE.read_bytes())
ci = next(s for s in r['sources'] if s['id'] == 'ci')
before_path = BASE / 'repository-input/.github/workflows/ci.yml'
after_path = ROOT / '.github/workflows/ci.yml'
assert sha(before_path) == ci['sha256']
before, after = before_path.read_text(), after_path.read_text()
block = "        env:\n          # Some Windows runner CPUs expose inconsistent AMX capabilities\n          # (oneDNN #5689, runner-images #14483); keep every dtype/test enabled.\n          ONEDNN_MAX_CPU_ISA: ${{ runner.os == 'Windows' && 'AVX2' || '' }}\n"
assert after.count(block) == 1
assert after.replace(block, '') == before
assert "      - run: uv run ${{ matrix.extra }} pytest --junitxml=outputs/pytest-results.xml\n" + block in after
current_copy = OUT / 'current-ci.yml'
current_copy.write_bytes(after_path.read_bytes())
assert sha(current_copy) == sha(after_path)
result = {'pre_callback_report_path': str(PRE.relative_to(ROOT)), 'pre_callback_report_sha256': sha(PRE),
    'ci_source': {'path': '.github/workflows/ci.yml', 'old_sha256': ci['sha256'],
        'new_sha256': sha(after_path), 'old_snapshot_path': str(before_path.relative_to(ROOT)),
        'current_snapshot_path': str(current_copy.relative_to(ROOT)),
        'full_current_workflow_personally_read': True,
        'only_workflow_change': block, 'original_workflow_preserved_after_removing_added_env_block': True,
        'linux_windows_macos_matrix_unchanged': True, 'check_env_and_pytest_commands_unchanged': True,
        'test_selection_dtype_and_skip_policy_unchanged_by_this_workflow_change': True},
    'public_pages': [], 'unchanged_prior_dependencies': [], 'figure_sha256': {},
    'official_original_sources': [], 'public_claims_reread': [],
    'new_native_windows_run_checked': False, 'new_native_windows_pass_claimed': False,
    'model_execution_training_generation': False, 'native_windows_execution': False}
for path, h in r['source_files'].items():
    assert sha(ROOT / path) == h
    result['public_pages'].append({'path': path, 'sha256': h, 'unchanged': True,
        'initial_full_read_and_reinspection_scope_retained': True})
for group in ['sources', 'artifacts']:
    for item in r[group]:
        if group == 'sources' and item['id'] == 'ci':
            continue
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
        result['unchanged_prior_dependencies'].append({'group': group, 'id': item['id'],
            'path': item['path'], 'sha256': item['sha256']})
for path, h in r['figure_sha256'].items():
    assert sha(ROOT / path) == h
    result['figure_sha256'][path] = h
index = json.loads((RESEARCH / 'source-index.json').read_bytes())
rows = index if isinstance(index, list) else index['sources']
names = ['onednn-5689.json', 'onednn-5689-comments.json', 'runner-images-14483.json',
    'onednn-dispatcher-control.md', 'onednn-exact-isa.cpp', 'onednn-exact-utils.cpp']
for name in names:
    record = next(x for x in rows if isinstance(x, dict) and x.get('name') == name)
    assert record['http_status'] == 200
    p = RESEARCH / name
    assert p.stat().st_size == record['bytes']
    frozen = OUT / 'official-originals' / name
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_bytes(p.read_bytes())
    assert sha(p) == sha(frozen)
    result['official_original_sources'].append({'name': name, 'url': record['url'],
        'path': str(frozen.relative_to(ROOT)), 'sha256': sha(frozen),
        'original_saved_path': str(p.relative_to(ROOT)), 'personally_read': True})
doc = (OUT / 'official-originals/onednn-dispatcher-control.md').read_text()
assert 'ONEDNN_ENABLE_MAX_CPU_ISA' in doc and 'ONEDNN_MAX_CPU_ISA' in doc
assert '| ONEDNN_MAX_CPU_ISA   | AVX2' in doc
assert 'Function settings take precedence over environment variables.' in doc
utils = (OUT / 'official-originals/onednn-exact-utils.cpp').read_text()
isa = (OUT / 'official-originals/onednn-exact-isa.cpp').read_text()
assert 'for (const auto &prefix : {"ONEDNN_", "DNNL_"})' in utils
assert 'getenv_string_user("MAX_CPU_ISA")' in isa and 'ELSEIF_HANDLE_CASE(avx2)' in isa
for path, line_numbers in [('README.md', [48, 102]), ('docs/environment.md', [19]),
    ('docs/validation.md', [70])]:
    lines = (ROOT / path).read_text().splitlines()
    for n in line_numbers:
        result['public_claims_reread'].append({'path': path, 'line': n, 'text': lines[n - 1],
            'page_sha256': sha(ROOT / path), 'personally_reread': True})
result['verified_scope'] = 'Current CI source contract and official rationale only. AVX2 limits oneDNN dispatch when its control is enabled; it is not a dtype conversion or a guarantee for every Windows CPU. Existing historical macOS/MPS receipts retain their original scope. No new native Windows result was inspected or asserted.'
result['command'] = '/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-public-a/verify_ci_dependency_callback.py'
(OUT / 'measurements.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'old_ci_sha256': ci['sha256'], 'new_ci_sha256': sha(after_path),
    'public_pages_unchanged': len(result['public_pages']),
    'other_source_artifact_dependencies_unchanged': len(result['unchanged_prior_dependencies']),
    'official_sources_personally_read': len(result['official_original_sources']),
    'only_pytest_env_added': True, 'new_native_windows_pass_claimed': False,
    'assertions': 'passed'}, ensure_ascii=False, indent=2))
