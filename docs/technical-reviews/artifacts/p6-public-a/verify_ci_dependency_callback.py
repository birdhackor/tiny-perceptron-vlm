"""Read-only CI source callback; no tests, model execution, or remote CI trigger."""
from pathlib import Path
import ast, hashlib, json

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
        if group == 'sources' and item['id'] in ['ci', 'pages', 'evaluate']:
            continue
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
        result['unchanged_prior_dependencies'].append({'group': group, 'id': item['id'],
            'path': item['path'], 'sha256': item['sha256']})
for path, h in r['figure_sha256'].items():
    assert sha(ROOT / path) == h
    result['figure_sha256'][path] = h
result['additional_changed_sources'] = {}
for sid, path in [('pages', '.github/workflows/pages.yml'), ('evaluate', 'scripts/selftrained/evaluate.py')]:
    old_source = next(s for s in r['sources'] if s['id'] == sid)
    old_path = BASE / 'repository-input' / path
    current_path = ROOT / path
    assert sha(old_path) == old_source['sha256']
    old, current = old_path.read_text(), current_path.read_text()
    if sid == 'pages':
        expected = old.replace('Fetch the LFS dataset cited by the preference review',
            'Fetch the exact LFS datasets cited by factual reviews').replace(
            'assets/training/ultrafeedback-dpo-v1.tar.gz" --exclude=',
            'assets/training/ultrafeedback-dpo-v1.tar.gz,assets/training/selftrained-v2.tar.gz" --exclude=')
        assert current == expected and 'timeout-minutes: 40' in current
        support = 'Only the LFS step name and include list changed: preference plus v2 original tar. The 40-minute budget, review gates, independent serial CPU kernels, strict build, link checks, artifact and deploy steps are byte-identical.'
    else:
        expected = old.replace('str(path.relative_to(root)) for path in sorted',
            'path.relative_to(root).as_posix() for path in sorted').replace(
            'with journal.receipt_path.open("rb") as handle:',
            'with journal.receipt_path.open("r+b") as handle:')
        assert current == expected
        old_ast, current_ast = ast.parse(old), ast.parse(current)
        def node(tree, name):
            return next(n for n in tree.body if getattr(n, 'name', None) == name)
        for name in ['score_reply', 'summarize', 'validate_protocol']:
            assert ast.dump(node(old_ast, name)) == ast.dump(node(current_ast, name))
        old_thresholds = next(n for n in old_ast.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'THRESHOLDS' for t in n.targets))
        current_thresholds = next(n for n in current_ast.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'THRESHOLDS' for t in n.targets))
        assert ast.dump(old_thresholds) == ast.dump(current_thresholds)
        support = 'Only fingerprint-key path rendering and fsync receipt descriptor mode changed. Thresholds, score_reply, summarize, validate_protocol and every other code line remain identical. Historical frozen scores still refer to their original method bytes; no new evaluation is implied.'
    copy_path = OUT / 'current-input' / path
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    copy_path.write_bytes(current_path.read_bytes())
    result['additional_changed_sources'][sid] = {'path': path, 'old_sha256': old_source['sha256'],
        'new_sha256': sha(current_path), 'old_snapshot_path': str(old_path.relative_to(ROOT)),
        'current_snapshot_path': str(copy_path.relative_to(ROOT)), 'verification': support}
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
    ('docs/validation.md', [13, 70]), ('docs/publishing.md', [57])]:
    lines = (ROOT / path).read_text().splitlines()
    for n in line_numbers:
        result['public_claims_reread'].append({'path': path, 'line': n, 'text': lines[n - 1],
            'page_sha256': sha(ROOT / path), 'personally_reread': True})
result['verified_scope'] = 'Source dependencies only: current CI contract and official rationale, Pages LFS include change, and evaluator path/fsync portability changes. AVX2 limits oneDNN dispatch when its control is enabled; it is not a dtype conversion or a guarantee for every Windows CPU. Historical macOS/MPS receipts and frozen model scores retain their original method/version scope. No new native Windows result was inspected or asserted.'
result['command'] = '/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-public-a/verify_ci_dependency_callback.py'
(OUT / 'measurements.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'old_ci_sha256': ci['sha256'], 'new_ci_sha256': sha(after_path),
    'public_pages_unchanged': len(result['public_pages']),
    'other_source_artifact_dependencies_unchanged': len(result['unchanged_prior_dependencies']),
    'official_sources_personally_read': len(result['official_original_sources']),
    'additional_changed_source_ids_checked': list(result['additional_changed_sources']),
    'only_pytest_env_added': True, 'new_native_windows_pass_claimed': False,
    'assertions': 'passed'}, ensure_ascii=False, indent=2))
