"""Issue-specific rechecks run only after preserving the fresh full original audit."""

import hashlib
import importlib.util
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SCRATCH = Path('/workspace/work/tokenizer-dynamic-20261027/tools-curriculum/historical-guards-01')
ART = 'docs/technical-reviews/artifacts/tokenizer-20261027/tools-curriculum/'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


test = load('historical_guard_fixture', ROOT / 'tests/test_historical_rechecks.py')
review, preflight = test.review, test.inventory.preflight
results = []


def run(issue, name, body, expected, old='old-token', new='new-token'):
    root = SCRATCH / name
    root.mkdir(parents=True, exist_ok=False)
    ref = test.save(root, 'current.md', body.encode())
    current = {'page_id': 'page', 'source': 'current.md', 'selector': 'whole-file', 'source_sha256': ref['sha256'], 'snapshot': ref['path'], 'figures_sha256': {}}
    patch = {'id': name, 'path': 'current.md', 'old': old, 'new': new, 'requested_by': 'author', 'requested_on': '2026-10-27', 'reason': 'Synthetic guard recheck.'}
    listed = test.save(root, 'patches.json', {'schema_version': 1, 'kind': review.PATCHES_KIND, 'statement': review.PATCHES_STATEMENT, 'patches': [patch]})
    freeze = test.save(root, 'freeze.json', {'pages': [current], 'figure_snapshots': {}})
    draft = {'author_patches': listed, 'source_freeze': freeze}
    row = {'issue_id': issue, 'case': name, 'input': body, 'patch': patch, 'expected': expected}
    try:
        patches = review.author_patches(root, draft, {'page': current})
        files = review.file_versions(root, patches)
        pages = review.page_versions(root, current, current, patches)
        intended = review.sha(body.replace(new, old).encode())
        row.update(observed='accepted', file_versions=files, page_versions=pages)
        assert files['current.md'][intended] == (name,)
        assert pages[intended] == (name,)
    except ValueError as error:
        row.update(observed='rejected', error=str(error))
    errors = []
    patches, files = preflight.patch_files(root, draft, {'page': current}, errors)
    earlier = preflight.patch_pages(root, draft, {'page': current}, patches, errors)
    row['local_preflight'] = 'rejected' if errors else 'accepted'
    row['local_errors'] = errors
    row['as_expected'] = row['observed'] == expected and row['local_preflight'] == expected
    results.append(row)


for name, body in {
    'link-target': '[連結](https://example.invalid/new-token)\n',
    'link-label': '[new-token](https://example.invalid/)\n',
    'reference-target': '[link][ref]\n\n[ref]: https://example.invalid/new-token\n',
    'image-target': '![figure](https://example.invalid/new-token.svg)\n',
    'html-image': '<img src="new-token.svg">\n',
    'inline-code': '`new-token`\n',
    'math-inline': '$x_{new-token}$\n',
    'indented-code': '    new-token\n',
    'html-pre': '<pre>new-token</pre>\n',
}.items():
    run('AUTHOR-PATCH-MARKUP', name, body, 'accepted')

run('AUTHOR-PATCH-CODEBLOCK', 'plain-inside', '```python\nprint("new-token")\n```\n', 'rejected')
for name, closer in {'indented-one': ' ```', 'indented-four': '    ```', 'quoted-closer': '> ```'}.items():
    run('AUTHOR-PATCH-CODEBLOCK', name, '```python\n' + closer + '\nprint("new-token")\n```\n', 'rejected')
run('AUTHOR-PATCH-CODEBLOCK', 'plain-outside', '```python\nprint(1)\n```\nnew-token\n', 'accepted')
run('AUTHOR-PATCH-CODEBLOCK', 'crlf-outside', '```python\r\nprint(1)\r\n```\r\nnew-token\r\n', 'accepted')

for label, mark in {'nbsp': '\u00a0', 'em-space': '\u2003', 'ideographic': '\u3000', 'vertical-tab': '\x0b', 'ascii-space': ' ', 'ascii-tab': '\t'}.items():
    for label2, fence in {'backtick': '```', 'tilde': '~~~'}.items():
        run('AUTHOR-PATCH-FENCE-WHITESPACE', label + '-' + label2, fence + 'python\n' + fence + mark + '\nprint("new-token")\n' + fence + '\n', 'rejected')

for name, body, old, new in [
    ('bad-info', '```bad`info\n```\nprint("new-token")\n```\n', 'old-token', 'new-token'),
    ('invalid-option', '```python invalid\n```\nprint("new-token")\n```\n', 'old-token', 'new-token'),
    ('unclosed-eof', 'new-token\n```python\nprint(1)\n', 'old-token', 'new-token'),
    ('reverse-invalid-fence', '用``X 開頭\n', '` 開頭', 'X 開頭'),
]:
    run('AUTHOR-PATCH-FENCE-STATE', name, body, 'rejected', old, new)

sha = lambda name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
prior = json.loads((ROOT / 'docs/course-review-20261027-tokenizer/reviews/technical-tools-callback/tool-audit-current-01.json').read_bytes())
old_files = {item['path']: item['sha256'] for item in prior['checked_files']}
comparison = {name: {'current': sha(name), 'second_wave': old_files.get(name), 'unchanged': sha(name) == old_files.get(name)} for name in ['docs/review-tools/incremental_revision.py', 'docs/course-review-20261019/tools/preflight.py', 'docs/course-review-20261019/tools/issue_inventory.py']}
output = {
    'reviewer_task': '/root/tech_tools_curriculum_20261027', 'created_at': datetime.now(UTC).isoformat(),
    'command': 'uv run python ' + ART + 'historical-guards-01.py',
    'environment': {'python': sys.version, 'platform': platform.platform(), 'device': 'CPU; offline metadata guards'},
    'scope': 'Current author_patches/file_versions/page_versions and local preflight wrappers. No rendering or complete publication manifest/all gate.',
    'policy_note': 'MARKUP remains fixed by explicit policy and code limits: link/image/inline/math/unfenced/HTML positions remain author responsibility and are expected to be accepted. Other historical fence cases must be rejected.',
    'second_wave_byte_comparison': comparison,
    'results': results, 'all_as_expected': all(row['as_expected'] for row in results),
}
(ROOT / (ART + 'historical-guards-01.json')).write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'cases': len(results), 'all_as_expected': output['all_as_expected'], 'second_wave_three_unchanged': all(row['unchanged'] for row in comparison.values()), 'unexpected': [row for row in results if not row['as_expected']]}, ensure_ascii=False, indent=2))
raise SystemExit(0 if output['all_as_expected'] else 1)
