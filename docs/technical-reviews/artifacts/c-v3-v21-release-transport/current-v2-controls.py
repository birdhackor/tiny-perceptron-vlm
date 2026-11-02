"""Owner's bounded pure-helper/AST controls; never call publish(), Git or network."""

import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path
import tempfile
import textwrap
from unittest.mock import patch

BOOK = Path('/tmp/tpb')
CODE = Path('/tmp/tpc')
ART = BOOK / 'docs/technical-reviews/artifacts/c-v3-v21-release-transport'
SCOPE = json.loads((ART / 'current-v2-read-scope.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location('bounded_release_current_v2', CODE / 'scripts/publish_c_v3_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
revision = 'a' * 40
calls = []


def server(suffix, payload):
    calls.append((suffix, copy.deepcopy(payload)))
    return {'id': release.RELEASE_ID,
            'tag_name': payload.get('tag_name', 'untagged-server-default'),
            'target_commitish': payload['target_commitish'],
            'draft': payload['draft'], 'assets': []}


update_cases = []
with patch.object(release, 'api', server):
    for draft in (True, False):
        result = release.update_release(revision, draft)
        assert result['tag_name'] == release.TAG and result['draft'] is draft
        assert calls[-1] == (f'releases/{release.RELEASE_ID}',
                             {'tag_name': release.TAG, 'target_commitish': revision, 'draft': draft})
        update_cases.append({'case': f'explicit_tag_draft_{draft}', 'expected': 'accepted', 'passed': True})

base = {'id': release.RELEASE_ID, 'tag_name': release.TAG,
        'target_commitish': revision, 'draft': True, 'assets': []}
for name, request_draft, response in [
    ('server_untagged_response', True, {**base, 'tag_name': 'untagged-server-default'}),
    ('wrong_release_id', True, {**base, 'id': release.RELEASE_ID + 1}),
    ('unknown_state', True, {**base, 'draft': 'true'}),
    ('draft_update_returned_public', True, {**base, 'draft': False}),
    ('public_update_still_draft', False, base),
    ('public_update_wrong_target', False, {**base, 'draft': False, 'target_commitish': 'b' * 40}),
]:
    with patch.object(release, 'api', lambda *args, value=response: copy.deepcopy(value)):
        try:
            release.update_release(revision, request_draft)
        except ValueError:
            update_cases.append({'case': name, 'expected': 'rejected', 'passed': True})
        else:
            raise AssertionError(name)

current_tree = ast.parse((CODE / 'scripts/publish_c_v3_release.py').read_text())
prior_tree = ast.parse((BOOK / 'docs/course-revision-20261101-c-v3/technical-v21/release-transport-owner/current-v1-source-snapshot/scripts/publish_c_v3_release.py').read_text())
functions = {node.name: node for node in current_tree.body if isinstance(node, ast.FunctionDef)}
prior_functions = {node.name: node for node in prior_tree.body if isinstance(node, ast.FunctionDef)}
for name in ['verify_implementation', 'check_tag_response', 'tag_preflight']:
    assert ast.dump(functions[name]) == ast.dump(prior_functions[name]), name
pub = functions['publish']
pub_calls = [node for node in ast.walk(pub) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
updates = sorted([node for node in pub_calls if node.func.id == 'update_release'], key=lambda n: n.lineno)
preflights = sorted([node for node in pub_calls if node.func.id == 'tag_preflight'], key=lambda n: n.lineno)
uploads = [node for node in pub_calls if node.func.id == 'gh' and isinstance(node.args[0], ast.List)]
uploads = [node for node in uploads if isinstance(node.args[0].elts[0], ast.Constant) and node.args[0].elts[0].value == 'release']
assert len(updates) == 2 and len(preflights) == 2 and len(uploads) == 1
assert updates[0].args[1].value is True and updates[1].args[1].value is False
assert preflights[0].lineno < updates[0].lineno < uploads[0].lineno
assert preflights[1].lineno < updates[1].lineno
api_calls = sorted([node for node in pub_calls if node.func.id == 'api'], key=lambda n: n.lineno)
assert all(len(node.args) == 1 and not node.keywords for node in api_calls)
assert 'git/ref/tags/' in ast.get_source_segment((CODE / 'scripts/publish_c_v3_release.py').read_text(), api_calls[-1])
source_guard = next(node for node in pub_calls if node.func.id == 'verify_implementation')
assert source_guard.lineno < min(node.lineno for node in api_calls)

descriptor = release.verify_implementation(CODE, CODE / 'docs/decision-product/v3/public-release.json')
assert len(descriptor['implementation']) == 17
release.verify_file(release.PACKAGE)

workflow = (CODE / '.github/workflows/c-v3-release.yml').read_text()
snippet = workflow.split("          python - <<'PY'\n", 1)[1].split('\n          PY', 1)[0]
snippet = textwrap.dedent(snippet)
compiled = compile(snippet, 'workflow-safe-receipt-step', 'exec')
fixture = {'created_at': 'fixture-only', 'repository': release.REPOSITORY,
           'workflow_revision': revision, 'workflow_run': 'fixture-not-a-run',
           'release_id': release.RELEASE_ID, 'tag': release.TAG, 'url': release.URL,
           'bytes': release.SIZE, 'sha256': release.DIGEST, 'training_started': False,
           'success': False, 'stage': 'upload_fixed_package'}
cwd = Path.cwd()
try:
    with tempfile.TemporaryDirectory(prefix='c3-release-current-v2-', dir='/tmp') as directory:
        os.chdir(directory)
        missing = io.StringIO()
        with contextlib.redirect_stdout(missing):
            exec(compiled, {})
        assert 'No publication receipt' in missing.getvalue()
        path = Path('outputs/c-v3-release-publication/result.json')
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(fixture))
        existing = io.StringIO()
        with contextlib.redirect_stdout(existing):
            exec(compiled, {})
        assert json.loads(existing.getvalue()) == fixture
finally:
    os.chdir(cwd)
assert 'if: always()' in workflow.split('- name: Show only the small non-secret publication receipt', 1)[1].split('- name: Preserve actual publication result', 1)[0]
assert 'GH_TOKEN: ${{ github.token }}' in workflow
assert 'git lfs pull --include="$C3_PACKAGE_PATH" --exclude=\'\'' in workflow
for row in SCOPE['files']:
    assert sha(CODE / row['path']) == row['sha256'] == sha(BOOK / row['snapshot'])

result = {
    'created_at': datetime.now(UTC).isoformat(),
    'reviewer_task': '/root/omni_p5_full_tools_review',
    'update_release_helper_cases': update_cases,
    'all8_update_controls_expected': True,
    'retained_001_002_helper_AST_exact_same_as_current1': True,
    'actual_qualified17_hash_guard_positive': True,
    'fixed_archive_bytes': release.SIZE,
    'fixed_archive_sha256': sha(release.PACKAGE),
    'first_preflight_line': preflights[0].lineno,
    'draft_bind_update_line': updates[0].lineno,
    'upload_line': uploads[0].lineno,
    'second_preflight_line': preflights[1].lineno,
    'publication_update_line': updates[1].lineno,
    'no_direct_PATCH_bypassing_update_release_in_publish': True,
    'postpublication_actual_tag_read_retained': True,
    'safe_receipt_inline_python': 'compiled and executed only in private fixture directory: missing and existing small receipt controls both passed',
    'receipt_fixture_is_not_actual_model_or_provider_result': True,
    'publish_called': False, 'actual_Git_or_gh_executed': False,
    'actual_network_or_model_execution': False,
    'source3_current_hashes_still_match_snapshot': True,
}
with (ART / 'current-v2-controls-result.json').open('x') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print('8 update_release API-mock controls passed; retained guards/order/actual17/ZIP and 2 private receipt controls passed; no publish/Git/network/model executed.')
