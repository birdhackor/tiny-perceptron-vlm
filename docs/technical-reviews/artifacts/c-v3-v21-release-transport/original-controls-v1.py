"""Pure guards, archive bytes and static order; never call publish/Git/API."""

import ast
import copy
import hashlib
import importlib.util
import json
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml


BOOK = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
PRODUCT = Path('/workspace/work/c-product-v3-20261101')
REVIEW = BOOK / 'docs/course-revision-20261101-c-v3/technical-v21/release-transport-owner'
SNAPSHOT = REVIEW / 'original-v1-source-snapshot'
SOURCE = SNAPSHOT / 'scripts/publish_c_v3_release.py'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(BOOK)), 'sha256': digest(raw), 'bytes': len(raw)}


spec = importlib.util.spec_from_file_location('reviewed_transport_original', SOURCE)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
assert digest(SOURCE.read_bytes()) == 'aaf8138f359e6dc40a8619595b95f97b76bc28a537bac392972a026248900f60'
assert digest((PRODUCT / 'scripts/publish_c_v3_release.py').read_bytes()) == digest(SOURCE.read_bytes())
package = PRODUCT / 'assets/releases/c-v3-20261101' / release.NAME
release.verify_file(package)
descriptor = json.loads((SNAPSHOT / 'docs/decision-product/v3/public-release.json').read_bytes())
assert len(descriptor['implementation']) == 17
with zipfile.ZipFile(package) as archive:
    for name, expected in descriptor['implementation'].items():
        candidates = [entry for entry in archive.namelist() if entry.endswith('/' + name)]
        assert len(candidates) == 1 and digest(archive.read(candidates[0])) == expected
        assert digest((PRODUCT / name).read_bytes()) == expected
    for name, expected in descriptor['bundle'].items():
        candidates = [entry for entry in archive.namelist() if entry.endswith('/bundle/' + name)]
        assert len(candidates) == 1 and digest(archive.read(candidates[0])) == expected

revision, other = 'a' * 40, 'b' * 40
base = {'id': release.RELEASE_ID, 'tag_name': release.TAG, 'draft': True, 'assets': []}
asset = {'id': 12, 'name': release.NAME, 'size': release.SIZE}
assert release.checked_release(base, revision) is None
assert release.checked_release({**base, 'assets': [asset]}, revision) == asset
assert release.checked_release({**base, 'draft': False, 'target_commitish': revision, 'assets': [asset]}, revision) == asset
rejected = []
for name, bad in [
    ('wrong_release_id', {**base, 'id': 123}),
    ('wrong_tag_name', {**base, 'tag_name': 'other'}),
    ('nonbool_draft', {**base, 'draft': 1}),
    ('wrong_remote_size', {**base, 'assets': [{**asset, 'size': 1}]}),
    ('duplicate_name', {**base, 'assets': [asset, asset]}),
    ('public_other_target_metadata', {**base, 'draft': False, 'target_commitish': other}),
]:
    try:
        release.checked_release(bad, revision)
    except ValueError as error:
        rejected.append({'case': name, 'rejected': True, 'error': str(error)})
    else:
        raise AssertionError(name)

tree = ast.parse(SOURCE.read_text())
publish = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'publish')
calls = [node for node in ast.walk(publish) if isinstance(node, ast.Call)]
publication = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'api'
                   and len(node.args) == 2 and isinstance(node.args[1], ast.Dict)
                   and any(isinstance(key, ast.Constant) and key.value == 'draft' and isinstance(value, ast.Constant)
                           and value.value is False for key, value in zip(node.args[1].keys, node.args[1].values)))
tag_read = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'api'
            and node.args and 'git/ref/tags/' in ast.unparse(node.args[0])]
assert len(tag_read) == 1 and tag_read[0].lineno > publication.lineno
# This exact draft metadata is accepted, although a preexisting actual tag can
# point elsewhere; the source contains no tag read until after public exposure.
accepted_draft = {**base, 'target_commitish': other, 'assets': [asset]}
assert release.checked_release(accepted_draft, revision) == asset
tag_order = {'publish_draft_false_line': publication.lineno, 'first_actual_tag_read_line': tag_read[0].lineno,
             'draft_metadata_accepted': accepted_draft,
             'preexisting_tag_counterexample': {'object': {'type': 'commit', 'sha': other}},
             'no_publish_or_provider_call_executed': True,
             'interpretation': 'target_commitish patch does not repoint a preexisting tag; actual tag rejection occurs after the release has been made public.'}

source_loop = next(node for node in ast.walk(publish) if isinstance(node, ast.For)
                   and "descriptor['implementation'].items()" == ast.unparse(node.iter))
loop = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(source_loop)], type_ignores=[]))
local_guard = compile(loop, str(SOURCE) + ':implementation-loop-only', 'exec')
with tempfile.TemporaryDirectory(prefix='c3-transport-source17-') as directory:
    root = Path(directory)
    manifest = {}
    for name in descriptor['implementation']:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('fixed:' + name).encode())
        manifest[name] = digest(path.read_bytes())

    def guard(mapping):
        # Only this existing local hash loop is compiled; not publish(), Git,
        # gh, urllib or any other provider operation.
        exec(local_guard, {'descriptor': {'implementation': mapping}, 'ROOT': root, 'hashlib': hashlib})

    guard(manifest)
    drift = 'tiny_perceptron/decision_product/service.py'
    (root / drift).write_bytes(b'changed implementation')
    try:
        guard(manifest)
    except ValueError:
        full_map_drift_rejected = True
    else:
        raise AssertionError('expected listed source drift rejection')
    missing = dict(manifest)
    missing.pop(drift)
    guard(missing)
    guard({})
    incomplete_guard = {'actual_loop_line': source_loop.lineno, 'positive_exact17_pass': True,
                        'listed_source_drift_rejected': full_map_drift_rejected,
                        'same_drift_omitted_from16entry_map_accepted': True,
                        'empty_map_accepted': True, 'publish_Git_or_network_called': False}

workflow = yaml.load((SNAPSHOT / '.github/workflows/c-v3-release.yml').read_text(), Loader=yaml.BaseLoader)
assert workflow['on']['push']['branches'] == ['revision/c-product-v3-20261101']
assert workflow['permissions'] == {'contents': 'write'}
job = workflow['jobs']['publish']
assert job['env']['GH_TOKEN'] == '${{ github.token }}'
assert job['runs-on'] == 'ubuntu-latest'
checkout = next(step for step in job['steps'] if step.get('uses', '').startswith('actions/checkout@'))
assert checkout['with']['lfs'] == 'false'
assert job['env']['C3_PACKAGE_PATH'] == 'assets/releases/c-v3-20261101/' + release.NAME
assert any(step.get('run') == 'git lfs pull --include="$C3_PACKAGE_PATH" --exclude=\'\'' for step in job['steps'])
assert '/assets/releases/c-v3-20261101/' + release.NAME + ' filter=lfs diff=lfs merge=lfs -text' in (SNAPSHOT / '.gitattributes').read_text()
assert '3 passed' in (ARTIFACTS / 'original-local-tests-v1.log').read_text()
result = {'reviewer_task': '/root/omni_p5_full_tools_review', 'created_at': datetime.now(UTC).isoformat(),
          'snapshot_script': ref(SOURCE), 'fixed_archive_bytes_and_SHA_verified': True,
          'actual_qualified_ZIP_source17_and_bundle4_match_descriptor_and_repo': True,
          'checked_release_positive_controls': 3, 'checked_release_negative_controls': rejected,
          'tag_publication_order_finding': tag_order, 'source17_completeness_finding': incomplete_guard,
          'workflow_fixed_branch_token_permissions_exactLFS_static_checks': True,
          'root_attrs_read_at_original_snapshot': ref(SNAPSHOT / '.gitattributes'),
          'author_announced_root_to_nested_attrs_change_requires_current_recheck': True,
          'selected_existing_pytest': ref(ARTIFACTS / 'original-local-tests-v1.log'),
          'existing_guard_test_calling_publish_deliberately_not_executed': 'test_local_process_cannot_publish',
          'publish_or_Git_or_network_write_or_model_called': False,
          'official_tag_documentation': ref(ARTIFACTS / 'official-release-tag-semantics-v1.json')}
with (ARTIFACTS / 'original-controls-result-v1.json').open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'existing_tests': 3, 'release_helper_positive': 3, 'release_helper_rejections': 6,
                  'necessary_tag_order_counterexample': True, 'necessary_source17_omission_counterexample': True,
                  'publish_Git_or_provider_write_executed': False}))
