"""Bounded fix replay: pure helpers/mock HTTP/AST; no publish/Git/write API."""

import ast
import fnmatch
import hashlib
import importlib.util
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import yaml


BOOK = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
PRODUCT = Path('/workspace/work/c-product-v3-20261101')
REVIEW = BOOK / 'docs/course-revision-20261101-c-v3/technical-v21/release-transport-owner'
SOURCE = PRODUCT / 'scripts/publish_c_v3_release.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path.relative_to(BOOK)), 'sha256': sha(path), 'bytes': path.stat().st_size}


scope = json.loads((ARTIFACTS / 'current-read-scope-v1.json').read_bytes())
for row in scope['files']:
    assert sha(PRODUCT / row['path']) == row['sha256']
assert sha(REVIEW / 'original-v1.json') == 'e99e41820b13b1aa5fca448636848ad8c2530f0acc5916d7309171e2b16f8382'
collection = json.loads((BOOK / 'docs/course-revision-20261101-c-v3/actual-release-transport-original-collected-v1.json').read_bytes())
assert collection['collection_index'][0]['sha256'] == sha(REVIEW / 'original-v1.json')
assert datetime.fromisoformat(collection['collection_index'][0]['received_at']) < datetime.fromisoformat(scope['created_at'])
spec = importlib.util.spec_from_file_location('current_Cv3_transport_guard', SOURCE)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
revision = 'a' * 40
good = {'ref': 'refs/tags/' + release.TAG, 'object': {'type': 'commit', 'sha': revision}}


def response(status, body, code=0, crlf=True):
    newline = b'\r\n' if crlf else b'\n'
    payload = json.dumps(body).encode()
    return SimpleNamespace(returncode=code, stdout=b'HTTP/2.0 ' + str(status).encode() + b' Status'
                           + newline + b'Content-Type: application/json' + newline + newline + payload, stderr=b'')


http_cases = [
    ('matching_draft_CRLF', response(200, good), True, True),
    ('matching_public_LF', response(200, good, crlf=False), False, True),
    ('explicit_404_draft', response(404, {'message': 'Not Found'}, code=1), True, False),
    ('404_public_rejected', response(404, {}, code=1), False, None),
    ('403_not_missing', response(403, {}, code=1), True, None),
    ('500_not_missing', response(500, {}, code=1), True, None),
    ('wrong_existing_commit', response(200, {**good, 'object': {'type': 'commit', 'sha': 'b' * 40}}), True, None),
    ('annotated_tag_not_verified_commit', response(200, {**good, 'object': {'type': 'tag', 'sha': revision}}), True, None),
    ('wrong_ref_name', response(200, {**good, 'ref': 'refs/tags/another-tag'}), True, None),
    ('unrecognized_HTTP_status', SimpleNamespace(returncode=1, stdout=b'network failed', stderr=b''), True, None),
]
real_run = release.subprocess.run
mock_calls = []
http_results = []
try:
    for label, supplied, draft, expected in http_cases:
        def fake_run(args, *, capture_output, timeout, check, result=supplied):
            assert args == ['gh', 'api', '--include', f'repos/{release.REPOSITORY}/git/ref/tags/{release.TAG}']
            assert capture_output is True and timeout == 120 and check is False
            mock_calls.append(label)
            return result

        release.subprocess.run = fake_run
        try:
            value = release.tag_preflight(revision, draft)
        except ValueError as error:
            assert expected is None, (label, str(error))
            http_results.append({'case': label, 'rejected': True, 'error': str(error)})
        else:
            assert expected is not None and value is expected, (label, value)
            http_results.append({'case': label, 'value': value, 'rejected': False})
finally:
    release.subprocess.run = real_run
assert len(mock_calls) == len(http_cases) == 10

descriptor_path = PRODUCT / 'docs/decision-product/v3/public-release.json'
descriptor = release.verify_implementation(PRODUCT, descriptor_path)
assert sha(descriptor_path) == release.DESCRIPTOR_DIGEST == 'ac8210926a00700a3756aa73f5e877ca292ec7f3c6dfae9c096e45dbe99c3859'
assert len(descriptor['implementation']) == 17


class NoSourceReadBeforePinnedDescriptor:
    def __truediv__(self, name):
        raise AssertionError('bad descriptor must reject before reading any source entry')


descriptor_controls = []
with tempfile.TemporaryDirectory(prefix='c3-transport-descriptor-') as directory:
    path = Path(directory) / 'descriptor.json'
    for label, mapping in [
        ('empty_map', {}),
        ('same_old_drifting_service_omitted_from16map', {name: digest for name, digest in descriptor['implementation'].items()
                                                      if name != 'tiny_perceptron/decision_product/service.py'}),
    ]:
        path.write_text(json.dumps({**descriptor, 'implementation': mapping}))
        try:
            release.verify_implementation(NoSourceReadBeforePinnedDescriptor(), path)
        except ValueError as error:
            assert '完整17檔' in str(error)
            descriptor_controls.append({'case': label, 'rejected_before_any_source_read': True, 'error': str(error)})
        else:
            raise AssertionError(label)

tree = ast.parse(SOURCE.read_text())
publish = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'publish')
calls = [node for node in ast.walk(publish) if isinstance(node, ast.Call)]
preflights = sorted(node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'tag_preflight')
patches = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'api'
           and len(node.args) > 1]
upload = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'gh'
          and node.args and isinstance(node.args[0], ast.List)
          and [getattr(value, 'value', None) for value in node.args[0].elts[:2]] == ['release', 'upload']]
publication = next(node for node in patches if isinstance(node.args[1], ast.Dict)
                   and any(isinstance(key, ast.Constant) and key.value == 'draft' and isinstance(value, ast.Constant)
                           and value.value is False for key, value in zip(node.args[1].keys, node.args[1].values)))
post_tag = [node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'api'
            and node.args and 'git/ref/tags/' in ast.unparse(node.args[0])]
assert len(preflights) == 2
assert preflights[0] < min(node.lineno for node in patches + upload)
assert preflights[1] < publication.lineno < min(post_tag)
implementation_guard = next(node.lineno for node in calls if isinstance(node.func, ast.Name)
                            and node.func.id == 'verify_implementation')
first_release_read = min(node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'api')
assert implementation_guard < first_release_read

package = PRODUCT / 'assets/releases/c-v3-20261101' / release.NAME
release.verify_file(package)
stage_path = Path('/workspace/work/c-product-stage-lfs-receipt-20261102.json')
stage = json.loads(stage_path.read_bytes())
expected_pointer = f'version https://git-lfs.github.com/spec/v1\noid sha256:{release.DIGEST}\nsize {release.SIZE}\n'
assert stage['git_index_pointer'] == expected_pointer
assert len(expected_pointer.encode()) == stage['ordinary_Git_pointer_bytes'] == 132
assert stage['model_worktree_bytes'] == package.stat().st_size == release.SIZE
nested = PRODUCT / 'assets/releases/.gitattributes'
assert sha(nested) == stage['nested_attributes']['sha256'] == '8adfcfe89844f2bbfc31f81c2328efdeb5071522ffa3f4eee3ee299b5870cf5d'
assert stage['git_check_attr_observed'] == 'filter=lfs,diff=lfs,merge=lfs,text=unset'
assert stage['root_attributes_unchanged_from_HEAD'] is True
root_attrs = (PRODUCT / '.gitattributes').read_bytes()
assert root_attrs == (BOOK / '.gitattributes').read_bytes()
old_root = (REVIEW / 'original-v1-source-snapshot/.gitattributes').read_text()
assert root_attrs == (old_root.split('# 固定成品的 workflow 傳輸；模型與原始輸出不存普通 Git。', 1)[0].rstrip() + '\n').encode()
rule = 'c-v3-20261101/*.zip filter=lfs diff=lfs merge=lfs -text'
assert rule in nested.read_text()
assert fnmatch.fnmatchcase('c-v3-20261101/' + release.NAME, rule.split()[0])
workflow_path = PRODUCT / '.github/workflows/c-v3-release.yml'
assert sha(workflow_path) == '207f1bcf5f25e40d5be0a0c970eba2d9cd4606205dbb2ddecb7a5808b6e28e20'
workflow = yaml.load(workflow_path.read_text(), Loader=yaml.BaseLoader)
assert workflow['on']['push']['branches'] == ['revision/c-product-v3-20261101']
job = workflow['jobs']['publish']
assert job['env']['GH_TOKEN'] == '${{ github.token }}'
assert job['env']['C3_PACKAGE_PATH'] == 'assets/releases/c-v3-20261101/' + release.NAME
assert any(step.get('run') == 'git lfs pull --include="$C3_PACKAGE_PATH" --exclude=\'\'' for step in job['steps'])
assert '5 passed, 1 deselected' in (ARTIFACTS / 'current-local-tests-v1.log').read_text()
for row in scope['files']:
    assert sha(PRODUCT / row['path']) == row['sha256']
assert sha(REVIEW / 'original-v1.json') == 'e99e41820b13b1aa5fca448636848ad8c2530f0acc5916d7309171e2b16f8382'
result = {'reviewer_task': '/root/omni_p5_full_tools_review', 'created_at': datetime.now(UTC).isoformat(),
          'current_read_scope': ref(ARTIFACTS / 'current-read-scope-v1.json'),
          'actual_original_collection': ref(BOOK / 'docs/course-revision-20261101-c-v3/actual-release-transport-original-collected-v1.json'),
          'real_tag_preflight_HTTP_parser_mock_controls': http_results,
          'all10_HTTP_controls_expected': True, 'first_preflight_before_any_remote_write_line': preflights[0],
          'second_preflight_line': preflights[1], 'publication_line': publication.lineno,
          'post_tag_read_line': min(post_tag), 'source_guard_before_first_API_line': implementation_guard,
          'actual_qualified17_source_positive': True, 'original_map_omission_controls': descriptor_controls,
          'root_attributes_restored_to_canonical_and_original_preaddition_bytes': True,
          'nested_attributes_and_exact_fetch_static_match': True,
          'author_actual_staged_LFS_receipt': {'external_path': str(stage_path), 'sha256': sha(stage_path),
                                              'bytes': stage_path.stat().st_size, 'execution_owner': '/root'},
          'author_LFS_pointer_parsed_and_matched_to_actual_fixed_archive': True,
          'pointer_bytes': 132, 'archive_bytes': release.SIZE, 'archive_sha256': release.DIGEST,
          'author_Git_operations_not_relabelled_as_reviewer_execution': True,
          'selected_existing_pytest': ref(ARTIFACTS / 'current-local-tests-v1.log'),
          'tests_passed': 5, 'tests_deselected': 1,
          'publish_Git_network_write_training_or_model_executed': False,
          'limits': 'Only pure helpers with subprocess.run mocked, AST order, actual file hashes and read-only author LFS receipt inspection. No publish(), actual gh/Git/API/runtime permission test or new publication. Source17/ZIP remained exact.'}
with (ARTIFACTS / 'current-controls-result-v1.json').open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'existing_helper_tests': 5, 'excluded_publish_test': 1, 'HTTP_mock_controls': 10,
                  'original_source17_omissions_now_rejected': True,
                  'root_and_nested_LFS_receipt_verified': True, 'publish_or_Git_called': False}))
