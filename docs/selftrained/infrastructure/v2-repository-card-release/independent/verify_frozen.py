"""Independent frozen-code preservation and evidence binding; read-only repo."""
import ast
import hashlib
import inspect
import json
import re
import subprocess
from datetime import datetime, UTC
from pathlib import Path

import huggingface_hub
from huggingface_hub import HfApi
from huggingface_hub._commit_api import _send_commit

OUT = Path('/tmp/p5-repository-card-engineering-review')
CANDIDATE = Path('/tmp/p5-repository-card-release-candidate')
ACTUAL = Path('/workspace/selftrained-v2')
BASE = '9482b50212cca4c6b68cf7ad2d54abe27c2b308b'
EXPECTED = {
    'candidate.patch': '10d06273609487db5c5d40e75affbdc5017f492e9dd77e81387de9e665b36ab4',
    'proof.json': 'bef959cf8d54a59760a7425308d7075a3fe3b93fcdd5e4624ee724000c24b068',
    'test_repository_card_release.py': 'e1d465fe769f5f6dc493abbcc89c1eed5ccc4b4fc523ee1565081857b91eaaba',
    'scripts/selftrained/modal_runner.py': 'ee7676871d1c24962905660775b869f50ac5c83818980934cef920f5ba1c2947',
    'scripts/selftrained/hf_transport.py': '500417c5f922cd1ce7ae0e6add8a31e2bc34bb7434a30bd845be93b47e831600',
}
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def git(*args):
    return subprocess.check_output(['git', *args], cwd=ACTUAL)
def functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
def dump(node):
    return ast.dump(node, include_attributes=False)

bound = {}
for path, expected in EXPECTED.items():
    raw = (CANDIDATE / path).read_bytes()
    assert sha(raw) == expected, path
    bound[path] = {'sha256': expected, 'bytes': len(raw)}
paths = re.findall(r'^diff --git a/(\S+) b/(\S+)$', (CANDIDATE / 'candidate.patch').read_text(), re.MULTILINE)
assert paths == [(path, path) for path in ('scripts/selftrained/modal_runner.py', 'scripts/selftrained/hf_transport.py')]
check = subprocess.run(['git', 'apply', '--check', str(CANDIDATE / 'candidate.patch')], cwd=ACTUAL, capture_output=True, text=True)
assert check.returncode == 0, check.stderr

preserved = {}
trees = {}
for name, changed in [('modal_runner', {'validate_job', 'register_modal'}), ('hf_transport', set())]:
    path = f'scripts/selftrained/{name}.py'
    original = git('show', f'{BASE}:{path}')
    assert (ACTUAL / path).read_bytes() == original
    old, new = ast.parse(original), ast.parse((CANDIDATE / path).read_bytes())
    old_functions, new_functions = functions(old), functions(new)
    unchanged = []
    for function, node in old_functions.items():
        if function not in changed:
            assert dump(node) == dump(new_functions[function]), function
            unchanged.append(function)
    preserved[name] = {'actual_still_equals_base': True, 'original_function_AST_unchanged': unchanged}
    trees[name] = (old, new)

old, new = trees['modal_runner']
old_nested = functions(functions(old)['register_modal'])
new_nested = functions(functions(new)['register_modal'])
for name in ('finish_remote', 'prepare_remote', 'gpu_remote', 'release_remote'):
    assert dump(old_nested[name]) == dump(new_nested[name]), name
for name in ('reserve_remote', 'finish_remote', 'prepare_remote', 'gpu_remote', 'release_remote'):
    assert [dump(d) for d in old_nested[name].decorator_list] == [dump(d) for d in new_nested[name].decorator_list], name

for path in ('scripts/selftrained/finance.py', 'scripts/selftrained/train.py', '.github/workflows/selftrained-assistant.yml', '.github/workflows/selftrained-inspect.yml', '.github/workflows/selftrained-data.yml', 'docs/selftrained/v2-gross-quota-20261006.json'):
    assert (ACTUAL / path).read_bytes() == git('show', f'{BASE}:{path}'), path
assert (CANDIDATE / 'scripts/selftrained/finance.py').read_bytes() == (ACTUAL / 'scripts/selftrained/finance.py').read_bytes()

sdk_create, sdk_send = inspect.getsource(HfApi.create_commit), inspect.getsource(_send_commit)
assert huggingface_hub.__version__ == '1.33.0'
assert sdk_create.index('if len(operations_without_no_op) == 0:') < sdk_create.index('commit_info = _send_commit(') < sdk_create.index('addition._is_committed = True')
assert 'retry_on_error: bool = False' in sdk_send
assert 'retry_on_error=' not in sdk_create
(OUT / 'hub-sdk-create-commit-source.txt').write_text(sdk_create)
(OUT / 'hub-sdk-send-commit-source.txt').write_text(sdk_send)
logs = {}
for path, result in [('baseline-pytest.txt', '85 passed'), ('author-tests-rerun.txt', '40 passed'), ('candidate-existing-regression.txt', '85 passed'), ('independent-extra-tests-final.txt', '7 passed')]:
    raw = (OUT / path).read_bytes()
    assert result in raw.decode()
    logs[path] = {'sha256': sha(raw), 'last_line': raw.decode().strip().splitlines()[-1]}
report = {
    'status': 'PASS_NO_BLOCKING_ENGINEERING_FINDING_FROZEN_PATCH_NOT_APPLIED',
    'reviewed_at': datetime.now(UTC).isoformat(),
    'base_revision': BASE,
    'current_repository_revision': git('rev-parse', 'HEAD').decode().strip(),
    'repository_status_at_completion': git('status', '--short').decode(),
    'frozen_input_bindings': bound,
    'patch_scope_exact_two_authorized_existing_scripts': paths,
    'git_apply_check_exit': check.returncode,
    'original_function_preservation': preserved,
    'nested_wrapper_and_resource_decorators_AST_preserved': True,
    'finance_trainer_workflow_policy_bytes_still_equal_base': True,
    'tests': logs,
    'existing_candidate_regression_adapter': {
        'path': str(OUT / 'candidate_plugin.py'),
        'sha256': sha((OUT / 'candidate_plugin.py').read_bytes()),
        'behavior': 'Substitute frozen candidate runner/transport in the existing 85-test infrastructure/quota suite; existing financial reserve fixture compiles actual frozen candidate reserve_remote AST; assertions unchanged; block socket connections.'
    },
    'independent_extra_checks': {
        'path': str(OUT / 'test_independent_extra.py'),
        'sha256': sha((OUT / 'test_independent_extra.py').read_bytes()),
        'count': 7,
        'real_pinned_sdk_flows': ['Actual HfApi.create_commit successful singleton Add sets _is_committed only after one _send_commit with pinned parent and no retry_on_error.', 'SDK race/no-op returned changed current HEAD without sending commit; candidate raises SDK skipped and performs no verification/success return.', 'Ambiguous _send_commit response invokes one send, propagates failure, leaves flag false, performs no recovery.'],
        'other_checks': ['Exact 65536-byte payload accepted; oversize and wrong payload types rejected.', 'Short dispatch revision and wrong source SHA rejected before public API.', 'Unknown modes and extra export path descriptor rejected.', 'Existing prepare/pretrain jobs do not acquire card source.'],
        'harness_correction': 'Initial independent SDK no-op fixture omitted actual preupload _upload_mode=regular, so _local_oid was None and no-op was not exercised. Corrected reviewer fixture; all final 7 tests pass. Frozen candidate was unchanged.'
    },
    'source_gates_review': 'Actual reserve_remote card gate precedes official snapshot/reserve_entry/sidecar/ledger writes and rejects no-op reservation. Actual execute card gate checks prior entry source SHA before running status, output creation or publisher. Client binds full actual dispatch Git revision, cat-file bounded blob size, local bytes equality, SHA and manifest.',
    'publish_contract_review': 'Anonymous authorized public HEAD must equal pinned parent. Exactly one CommitOperationAdd README.md on main with create_pr=False and parent_commit CAS. Returned full new immutable SHA and _is_committed true required. Anonymous metadata commit/size and immutable README download hash verified. Normal no-op returns no_op true/new_commit_created false/recovered false. No catch/retry/recovery in card publisher.',
    'financial_preservation_review': 'Same live-rate CPU1/memory2GiB/600-second bounded release reservation, root serialization, retries0 and codex_cloud Secret. Financial functions and policy unchanged: exact additional30, actualgross before credits, sidecar round bounds, USD0.54 carry retained. New card job requires explicit gross policy and rejects model/checkpoint options.',
    'limits': ['Synthetic/local CPU only; no production card or HF/Modal API/account operations, no paid/GPU work, no model/checkpoint/test-gold reads.', 'No repository edits or patch application by this reviewer; all reviewer artifacts in /tmp.', 'Review certifies candidate engineering behavior and preservation, not production financial readiness or actual public publication.', 'SDK private flag is a pinned 1.33.0 fail-closed dependency, now directly checked with actual SDK local control flow.']
}
(OUT / 'actualproof.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'proof': str(OUT / 'actualproof.json'), 'proof_sha256': sha((OUT / 'actualproof.json').read_bytes()), 'frozen_patch_sha256': EXPECTED['candidate.patch'], 'candidate_tests': [logs[name]['last_line'] for name in ('author-tests-rerun.txt', 'candidate-existing-regression.txt', 'independent-extra-tests-final.txt')]}))
