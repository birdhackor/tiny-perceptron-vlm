"""Reviewer controls for frozen metadata and local real bytes; never predict or publish."""
import contextlib
import hashlib
import importlib.util
import io
import json
import pickle
import platform
import shutil
import subprocess
import tempfile
import tomllib
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.omni_revision import core_final, cpu_final_execution, public_demo, public_download

ROOT = Path(__file__).resolve().parents[5]
SNAPSHOT = Path('/workspace/work/omni-P5-full-tools-review-source-v3')
MODELS = Path('/workspace/work/omni-models-v1')
ARCHIVES = Path('/workspace/work/omni-core-early-artifacts-40988311100')
PROVIDER = Path('/workspace/work/omni-public-delivery-review-current-v2/provider-snapshots')
BASE = Path('docs/course-revision-20261029-omni')
OUT = Path(__file__).with_name('scope333-metadata-controls-result-v1.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def reject(name, function):
    try:
        function()
    except ValueError as error:
        return {'case': name, 'rejected': True, 'error': str(error)}
    raise AssertionError('accepted negative: ' + name)


inventory_path = BASE / 'reviews/P5-tools/scope-inventory-v3.json'
inventory = read(ROOT / inventory_path)
files = inventory['checked_file_scope']
assert len(files) == 333 and sum(row['bytes'] for row in files) == 38971359
for row in files:
    for root in (SNAPSHOT, ROOT):
        path = root / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']

location_path = BASE / 'infrastructure/final-evaluation-evidence-locations-v2.json'
locations = read(ROOT / location_path)
for row in locations['canonical_files']:
    path = ROOT / row['canonical_path']
    assert sha(path) == row['canonical_sha256'] == row['sha256']
    assert path.stat().st_size == row['bytes']
for reference in (locations['previous_map'], locations['portable_index']):
    assert sha(ROOT / reference['path']) == reference['sha256']
old_test = next(row for row in locations['canonical_files'] if row['archive_path'] == 'tests/test_omni_final_measurements.py')
old_git = subprocess.check_output(['git', 'show', '0d806a62410bc94483cadee15d93b9cd2423a230:tests/test_omni_final_measurements.py'], cwd=MODELS)
assert old_git == (ROOT / old_test['canonical_path']).read_bytes()

audit_path = BASE / 'infrastructure/cpu-final-execution-audit-v1.json'
audit = read(ROOT / audit_path)
expected = {name: sha(ROOT / name) for name in cpu_final_execution.source_paths()}
assert len(expected) == 98 and audit['files'] == expected
scope_names = {row['path'] for row in files}
assert set(expected) <= scope_names
for row in [audit['original_review'], *audit['evidence']]:
    assert sha(ROOT / row['path']) == row['sha256']
    assert (ROOT / row['path']).stat().st_size == row['bytes']

lock = tomllib.loads((ROOT / 'uv.lock').read_text())
project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
assert lock['requires-python'] == project['project']['requires-python']
packages = lock['package']
package_names = {row['name'] for row in packages}
artifact_count = 0
hosts = set()
for row in packages:
    for dependency in row.get('dependencies', []):
        assert dependency['name'] in package_names
    for artifact in [*row.get('wheels', []), *([row['sdist']] if 'sdist' in row else [])]:
        assert len(artifact['hash']) == 71 and artifact['hash'].startswith('sha256:')
        assert int(artifact['hash'][7:], 16) >= 0 and artifact['size'] > 0
        parsed = urllib.parse.urlparse(artifact['url'])
        assert parsed.scheme == 'https'
        hosts.add(parsed.hostname)
        artifact_count += 1
torch_rows = [{'version': row['version'], 'source': row['source']} for row in packages if row['name'] == 'torch']
assert len(torch_rows) == 5

spec = importlib.util.spec_from_file_location('p5_visuals_scope333', ROOT / 'scripts/build_visuals.py')
visuals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visuals)
with tempfile.TemporaryDirectory(prefix='p5-scope333-visuals-') as directory:
    private = Path(directory)
    shutil.copytree(ROOT / 'course/figures', private / 'course/figures')
    metadata = Path('docs/lesson-products/results/publication-spec-v1.json')
    (private / metadata).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / metadata, private / metadata)
    before = {path.name: sha(path) for path in (private / 'course/figures').glob('*.svg')}
    visuals.ROOT = private
    visuals.DIRECTORY = private / 'course/figures'
    with contextlib.redirect_stdout(io.StringIO()):
        visuals.build()
    after = {path.name: sha(path) for path in (private / 'course/figures').glob('*.svg')}
    assert before == after and len(before) == 215
    assert (private / 'course/figures/index.json').read_bytes() == (ROOT / 'course/figures/index.json').read_bytes()

forward_attempts = []
pickle_attempts = []


def forbid_forward(module, *args, **kwargs):
    forward_attempts.append(type(module).__name__)
    raise RuntimeError('review forbids NN module calls')


def forbid_pickle(*args, **kwargs):
    pickle_attempts.append(True)
    raise RuntimeError('review forbids continuation deserialization')


def forbid_network(*args, **kwargs):
    raise RuntimeError('review forbids network transport')


def local_fetch(value, asset, target):
    source = ARCHIVES / asset['name']
    assert sha(source) == asset['sha256'] and source.stat().st_size == asset['bytes']
    shutil.copyfile(source, target)


selected_path = ROOT / 'docs/omni-products/v1/model-selection.json'
selected = read(selected_path)
selection_controls = []
with patch.object(torch.nn.Module, '_call_impl', forbid_forward), patch.object(torch, 'load', forbid_pickle), patch.object(pickle, 'load', forbid_pickle), patch.object(pickle, 'loads', forbid_pickle), patch.object(urllib.request, 'urlopen', forbid_network):
    # Positively self-test the forward trap, then count only actual integration attempts.
    try:
        torch.nn.Identity()(torch.tensor(1.0))
    except RuntimeError:
        pass
    assert forward_attempts == ['Identity']
    forward_attempts.clear()
    with patch.object(core_final, 'ROOT', MODELS), patch.object(core_final, 'FROZEN', ROOT / 'docs/omni-products/v1'):
        actual_selection = core_final.checked_selection(MODELS / 'outputs/omni-formal-core-v1', sha(selected_path))
        assert actual_selection == selected
        for name in ['missing_formal_job', 'wrong_dev_winner', 'foreign_weight_binding', 'non_grid_T', 'feasible_without_threshold', 'foreign_receipt']:
            with tempfile.TemporaryDirectory(prefix='p5-selection-negative-') as directory:
                private = Path(directory)
                for path in (ROOT / 'docs/omni-products/v1').glob('*.json'):
                    shutil.copyfile(path, private / path.name)
                bad = json.loads(json.dumps(selected))
                if name == 'missing_formal_job':
                    del bad['all_formal_jobs']['av-additive-s11']
                elif name == 'wrong_dev_winner':
                    bad['selected']['av'] = 'av-additive-s11'
                elif name == 'foreign_weight_binding':
                    bad['model_files']['av']['weights']['sha256'] = 'f' * 64
                elif name == 'non_grid_T':
                    bad['temperature'] = 1.23456789
                elif name == 'feasible_without_threshold':
                    bad['threshold'] = None
                else:
                    bad['all_formal_jobs']['av-additive-s11']['job_result_sha256'] = 'f' * 64
                (private / 'model-selection.json').write_text(json.dumps(bad))
                with patch.object(core_final, 'FROZEN', private):
                    selection_controls.append(reject(name, lambda: core_final.checked_selection(MODELS / 'outputs/omni-formal-core-v1', sha(private / 'model-selection.json'))))
    descriptor = public_download.descriptor(ROOT / 'docs/omni-products/v1/public-release.json')
    provider_release = read(PROVIDER / 'provider-release-gh.json')
    ledger_wrapper = read(PROVIDER / 'provider-ledger-response.json')
    ledger = json.loads(ledger_wrapper['structuredContent']['content'])
    row = next(row for row in ledger['attempts'] if row['attempt_id'] == 'cpu-core-40988311100-1')
    assert row['status'] == 'completed' and row['receipt']['success'] is True
    assert provider_release['draft'] is True and provider_release['published_at'] is None
    assert row['receipt']['release_id'] == provider_release['id']
    assert row['receipt']['release_tag'] == provider_release['tag_name'] == descriptor['release_tag']
    ledger_assets = {row['name']: row for row in row['receipt']['assets']}
    provider_assets = {row['name']: row for row in provider_release['assets']}
    archive_checks = []
    for kind, asset in descriptor['components'].items():
        original = ledger_assets[asset['name']]
        official = provider_assets[asset['name']]
        assert asset['asset_id'] == original['id'] == official['id']
        assert asset['bytes'] == original['bytes'] == official['size']
        assert asset['sha256'] == original['sha256'] == official['digest'].removeprefix('sha256:')
        with zipfile.ZipFile(ARCHIVES / asset['name']) as archive:
            members = archive.infolist()
            bindings = json.loads(archive.read('package-file-manifest.json'))
            assert len(members) == len({member.filename for member in members}) == len(bindings) + 1
            for name, binding in bindings.items():
                raw = archive.read(name)
                assert len(raw) == binding['bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256']
        archive_checks.append({'kind': kind, 'name': asset['name'], 'sha256': sha(ARCHIVES / asset['name']), 'inner_payloads': len(bindings)})
    with tempfile.TemporaryDirectory(prefix='p5-real-selected-install-') as directory, patch.object(public_download, 'fetch', local_fetch):
        private = Path(directory)
        receipt = public_download.download(ROOT / 'docs/omni-products/v1/public-release.json', private / 'infer', include_control=True)
        assert set(receipt['assets']) == public_download.KINDS and not list((private / 'infer').rglob('*.pt'))
        assert public_download.download(ROOT / 'docs/omni-products/v1/public-release.json', private / 'infer', include_control=True) == receipt
        assistant, actual_descriptor = public_demo.assistant(private / 'infer', ROOT / 'docs/omni-products/v1/public-release.json')
        control = core_final.load_trained(private / 'infer/core/c-compact/training', 'choice')
        assert type(control).__name__ == 'ChoiceModel' and actual_descriptor['_model_selection'] == selected
        full = public_download.download(ROOT / 'docs/omni-products/v1/public-release.json', private / 'full', include_control=True, full_state=True)
        assert len(list((private / 'full').rglob('*.pt'))) == 10
    assert forward_attempts == [] and pickle_attempts == []

opened = read(ROOT / 'docs/omni-products/v1/final-opening.json')
assert len(opened['evaluation_sources_sha256']) == 57
assert opened['evaluation_sources_sha256'] == core_final.evaluation_sources()
assert opened['model_selection_sha256'] == sha(selected_path)
assert opened['independent_audit_sha256'] == sha(core_final.AUDIT)
assert opened['runtime'] == {'device': 'cpu', 'torch': str(torch.__version__), 'python': platform.python_version()}
assert opened['final_opened'] is True and datetime.fromisoformat(opened['created_at']) > datetime.fromisoformat(selected['created_at'])
for key, name in [('data_definition_sha256', 'data-definition.json'), ('metrics_sha256', 'metrics.json')]:
    assert opened[key] == sha(ROOT / 'docs/omni-products/v1' / name)

result = {
    'reviewer_task': '/root/omni_p5_full_tools_review', 'created_at': datetime.now(UTC).isoformat(),
    'inventory_sha256': sha(ROOT / inventory_path), 'all333_snapshot_and_live_bytes_match': True,
    'v2_historical_map': {'canonical_entries_verified': len(locations['canonical_files']), 'v1_map_preserved': True, 'exact_old_test_equal_true_0d806_Git_object': True},
    'cpu_wrapper_audit': {'all98_sources_exact': True, 'all98_within333': True, 'original_and_evidence_hashes_verified': True, 'actual_production_commit_and_dispatch_tested': False},
    'lockfile': {'packages': len(packages), 'dependency_targets_exist': True, 'all_artifact_hashes_sizes_https_verified': artifact_count, 'hosts': sorted(hosts), 'torch_variants': torch_rows, 'actual_native_all_platform_install_claimed': False},
    'figures': {'private_build': True, 'all215_SVG_bytes_preserved': True, 'index_exact_bytes_equal': True, 'index_sha256': sha(ROOT / 'course/figures/index.json')},
    'real_selection': {'all10_formal_jobs_weights_and_receipts_checked_with_actual_loader': True, 'selected': selected['selected'], 'temperature': selected['temperature'], 'threshold': selected['threshold'], 'negative_controls': selection_controls},
    'real_local_public': {'preserved_provider_snapshot_only': True, 'release_draft': True, 'selected_archives': archive_checks, 'all5_inference_install_cache_load_compose_pass': True, 'full5_opaque_copy_with10_pt_pass': True, 'anonymous_network_success_claimed': False},
    'opening': {'sha256': sha(ROOT / 'docs/omni-products/v1/final-opening.json'), 'all57_source_hashes_and_independent_audit_exact': True, 'backend_exact': True, 'created_after_selection': True, 'opening_created_by_this_reviewer': False, 'opening_is_not_NN_execution': True},
    'forward_trap_positive_control': 'Identity blocked; counter reset before integration', 'actual_integration_forward_attempts': forward_attempts, 'continuation_deserialization_attempts': pickle_attempts,
    'limits': ['No final predictions, training, generated examples, anonymous HTTP delivery, live release publication, workflow dispatch, provider mutation, or 369-page publication gate.', 'Actual selected weights/archives and source snapshots stay outside Git. Local transport receipts do not prove anonymous delivery. Six semantic negatives use fresh private copies and explicit mutated metadata.']
}
with OUT.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
