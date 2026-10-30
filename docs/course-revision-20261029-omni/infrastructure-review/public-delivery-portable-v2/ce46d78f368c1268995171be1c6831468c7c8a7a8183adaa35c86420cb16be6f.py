"""Real selected archives and safetensors; forbid NN execution and network."""
from pathlib import Path
from unittest.mock import patch
import datetime
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import zipfile

import torch

ROOT = Path('/workspace/work/omni-models-v1')
E = Path('/workspace/work/omni-public-delivery-review-current-v2')
ARCHIVES = Path('/workspace/work/omni-core-early-artifacts-40988311100')
PROVIDER = E / 'provider-snapshots'
sys.path.insert(0, str(ROOT))
from scripts.omni_revision import public_download as delivery, public_demo as demo
from tiny_perceptron.omni_products import inference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


manifest_path = ROOT / 'docs/omni-products/v1/public-release.json'
selection_path = ROOT / 'docs/omni-products/v1/model-selection.json'
codec_path = ROOT / 'docs/omni-products/v1/codec-selection.json'
checked = delivery.descriptor(manifest_path)
selection = checked['_model_selection']
assert sha(manifest_path) == '12282d37c33bd3769aba75d199bf26110976e3c46cc0206c6941e3614a6c166e'
assert sha(selection_path) == checked['model_selection']['sha256'] == '935594413eda401f4fddc81b57f0245cd6a28219dda19e4e27b9d0f751385e20'
assert sha(codec_path) == selection['codec_selection_sha256']
committed = {}
for path in (selection_path, codec_path):
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(['git', 'show', '671b5340a54c3dcb87d65494b12a124affb46a83:' + relative], cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest() == sha(path)
    committed[relative] = sha(path)

# These are preserved provider responses, not a new network query by this reviewer.
ledger_raw = json.loads((PROVIDER / 'provider-ledger-response.json').read_bytes())
ledger = json.loads(ledger_raw['structuredContent']['content'])
rows = [r for r in ledger['attempts'] if r['attempt_id'] == 'cpu-core-40988311100-1']
assert len(rows) == 1 and rows[0]['status'] == 'completed'
receipt = rows[0]['receipt']
release = json.loads((PROVIDER / 'provider-release-gh.json').read_bytes())
assert release['id'] == receipt['release_id'] == 428573191
assert release['tag_name'] == receipt['release_tag'] == checked['release_tag']
assert release['target_commitish'] == receipt['source_revision'] == rows[0]['revision']
assert receipt['attempt'] in release['body'] and receipt['source_revision'] in release['body']
assert release['draft'] is True and release['published_at'] is None
assert receipt['success'] is True and receipt['final_opened'] is False
ledger_assets = {a['name']: a for a in receipt['assets']}
release_assets = {a['name']: a for a in release['assets']}
assert len(receipt['assets']) == len(ledger_assets) == len(release['assets']) == len(release_assets) == 11
assert ledger_assets.keys() == release_assets.keys()
all_assets = []
for name, asset in ledger_assets.items():
    official = release_assets[name]
    source = ARCHIVES / name
    assert asset['id'] == official['id'] and asset['bytes'] == official['size'] == source.stat().st_size
    assert official['digest'] == 'sha256:' + asset['sha256']
    assert sha(source) == asset['sha256'] and asset['roundtrip_verified'] is True
    all_assets.append({'name': name, 'id': asset['id'], 'bytes': asset['bytes'], 'sha256': asset['sha256']})

inner = {}
for kind, asset in checked['components'].items():
    origin = ledger_assets[asset['name']]
    assert asset['job'] == selection['selected'][kind]
    assert asset['asset_id'] == origin['id'] and asset['sha256'] == origin['sha256'] and asset['bytes'] == origin['bytes']
    with zipfile.ZipFile(ARCHIVES / asset['name']) as archive:
        members = archive.namelist()
        bindings = json.loads(archive.read('package-file-manifest.json'))
        assert set(members) == set(bindings) | {'package-file-manifest.json'}
        assert len(members) == len(set(members))
        for name, binding in bindings.items():
            raw = archive.read(name)
            assert len(raw) == binding['bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256']
        # Expected selected identity is reconstructed directly from actual selection.
        job = selection['selected'][kind]
        expected = {
            f'core/{job}/training/weights.safetensors': selection['model_files'][kind]['weights'],
            f'core/{job}/training/training-result.json': {'sha256': selection['model_files'][kind]['training_result_sha256']},
            f'core/{job}/job-result.json': {'sha256': selection['all_formal_jobs'][job]['job_result_sha256']},
        }
        assert delivery.selected_files(selection, kind) == expected
        for name, binding in expected.items():
            for key, value in binding.items():
                assert bindings[name][key] == value
        inner[kind] = {'job': job, 'members': len(members), 'verified_files': bindings, 'selected3': expected}

forward_attempts = []
guard_control = []


class ForbiddenForward(RuntimeError):
    pass


def forbid_forward(self, *args, **kwargs):
    forward_attempts.append(type(self).__name__)
    raise ForbiddenForward('NN execution forbidden during delivery integration review')


def forbidden_execution(*args, **kwargs):
    raise AssertionError('network / optimizer deserialization forbidden')


fetches = []


def local_transport(value, asset, target):
    # Exact original bytes, no substitution of archive/manifest/loader/weights.
    source = ARCHIVES / asset['name']
    assert sha(source) == asset['sha256'] and source.stat().st_size == asset['bytes']
    fetches.append(asset['name'])
    shutil.copyfile(source, target)


original_parser = inference.load_file
parses = []


def actual_parser(path, device):
    parses.append({'path': str(path), 'sha256': sha(Path(path)), 'device': device})
    return original_parser(path, device=device)


cases = []
outputs = {}
with patch.object(torch.nn.Module, '_call_impl', forbid_forward), \
     patch.object(torch, 'load', forbidden_execution), \
     patch.object(pickle, 'load', forbidden_execution), \
     patch.object(pickle, 'loads', forbidden_execution), \
     patch.object(delivery.urllib.request, 'urlopen', forbidden_execution), \
     patch.object(delivery, 'fetch', local_transport), \
     patch.object(inference, 'load_file', actual_parser):
    try:
        torch.nn.Identity()(torch.zeros(1))
    except ForbiddenForward:
        guard_control.extend(forward_attempts)
    else:
        raise AssertionError('forward guard not active')
    assert guard_control == ['Identity']
    forward_attempts.clear()
    for label, control, full in [('default4', False, False), ('inference5', True, False), ('full5', True, True)]:
        output = E / ('real-install-' + label)
        assert not output.exists()
        result = delivery.download(manifest_path, output, control, full)
        assert len(result['assets']) == (5 if control else 4)
        assert delivery.download(manifest_path, output, control, full) == result
        for kind in result['assets']:
            delivery.verify_installed(output, selection, kind)
            for name, binding in inner[kind]['verified_files'].items():
                path = output / name
                if name.endswith('.pt') and not full:
                    assert not path.exists()
                else:
                    assert path.is_file() and not path.is_symlink()
                    assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256']
        outputs[label] = output
        cases.append({'case': label, 'components': list(result['assets']), 'exact_cache_reuse': True,
                      'receipt_sha256': sha(output / 'download-receipt.json'), 'pt_copied_only_if_requested': True})
    # Real constructors, safe parser, metadata checks and composer, no stubs.
    product, published = demo.assistant(outputs['inference5'], manifest_path)
    control = inference.load_trained(outputs['inference5'] / 'core' / selection['selected']['choice_control'] / 'training', 'choice')
    models = {'av': product.av, 'choice': product.c.model, 'codec': product.codec, 'talker': product.talker, 'choice_control': control}
    assert isinstance(product, inference.OmniAssistant)
    assert product.c.temperature == selection['temperature'] and product.c.threshold == selection['threshold']
    models_report = {}
    for kind, model in models.items():
        assert model.training is False
        parameters = list(model.parameters())
        assert all(parameter.dtype == torch.float32 and parameter.device.type == 'cpu' and not parameter.requires_grad for parameter in parameters)
        models_report[kind] = {'type': type(model).__name__, 'parameters': sum(p.numel() for p in parameters), 'fp32_cpu_eval_frozen': True}
    assert len(parses) == 5
    assert forward_attempts == []

report = {
    'schema': 'omni-independent-real-public-artifact-integration-current-v2',
    'reviewer_task': '/root/omni_final_definition_review',
    'completed_at_utc': datetime.datetime.now(datetime.UTC).isoformat(),
    'descriptor_sha256': sha(manifest_path), 'model_selection_sha256': sha(selection_path),
    'selections_committed_at_671b5340': committed,
    'preserved_provider_ledger_and_release_all11_match_actual_local_bytes': all_assets,
    'provider_snapshot_release': {'id': release['id'], 'tag': release['tag_name'], 'target': release['target_commitish'], 'draft': release['draft'], 'published_at': release['published_at']},
    'selected5_inner_all_SHA_and_actual_selection3_reconstruction': inner,
    'installation_cases': cases, 'local_transport_fetches': len(fetches),
    'real_safetensors_parser_calls': parses, 'actual_models': models_report,
    'preserved_selection': {'selected': selection['selected'], 'temperature': product.c.temperature, 'threshold': product.c.threshold, 'policy_feasible': selection['policy_feasible']},
    'forward_guard_self_test_blocked': guard_control, 'integration_forward_attempts': forward_attempts,
    'continuations_deserialized': False, 'new_network_or_provider_calls': 0,
    'new_predictions_or_training_or_final_opening_by_reviewer': False,
    'actual_anonymous_live_download_verified': False,
    'scope': 'Exact real draft archive install, selected binding and real safetensors load integration; transport redirected to original local ZIP bytes only. No NN execution, new network, live publication, capability or final inference.'
}
path = E / 'real-artifact-integration-evidence-current-v2.json'
dump_new(path, report)
print(json.dumps({'path': str(path), 'sha256': sha(path), 'actual_models': models_report,
                  'integration_forward_attempts': forward_attempts, 'anonymous_live_verified': False}, indent=2))
