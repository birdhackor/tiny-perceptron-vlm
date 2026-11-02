"""19.10 independent bounded CPU verification; no training or quality benchmark."""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import safetensors
import torch
from safetensors.torch import load_file

from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig


ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-technical-technical_19_10-v1')
OUTPUT = Path(__file__).with_name('execution.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numel(module):
    return sum(p.numel() for p in module.parameters())


def tensor_info(value):
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': value.numel() * value.element_size()}


torch.set_num_threads(1)
torch.manual_seed(20261029)
freeze = json.loads((ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json').read_bytes())
page = next(p for p in freeze['pages'] if p['page_id'] == '19.10')
snapshot = ROOT / page['snapshot']
body = snapshot.read_text()
current = re.search(r'^## 19\.10 .*?(?=^## |\Z)', (ROOT / page['source']).read_text(), re.M | re.S)[0]
assert sha(snapshot) == page['source_sha256'] == hashlib.sha256(current.encode()).hexdigest()
assert page['figures_sha256'] == {}
inline = re.findall(r'```python\n(.*?)```', body, re.S)
assert len(inline) == 1
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(inline[0], str(snapshot) + ':inline-code', 'exec'), {})
assert stdout.getvalue() == 'MoE FP32參數bytes 21788428 約MiB 20.78\nDense FP32參數bytes 9152268 約MiB 8.73\n'

result = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_19_10-v1/verify.py',
    'reviewer_task': '/root/omni_technical_coordinator/technical_19_10',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'torch_git_version': torch.version.git_version, 'safetensors': safetensors.__version__,
                    'device': 'cpu', 'threads': '1', 'seed': '20261029'},
    'source_sha256': sha(snapshot),
    'inline_code_sha256': hashlib.sha256(inline[0].encode()).hexdigest(),
    'inline_stdout': stdout.getvalue(),
    'float_element_bytes': {'float32': torch.empty((), dtype=torch.float32).element_size(),
                            'float16': torch.empty((), dtype=torch.float16).element_size()},
    'models': {},
    'scope': 'Exact inline arithmetic and frozen public tiny-model storage/forward checks only. No training, latency/RAM-peak measurement, or quality evaluation.',
}

for arch, expected_count, expected_bytes, expected_sha in [
    ('moe', 5_447_107, 22_366_908, '26102d2edbd4574bef6733cba9e0dfd039118b206babdc66dbd16d1e57032d8c'),
    ('dense', 2_288_067, 9_725_076, '5f8b5b0404560ed12c7733dc234c710beb968a97fd81b0227a1a0c3ec8fe8bcf'),
]:
    config_path = CACHE / f'{arch}-model-config.json'
    manifest_path = CACHE / f'{arch}-inference-manifest.json'
    weights_path = CACHE / f'{arch}-model.safetensors'
    config = json.loads(config_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    for name in ['model-config.json', 'model.safetensors', 'tokenizer.json']:
        assert sha(CACHE / f'{arch}-{name}') == manifest['files'][name]
    assert weights_path.stat().st_size == expected_bytes
    assert sha(weights_path) == expected_sha
    raw = weights_path.read_bytes()
    header_length = int.from_bytes(raw[:8], 'little')
    header = json.loads(raw[8:8 + header_length])
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    assert {v['dtype'] for v in tensors.values()} == {'F32'}
    intervals = sorted(v['data_offsets'] for v in tensors.values())
    assert intervals[0][0] == 0
    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    payload_bytes = sum(b - a for a, b in intervals)
    assert payload_bytes == len(raw) - 8 - header_length
    assert all(v['data_offsets'][1] - v['data_offsets'][0] == math.prod(v['shape']) * 4 for v in tensors.values())
    state = load_file(weights_path, device='cpu')
    assert all(torch.isfinite(t).all() for t in state.values())
    assert torch.equal(state['lm.embedding.weight'], state['lm.output.weight'])
    model = LimitedAssistant(SelftrainedConfig(**config)).eval()
    assert set(state) == set(model.state_dict())
    assert all(t.shape == model.state_dict()[k].shape and t.dtype == model.state_dict()[k].dtype for k, t in state.items())
    loaded = model.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert numel(model) == expected_count
    assert model.lm.embedding.weight is model.lm.output.weight
    assert model.lm.embedding.weight.shape == (550, 256)
    duplicate_bytes = state['lm.embedding.weight'].numel() * state['lm.embedding.weight'].element_size()
    assert payload_bytes == expected_count * 4 + duplicate_bytes
    model_pointers = {k: t.data_ptr() for k, t in model.named_parameters()}
    assert model.lm.embedding.weight.data_ptr() != state['lm.embedding.weight'].data_ptr()
    with torch.no_grad():
        first = model(torch.tensor([[10, 11, 12]], dtype=torch.long))
        second = model(torch.tensor([[13]], dtype=torch.long), cache=first['cache'])
        image_tokens, _ = model.vision_encoder(torch.zeros(2, 1, 32, 32), torch.tensor([[.25, .5], [.75, .5]]))
        ocr_tokens, _ = model.ocr_encoder(torch.zeros(1, 32, 128))
        audio_tokens, _ = model.audio_encoder(torch.zeros(17, 40))
    assert model_pointers == {k: t.data_ptr() for k, t in model.named_parameters()}
    cache_bytes = lambda out: sum(t.numel() * t.element_size() for pair in out['cache'] for t in pair)
    assert first['logits'].shape == (1, 3, 550) and second['logits'].shape == (1, 1, 550)
    details = {
        'config': config,
        'manifest_sha256': sha(manifest_path),
        'weights_sha256': sha(weights_path),
        'unique_parameters': numel(model),
        'components': {'language': numel(model.lm), 'image': numel(model.vision_encoder),
                       'ocr': numel(model.ocr_encoder), 'audio': numel(model.audio_encoder)},
        'all_parameters_float32': all(p.dtype == torch.float32 for p in model.parameters()),
        'all_parameters_cpu': all(p.device.type == 'cpu' for p in model.parameters()),
        'all_parameter_pointers_retained_after_forward': True,
        'raw_parameter_bytes': expected_count * 4,
        'raw_parameter_mib': expected_count * 4 / 1024**2,
        'parameters_in_millions': expected_count / 1_000_000,
        'ideal_two_byte_parameter_bytes': expected_count * 2,
        'ideal_two_byte_parameter_mib': expected_count * 2 / 1024**2,
        'ideal_two_byte_ratio': (expected_count * 2) / (expected_count * 4),
        'file_bytes': len(raw), 'file_mib': len(raw) / 1024**2,
        'tensor_entries': len(tensors), 'serialized_dtype': 'F32',
        'serialized_tensor_elements': sum(math.prod(v['shape']) for v in tensors.values()),
        'serialized_payload_bytes': payload_bytes,
        'header_json_bytes': header_length, 'header_length_prefix_bytes': 8,
        'embedding_output_tied_in_model': True,
        'embedding_output_equal_in_file': True,
        'embedding_shape': list(model.lm.embedding.weight.shape),
        'one_embedding_duplicate_bytes': duplicate_bytes,
        'one_embedding_duplicate_mib': duplicate_bytes / 1024**2,
        'embedding_file_data_offsets': {k: tensors[k]['data_offsets'] for k in ['lm.embedding.weight', 'lm.output.weight']},
        'loaded_model_has_separate_storage_from_file_tensor': True,
        'logits': tensor_info(first['logits']),
        'kv_cache_first_key': tensor_info(first['cache'][0][0]),
        'kv_cache_bytes_three_tokens': cache_bytes(first),
        'kv_cache_bytes_four_tokens': cache_bytes(second),
        'synthetic_modality_features': {k: tensor_info(v) for k, v in [('image', image_tokens), ('ocr', ocr_tokens), ('audio', audio_tokens)]},
        'final_training_objective': {'policy': manifest['language_objective_policy']['version'],
                                     'tool': manifest['tool_loss_weight'], 'numeric': manifest['numeric_run_loss_weight'],
                                     'native_voice': manifest.get('native_voice_loss_weight', 1.0)},
        'initialization_origin': manifest['origin'],
        'selected_checkpoint_sha256': manifest['selected_checkpoint_sha256'],
    }
    if arch == 'moe':
        assert all(block.ffn.experts.__len__() == 4 for block in model.lm.blocks)
        assert all(route['chosen'].shape == (1, 3, 2) and route['counts'].sum().item() == 6 for route in first['routing'])
        experts = [p for k, p in model.named_parameters() if '.experts.' in k]
        with torch.no_grad():
            probe = model.lm.blocks[0].ffn(torch.eye(256)[None])[2]['chosen'][0]
        pairs = torch.unique(probe.sort(dim=-1).values, dim=0)
        assert len(pairs) > 1
        details['moe_routing'] = {'experts_per_layer': 4, 'selected_per_token_per_layer': 2,
                                   'layers': 4, 'resident_expert_parameters': sum(p.numel() for p in experts),
                                   'resident_expert_parameter_bytes': sum(p.numel() * p.element_size() for p in experts),
                                   'first_layer_probe_distinct_top2_pairs': pairs.tolist(),
                                   'first_forward_chosen_per_layer': [r['chosen'].tolist() for r in first['routing']]}
    result['models'][arch] = details

tree = json.loads((CACHE / 'public-v2-tree.json').read_bytes())
files = [entry for entry in tree if entry['type'] == 'file']
assert len(files) == 16
result['release_inventory'] = {'revision': '979cdfacc588ad0536f1c64fff96f264571cf054',
                                'files': len(files), 'paths': [x['path'] for x in files],
                                'sha256': sha(CACHE / 'public-v2-tree.json'),
                                'pagination_link': None,
                                'no_final_quantized_or_distilled_export_in_v2_release': True}
assert result['models']['moe']['final_training_objective'] != result['models']['dense']['final_training_objective']
assert result['models']['moe']['selected_checkpoint_sha256'] != result['models']['dense']['selected_checkpoint_sha256']
result['result'] = 'All assertions passed; exact lesson output, counts, FP32 payloads, duplicate embedding, header arithmetic, and bounded synthetic CPU forwards verified.'
with OUTPUT.open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'output': str(OUTPUT.relative_to(ROOT)), 'sha256': sha(OUTPUT),
                  'inline_stdout': result['inline_stdout'],
                  'parameters': {k: v['unique_parameters'] for k, v in result['models'].items()},
                  'all_assertions_passed': True}, ensure_ascii=False))
