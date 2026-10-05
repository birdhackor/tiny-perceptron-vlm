"""Fresh reviewer CPU checks for the complete 12.1 fence and exercise."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import hashlib
import json
import math
import re
import sys
import torch
from tiny_perceptron.multimodal import tone, log_mel, AudioEncoder
from tiny_perceptron.data import ByteTokenizer

torch.set_num_threads(1)
raw = Path('course/chapters/12.md').read_text()
section = raw[raw.index('## 12.1 '):raw.index('## 12.2 ')]
fence = re.search(r'```python\n(.*?)```', section, re.S).group(1)
results = {
    'environment': {'python': sys.version, 'torch': torch.__version__,
                    'torch_git': torch.version.git_version, 'device': 'cpu',
                    'default_dtype': str(torch.get_default_dtype()), 'threads': 1},
    'prediction_before_execution': {
        '440': {'shape': [1600], 'first3_rounded': [0.0, 0.086, 0.169], 'rounded_extrema': [-0.5, 0.5], 'cycles_per_0_1s': 44},
        '880': {'shape': [1600], 'first3_rounded': [0.0, 0.169, 0.319], 'rounded_extrema': [-0.5, 0.5], 'cycles_per_0_1s': 88}},
    'tone': {},
}
for frequency in (440, 880):
    code = fence if frequency == 440 else fence.replace('440.0', '880.0')
    namespace = {}
    output = StringIO()
    with redirect_stdout(output):
        exec(compile(code, f'course/chapters/12.md#12.1-{frequency}', 'exec'), namespace)
    wave = namespace['wave']
    scalar = [0.5 * math.sin(2 * math.pi * frequency * n / 16000) for n in range(1600)]
    error = max(abs(a - b) for a, b in zip(wave.tolist(), scalar, strict=True))
    record = {
        'executed_exact_fence': frequency == 440,
        'exercise_edit': 'replace only 440.0 with 880.0' if frequency == 880 else 'none',
        'stdout': output.getvalue(), 'shape': list(wave.shape), 'dtype': str(wave.dtype),
        'sample0_1_2': wave[:3].tolist(), 'scalar_double_sample0_1_2': scalar[:3],
        'min': wave.min().item(), 'max': wave.max().item(),
        'independent_scalar_max_abs_error': error,
        'upward_zero_crossings_including_start': int(((wave[:-1] <= 0) & (wave[1:] > 0)).sum()),
        'sample1_time_seconds': 1 / 16000, 'sample1_cycle_fraction': frequency / 16000,
        'last_sample_time_seconds': 1599 / 16000,
        'nominal_record_duration_seconds': len(wave) / 16000,
    }
    expected = results['prediction_before_execution'][str(frequency)]
    assert record['shape'] == expected['shape']
    assert [round(v, 3) for v in wave[:3].tolist()] == expected['first3_rounded']
    assert [round(record['min'], 2), round(record['max'], 2)] == expected['rounded_extrema']
    assert record['upward_zero_crossings_including_start'] == expected['cycles_per_0_1s']
    assert error < 4e-5, error
    assert torch.equal(wave, tone(float(frequency))), 'generator must be deterministic'
    results['tone'][str(frequency)] = record

results['continuous_phase_derivation'] = [
    {'cycle_fraction': u, 'amplitude': 0.5 * math.sin(2 * math.pi * u)}
    for u in (0, 0.25, 0.5, 0.75, 1)]
for actual, expected in zip(results['continuous_phase_derivation'], [0, 0.5, 0, -0.5, 0], strict=True):
    assert abs(actual['amplitude'] - expected) < 1e-12
wave = tone()
batch = torch.stack([wave, wave])
stereo = torch.stack([wave, -wave])
assert tuple(batch.shape) == (2, 1600)
assert tuple(stereo.shape) == (2, 1600)
features = log_mel(batch)
encoded = AudioEncoder()(batch)
results['axes_and_frontend'] = {
    'mono': list(wave.shape), 'equal_length_batch': list(batch.shape),
    'stereo_with_channel_first_convention': list(stereo.shape),
    'channel_time_concatenation_shape': list(torch.cat([wave, -wave]).shape),
    'log_mel_batch_band_time_shape': list(features.shape),
    'AudioEncoder_batch_frame_feature_shape': list(encoded.shape),
    'torchaudio_or_whisper_imported': any(x.startswith(('torchaudio', 'whisper')) for x in sys.modules),
}
assert not results['axes_and_frontend']['torchaudio_or_whisper_imported']

result_path = Path('docs/course-experiments/results/real_modal.json')
run = json.loads(result_path.read_text())
fsdd = run['results']['fsdd']
splits = fsdd['data']['splits']
summary = {}
for name, split in splits.items():
    records = split['records']
    families = [r['family'] for r in records]
    speakers = sorted({f.split('_')[1] for f in families})
    digits = sorted({r['answer'] for r in records})
    assert len(records) == split['count'] == 20
    assert digits == list('0123456789')
    assert all(r['answer'] == r['family'].split('_')[0] for r in records)
    summary[name] = {'count': len(records), 'speakers': speakers, 'digits': digits}
assert set(summary['train']['speakers']).isdisjoint(summary['validation']['speakers'])
assert set(summary['train']['speakers']).isdisjoint(summary['test']['speakers'])
assert set(summary['validation']['speakers']).isdisjoint(summary['test']['speakers'])
assert fsdd['training']['steps'] == len(fsdd['training']['history']) == 250
assert [x['step'] for x in fsdd['training']['history']] == list(range(1, 251))
assert sum(x['effective_targets'] for x in fsdd['training']['history']) == fsdd['training']['effective_tokens'] == 2000
assert fsdd['training']['weights_changed'] and fsdd['training']['nonzero_gradient_seen']
evaluation = {}
tok = ByteTokenizer()
for name in ('validation', 'test'):
    saved = fsdd[name]
    samples = saved['samples']
    assert [r['family'] for r in samples] == [r['family'] for r in splits[name]['records']]
    correct = 0
    for row in samples:
        ids = row['generated_ids']
        raw_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        match = raw_ids == tok.encode(row['target'])
        assert match == row['exact_match']
        assert tok.decode(raw_ids) == row['generated']
        correct += match
    assert correct == saved['correct']
    assert len(samples) == saved['examples'] == 20
    assert correct / len(samples) == saved['exact_match']
    evaluation[name] = {'n': len(samples), 'independently_recomputed_correct': correct,
                        'fraction': correct / len(samples),
                        'rule': 'generated token IDs before first EOS exactly equal ByteTokenizer.encode(target)'}
results['introduction_original_run_record_check'] = {
    'record_path': str(result_path), 'record_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
    'seed': run['seed'], 'recorded_training_device': run['device'],
    'recorded_torch_version': run['torch_version'],
    'recorded_updates': fsdd['training']['steps'], 'recorded_effective_training_targets': fsdd['training']['effective_tokens'],
    'splits': summary, 'evaluation': evaluation,
    'scope': 'CPU validation of original saved run records and split identities; no GPU training or checkpoint-quality replication.'}
print(json.dumps(results, ensure_ascii=False, indent=2))
