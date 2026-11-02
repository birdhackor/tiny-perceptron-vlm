"""19.22 technical review: execute the frozen lesson's actual Python block."""
import ast
import contextlib
import datetime
import hashlib
import inspect
import io
import json
import os
import platform
import re
import sys
import wave
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/reference/omni-technical-19-22-v1')
FREEZE = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json'
freeze = json.loads(FREEZE.read_text())
page = next(p for p in freeze['pages'] if p['page_id'] == '19.22')
raw = (ROOT / page['snapshot']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == page['source_sha256']
blocks = re.findall(r'```python\n(.*?)\n```', raw.decode(), re.S)
assert len(blocks) == 1
code = blocks[0] + '\n'
expected_text = re.search(r'```text\n(.*?)\n```', raw.decode(), re.S)[1]
records = {}
for name, replacement in [('baseline', None), ('unknown', 'unknown'), ('unsupported', 'triangle')]:
    actual_code = code
    if replacement is not None:
        needle = 'a_result = {"semantic": "circle", "source": "manual_pipeline_fixture"}'
        assert actual_code.count(needle) == 1
        actual_code = actual_code.replace(needle, needle.replace('"circle"', json.dumps(replacement)))
    destination = CACHE / name
    destination.mkdir(parents=True, exist_ok=True)
    namespace = {'__name__': '__lesson_review__'}
    output = io.StringIO()
    old_cwd = os.getcwd()
    try:
        os.chdir(destination)
        with contextlib.redirect_stdout(output):
            exec(compile(actual_code, str(ROOT / page['snapshot']) + f':python:{name}', 'exec'), namespace)
    finally:
        os.chdir(old_cwd)
    ns = namespace
    chain = ns['rows'][0]['chain']
    gold = ns['rows'][0]['gold']
    assert ns['targets']['circle'].tolist() == [1, 1, 0, 2, 2, 3]
    assert ns['targets']['square'].tolist() == [2, 2, 0, 1, 1, 3]
    assert ns['train_inputs'][0].tolist() == [4, 1, 1, 0, 2, 2]
    assert ns['labels'][0].tolist() == [1, 1, 0, 2, 2, 3]
    assert list(ns['logits'].shape) == [2, 6, 4]
    assert not any(p.requires_grad for p in ns['codec'].parameters())
    assert torch.equal(ns['frozen_table'], ns['codec'].table)
    assert all(id(p) != id(ns['codec'].table) for g in ns['optimizer'].param_groups for p in g['params'])
    assert 'targets' not in inspect.signature(ns['generate']).parameters
    assert gold['codes'] == [2] and gold['stop'] == 'EOS' and gold['waveform'].numel() == 800
    row = {'stdout': output.getvalue(), 'gold_codes': gold['codes'], 'gold_stop': gold['stop'],
           'chain_status': chain['status'], 'panel_denominator': len(ns['panel']),
           'chain_denominator': len(ns['rows']), 'loss_before_update': ns['loss'].item(),
           'logits_shape': list(ns['logits'].shape), 'codec_unchanged': True,
           'codec_excluded_from_optimizer': True, 'generate_signature': str(inspect.signature(ns['generate']))}
    assert row['panel_denominator'] == row['chain_denominator'] == 1
    if name == 'baseline':
        assert output.getvalue().startswith(expected_text + '\n')
        assert chain['codes'] == [2] and chain['stop'] == 'EOS'
        waveform = chain['waveform']
        assert waveform.numel() == 800
        assert torch.isfinite(waveform).all()
        frequency = torch.fft.rfftfreq(800, 1 / 8000)[torch.fft.rfft(waveform).abs().argmax()].item()
        assert frequency == 880.0
        row.update({'chain_codes': chain['codes'], 'chain_stop': chain['stop'],
                    'waveform_samples': waveform.numel(), 'duration_seconds': waveform.numel()/8000,
                    'fft_peak_hz': frequency, 'amplitude_max': waveform.abs().max().item()})
        wav_path = destination / 'demo-sound.wav'
        with wave.open(str(wav_path), 'rb') as wav:
            params = {'channels': wav.getnchannels(), 'sample_width_bytes': wav.getsampwidth(),
                      'sample_rate': wav.getframerate(), 'frames': wav.getnframes(), 'compression': wav.getcomptype()}
            assert params == {'channels': 1, 'sample_width_bytes': 2, 'sample_rate': 8000, 'frames': 800, 'compression': 'NONE'}
        row['wav'] = {'path': str(wav_path), 'sha256': hashlib.sha256(wav_path.read_bytes()).hexdigest(), 'parameters': params}
        # Changing gold must leave the actual A-prediction branch unchanged.
        alternate = ns['run_case']({'gold': 'circle', 'a_result': ns['a_result']})
        assert alternate['chain']['codes'] == chain['codes']
        assert torch.equal(alternate['chain']['waveform'], waveform)
        row['gold_change_leaves_chain_unchanged'] = True
        # Target/reference reconstruction is checked separately from free generation.
        reference_codes = ns['targets']['circle'][:-1].tolist()
        reference_wave = ns['codec'].decode(reference_codes)
        row['reference_decode_samples'] = reference_wave.numel()
        assert reference_wave.numel() == 4000
        # Each block is independent: changing a later code cannot alter an earlier prefix.
        assert torch.equal(ns['codec'].decode([1, 2])[:800], ns['codec'].decode([1, 0])[:800])
        row['block_decoder_zero_lookahead_checked'] = True
    else:
        assert chain['status'] == ('transfer' if name == 'unknown' else 'parse_error')
        assert 'waveform' not in chain
        row['waveform_absent'] = True
    records[name] = row

tree = ast.parse(code)
run_case = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_case')
records['inspection'] = {'run_case_ast_sha256': hashlib.sha256(ast.dump(run_case).encode()).hexdigest(),
                         'gold_access_only_for_gold_branch': "run_case uses case['gold'] once to create gold_run; condition comes only from case['a_result'].",
                         'training_configuration': 'seed=0; CPU; SGD lr=0.01; 2 conditions; 6 targets per condition incl EOS; one optimizer.step; no audio/model downloads.'}
result = {'created_at': datetime.datetime.now(datetime.UTC).isoformat(),
          'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-19-22-v1/verify_embedded.py',
          'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'platform': platform.platform(), 'device': 'cpu', 'threads': str(torch.get_num_threads())},
          'freeze': {'path': str(FREEZE.relative_to(ROOT)), 'sha256': hashlib.sha256(FREEZE.read_bytes()).hexdigest()},
          'source_snapshot': {'path': page['snapshot'], 'sha256': hashlib.sha256(raw).hexdigest()},
          'embedded_code_sha256': hashlib.sha256(code.encode()).hexdigest(),
          'result': 'PASS: unmodified embedded baseline and both specified exercise variants executed with all assertions satisfied.',
          'runs': records}
out = HERE / 'embedded-execution-v1.json'
with out.open('x') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(json.dumps({'path': str(out.relative_to(ROOT)), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'result': result['result']}, ensure_ascii=False))
