"""T.6 independent, bounded CPU verification. Models and input fixtures stay outside Git."""

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import PIL
import soundfile as sf
import torch

from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM, scene, tone
from tiny_perceptron.training import load_checkpoint, save_checkpoint, seed_everything


ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
WORK = Path('/workspace/work/reference/omni-technical-t-6-v1/local-cpu')
WORK.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
ENV = os.environ.copy()
ENV.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
result = {
    'started_at': datetime.now(UTC).isoformat(),
    'scope': 'Short teaching CPU only; no fixed GPU experiment, Modal, publication, or general quality conclusion.',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'torch_git_version': torch.version.git_version, 'device': 'cpu',
                    'Pillow': PIL.__version__, 'soundfile': sf.__version__, 'threads': '2'},
    'commands': [],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(label, args, expected=0):
    command = [sys.executable, *map(str, args)]
    completed = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=60)
    log = ART / f'{label}.log'
    log.write_text(completed.stdout + completed.stderr, encoding='utf-8')
    row = {'label': label, 'command': shlex.join(command), 'returncode': completed.returncode,
           'expected_returncode': expected, 'log': str(log.relative_to(ROOT)), 'log_sha256': sha(log)}
    if completed.stdout.strip():
        try:
            row['last_json'] = json.loads(completed.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            row['last_line'] = completed.stdout.strip().splitlines()[-1]
    result['commands'].append(row)
    if completed.returncode != expected:
        raise RuntimeError(f'{label}: {completed.returncode}, expected {expected}: {completed.stderr[-500:]}')
    print(label, completed.returncode, flush=True)
    return row.get('last_json')


# Execute the linked T.4 dialogue route using precisely its data generator, seed,
# model configuration, 500 updates, and CPU. Only output paths are relocated.
run('prepare-attributes', ['scripts/prepare_data.py', '--kind', 'attributes-sft', '--seed', 42,
                           '--output', WORK / 'data'])
seed_everything(42)
save_checkpoint(WORK / 'start.pt', TinyLM(ModelConfig(width=32, layers=1, heads=1, max_length=128)),
                step=0, metadata={'seed': 42, 'note': 'untrained baseline'})
run('t4-dialogue-500', ['scripts/train.py', '--task', 'sft', '--data', WORK / 'data/attributes-sft/train.jsonl',
                       '--checkpoint', WORK / 'start.pt', '--seed', 42, '--device', 'cpu', '--train',
                       '--steps', 500, '--output', WORK / 'attributes.pt'])

for modality in ('vision', 'audio'):
    run(f'pretrain-{modality}-300', ['scripts/pretrain_encoders.py', '--modality', modality, '--train',
                                   '--steps', 300, '--device', 'cpu', '--output', WORK / f'{modality}-encoder.pt'])
run('pretrain-vision-dry', ['scripts/pretrain_encoders.py', '--modality', 'vision', '--steps', 300,
                           '--device', 'cpu', '--output', WORK / 'must-not-exist.pt'])
result['dry_run_did_not_save'] = not (WORK / 'must-not-exist.pt').exists()

for task in ('vision', 'audio', 'joint'):
    args = ['scripts/train.py', '--task', task, '--checkpoint', WORK / 'attributes.pt']
    if task != 'audio':
        args += ['--vision-encoder', WORK / 'vision-encoder.pt']
    if task != 'vision':
        args += ['--audio-encoder', WORK / 'audio-encoder.pt']
    args += ['--freeze', 'partial' if task == 'joint' else 'projector', '--train', '--steps', 500,
             '--device', 'cpu', '--output', WORK / f'{task}.pt']
    run(f'train-{task}-500', args)

inference = {
    'vision': ['--color', 'blue', '--shape', 'circle', '--prompt', 'shape?'],
    'audio': ['--frequency', 880, '--prompt', 'pitch?'],
    'joint': ['--color', 'red', '--shape', 'square', '--frequency', 220, '--prompt', 'joint?'],
}
for task, args in inference.items():
    run(f'infer-{task}', ['scripts/infer_modal.py', WORK / f'{task}.pt', *args, '--tokens', 16, '--device', 'cpu'])
run('reject-text-evaluate', ['scripts/evaluate.py', WORK / 'vision.pt', '--mode', 'sft', '--device', 'cpu',
                             '--output', WORK / 'not-a-text-result.json'], expected=1)

# Make the first/last distinction observable beyond the one-layer T.4 model.
seed_everything(42)
save_checkpoint(WORK / 'three-block-start.pt', TinyLM(ModelConfig(width=32, layers=3, heads=1, max_length=128)))
freeze = {}
for scope in ('projector', 'partial', 'none'):
    out = WORK / f'freeze-{scope}.pt'
    run(f'freeze-{scope}', ['scripts/train.py', '--task', 'vision', '--checkpoint', WORK / 'three-block-start.pt',
                           '--freeze', scope, '--train', '--steps', 1, '--device', 'cpu', '--output', out])
    model, payload = load_checkpoint(out)
    names = payload['trainable_parameters']
    freeze[scope] = {
        'trainable_names': names,
        'format': payload['format_version'],
        'checkpoint_keys': sorted(payload),
        'step': payload['step'],
        'optimizer_present': payload['optimizer'] is not None,
        'unused_audio_optimizer_state_count': sum(
            1 for index,n in zip(payload['optimizer']['param_groups'][0]['params'], names, strict=True)
            if n.startswith(('audio.', 'audio_projector.')) and index in payload['optimizer']['state']),
    }
    if scope == 'projector':
        assert all(n.startswith(('image_projector.', 'audio_projector.')) for n in names)
    if scope == 'partial':
        assert any(n.startswith('language.blocks.0.') for n in names)
        assert any(n.startswith('language.blocks.2.') for n in names)
        assert not any(n.startswith('language.blocks.1.') for n in names)
    if scope == 'none':
        assert len(names) == len(list(model.named_parameters()))
result['freeze_inspection'] = freeze

# Compare interrupted versus uninterrupted joint training with a fixed total
# schedule. This verifies only this short deterministic CPU route.
for label, extra in [('resume-full', []), ('resume-part', ['--stop-after', 2])]:
    run(label, ['scripts/train.py', '--task', 'joint', '--checkpoint', WORK / 'attributes.pt',
                '--vision-encoder', WORK / 'vision-encoder.pt', '--audio-encoder', WORK / 'audio-encoder.pt',
                '--freeze', 'partial', '--train', '--steps', 4, '--device', 'cpu',
                '--output', WORK / f'{label}.pt', *extra])
run('resume-rest', ['scripts/train.py', '--task', 'joint', '--checkpoint', WORK / 'resume-part.pt', '--resume',
                    '--freeze', 'partial', '--train', '--steps', 4, '--device', 'cpu',
                    '--output', WORK / 'resume-rest.pt'])
full = torch.load(WORK / 'resume-full.pt', weights_only=True)
resumed = torch.load(WORK / 'resume-rest.pt', weights_only=True)
result['resume_comparison'] = {
    'full_step': full['step'], 'resumed_step': resumed['step'],
    'all_model_tensors_equal': all(torch.equal(v, resumed['model'][n]) for n,v in full['model'].items()),
    'max_abs_difference': max(float((v-resumed['model'][n]).abs().max()) for n,v in full['model'].items()),
}
assert result['resume_comparison']['all_model_tensors_equal']
run('resume-reload-encoder-rejected', ['scripts/train.py', '--task', 'joint', '--checkpoint', WORK / 'resume-part.pt',
                                      '--resume', '--vision-encoder', WORK / 'vision-encoder.pt', '--train',
                                      '--steps', 4, '--device', 'cpu'], expected=1)

# Local OCR data and all three ablations: software behavior only, no OCR quality
# claim. Use the T.6 toy vision checkpoint, which has not learned digit reading.
run('prepare-ocr', ['scripts/prepare_ocr.py', '--output', WORK / 'ocr'])
for ablation in ('none', 'blank', 'shuffle'):
    run(f'ocr-ablation-{ablation}', ['scripts/evaluate_modal.py', WORK / 'vision.pt', '--data', WORK / 'ocr/test.jsonl',
                                    '--ablation', ablation, '--limit', 3, '--tokens', 4, '--device', 'cpu',
                                    '--output', WORK / f'ocr-{ablation}.json'])
result['ocr_samples'] = {a:json.loads((WORK / f'ocr-{a}.json').read_text())
                         for a in ('none', 'blank', 'shuffle')}
result['ocr_donors_target_preserved'] = all(
    s['donor_row'] != s['row'] and s['target'] == result['ocr_samples']['none']['samples'][s['row']]['target']
    for s in result['ocr_samples']['shuffle']['samples'])

# File input shape/rate validation from the exact local data loader.
from tiny_perceptron.modal_data import modal_example
import numpy as np
record = {'image': 'ocr/images/12-0.png', 'question':'read digits', 'answer':'12'}
# Pick a known generated record instead of assuming which split contains 12.
record = json.loads((WORK / 'ocr/test.jsonl').read_text().splitlines()[0])
x,y,image,wave = modal_example(record, WORK/'ocr', 'vision')
result['local_image_input'] = {'shape':list(image.shape),'dtype':str(image.dtype),
                               'question_in_input': True, 'path_relative_to_jsonl': True}
rate_checks = {}
for rate,channels,length in [(16000,1,1600),(8000,1,800),(16000,2,1600),(16000,1,0)]:
    path=WORK/f'audio-{rate}-{channels}-{length}.wav'
    values=np.zeros((length,) if channels==1 else (length,channels),dtype='float32')
    sf.write(path,values,rate)
    try:
        _,_,_,w=modal_example({'audio':path.name,'question':'pitch?','answer':'low'},WORK,'audio')
        rate_checks[path.name]={'accepted':True,'samples':w.numel()}
    except ValueError as error:
        rate_checks[path.name]={'accepted':False,'error':str(error)}
result['audio_file_validation'] = rate_checks
result['tone'] = {'frequency_Hz':880,'sample_rate_Hz':16000,'samples':tone(880).numel(),
                  'seconds':0.1,'image_shape':list(scene('blue','circle').shape)}

# Safe CLI help only for the fixed long route; no invocation of formal training.
run('fixed-experiment-help', ['-m', 'scripts.course_experiments.run', '--help'])
result['finished_at'] = datetime.now(UTC).isoformat()
(ART/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'result':'completed','commands':len(result['commands']),
                  'resume':result['resume_comparison'],
                  'artifact':str(ART/'verification.json')},ensure_ascii=False))
