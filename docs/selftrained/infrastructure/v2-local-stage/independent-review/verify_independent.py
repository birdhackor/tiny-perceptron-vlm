"""Bounded independent CPU review of the frozen local-stage wrapper."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

BASE = Path('/tmp/p5-local-wrapper-independent')
REPO = Path('/workspace/selftrained-v2')
PYTHON = '/workspace/tiny-perceptron-vlm/.venv/bin/python'
SCRIPT = BASE / 'frozen/train_local_stage.py'
DATA = BASE / 'data'
RUNS = BASE / 'runs'
DATA.mkdir(exist_ok=True)
RUNS.mkdir(exist_ok=True)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def entry(name):
    path = DATA / name
    return {'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)}

Image.new('L', (64, 32), 240).save(DATA / 'image.png')
sf.write(DATA / 'audio.wav', np.sin(np.arange(1600) * 0.05).astype('float32'), 16000)
rows = []
for split in ('train', 'validation'):
    for task in ('text', 'tool_call', 'vision_clothing', 'ocr', 'voice_qa'):
        row = {'id': split + '-' + task, 'group_id': split + '-' + task,
               'split': split, 'task': task,
               'messages': [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': '2'}]}
        if task == 'vision_clothing':
            row.update(image='image.png', supervision={'vision_labels': [0, 1]})
        if task == 'ocr':
            row.update(image='image.png', roi=[0, 0, 32, 32], supervision={'ocr_text': '大'})
        if task == 'voice_qa':
            row.update(audio='audio.wav', supervision={'intent_id': 0})
        rows.append(row)
(DATA / 'records.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
extra = [dict(row, id='extra-' + row['id'], group_id='extra-' + row['group_id']) for row in rows if row['task'] == 'text']
(DATA / 'undeclared.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in extra))
manifest = {'schema_version': 1, 'initialization': 'random',
            'model_config': {'width': 16, 'layers': 1, 'heads': 2, 'kv_heads': 1, 'ffn_hidden': 32,
                             'experts': 2, 'top_k': 1, 'max_length': 512, 'backend': 'sdpa'},
            'records': [entry('records.jsonl')], 'assets': [entry('image.png'), entry('audio.wav')]}
MANIFEST = BASE / 'manifest.json'
write(MANIFEST, manifest)
write(BASE / 'manifest-omitted-assets.json', dict(manifest, assets=[]))
results = []

def run(name, extra_args, manifest_path=MANIFEST, repo=REPO):
    output = RUNS / name
    command = [PYTHON, str(SCRIPT), '--repo-root', str(repo), '--manifest', str(manifest_path),
               '--manifest-sha256', sha(manifest_path), '--data-root', str(DATA), '--',
               '--architecture', 'moe', '--batch-size', '16', '--context', '512', '--learning-rate', '0.0002',
               '--seed', '20261006', '--eval-every', '1', '--save-every', '1', '--threads', '1', '--device', 'cpu',
               '--output-dir', str(output), *extra_args]
    completed = subprocess.run(command, cwd='/tmp', capture_output=True, text=True, timeout=120)
    (BASE / (name + '.raw.log')).write_text(completed.stdout + completed.stderr)
    item = {'name': name, 'command': command, 'exit_code': completed.returncode, 'output_created': output.exists(),
            'raw_log_sha256': sha(BASE / (name + '.raw.log'))}
    if (output / 'execution.json').is_file():
        item['execution'] = json.loads((output / 'execution.json').read_text())
    if (output / 'train-receipt.json').is_file():
        item['training'] = json.loads((output / 'train-receipt.json').read_text())
    results.append(item)
    write(BASE / 'independent-actual-results.json', results)
    print(json.dumps({'name': name, 'exit_code': completed.returncode, 'output_created': output.exists()}), flush=True)
    return output

pre = run('pretrain', ['--stage', 'pretrain', '--steps', '1'])
sft = run('sft', ['--stage', 'sft', '--steps', '1', '--init-checkpoint', str(pre / 'best.pt')])
run('resume-selected-best', ['--stage', 'sft', '--steps', '2', '--resume', str(sft / 'best.pt')])
run('undeclared-record-abbreviation', ['--stage', 'pretrain', '--steps', '1', '--record', str(DATA / 'undeclared.jsonl')])
run('undeclared-assets', ['--stage', 'pretrain', '--steps', '1'], BASE / 'manifest-omitted-assets.json')
tampered = BASE / 'tampered-source'
tampered.mkdir()
gitdir = subprocess.check_output(['git', 'rev-parse', '--absolute-git-dir'], cwd=REPO, text=True).strip()
(tampered / '.git').write_text('gitdir: ' + gitdir + '\n')
tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD', 'scripts/selftrained', 'tiny_perceptron', 'pyproject.toml', 'uv.lock'], cwd=REPO, text=True).splitlines()
for relative in tracked:
    target = tampered / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / relative, target)
with (tampered / 'scripts/selftrained/train.py').open('a') as handle:
    handle.write('\n# independent review source tamper test\n')
run('reject-modified-source', ['--stage', 'pretrain', '--steps', '1'], repo=tampered)
print(json.dumps({'result_path': str(BASE / 'independent-actual-results.json'), 'frozen_script_sha256': sha(SCRIPT)}), flush=True)
