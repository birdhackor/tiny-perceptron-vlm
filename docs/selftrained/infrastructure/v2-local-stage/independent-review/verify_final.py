"""Independent bounded rerun on the final frozen wrapper; no production data."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

BASE = Path('/tmp/p5-local-wrapper-independent')
REPO = Path('/workspace/selftrained-v2')
PYTHON = '/workspace/tiny-perceptron-vlm/.venv/bin/python'
SCRIPT = BASE / 'frozen-final/train_local_stage.py'
DATA = BASE / 'data'
RUNS = BASE / 'final-runs'
RUNS.mkdir()
results = []

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(name, extra, expected=0, manifest=BASE / 'manifest.json', pin=None, repo=REPO):
    output = RUNS / name
    command = [PYTHON, str(SCRIPT), '--repo-root', str(repo), '--manifest', str(manifest),
               '--manifest-sha256', sha(manifest) if pin is None else pin, '--data-root', str(DATA), '--',
               '--architecture', 'moe', '--batch-size', '16', '--context', '512', '--learning-rate', '0.0002',
               '--seed', '20261006', '--eval-every', '1', '--save-every', '1', '--threads', '1', '--device', 'cpu',
               '--output-dir', str(output), *extra]
    actual = subprocess.run(command, cwd='/tmp', capture_output=True, text=True, timeout=120)
    raw_path = BASE / (name + '.final.raw.log')
    raw_path.write_text(actual.stdout + actual.stderr)
    item = {'name': name, 'command': command, 'expected_exit': expected, 'actual_exit': actual.returncode,
            'output_created': output.exists(), 'raw_log_sha256': sha(raw_path)}
    if expected == 0:
        execution = json.loads((output / 'execution.json').read_text())
        receipt = json.loads((output / 'receipt.json').read_text())
        training = json.loads((output / 'train-receipt.json').read_text())
        assert execution['status'] == 'completed' and execution['returncode'] == 0
        assert execution['wrapper_sha256'] == sha(SCRIPT)
        assert all(receipt[key] == value for key, value in execution.items())
        for entry in receipt['files']:
            path = output / entry['path']
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        assert training['completed_requested_steps'] is True and training['interrupted'] is False
        rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        item.update(saved_step=training['steps'], new_train_rows=sum(row.get('event') == 'train' for row in rows),
                    source=execution['parent'], all_actual_output_hashes_verified=True)
    else:
        assert not output.exists(), name
    assert actual.returncode == expected, name
    results.append(item)
    (BASE / 'final-independent-actual-results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({'name': name, 'actual_exit': actual.returncode, 'output_created': output.exists()}), flush=True)
    return output

pre = run('pretrain', ['--stage', 'pretrain', '--steps', '1'])
sft = run('sft', ['--stage', 'sft', '--steps', '1', '--init-checkpoint', str(pre / 'best.pt')])
run('latest-exact-resume', ['--stage', 'sft', '--steps', '2', '--resume', str(sft / 'latest.pt')])
run('reject-best-resume', ['--stage', 'sft', '--steps', '2', '--resume', str(sft / 'best.pt')], expected=1)
run('reject-record-abbreviation', ['--stage', 'pretrain', '--steps', '1', '--record', str(DATA / 'undeclared.jsonl')], expected=2)
run('reject-undeclared-assets', ['--stage', 'pretrain', '--steps', '1'], expected=1, manifest=BASE / 'manifest-omitted-assets.json')
run('reject-manifest-pin', ['--stage', 'pretrain', '--steps', '1'], expected=1, pin='0' * 64)
run('reject-modified-source', ['--stage', 'pretrain', '--steps', '1'], expected=1, repo=BASE / 'tampered-source')
copied = BASE / 'tampered-artifact'
shutil.copytree(sft, copied)
with (copied / 'latest.pt').open('ab') as handle:
    handle.write(b'independent artifact tamper')
run('reject-tampered-latest', ['--stage', 'sft', '--steps', '2', '--resume', str(copied / 'latest.pt')], expected=1)
print(json.dumps({'checks': len(results), 'new_cpu_optimizer_steps': sum(item.get('new_train_rows', 0) for item in results),
                  'final_wrapper_sha256': sha(SCRIPT), 'results_sha256': sha(BASE / 'final-independent-actual-results.json')}), flush=True)
