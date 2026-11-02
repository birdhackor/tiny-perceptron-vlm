"""T.8 frozen CLI verification; full logs/checkpoints stay outside Git."""

import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path('/workspace/tiny-perceptron-vlm')
CACHE = Path('/workspace/work/reference/omni-technical-t-8-v1')
WORK = CACHE / 'sandbox'
OUT = ROOT / 'docs/technical-reviews/artifacts/omni-technical-t-8-v1'
SNAPSHOT = ROOT / 'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.8.md'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(argv, name):
    completed = subprocess.run(argv, cwd=WORK, capture_output=True, text=True, timeout=45)
    raw = CACHE / f'{name}.log'
    raw.write_text(completed.stdout + completed.stderr)
    record = {'argv': argv, 'cwd': str(WORK), 'returncode': completed.returncode,
              'raw_path': str(raw), 'raw_sha256': digest(raw), 'raw_bytes': raw.stat().st_size}
    if completed.returncode:
        raise RuntimeError(json.dumps(record))
    return completed.stdout, record


def main():
    assert not torch.cuda.is_available(), 'CPU-only review must never select GPU'
    assert not torch.backends.mps.is_available()
    torch.set_num_threads(1)
    WORK.mkdir(parents=True, exist_ok=True)
    if not (WORK / 'scripts').exists():
        (WORK / 'scripts').symlink_to(ROOT / 'scripts', target_is_directory=True)
    env = {'python': platform.python_version(), 'torch': str(torch.__version__),
           'torch_git': torch.version.git_version, 'device': 'cpu',
           'platform': platform.platform(), 'OMP_NUM_THREADS': os.getenv('OMP_NUM_THREADS', ''),
           'MKL_NUM_THREADS': os.getenv('MKL_NUM_THREADS', '')}
    blocks = re.findall(r'```bash\n(.*?)```', SNAPSHOT.read_text(), re.S)
    cli = [line.strip() for block in blocks for line in block.splitlines() if line.strip()]
    short_cli = cli[:7]
    assert len(short_cli) == 7
    summary = {'created_at': datetime.now(UTC).isoformat(), 'environment': env,
               'snapshot_sha256': digest(SNAPSHOT),
               'execution_policy': 'Exact seven frozen local-command argument vectors; only interpreter changed to the uv-selected Python. Isolated working directory. No GPU/paid work. No fixed-course GPU recipe rerun.',
               'commands': [], 'training': {}, 'evaluation': {}}
    for i, line in enumerate(short_cli):
        argv = shlex.split(line)
        assert argv[0] == '.venv/bin/python'
        argv[0] = sys.executable
        stdout, record = run(argv, f'command-{i}')
        record['textbook_command'] = line
        summary['commands'].append(record)
        if i == 0:
            manifest = json.loads(stdout)
            assert {k: v['records'] for k, v in manifest['splits'].items()} == {'train': 9, 'validation': 1, 'test': 2}
            summary['dataset'] = manifest
        elif i <= 3:
            result = json.loads(stdout.strip().splitlines()[-1])
            name = ['baseline', 'rms', 'moe'][i - 1]
            assert result['device'] == 'cpu' and result['mode'] == 'train'
            assert len(result['history']) == 200 and result['history'][-1]['step'] == 200
            assert (result['config']['width'], result['config']['layers'], result['config']['heads']) == (32, 1, 1)
            ckpt = WORK / 'checkpoints' / f'{name}.pt'
            payload = torch.load(ckpt, map_location='cpu', weights_only=True)
            assert payload['step'] == 200
            summary['training'][name] = {'config': result['config'], 'parameters': result['parameters'],
                'steps': len(result['history']), 'batch_size': payload['metadata']['batch_size'],
                'effective_tokens': sum(q['effective_tokens'] for q in result['history']),
                'first_loss': result['history'][0]['loss'], 'last_loss': result['history'][-1]['loss'],
                'seconds': result['seconds'], 'checkpoint': str(ckpt), 'checkpoint_sha256': digest(ckpt),
                'checkpoint_bytes': ckpt.stat().st_size}
        else:
            name = ['baseline', 'rms', 'moe'][i - 4]
            path = WORK / 'outputs' / f'{name}-validation.json'
            result = json.loads(path.read_text())
            assert result['records_read'] == result['records_selected'] == result['generation_evaluated_records'] == 1
            assert result['limit'] == 'all' and not result['skipped']
            data = [json.loads(q) for q in (WORK / 'data/generated/toy-text/validation.jsonl').read_text().splitlines()]
            expected_count = sum(len(r['text'].encode()) + 1 for r in data)
            assert result['effective_tokens'] == expected_count
            from tiny_perceptron.data import ByteTokenizer
            from tiny_perceptron.training import load_checkpoint
            model, _ = load_checkpoint(WORK / 'checkpoints' / f'{name}.pt', 'cpu')
            tok = ByteTokenizer()
            ids = torch.tensor([[tok.bos_id] + tok.encode(data[0]['text']) + [tok.eos_id]])
            with torch.no_grad():
                logits = model(ids[:, :-1])['logits']
                nll = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten()).item()
            assert abs(nll - result['mean_token_nll']) < 1e-6
            assert all(len(s['generated_ids']) <= 32 for s in result['samples'])
            keys = ['mean_token_nll', 'effective_tokens', 'records_read', 'records_selected',
                    'loss_evaluated_records', 'generation_evaluated_records', 'skipped', 'metric_denominators', 'samples']
            summary['evaluation'][name] = {k: result[k] for k in keys}
            summary['evaluation'][name].update(independent_mean_nll=nll, nll_absolute_error=abs(nll-result['mean_token_nll']),
                                              result_path=str(path), result_sha256=digest(path))
    help_stdout, help_record = run([sys.executable, '-m', 'scripts.course_experiments.run', '--help'], 'runner-help')
    assert '--experiment ID' in help_stdout and '{cpu,cuda}' in help_stdout
    summary['runner_help'] = help_record
    summary['checks_passed'] = True
    path = OUT / 'cpu-result.json'
    assert not path.exists(), 'Do not overwrite saved evidence'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'result': str(path.relative_to(ROOT)), 'sha256': digest(path), 'checks_passed': True,
                      'parameters': {n: r['parameters'] for n,r in summary['training'].items()},
                      'mean_token_nll': {n: r['mean_token_nll'] for n,r in summary['evaluation'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
