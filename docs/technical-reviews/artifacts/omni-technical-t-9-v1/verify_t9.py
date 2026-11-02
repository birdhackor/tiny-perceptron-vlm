"""T.9 bounded CPU software inspection; raw checkpoints/results stay outside Git."""

import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from tiny_perceptron.data import render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.training import save_checkpoint, seed_everything

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-technical-t-9-v1')
WORK = CACHE / 'execution-v3'
SUMMARY = Path(__file__).with_name('execution-summary-v1.json')
SNAPSHOT = ROOT / 'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.9.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(content)


def main():
    torch.set_num_threads(1)
    WORK.mkdir(parents=True, exist_ok=True)
    for name in ['scripts', '.venv']:
        link = WORK / name
        if not link.exists():
            link.symlink_to(ROOT / name, target_is_directory=True)
    commands = []

    def run(command, name):
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        completed = subprocess.run(shlex.split(command), cwd=WORK, env=env, text=True,
                                   capture_output=True, timeout=35)
        raw = WORK / 'raw' / (name + '.txt')
        write_new(raw, completed.stdout + completed.stderr)
        commands.append({'command': command, 'cwd': str(WORK), 'returncode': completed.returncode,
                         'raw_path': str(raw), 'raw_sha256': sha(raw)})
        if completed.returncode:
            raise RuntimeError(f'{name} failed: {completed.stderr[:600]}')
        return completed.stdout

    run('.venv/bin/python scripts/prepare_data.py --kind attributes-sft --seed 42', 'prepare')
    parts = {split: [json.loads(line) for line in (WORK / f'data/generated/attributes-sft/{split}.jsonl').read_text().splitlines()]
             for split in ['train', 'validation', 'test']}
    seed_everything(42)
    model = TinyLM(ModelConfig(width=32, layers=1, heads=1, max_length=128))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x, y = render_chat(parts['train'][0]['messages'])
    optimizer.zero_grad()
    loss = masked_loss(model(x[None])['logits'], y[None])
    loss.backward()
    optimizer.step()
    checkpoint = WORK / 'checkpoints/attributes.pt'
    save_checkpoint(checkpoint, model, optimizer=optimizer, step=1,
                    metadata={'seed': 42, 'scope': 'one-update software fixture; not T.4 500-step training or quality evidence'})
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    source_hash = sha(checkpoint)
    text = SNAPSHOT.read_text()
    blocks = re.findall(r'```bash\n(.*?)```', text, flags=re.S)
    for block in blocks:
        for command in block.strip().splitlines():
            if any(('scripts/' + tool + '.py') in command for tool in ['quantize', 'infer', 'evaluate']):
                run(command, f'command-{len(commands):02}')
    assert sha(checkpoint) == source_hash

    # Execute the exact Python block embedded in the frozen lesson.
    inline = re.findall(r'```python\n(.*?)```', text, flags=re.S)
    assert len(inline) == 1
    inline_file = WORK / 'outputs/size_check.py'
    write_new(inline_file, inline[0])
    size_stdout = run('.venv/bin/python outputs/size_check.py', 'inline-size-check')
    assert len(size_stdout.strip().splitlines()) == 3

    # No fixed recipe training: inspect only its CLI.
    run('.venv/bin/python -m scripts.course_experiments.run --help', 'fixed-recipe-help')

    states = {}
    conversion = {}
    for bits in [4, 8]:
        path = WORK / f'checkpoints/attributes-int{bits}.pt'
        packed = torch.load(path, map_location='cpu', weights_only=True)
        state = packed['model']
        states[bits] = state
        layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
        weights = {name + '.weight' for name in layers}
        retained = [key for key in payload['model'] if key not in weights]
        assert all(torch.equal(payload['model'][key], state[key]) for key in retained)
        assert all(state[name + '.values'].dtype == (torch.uint8 if bits == 4 else torch.int8) for name in layers)
        assert all(state[name + '.scale'].dtype == torch.float32 for name in layers)
        expected_codes = sum((payload['model'][name + '.weight'].numel() + 1) // 2 if bits == 4
                             else payload['model'][name + '.weight'].numel() for name in layers)
        observed_codes = sum(state[name + '.values'].numel() * state[name + '.values'].element_size() for name in layers)
        assert expected_codes == observed_codes
        conversion[str(bits)] = {
            'source_checkpoint_sha256': source_hash, 'metadata': packed['metadata'],
            'quantized_checkpoint_sha256': sha(path), 'linear_count': len(layers),
            'code_dtype': str(state[layers[0] + '.values'].dtype),
            'code_bytes': observed_codes,
            'scale_bytes': sum(state[name + '.scale'].numel() * 4 for name in layers),
            'retained_item_count': len(retained), 'retained_items_bitwise_equal': True,
            'model_tensor_bytes': sum(t.numel() * t.element_size() for t in state.values()),
            'file_bytes': path.stat().st_size,
        }

    evaluations = {}
    for split in ['validation', 'test']:
        entries = {}
        for variant in ['fp32', 'int4', 'int8']:
            path = WORK / f'outputs/attributes-{variant}-{split}.json'
            report = json.loads(path.read_text())
            assert report['declared_split'] == split
            assert report['limit'] == 'all' and report['unselected_records'] == 0
            assert report['generation_evaluated_records'] == len(parts[split])
            assert report['generation_skipped_records'] == 0
            assert [s['row'] for s in report['samples']] == list(range(len(parts[split])))
            assert all('generated_ids' in s and 'generation_status' in s for s in report['samples'])
            entries[variant] = {'sha256': sha(path), 'path': str(path),
                                **{key: report[key] for key in ['records_read', 'generation_evaluated_records',
                                   'generation_skipped_records', 'effective_tokens', 'metric_denominators',
                                   'exact_match', 'completed_exact_match', 'eos_rate']}}
        evaluations[split] = entries

    probe = 'color=blue;shape=circle;pitch=low;color?'
    matches = [{'split': split, 'row': i, 'family': record['family'],
                'question': record['messages'][0]['content'], 'target': record['messages'][-1]['content']}
               for split, rows in parts.items() for i, record in enumerate(rows)
               if record['messages'][0]['content'] == probe]
    assert matches == [{'split': 'test', 'row': 7, 'family': 'blue:circle:low', 'question': probe, 'target': 'blue'}]
    float_tensor = torch.tensor([0.125, 0.375, -0.625], dtype=torch.float32)
    rounded = float_tensor.round()
    assert rounded.dtype == float_tensor.dtype and rounded.numel() * rounded.element_size() == 12

    summary = {
        'created_at': datetime.now(UTC).isoformat(), 'reviewer_task': '/root/omni_technical_coordinator/technical_t_9',
        'scope': 'bounded CPU interface/format smoke; one synthetic AdamW update only; no claim about trained accuracy, speed or memory',
        'environment': {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                        'torch_git_version': torch.version.git_version, 'device': 'cpu',
                        'platform': platform.platform(), 'torch_num_threads': '1'},
        'snapshot_sha256': sha(SNAPSHOT), 'inline_code_sha256': sha(inline_file),
        'fixture': {'config': asdict(model.config), 'seed': 42, 'optimizer_updates': 1,
                    'training_records': 1, 'effective_training_targets': int((y != -100).sum()),
                    'checkpoint_sha256': source_hash, 'optimizer_present': payload['optimizer'] is not None,
                    'rng_outside_model': 'torch_rng' in payload,
                    'model_tensor_bytes': sum(t.numel() * t.element_size() for t in payload['model'].values()),
                    'file_bytes': checkpoint.stat().st_size},
        'commands': commands, 'conversions': conversion,
        'inline_size_stdout': size_stdout, 'evaluations': evaluations,
        'probe_membership': matches,
        'data': {'generator_sha256': sha(ROOT / 'scripts/prepare_data.py'),
                 'manifest_sha256': sha(WORK / 'data/generated/attributes-sft/manifest.json'),
                 'splits': json.loads((WORK / 'data/generated/attributes-sft/manifest.json').read_text())['splits']},
        'rounding_example': {'input_dtype': str(float_tensor.dtype), 'output_dtype': str(rounded.dtype),
                              'before_tensor_bytes': 12, 'after_tensor_bytes': 12},
        'prior_harness_attempt': {'path': str(CACHE / 'attempt1-failure.txt'),
                                  'sha256': sha(CACHE / 'attempt1-failure.txt'),
                                  'resolution': 'Expected actual generation_status field; original raw outputs preserved.'},
        'prior_harness_attempt2': {'path': str(CACHE / 'attempt2-failure.txt'),
                                   'sha256': sha(CACHE / 'attempt2-failure.txt'),
                                   'resolution': 'Current prepare_data row is 7, distinct from historical fixed-recipe row2 ordering.'},
    }
    write_new(SUMMARY, json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'summary': str(SUMMARY), 'sha256': sha(SUMMARY), 'commands': len(commands),
                      'probe_membership': matches, 'model_tensor_bytes': {
                          'fp32': summary['fixture']['model_tensor_bytes'],
                          **{bits: item['model_tensor_bytes'] for bits, item in conversion.items()}},
                      'scope': summary['scope']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
