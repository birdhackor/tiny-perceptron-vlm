"""15.1 independent CPU checks; no optimizer, backward, training, or GPU use."""

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import torch

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import DenseFFN


ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-technical-technical_15_1-v1')
SOURCE = ROOT / 'docs/course-review-20261019/freeze-01/sources/15.1.md'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run_block(code):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, str(SOURCE), 'exec'), namespace)
    return namespace, output.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite evidence: ' + str(args.output))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    raw = SOURCE.read_bytes()
    assert digest(raw) == '08209b8a0bcb4586de9d24243c264bcf5dba65b6d7705edd7a02fa9b1ea85f87'
    blocks = re.findall(r'```python\n(.*?)```', raw.decode(), re.S)
    assert len(blocks) == 1
    code = blocks[0]
    env, printed = run_block(code)
    x, up, down, ffn = (env[key] for key in ('x', 'up', 'down', 'ffn'))
    assert x.dtype == up.dtype == down.dtype == torch.float32
    assert all(t.device.type == 'cpu' for t in (x, up, down))
    assert (x.shape, up.shape, down.shape) == ((2, 2), (2, 3), (3, 2))
    assert torch.equal(ffn(x), torch.tensor([[4., 5.], [5., 4.]]))
    assert up.numel() + down.numel() == 12
    assert printed == ('輸出 [[4.0, 5.0], [5.0, 4.0]]\n'
                       'token數 2 權重數 12\n'
                       '輸出 [[4.0, 5.0], [5.0, 4.0], [4.0, 5.0], [5.0, 4.0]]\n'
                       'token數 4 權重數 12\n')
    weight_ptrs = [up.data_ptr(), down.data_ptr()]
    six_env, six_output = run_block(code.replace('torch.cat([x, x], dim=0)', 'torch.cat([x, x, x], dim=0)'))
    assert six_env['batch'].shape == (6, 2)
    assert torch.equal(six_env['ffn'](six_env['batch']), torch.tensor([[4., 5.], [5., 4.]] * 3))
    neg_code = code.replace('torch.cat([x, x], dim=0)', 'torch.cat([x, x, x], dim=0)').replace(
        'x = torch.tensor([[1.0, 2.0], [2.0, 1.0]])',
        'x = torch.tensor([[1.0, -2.0], [2.0, 1.0]])')
    negative, negative_output = run_block(neg_code)
    neg_x = negative['x']
    neg_up = neg_x @ negative['up']
    neg_relu = torch.relu(neg_up)
    neg_result = negative['ffn'](neg_x)
    assert neg_up[0].tolist() == [1., -2., -1.]
    assert neg_relu[0].tolist() == [1., 0., 0.]
    assert neg_result[0].tolist() == [1., 0.]
    assert ffn(x)[0].tolist() == [4., 5.]
    assert torch.equal(ffn(neg_x)[1], ffn(x)[1])
    assert weight_ptrs == [up.data_ptr(), down.data_ptr()]
    assert torch.relu(torch.tensor([-2., 0., 2.])).tolist() == [0., 0., 2.]

    layer = DenseFFN(2, hidden=3)
    assert layer.activation == 'gelu'
    assert layer.up.bias.shape == (3,) and layer.down.bias.shape == (2,)
    assert sum(p.numel() for p in layer.parameters()) == 17
    with torch.no_grad():
        layer.up.weight.copy_(up.T)
        layer.down.weight.copy_(down.T)
        layer.up.bias.zero_()
        layer.down.bias.zero_()
        equivalent = layer.down(torch.relu(layer.up(x)))
        gelu = layer(x)
    assert torch.equal(equivalent, ffn(x))
    assert not torch.equal(gelu, equivalent)
    wider_up = torch.ones(2, 4)
    try:
        torch.relu(x @ wider_up) @ down
    except RuntimeError:
        one_table_only_rejected = True
    else:
        raise AssertionError('Expected the unmatched hidden dimensions to fail')
    assert (torch.relu(x @ wider_up) @ torch.ones(4, 2)).shape == (2, 2)

    experiment_path = ROOT / 'docs/course-experiments/results/moe.json'
    experiment_raw = experiment_path.read_bytes()
    report = json.loads(experiment_raw)
    result = report['results']['variants']['dense_active_top1']
    config = ModelConfig(**result['model']['config'])
    model = TinyLM(config)
    total_parameters = sum(p.numel() for p in model.parameters())
    ffn_counts = [sum(p.numel() for p in block.ffn.parameters()) for block in model.blocks]
    assert config.width == 64 and config.layers == 2
    assert 64 * 256 + 256 + 256 * 64 + 64 == 33088
    assert ffn_counts == [33088, 33088]
    assert total_parameters == result['model']['parameters'] == 141568
    groups = {
        'token_embedding': sum(p.numel() for p in model.embedding.parameters()),
        'position_embedding': sum(p.numel() for p in model.position.parameters()),
        'block_attention': [sum(p.numel() for p in b.attention.parameters()) for b in model.blocks],
        'block_norms': [sum(p.numel() for p in b.norm1.parameters()) + sum(p.numel() for p in b.norm2.parameters()) for b in model.blocks],
        'block_ffn': ffn_counts,
        'final_norm': sum(p.numel() for p in model.final_norm.parameters()),
        'output': sum(p.numel() for p in model.output.parameters()),
    }
    training = result['training']
    assert training['requested_steps'] == training['steps'] == training['optimizer_updates'] == 180
    assert training['skipped_updates'] == 0 and training['status'] == 'completed'
    heldout = {}
    for split, rounded in [('validation', '2.29999'), ('test', '2.32093')]:
        row = result['heldout'][split]
        recomputed = row['nll_sum'] / row['effective_tokens']
        assert abs(recomputed - row['nll']) < 1e-15
        assert f'{recomputed:.5f}' == rounded
        heldout[split] = {k: row[k] for k in ['records', 'examples', 'nll', 'nll_sum', 'effective_tokens']}
        heldout[split]['recomputed_nll'] = recomputed
        heldout[split]['rounded_five_decimals'] = f'{recomputed:.5f}'

    archive = ROOT / 'assets/training/tinystories-v1.tar.gz'
    archive_raw = archive.read_bytes()
    assert digest(archive_raw) == report['assets'][0]['archive_sha256']
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode='r:gz') as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith('tinystories-train-512.jsonl'))
        records_raw = tar.extractfile(member).read()
    records = [json.loads(line) for line in records_raw.decode().splitlines() if line.strip()]
    for record in records:
        record['family'] = record.get('text_sha256', records_sha256([{'text': record['text']}]))
    data = split_records(records, report['seed'])
    independent_splits = {}
    for split, rows in data.items():
        count = len(rows)
        split_sha = records_sha256(rows)
        assert count == report['results']['dataset'][split]['records']
        assert split_sha == report['results']['dataset'][split]['sha256']
        examples = text_examples(rows, max_length=128)
        targets = sum(int((labels != IGNORE).sum()) for _, labels in examples)
        expected = training['initial'] if split == 'train' else result['heldout'][split]
        assert len(examples) == expected['examples']
        assert targets == expected['effective_tokens']
        independent_splits[split] = {'records': count, 'sha256': split_sha, 'examples': len(examples), 'effective_targets': targets}
    serialized_data = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    dataset_sha = next(a['sha256'] for a in report['artifacts'] if a['path'] == 'dataset.json')
    assert digest(serialized_data) == dataset_sha
    historical = {}
    for path in ['scripts/course_experiments/architecture.py', 'scripts/course_experiments/common.py',
                 'tiny_perceptron/model.py', 'tiny_perceptron/modern.py', 'tiny_perceptron/data.py', 'tiny_perceptron/attention.py']:
        original = subprocess.check_output(['git', 'show', report['revision'] + ':' + path], cwd=ROOT)
        original_sha = digest(original)
        assert original_sha == report['code_sha256'][path]
        historical[path] = {'record_sha256': original_sha, 'current_sha256': digest((ROOT/path).read_bytes())}

    output = {
        'created_at': datetime.datetime.now(datetime.UTC).isoformat(),
        'command': 'uv run --extra cpu python ' + str(Path(__file__).relative_to(ROOT)) + ' --output ' + str(args.output),
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'torch_git_version': str(torch.version.git_version), 'device': 'cpu', 'dtype': 'torch.float32', 'threads': str(torch.get_num_threads()), 'seed': '42'},
        'source': {'path': str(SOURCE.relative_to(ROOT)), 'sha256': digest(raw), 'embedded_python_blocks': 1, 'embedded_code_sha256': digest(code.encode()), 'figure_count': 0},
        'lesson_execution': {'original_stdout': printed, 'six_rows_stdout': six_output, 'negative_stdout': negative_output,
                             'input_shape': list(x.shape), 'up_shape': list(up.shape), 'down_shape': list(down.shape),
                             'up_result': (x @ up).tolist(), 'output': ffn(x).tolist(), 'weight_elements': 12,
                             'negative_up': neg_up[0].tolist(), 'negative_relu': neg_relu[0].tolist(), 'negative_output': neg_result[0].tolist(),
                             'row_independence': True, 'same_weight_objects': True,
                             'scalar_multiplication_terms_for_two_matmuls': {'2_rows': 24, '4_rows': 48, '6_rows': 72},
                             'linear_equivalent_with_relu_and_zero_bias': equivalent.tolist(), 'default_gelu_output': gelu.tolist(),
                             'DenseFFN_2_3_default_parameters': 17, 'one_table_only_hidden_widening_rejected': one_table_only_rejected},
        'historical_record_audit': {'path': str(experiment_path.relative_to(ROOT)), 'sha256': digest(experiment_raw),
                                    'revision': report['revision'], 'variant': 'dense_active_top1',
                                    'historical_environment': {key: report[key] for key in ['device', 'seed', 'torch_version', 'python_version', 'gpu']},
                                    'parameters_recounted': total_parameters, 'parameter_groups': groups,
                                    'training': {key: training[key] for key in ['requested_steps', 'steps', 'optimizer_updates', 'skipped_updates', 'effective_tokens', 'batch_size', 'learning_rate', 'status']},
                                    'heldout': heldout, 'independent_splits': independent_splits,
                                    'dataset_json_sha256_rebuilt': digest(serialized_data), 'archive_sha256': digest(archive_raw),
                                    'historical_code_hashes': historical,
                                    'limit': 'The stored historical run is audited, not retrained. NLL is recomputed from original nll_sum/effective_tokens; logits are not independently re-evaluated without the private checkpoint. This CPU check makes no GPU speed or general language-quality claim.'},
        'original_source_fetch_receipts': json.loads((CACHE/'fetch-receipts-v1.json').read_text()),
        'checkpoint_access_attempt': json.loads((CACHE/'hf-fetch-v1.json').read_text()),
        'result': 'All asserted lesson-code, exercise, shape, parameter, split, denominator, provenance, and reported-NLL arithmetic checks passed; no training or GPU used.'
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': digest(args.output.read_bytes()), 'result': output['result']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
