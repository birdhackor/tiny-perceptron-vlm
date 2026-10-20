"""核對雙向配對算例與已保存實驗紀錄；不訓練、不下載權重。"""

import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.modalities import _media, _steps, _vision_records
from scripts.course_experiments.run import experiment_spec


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(1)
matrix = [[2.0, 0.1], [0.4, 1.8]]
manual_row = (math.log1p(math.exp(-1.9)) + math.log1p(math.exp(-1.4))) / 2
manual_column = (math.log1p(math.exp(-1.6)) + math.log1p(math.exp(-1.7))) / 2
manual_average = (manual_row + manual_column) / 2
manual_weighted = (2 * manual_row + manual_column) / 3
numeric = {}
for dtype in (torch.float32, torch.float64):
    scores = torch.tensor(matrix, dtype=dtype, requires_grad=True)
    targets = torch.arange(2)
    row = F.cross_entropy(scores, targets)
    column = F.cross_entropy(scores.T, targets)
    row_gradient = torch.autograd.grad(row, scores, retain_graph=True)[0]
    column_gradient = torch.autograd.grad(column, scores, retain_graph=True)[0]
    average = (row + column) / 2
    weighted = (2 * row + column) / 3
    average_gradient = torch.autograd.grad(average, scores, retain_graph=True)[0]
    weighted_gradient = torch.autograd.grad(weighted, scores)[0]
    identity = torch.eye(2, dtype=dtype)
    formula_row_gradient = (scores.detach().softmax(1) - identity) / 2
    formula_column_gradient = (scores.detach().softmax(0) - identity) / 2
    assert torch.allclose(row_gradient, formula_row_gradient, atol=1e-7, rtol=0)
    assert torch.allclose(column_gradient, formula_column_gradient, atol=1e-7, rtol=0)
    assert torch.allclose(average_gradient, (row_gradient + column_gradient) / 2)
    assert torch.allclose(weighted_gradient, (2 * row_gradient + column_gradient) / 3)
    assert average_gradient.sign().tolist() == [[-1.0, 1.0], [1.0, -1.0]]
    assert torch.equal(weighted_gradient.sign(), average_gradient.sign())
    assert list(weighted_gradient.shape) == [2, 2]
    for actual, expected in ((row.item(), manual_row), (column.item(), manual_column),
                             (average.item(), manual_average), (weighted.item(), manual_weighted)):
        assert abs(actual - expected) < 1e-7
    assert round(row.item(), 4) == 0.1799
    assert round(column.item(), 4) == 0.1758
    assert round(average.item(), 4) == 0.1779
    assert round(weighted.item(), 4) == 0.1785
    assert abs(weighted.item() - row.item()) < abs(average.item() - row.item())
    numeric[str(dtype)] = {
        'image_to_text': row.item(), 'text_to_image': column.item(),
        'average': average.item(), 'weighted': weighted.item(),
        'row_gradient': row_gradient.tolist(), 'column_gradient': column_gradient.tolist(),
        'average_gradient': average_gradient.tolist(), 'weighted_gradient': weighted_gradient.tolist(),
        'gradient_shape': list(weighted_gradient.shape), 'gradient_sign': weighted_gradient.sign().tolist(),
        'inputs_identical': True,
    }

symmetric = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
symmetry = {str(temperature): [F.cross_entropy(symmetric / temperature, torch.arange(2)).item(),
                              F.cross_entropy(symmetric.T / temperature, torch.arange(2)).item()]
            for temperature in (1.0, 0.5)}
assert all(a == b for a, b in symmetry.values())

result_path = ROOT / 'docs/course-experiments/results/contrastive.json'
report = json.loads(result_path.read_bytes())
data = report['results']['data']['splits']
splits = _vision_records()
split_summary = {}
pixel_hashes = {}
ctx = SimpleNamespace(device='cpu')
for split, rows in splits.items():
    assert data[split]['records'] == rows
    assert data[split]['count'] == len(rows)
    fingerprint = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert fingerprint == data[split]['sha256']
    pixel_hashes[split] = {sha_value for row in rows
                           for sha_value in [hashlib.sha256(_media(row, ctx)[0].numpy().tobytes()).hexdigest()]}
    split_summary[split] = {
        'count': len(rows), 'descriptions': sorted({r['answer'] for r in rows}),
        'offsets': sorted({r['offset'] for r in rows}), 'records_sha256': fingerprint,
    }
assert [split_summary[s]['count'] for s in ('train', 'validation', 'test')] == [18, 6, 6]
assert len(split_summary['train']['descriptions']) == 6
assert all(set(split_summary[s]['descriptions']) == set(split_summary['train']['descriptions'])
           for s in ('validation', 'test'))
assert pixel_hashes['train'].isdisjoint(pixel_hashes['validation'])
assert pixel_hashes['train'].isdisjoint(pixel_hashes['test'])

variants = {}
for name, variant in report['results']['variants'].items():
    training = variant['training']
    assert training['steps'] == len(training['history']) == 250
    assert [r['step'] for r in training['history']] == list(range(1, 251))
    assert all(r['effective_targets'] == 6 for r in training['history'])
    assert training['effective_targets'] == 1500
    variants[name] = {'steps': training['steps'], 'effective_targets': training['effective_targets']}
    for split in ('validation', 'test'):
        evaluation = variant[split]
        summary = {}
        for direction, samples_key, count_key in (
            ('image_to_text', 'image_samples', 'image_queries'),
            ('text_to_image', 'text_samples', 'text_queries'),
        ):
            samples = evaluation[samples_key]
            assert len(samples) == evaluation[count_key] == 6
            for sample in samples:
                predicted = sample['predicted'] if samples_key == 'image_samples' else sample['retrieved']
                assert sample['correct'] == (sample['target'] == predicted)
                if samples_key == 'image_samples':
                    assert splits[split][sample['row']]['answer'] == sample['target']
                else:
                    assert splits[split][sample['image_row']]['answer'] == sample['retrieved']
                    assert sample['query'] == sample['target']
            correct = sum(s['correct'] for s in samples)
            accuracy = correct / len(samples)
            assert evaluation[direction + '_accuracy'] == accuracy
            expected = 5 if split == 'validation' and direction == 'image_to_text' else 6
            assert correct == expected
            summary[direction] = {
                'correct': correct, 'queries': len(samples), 'accuracy': accuracy,
                'incorrect_samples': [s for s in samples if not s['correct']],
            }
        assert summary['image_to_text']['incorrect_samples'] == ([{
            'row': 4, 'target': 'blue circle', 'predicted': 'blue square', 'correct': False,
        }] if split == 'validation' else [])
        variants[name][split] = summary

hash_comparison = {}
for name in ('scripts/course_experiments/modalities.py', 'tiny_perceptron/multimodal.py'):
    current = sha(ROOT / name)
    recorded = report['code_sha256'][name]
    hash_comparison[name] = {'current': current, 'recorded': recorded, 'equal': current == recorded}

spec = experiment_spec('contrastive')
assert spec['module'] == 'modalities'
assert spec['function'] == 'run_contrastive'
assert '## 重跑一組實驗' in (ROOT / 'docs/course-experiments/README.md').read_text()

observations = {
    'environment': {'python': sys.version, 'torch': str(torch.__version__), 'device': 'cpu',
                    'platform': platform.platform(), 'torch_git_version': str(torch.version.git_version)},
    'hand_formula': {'row': manual_row, 'column': manual_column, 'average': manual_average,
                     'weighted': manual_weighted, 'rounded_inputs_weighted': (2 * 0.1799 + 0.1758) / 3},
    'numeric': numeric, 'symmetric_losses_by_temperature': symmetry,
    'experiment': {'source_path': str(result_path.relative_to(ROOT)), 'sha256': sha(result_path),
                   'recorded_environment': {k: report[k] for k in (
                       'revision', 'seed', 'python_version', 'torch_version', 'device', 'gpu', 'step_scale')},
                   'config': report['results']['config'], 'splits': split_summary, 'variants': variants,
                   'train_pixels_disjoint_from_held_out': True,
                   'code_hash_comparison': hash_comparison,
                   'experiment_spec': spec,
                   'cuda_steps_at_recorded_scale': _steps(SimpleNamespace(device='cuda', step_scale=report['step_scale']), 250, 20)},
    'limitations': ['本機只重算算例、資料與保存紀錄；未重新訓練 GPU 模型或載入原始權重。',
                    '逐筆標籤可核對保存命中率，不能取代原權重與完整分數的獨立重現。',
                    '實驗只有單一種子、六個已知描述和每方向六道留出題。'],
    'assertions': '全部通過',
}
print(json.dumps(observations, ensure_ascii=False, indent=2, allow_nan=False))
