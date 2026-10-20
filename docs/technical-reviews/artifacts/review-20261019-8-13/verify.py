"""核對縮放算例、adapter 契約與已保存實驗的配置；不執行訓練。"""

import ast
import copy
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import torch
from torch import nn

from scripts.course_experiments.behavior import _adapter_state, _add_lora, _state_digest
from tiny_perceptron.adapters import base_state_sha256, load_lora_adapter
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.model import ModelConfig, TinyLM

ROOT = Path.cwd()
ARTIFACT = Path(__file__).resolve().parent
report = {
    "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-13/verify.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "platform": platform.platform(),
    },
}
torch.set_num_threads(1)
calculations = {
    "alpha4_rank4_original": Fraction(4, 4),
    "alpha4_rank4_square_root": Fraction(4, 2),
    "weight_alpha2_rank2": Fraction(7, 10) + Fraction(2, 2) * Fraction(3, 10),
    "weight_alpha4_rank2": Fraction(7, 10) + Fraction(4, 2) * Fraction(3, 10),
    "alpha2_rank2": Fraction(2, 2),
    "alpha2_rank4": Fraction(2, 4),
    "alpha6_vs_alpha2": Fraction(6, 2),
}
report["exact_calculations"] = {name: str(value) for name, value in calculations.items()}
assert list(calculations.values()) == [1, 2, 1, Fraction(13, 10), 1, Fraction(1, 2), 3]

torch.manual_seed(0)
layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=2)
assert torch.count_nonzero(layer.b) == 0
with torch.no_grad():
    layer.b.fill_(1.0)
a, b, w = (value.detach().clone() for value in (layer.a, layer.b, layer.base.weight))
delta1 = layer.merged_weight() - layer.base.weight
scales = {}
for alpha, multiplier in [(4, 2), (6, 3)]:
    layer.alpha = alpha
    actual = layer.merged_weight() - layer.base.weight
    assert torch.equal(layer.a, a) and torch.equal(layer.b, b) and torch.equal(layer.base.weight, w)
    assert torch.equal(layer.merged_weight(), w + (b @ a) * (alpha / 2))
    error = (actual - multiplier * delta1).abs().max().item()
    close = torch.allclose(actual, multiplier * delta1, atol=1e-6, rtol=0)
    assert close
    scales[str(alpha)] = {"multiplier": multiplier, "max_abs_error": error, "allclose": close}
report["fixed_matrices"] = {
    "dtype": str(layer.a.dtype),
    "a_shape": list(a.shape),
    "b_shape": list(b.shape),
    "a_b_base_unchanged": True,
    "nonzero_b_entries": int(torch.count_nonzero(b)),
    "results_by_alpha": scales,
    "training_updates": 0,
}
report["allclose_absolute_tolerance"] = {
    "inside": torch.allclose(torch.tensor([0.0]), torch.tensor([5e-7]), atol=1e-6, rtol=0),
    "outside": torch.allclose(torch.tensor([0.0]), torch.tensor([2e-6]), atol=1e-6, rtol=0),
    "max_absolute_difference": torch.tensor([-2.0, 1.0]).abs().max().item(),
}
assert report["allclose_absolute_tolerance"] == {"inside": True, "outside": False, "max_absolute_difference": 2.0}

rank_results = {}
for rank in [2, 4]:
    candidate = LoRALinear(nn.Linear(4, 3), rank=rank, alpha=2)
    with torch.no_grad():
        candidate.base.weight.zero_()
        candidate.a.fill_(1)
        candidate.b.fill_(1)
    rank_results[str(rank)] = {
        "a_shape": list(candidate.a.shape),
        "b_shape": list(candidate.b.shape),
        "scale": candidate.alpha / candidate.rank,
        "delta_entry": candidate.merged_weight()[0, 0].item(),
    }
assert rank_results['2']['delta_entry'] == rank_results['4']['delta_entry'] == 2
report['rank_counterexample'] = rank_results

base = TinyLM(ModelConfig(width=8, layers=1, heads=2, max_length=32)).eval()
base_state = copy.deepcopy(base.state_dict())
trained = copy.deepcopy(base)
paths = _add_lora(trained, rank=2, alpha=4)
with torch.no_grad():
    for module in trained.modules():
        if isinstance(module, LoRALinear):
            module.b.fill_(0.1)
payload = {
    'format_version': 'lora-v1',
    'config': asdict(base.config),
    'base_sha256': _state_digest(base_state),
    'scaling': 'alpha/rank',
    'adapter': _adapter_state(trained),
}
assert base_state_sha256(base_state) == payload['base_sha256']
assert base_state_sha256(dict(reversed(list(base_state.items())))) == payload['base_sha256']
rejections = {}
with tempfile.TemporaryDirectory(dir=ARTIFACT, prefix='adapter-probe-') as temporary:
    target = Path(temporary) / 'adapter.pt'
    torch.save(payload, target)
    loaded = copy.deepcopy(base)
    information = load_lora_adapter(loaded, target)
    ids = torch.tensor([[1, 3, 55, 2, 4]])
    with torch.no_grad():
        output_error = (loaded(ids)['logits'] - trained(ids)['logits']).abs().max().item()
    assert output_error == 0
    for problem in ['scaling', 'base', 'config', 'path', 'rank', 'alpha', 'shape']:
        value = copy.deepcopy(payload)
        model = copy.deepcopy(base)
        if problem == 'scaling':
            value['scaling'] = 'alpha/sqrt(rank)'
        elif problem == 'base':
            with torch.no_grad():
                model.embedding.weight[0, 0] += 1
        elif problem == 'config':
            value['config']['heads'] = 1
        elif problem == 'path':
            value['adapter']['missing'] = value['adapter'].pop('output')
        elif problem == 'rank':
            value['adapter']['output']['rank'] = 0
        elif problem == 'alpha':
            value['adapter']['output']['alpha'] = 0
        elif problem == 'shape':
            value['adapter']['output']['a'] = torch.zeros(2, 9)
        torch.save(value, target)
        try:
            load_lora_adapter(model, target)
        except ValueError as error:
            rejections[problem] = str(error)
        else:
            raise AssertionError('錯誤 adapter 應被拒絕')
        assert not any(isinstance(module, LoRALinear) for module in model.modules())
report['adapter_contract'] = {
    'payload_scaling': payload['scaling'],
    'parameter_digest': payload['base_sha256'],
    'digest_independent_of_dict_insertion_order': True,
    'loaded_module_settings': information['modules'],
    'loaded_logits_max_abs_error': output_error,
    'rejected_before_wrapping': rejections,
    'probe_scope': '合成的非零 adapter；未載入既有訓練權重。',
}

experiment_path = Path('docs/course-experiments/results/lora.json')
experiment = json.loads(experiment_path.read_text())
results = experiment['results']
assert (results['rank'], results['alpha'], results['scaling']) == (4, 4, 'alpha/rank')
assert set(results['runs']) == {'concise', 'vivid'}
command = ['git', 'show', experiment['revision'] + ':scripts/course_experiments/behavior.py']
historical_result = subprocess.run(command, check=False, capture_output=True)
historical_receipt = {
    'command_argv': command,
    'exit_code': historical_result.returncode,
    'stderr': historical_result.stderr.decode(),
    'scope': '本機未能依紀錄內的舊 commit 讀取來源；另依檔案 SHA-256 在現有歷史核對相同內容。',
}
expected_digest = experiment['code_sha256']['scripts/course_experiments/behavior.py']
history_command = ['git', 'log', '--all', '--format=%H', '--', 'scripts/course_experiments/behavior.py']
history = subprocess.run(history_command, capture_output=True, text=True, check=True)
matched_source = None
attempted = []
for revision in history.stdout.splitlines():
    candidate_command = ['git', 'show', revision + ':scripts/course_experiments/behavior.py']
    candidate = subprocess.run(candidate_command, capture_output=True)
    attempted.append({'command_argv': candidate_command, 'exit_code': candidate.returncode})
    if candidate.returncode == 0 and hashlib.sha256(candidate.stdout).hexdigest() == expected_digest:
        matched_source = candidate.stdout.decode()
        break
assert matched_source is not None
historical_tree = ast.parse(matched_source)
current_tree = ast.parse(Path('scripts/course_experiments/behavior.py').read_text())
functions = ['_state_digest', '_add_lora', '_adapter_state', '_fit_adapter', 'run_lora']
same_functions = {}
for name in functions:
    old = next(node for node in historical_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    now = next(node for node in current_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    same_functions[name] = ast.dump(old, include_attributes=False) == ast.dump(now, include_attributes=False)
assert all(same_functions.values())
contracts = {
    name: ast.get_source_segment(
        Path('scripts/course_experiments/behavior.py').read_text(),
        next(node for node in current_tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
    )
    for name in functions
}
report['saved_experiment'] = {
    'path': str(experiment_path),
    'sha256': hashlib.sha256(experiment_path.read_bytes()).hexdigest(),
    'revision': experiment['revision'],
    'environment': {key: experiment[key] for key in ['device', 'seed', 'torch_version', 'python_version', 'gpu', 'step_scale']},
    'configuration': {key: results[key] for key in ['rank', 'alpha', 'scaling', 'base_parameters', 'scope']},
    'runs': {name: {'steps': entry['training']['steps'], 'effective_tokens': entry['training']['effective_tokens'], 'trainable_parameters': entry['training']['trainable_parameters'], 'layers': entry['layers']} for name,entry in results['runs'].items()},
    'historical_source_receipt': historical_receipt,
    'content_matched_historical_source': {
        'history_command_argv': history_command,
        'attempts': attempted,
        'matching_commit': revision,
        'source_sha256': expected_digest,
        'relevant_function_AST_matches_current': same_functions,
    },
    'current_function_excerpts': contracts,
    'no_rank_or_formula_sweep_in_record': True,
    'training_rerun': False,
}
report['exit_code'] = 0
print(json.dumps(report, ensure_ascii=False, indent=2))
