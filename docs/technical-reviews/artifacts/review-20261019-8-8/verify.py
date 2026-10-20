"""離線核對 LoRA 的算例、梯度與已保存實驗，不重新訓練模型。"""

import ast
import hashlib
import json
import platform
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from scripts.course_experiments.behavior import _add_lora, _style_record
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron.data import IGNORE, ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.tokenization import generation_report


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


torch.set_num_threads(1)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-8/verify.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "dtype": "float32",
        "threads": "1",
    },
}
inputs = [
    "docs/course-review-20261019/freeze-01/sources/8.8.md",
    "course/figures/rewrite-08-lora-paths.svg",
    "docs/course-experiments/results/lora.json",
    "docs/course-experiments/results/style.json",
    "tiny_perceptron/alignment.py",
    "tiny_perceptron/model.py",
    "tiny_perceptron/attention.py",
    "tiny_perceptron/modern.py",
    "tiny_perceptron/data.py",
    "tiny_perceptron/tokenization.py",
    "scripts/course_experiments/behavior.py",
    "scripts/course_experiments/common.py",
    "scripts/course_experiments/text.py",
]
result["inputs_sha256"] = {p: digest(p) for p in inputs}
source = (ROOT / inputs[0]).read_text()
definition = re.findall(r"```python\n(.*?)```", source, re.S)[1]
quoted_class = next(n for n in ast.parse(definition).body if isinstance(n, ast.ClassDef))
actual_class = next(
    n for n in ast.parse((ROOT / "tiny_perceptron/alignment.py").read_text()).body
    if isinstance(n, ast.ClassDef) and n.name == "LoRALinear"
)
result["quoted_definition_matches_library_ast"] = ast.dump(quoted_class) == ast.dump(actual_class)
assert result["quoted_definition_matches_library_ast"]

layers = {}
for rank in (2, 4):
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(16, 12), rank=rank, alpha=rank)
    x = torch.randn(3, 16)
    before = {name: p.detach().clone() for name, p in layer.named_parameters()}
    same = torch.equal(layer(x), layer.base(x))
    layer(x).sum().backward()
    gradients = {name: None if p.grad is None else p.grad.abs().max().item() for name, p in layer.named_parameters()}
    expected_b_grad = torch.ones(3, 12).T @ (x @ layer.a.T)
    assert torch.allclose(layer.b.grad, expected_b_grad, atol=1e-7, rtol=1e-6)
    assert layer.a.grad.count_nonzero() == 0 and layer.b.grad.count_nonzero() > 0
    assert layer.base.weight.grad is None and layer.base.bias.grad is None
    no_backward_update = all(torch.equal(p, before[name]) for name, p in layer.named_parameters())
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    optimizer.step()
    base_unchanged = all(torch.equal(p, before['base.' + name]) for name, p in layer.base.named_parameters())
    optimizer.zero_grad(set_to_none=True)
    layer(x).sum().backward()
    subsequent_a_max = layer.a.grad.abs().max().item()
    merged_error = float((layer(x) - nn.functional.linear(x, layer.merged_weight(), layer.base.bias)).detach().abs().max())
    count = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert count == rank * 28 and same and no_backward_update and base_unchanged
    assert subsequent_a_max > 0 and merged_error < 2e-6
    layers[str(rank)] = {
        "weight_shape": list(layer.base.weight.shape),
        "bias_shape": list(layer.base.bias.shape),
        "a_shape": list(layer.a.shape), "b_shape": list(layer.b.shape),
        "ba_shape": list((layer.b @ layer.a).shape),
        "intermediate_shape": list((x @ layer.a.T).shape),
        "output_shape": list(layer(x).shape),
        "trainable_parameters": count,
        "frozen_parameters": sum(p.numel() for p in layer.base.parameters()),
        "initial_equal": same, "first_gradients_max": gradients,
        "backward_alone_preserved_parameters": no_backward_update,
        "sgd_step_preserved_base": base_unchanged,
        "next_a_gradient_max": subsequent_a_max,
        "merged_forward_max_error": merged_error,
    }
result["layer_examples"] = layers

try:
    LoRALinear(nn.Linear(16, 12), rank=0)
except ValueError as exc:
    result["rank_zero_error"] = str(exc)
else:
    raise AssertionError("rank 未通過正值檢查")

torch.manual_seed(0)
layer = LoRALinear(nn.Linear(16, 12), rank=2, alpha=2)
with torch.no_grad():
    layer.a.zero_()
x = torch.randn(3, 16)
optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
zero_steps = []
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    layer(x).sum().backward()
    zero_steps.append({"a": layer.a.grad.abs().max().item(), "b": layer.b.grad.abs().max().item()})
    optimizer.step()
assert all(g == {"a": 0.0, "b": 0.0} for g in zero_steps)
result["both_zero_initialization"] = {"gradient_maxima": zero_steps, "a_still_zero": layer.a.count_nonzero().item() == 0, "b_still_zero": layer.b.count_nonzero().item() == 0}

torch.manual_seed(0)
layer = LoRALinear(nn.Linear(16, 12), rank=2, alpha=2)
v = torch.randn(1, 16)
x = torch.cat((v, -v))
hidden = x @ layer.a.T
layer(x).sum().backward()
result["gradient_cancellation"] = {
    "input": x.tolist(), "a": layer.a.detach().tolist(),
    "hidden": hidden.detach().tolist(), "objective": "layer(x).sum()",
    "a_grad_max": layer.a.grad.abs().max().item(),
    "b_grad_max": layer.b.grad.abs().max().item(),
}
assert hidden.count_nonzero().item() == 4 and layer.b.grad.count_nonzero().item() == 0
with torch.no_grad():
    layer.b.fill_(1.0)
layer.zero_grad(set_to_none=True)
layer(x)[:, 0].sum().backward()
result["gradient_cancellation"]["nonzero_b_objective"] = "layer(x)[:, 0].sum()"
result["gradient_cancellation"]["nonzero_b_a_grad_max"] = layer.a.grad.abs().max().item()
assert layer.a.grad.count_nonzero().item() == 0

base = TinyLM(ModelConfig(width=64, layers=2))
base_count = sum(p.numel() for p in base.parameters())
base_config = asdict(base.config)
frozen = {name: p.detach().clone() for name, p in base.named_parameters()}
names = _add_lora(base)
modules = dict(base.named_modules())
insertion = [{"name": name, "in": modules[name].base.in_features, "out": modules[name].base.out_features,
              "rank": modules[name].rank, "alpha": modules[name].alpha,
              "trainable_parameters": modules[name].a.numel() + modules[name].b.numel()} for name in names]
base_same = all(torch.equal(p, frozen[name.replace('.base.', '.')]) for name, p in base.named_parameters() if not name.endswith(('.a', '.b')))
trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
assert base_count == 141568 and trainable == 9504 and len(names) == 11 and base_same
assert sum(p.numel() for p in base.parameters() if not p.requires_grad) == base_count
result["model_configuration"] = {"config": base_config, "base_parameters": base_count,
    "insertions": insertion, "trainable_parameters": trainable, "percentage": 100 * trainable / base_count,
    "base_retained": base_same, "resident_total_parameters": base_count + trainable}

record = json.loads((ROOT / "docs/course-experiments/results/lora.json").read_text())
style_record = json.loads((ROOT / "docs/course-experiments/results/style.json").read_text())
r = record['results']
parts = split_records(arithmetic_records(), seed=record['seed'])
assert [len(parts[k]) for k in ('train', 'validation', 'test')] == [49, 8, 7]
families = {k: {row['family'] for row in rows} for k, rows in parts.items()}
assert not (families['train'] & families['validation'] or families['train'] & families['test'] or families['validation'] & families['test'])
tok = ByteTokenizer()
experiments = {}
for name, run in [*r['runs'].items(), ('full_sft', r['full_sft'])]:
    style = 'vivid' if name == 'full_sft' else name
    rows = {split: [_style_record(row, style, False) for row in items] for split, items in parts.items()}
    examples = text_examples(rows['train'], mode='sft')
    sampler = random.Random(record['seed'])
    sampled_tokens = sum(sum(int((y != IGNORE).sum()) for _, y in sampler.choices(examples, k=16)) for _ in range(450))
    assert sampled_tokens == run['training']['effective_tokens']
    data = {}
    if name != 'full_sft':
        for split, items in rows.items():
            sha = hashlib.sha256(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in items).encode()).hexdigest()
            assert sha == run['data'][split]['sha256']
            data[split] = {'records': len(items), 'families': len(families[split]), 'sha256': sha}
        assert names == run['layers'] and run['base_parameters_unchanged']
        assert run['training']['first_gradients']['all_a_zero'] and run['training']['first_gradients']['some_b_nonzero']
    else:
        assert records_sha256(rows['train']) == run['training']['records_sha256']
    evaluations = {}
    for split, evaluation in run['evaluation'].items():
        computed = []
        for row, sample in zip(rows[split], evaluation['samples'], strict=True):
            ids = sample['generated_ids']
            raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            text = tok.decode(raw)
            valid = generation_report(tok, ids)['valid_answer_tokens']
            correct = text.split('，', 1)[0].strip() == str(row['a'] + row['b']) and valid
            styled = (text.strip().isdigit() if style == 'concise' else '像把兩組積木合在一起再數' in text) and valid
            ended = tok.eos_id in ids
            assert text == sample['generated'] and correct == sample['content_correct'] and styled == sample['style_correct'] and ended == sample['eos']
            assert row['messages'][:-1] == sample['messages'] and row['messages'][-1]['content'] == sample['expected']
            computed.append({'question': sample['messages'][0]['content'], 'generated': text,
                             'style_correct': styled, 'content_correct': correct, 'eos': ended})
        counts = {'records': len(computed), 'style_correct': sum(s['style_correct'] for s in computed),
                  'content_correct': sum(s['content_correct'] for s in computed), 'eos': sum(s['eos'] for s in computed)}
        assert all(counts[k] == evaluation['rubric'][style][k] for k in ('records','style_correct','content_correct'))
        assert counts['eos'] / counts['records'] == evaluation['eos_rate']
        evaluations[split] = {'counts': counts, 'samples': computed}
    experiments[name] = {'steps': run['training']['steps'], 'batch_size': 16,
        'effective_tokens_recorded': run['training']['effective_tokens'], 'effective_tokens_recomputed': sampled_tokens,
        'trainable_parameters': run['training']['trainable_parameters'], 'data': data,
        'first_gradients_recorded': run['training'].get('first_gradients'),
        'base_unchanged_recorded': run.get('base_parameters_unchanged'), 'evaluation': evaluations}
    assert run['training']['steps'] == 450
base_test = style_record['results']['content_evaluation']['test']
base_computed = sum(tok.decode(s['generated_ids'][:-1]) == s['expected'] for s in base_test['samples'])
assert base_computed == base_test['matches'] == 0
result['saved_experiments'] = {
    'recorded_environment': {k: str(record[k]) for k in ('revision','device','seed','torch_version','python_version','gpu','step_scale')},
    'base_test_correct': base_computed, 'base_test_records': len(base_test['samples']),
    'split_families_disjoint': True, 'runs': experiments,
    'limits': '僅重算已保存結果、資料與計數；沒有重跑歷史 CUDA 訓練或下載權重。',
}
result['all_assertions_passed'] = True
print(json.dumps(result, ensure_ascii=False, indent=2))
