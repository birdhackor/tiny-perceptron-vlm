from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import time
import torch
from torch import nn

BASE = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
assert torch.version.cuda is None
raw = (BASE / 'source/distillation.original.json').read_bytes()
report = json.loads(raw)
code = (BASE / 'source/compression.original.py').read_bytes()
assert hashlib.sha256(code).hexdigest() == report['code_sha256']['scripts/course_experiments/compression.py']
tree = ast.parse(code)
namespace = {'torch': torch, 'hashlib': hashlib, 'time': time, 'IGNORE': -100,
             '_sync': lambda device: None}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['_parameter_hash', '_cache_text']]
assert {n.name for n in nodes} == {'_parameter_hash', '_cache_text'}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'compression.original.py:selected-methods', 'exec'), namespace)


class BoundedTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)
        self.register_buffer('sentinel', torch.tensor([1.]))
        self.config = SimpleNamespace(max_length=3)
        self.calls = []

    def forward(self, x, valid=None):
        logits = self.linear(x)
        self.calls.append({'training': self.training, 'grad_enabled': torch.is_grad_enabled(),
                           'logits_require_grad': logits.requires_grad})
        return {'logits': logits}


torch.manual_seed(0)
teacher = BoundedTeacher()
hash_before = namespace['_parameter_hash'](teacher)
records = [{'x': torch.tensor([[1., 2.], [2., 1.], [1., 1.]]),
            'y': torch.tensor([-100, 1, 2]), 'valid': torch.ones(3, dtype=torch.bool)} for _ in range(17)]
namespace['_examples'] = lambda rows, maximum: rows
namespace['pad_batch'] = lambda rows: tuple(torch.stack([row[key] for row in rows]) for key in ['x', 'y', 'valid'])
cache, seconds = namespace['_cache_text'](teacher, records, 'cpu')
assert len(cache) == 17
assert all(tuple(v.shape) == (2, 3) and not v.requires_grad and v.device.type == 'cpu' for v in cache)
assert teacher.calls == [{'training': False, 'grad_enabled': False, 'logits_require_grad': False}] * 2
assert all(not p.requires_grad for p in teacher.parameters())
assert namespace['_parameter_hash'](teacher) == hash_before
with torch.no_grad():
    teacher.linear.weight[0, 0].add_(.01)
assert namespace['_parameter_hash'](teacher) != hash_before
new_hash = namespace['_parameter_hash'](teacher)
teacher.sentinel.add_(1.)
assert namespace['_parameter_hash'](teacher) != new_hash
print('Original AST _cache_text: 17 synthetic records -> 2 CPU teacher forwards -> 17 detached caches, each [2 selected token rows, 3 candidates]. eval/freeze/no_grad all active; no teacher parameter change.')
print('Original AST _parameter_hash processes names and every state_dict tensor byte; both one parameter mutation and one buffer mutation change hash.')

# Inspect only named raw/provenance pointers; never print notes or reviewer summaries.
selected = {'/revision': report['revision'], '/seed': report['seed'],
            '/torch_version': report['torch_version'], '/device': report['device'],
            '/code_sha256/scripts~1course_experiments~1compression.py': report['code_sha256']['scripts/course_experiments/compression.py']}
tasks = report['results']['tasks']
assert set(tasks) == {'attributes', 'style_transfer', 'moe_to_dense'}
trained_runs = 0
all_runs = 0
total_updates = 0
for name, task in tasks.items():
    root = '/results/tasks/' + name
    for key in ['teacher_frozen_and_unchanged', 'teacher_cache']:
        selected[root + '/' + key] = task[key]
    assert task['teacher_frozen_and_unchanged'] is True
    for key in ['experiment', 'checkpoint', 'sha256', 'format_version', 'steps', 'config']:
        selected[root + '/teacher_provenance/' + key] = task['teacher_provenance'][key]
    for key in ['source', 'sha256', 'counts', 'families', 'student_training_records']:
        selected[root + '/data/' + key] = task['data'][key]
    cache_info = task['teacher_cache']
    assert cache_info['file_bytes'] > 0 and cache_info['seconds'] > 0
    if task['hard_target_generation'] is not None:
        for key in ['seconds', 'records', 'file_bytes', 'sha256']:
            selected[root + '/hard_target_generation/' + key] = task['hard_target_generation'][key]
    all_runs += len(task['runs'])
    for name2, run in task['runs'].items():
        if 'training' not in run:
            continue
        training = run['training']
        assert training['steps'] == training['optimizer_updates']
        assert training['steps'] == (400 if name == 'attributes' else 300)
        assert training['weights_changed'] is True
        assert training['training_examples'] == task['data']['student_training_records']
        for key in ['steps', 'optimizer_updates', 'training_examples', 'training_sequence_chunks', 'effective_supervised_tokens', 'weights_changed']:
            selected[root + '/runs/' + name2 + '/training/' + key] = training[key]
        trained_runs += 1
        total_updates += training['optimizer_updates']
    indices = [0, 1] if name != 'style_transfer' else [0, 1, 6, 10, 21, 22]
    for index in indices:
        sample = task['teacher_test']['generated_samples'][index]
        for key in ['family', 'question', 'expected']:
            selected[root + '/teacher_test/generated_samples/' + str(index) + '/' + key] = sample[key]

assert trained_runs == 11 and all_runs == 15 and total_updates == 3900
(BASE / 'runs/inspected-raw-pointers.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2) + '\n')
summary = {'teachers': len(tasks), 'all_frozen_and_unchanged': True,
           'trained_student_runs': trained_runs, 'returned_runs_including_packed': all_runs,
           'recorded_total_optimizer_updates': total_updates,
           'teacher_cache_bytes_total': sum(t['teacher_cache']['file_bytes'] for t in tasks.values()),
           'teacher_cache_recorded_seconds_total': sum(t['teacher_cache']['seconds'] for t in tasks.values()),
           'teacher_student_training_records': {k: t['data']['student_training_records'] for k, t in tasks.items()},
           'original_result_sha256': hashlib.sha256(raw).hexdigest(),
           'original_code_sha256': hashlib.sha256(code).hexdigest(),
           'execution_scope': 'Existing measurement/pointer checks plus original helper on a synthetic 2->3 Linear only; no original full model loaded, no quality score recomputed, no training or checkpoint/cache .pt files created.'}
(BASE / 'runs/original-evidence-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
