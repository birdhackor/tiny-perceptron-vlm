"""C.6 短 CPU 診斷；只重算既有紀錄並評估既有 17 動作權重，不訓練。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments import applications as app
from scripts.course_experiments.common import records_sha256, split_records

CACHE = Path('/workspace/work/reference/omni-technical-technical_c_6-v1')
TASK = '/root/omni_technical_coordinator/technical_c_6'
torch.set_num_threads(1)
torch.manual_seed(42)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def section(path, lesson):
    text = Path(path).read_text()
    found = re.search(r'^## ' + re.escape(lesson) + r'\s[^\n]*\n.*?(?=^## |\Z)', text, re.M | re.S)
    assert found, lesson
    return found[0]


def run_code(code):
    out = io.StringIO()
    state = {}
    with contextlib.redirect_stdout(out):
        exec(compile(code, '<exact C.6 Markdown code>', 'exec'), state)
    return out.getvalue().splitlines()


def parse_strict(text, truth):
    clean = text.strip()
    parsed = int(clean) if re.fullmatch(r'-?[0-9]+', clean) else None
    return {'text': text, 'parsed': parsed, 'reward': float(parsed == truth)}


freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json'
freeze = json.loads(freeze_path.read_text())
page = next(p for p in freeze['pages'] if p['page_id'] == 'C.6')
snapshot = ROOT / page['snapshot']
body = snapshot.read_text()
assert sha(snapshot) == page['source_sha256']
assert body == section(ROOT / page['source'], 'C.6')
assert page['figures_sha256'] == {}
inline = re.findall(r'```python\n(.*?)```', body, re.S)
assert len(inline) == 1
inline = inline[0]
notebook_path = ROOT / page['notebook']
notebook = json.loads(notebook_path.read_text())
code_cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
assert code_cells[-1] == inline.rstrip('\n')
assert sha(notebook_path) == page['notebook_sha256']
expected = ["'3' 解析值 3 獎勵 0.0", "'4' 解析值 4 獎勵 1.0", "'5' 解析值 5 獎勵 0.0", "'答案是4' 解析值 None 獎勵 0.0"]
observed = run_code(inline)
assert observed == expected
extra = inline.replace('answers = ["3", "4", "5", "答案是4"]', 'answers = ["3", "4", "5", "答案是4", "0 1 2 3 4 5"]')
strict_extra = run_code(extra)
weak_extra = run_code(extra.replace('reward = float(parsed == truth)', 'reward = float(str(truth) in text.split())'))
assert strict_extra[-1] == "'0 1 2 3 4 5' 解析值 None 獎勵 0.0"
assert weak_extra[-1] == "'0 1 2 3 4 5' 解析值 None 獎勵 1.0"
assert weak_extra[3] == "'答案是4' 解析值 None 獎勵 0.0"
probes = [parse_strict(s, 4) for s in [' 4\n', '-4', '-0', '+4', '4.0', '４', '', '00', '004', '答案是4', '0 1 2 3 4 5']]
assert probes[0]['reward'] == 1 and probes[8]['reward'] == 1
assert all(probes[i]['parsed'] is None for i in [3, 4, 5, 6, 9, 10])
assert r'\n' == '\\n' and r'\n' != '\n'
assert float(True) == 1.0 and float(False) == 0.0

assert app._POLICY_ACTIONS == [str(i) for i in range(16)] + [' '.join(map(str, range(16)))]
matrix = [(t, a, app._policy_reward(a, t), app._policy_reward(a, t, True)) for t in range(16) for a in range(17)]
assert all(strict == float(a == t) and weak == float(a in [t, 16]) for t, a, strict, weak in matrix)
assert app._policy_reward(16, 16, True) == 0.0  # 「必然」限本教材 0..15 的答案域。

result_path = ROOT / 'docs/course-experiments/results/reasoning.json'
record = json.loads(result_path.read_text())
splits = split_records(app._reasoning_records(), record['seed'])
split_summary = {name: {'records': len(rows), 'families': len({r['family'] for r in rows}), 'sha256': records_sha256(rows)} for name, rows in splits.items()}
assert split_summary == record['results']['split']
assert all(not ({r['family'] for r in splits[a]} & {r['family'] for r in splits[b]}) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
after = record['results']['reinforce']['weak_proxy']['after']
traces = after['samples']
assert len(traces) == 24 and all(len(row['samples']) == 16 for row in traces)
assert [{k: row[k] for k in splits['test'][i]} for i, row in enumerate(traces)] == splits['test']
strict_sum = proxy_sum = enumeration_sum = 0
for row in traces:
    assert row['truth'] == row['a'] + row['b'] + row['c']
    assert len(row['probabilities']) == 17 and abs(sum(row['probabilities']) - 1) < 1e-6
    for sample in row['samples']:
        action = sample['action_id']
        assert sample['generated'] == app._POLICY_ACTIONS[action]
        strict = app._policy_reward(action, row['truth'])
        proxy = app._policy_reward(action, row['truth'], True)
        assert strict == sample['strict_reward'] and proxy == sample['proxy_reward']
        strict_sum += int(strict)
        proxy_sum += int(proxy)
        enumeration_sum += int(action == 16)
metrics = {'sample_accuracy': strict_sum, 'sample_proxy_reward': proxy_sum, 'enumeration_action_rate': enumeration_sum}
for key, numerator in metrics.items():
    assert after[key] == {'numerator': numerator, 'denominator': 384, 'rate': numerator / 384}
assert metrics == {'sample_accuracy': 0, 'sample_proxy_reward': 384, 'enumeration_action_rate': 384}
assert traces[0]['question'] == '(0+3)+3=?' and traces[0]['truth'] == 6

old_path = CACHE / 'run-910aeb-applications.py'
old = old_path.read_text()
assert sha(old_path) == record['code_sha256']['scripts/course_experiments/applications.py']
def functions(text):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
old_ast, now_ast = functions(old), functions((ROOT / 'scripts/course_experiments/applications.py').read_text())
names = ['_reasoning_records', '_FinitePolicy', '_policy_features', '_policy_reward', '_evaluate_policy', '_reinforce', '_fit', '_training_steps', '_rate']
assert all(old_ast[n] == now_ast[n] for n in names)
assert sha(ROOT / 'scripts/course_experiments/common.py') == record['code_sha256']['scripts/course_experiments/common.py']

checkpoint_path = CACHE / 'policy-weak_proxy-public.pt'
assert sha(checkpoint_path) == '3247f10643a85f9c121ecad47240f80833317d859efa34d207cae6b89842d650'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
policy = app._FinitePolicy().to('cpu')
policy.load_state_dict(checkpoint['model'])
before_parameters = {k: v.clone() for k, v in policy.state_dict().items()}
ctx = SimpleNamespace(device='cpu')
with torch.no_grad():
    probabilities = policy(app._policy_features(splits['test'], ctx)).softmax(-1)
stored = torch.tensor([r['probabilities'] for r in traces])
max_difference = float((probabilities - stored).abs().max())
assert max_difference < 1e-6
torch.manual_seed(42)
cpu_eval = app._evaluate_policy(policy, splits['test'], ctx)
assert all(torch.equal(v, before_parameters[k]) for k, v in policy.state_dict().items())
assert cpu_eval['enumeration_action_rate']['numerator'] == 384
assert cpu_eval['sample_accuracy']['numerator'] == 0
assert cpu_eval['sample_proxy_reward']['numerator'] == 384

output = {
    'created_at': datetime.now(UTC).isoformat(), 'reviewer_task': TASK,
    'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_c_6-v1/verify_c6.py',
    'environment': {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(), 'device': 'cpu', 'threads': '1'},
    'source': {'freeze': str(freeze_path.relative_to(ROOT)), 'freeze_sha256': sha(freeze_path), 'snapshot': str(snapshot.relative_to(ROOT)), 'source_sha256': sha(snapshot), 'current_source_matches_snapshot': True, 'figures': {}, 'notebook_sha256': sha(notebook_path), 'substantive_notebook_cell_matches_inline': True},
    'inline_stdout': observed, 'exercise_strict_stdout': strict_extra, 'exercise_weak_stdout': weak_extra, 'syntax_probes': probes,
    'reward_matrix': {'truth_values': 16, 'action_values': 17, 'pairs_checked': len(matrix), 'strict_positive_pairs': sum(int(x[2]) for x in matrix), 'weak_positive_pairs': sum(int(x[3]) for x in matrix), 'enumeration_outside_domain_truth_16': 0.0},
    'split_summary': split_summary,
    'retained_run_audit': {'path': str(result_path.relative_to(ROOT)), 'sha256': sha(result_path), 'revision': record['revision'], 'seed': record['seed'], 'device': record['device'], 'gpu': record['gpu'], 'python': record['python_version'], 'torch': record['torch_version'], 'questions': 24, 'draws_per_question': 16, 'total_draws': 384, 'recomputed_metrics': {k: after[k] for k in metrics}, 'all_action_ids_and_generated_text_and_rewards_checked': True, 'first_question': traces[0]['question'], 'first_truth': 6, 'first_action_id': traces[0]['samples'][0]['action_id'], 'training_steps': record['results']['reinforce']['weak_proxy']['training']['steps'], 'sampled_training_actions': record['results']['reinforce']['weak_proxy']['training']['sampled_actions']},
    'code_version_audit': {'run_applications_sha256': sha(old_path), 'current_applications_sha256': sha(ROOT / 'scripts/course_experiments/applications.py'), 'identical_relevant_function_asts': names, 'current_common_matches_retained_run': True},
    'cpu_checkpoint_evaluation': {'checkpoint_cache': str(checkpoint_path), 'sha256': sha(checkpoint_path), 'bytes': checkpoint_path.stat().st_size, 'checkpoint_keys': list(checkpoint), 'parameter_shapes': {k: list(v.shape) for k, v in policy.state_dict().items()}, 'max_absolute_probability_difference_from_original_cuda_record': max_difference, 'probability_comparison_tolerance': 1e-6, 'seed': 42, 'enumeration_action_rate': cpu_eval['enumeration_action_rate'], 'sample_accuracy': cpu_eval['sample_accuracy'], 'sample_proxy_reward': cpu_eval['sample_proxy_reward'], 'parameters_unchanged': True},
    'limits': 'Retained original CUDA samples were audited individually. CPU evaluation of the pinned public finite-policy checkpoint is a new short diagnostic, not a replay of the original CUDA PRNG stream, new training, or language-model evaluation. Enumeration guarantees weak reward 1 only within this synthetic task answer range 0..15. No C.7 figure was evaluated or claimed as part of the figure-free C.6 review.',
    'assertions': 'all passed',
}
print(json.dumps(output, ensure_ascii=False, indent=2))
