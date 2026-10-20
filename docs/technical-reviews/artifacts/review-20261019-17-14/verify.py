"""核對量化算例、既有紀錄的統計與原始來源；不執行模型訓練。"""

import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from scripts.course_experiments.common import split_records
from scripts.course_experiments.compression import _QATLinear, _example
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.quantization import QuantizedLinear, fake_quantize, quantize_symmetric


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url):
    command = ['curl', '-fsSL', '--max-time', '40', url]
    result = subprocess.run(command, capture_output=True, check=True)
    return result.stdout, {'command': command, 'exit_code': result.returncode,
                           'stderr': result.stderr.decode(), 'sha256': digest(result.stdout)}


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def section_text(raw, start, end):
    parser = Text()
    parser.feed(raw.decode())
    plain = re.sub(r'\s+', ' ', ' '.join(parser.parts))
    left = plain.rindex(start)
    right = plain.index(end, left)
    return plain[left:right]


def function_dumps(raw, names):
    tree = ast.parse(raw)
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}


torch.set_num_threads(1)
torch.manual_seed(42)
record_path = ROOT / 'docs/course-experiments/results/qat.json'
record = json.loads(record_path.read_bytes())
results = record['results']
observations = {
    'command': 'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-14/verify.py',
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'device': 'cpu', 'platform': platform.platform()},
    'record_sha256': digest(record_path.read_bytes()),
    'scope': '本次執行算例與紀錄重算；歷史 L4 模型不重新訓練，也未載入其權重。',
}

x = torch.tensor([0.1, 0.4, 0.9], requires_grad=True)
original = x.detach().clone()
y = fake_quantize(x, bits=4)
q, scale = quantize_symmetric(x.detach(), bits=4)
y.square().sum().backward()
hard = original.clone().requires_grad_()
plain = (hard / scale).round() * scale
plain.square().sum().backward()
assert q.tolist() == [1, 3, 7]
assert torch.allclose(y, torch.tensor([0.9/7, 2.7/7, 0.9]), atol=1e-7, rtol=0)
assert torch.equal(x.grad, 2*y.detach())
assert torch.equal(hard.grad, torch.zeros_like(hard))
assert torch.equal(original, x.detach())
observations['vector'] = {'scale': scale.item(), 'codes': q.tolist(), 'forward': y.detach().tolist(),
                          'squared_loss_ste_gradient': x.grad.tolist(),
                          'squared_loss_round_gradient': hard.grad.tolist(),
                          'parameters_unchanged_after_backward': torch.equal(original, x.detach()),
                          'constant_scale_derivation': 's=9/70；x/s=[7/9,28/9,7]；round=[1,3,7]；q*s=[9/70,27/70,9/10]。平方代價導數為 2y；STE 的局部替代導數為 1。'}
linear = nn.Linear(4, 3)
qat = _QATLinear(linear)
packed = QuantizedLinear(linear, bits=4)
inputs = torch.randn(2, 4)
qat_out, packed_out = qat(inputs), packed(inputs)
max_error = float((qat_out - packed_out).abs().max().detach())
assert max_error <= 1e-6
before = qat.weight.detach().clone()
qat_out.sum().backward()
after_backward = qat.weight.detach().clone()
optimizer = torch.optim.SGD(qat.parameters(), lr=0.01)
optimizer.step()
assert torch.equal(before, after_backward)
assert not torch.equal(before, qat.weight.detach())
observations['linear_probe'] = {'input_shape': list(inputs.shape), 'weight_shape': list(before.shape),
                                'scale_shape': list(packed.scale.shape), 'dtype': str(before.dtype),
                                'max_output_difference': max_error, 'tolerance': 1e-6,
                                'backward_preserved_weights': True, 'step_changed_weights': True}

parts = split_records(generate_records('attributes-sft'), seed=42)
dataset_raw = (json.dumps(parts, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
assert digest(dataset_raw) == results['data']['sha256']
families = {key: {r['family'] for r in rows} for key, rows in parts.items()}
assert all(not families[a] & families[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
token_counts = {split: sum(int((_example(row,128)[1]!=IGNORE).sum()) for row in rows)
                for split,rows in parts.items()}
rng = random.Random(42)
plan = [[rng.randrange(len(parts['train'])) for _ in range(16)] for _ in range(350)]
plan_sha = digest(json.dumps(plan).encode())
effective = sum(int((_example(parts['train'][i],128)[1]!=IGNORE).sum()) for batch in plan for i in batch)
assert effective == 39348
assert all(v['batch_plan_sha256'] == plan_sha for v in results['training'].values())
assert len({v['initialization_sha256'] for v in results['training'].values()}) == 1
observations['dataset'] = {'sha256': digest(dataset_raw), 'matches_record': True,
                           'counts': {k: len(v) for k,v in parts.items()},
                           'families': {k: sorted(v) for k,v in families.items()},
                           'effective_targets_per_split': token_counts,
                           'sample_plan_sha256': plan_sha,
                           'training_targets_in_sample_plan': effective,
                           'training_record': results['training']}
tok = ByteTokenizer()
summary = {}
for name, variant in results['runs'].items():
    summary[name] = {}
    for split in ('validation','test'):
        value = variant[split]
        samples = value['generated_samples']
        recomputed_correct = 0
        recomputed_eos = 0
        unexpected = []
        for index, (sample, row) in enumerate(zip(samples, parts[split], strict=True)):
            assert sample['question'] == row['messages'][0]['content']
            assert sample['expected'] == row['messages'][-1]['content']
            ids = sample['generated_ids']
            raw_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            hit = raw_ids == tok.encode(sample['expected'])
            assert hit == sample['exact']
            assert tok.decode(raw_ids) == sample['generated']
            recomputed_correct += hit
            recomputed_eos += tok.eos_id in ids
            unexpected.extend([{'sample': index, 'id': i} for i in raw_ids if i < 8])
        mean = value['nll_sum']/token_counts[split]
        assert abs(mean-value['answer_nll']) <= 1e-12
        assert value['supervised_tokens'] == token_counts[split]
        assert value['examples'] == len(samples) == len(parts[split])
        assert value['correct'] == recomputed_correct
        assert value['eos_count'] == recomputed_eos
        summary[name][split] = {'nll_sum': value['nll_sum'], 'supervised_tokens': token_counts[split],
                                'mean_nll_recomputed': mean, 'rounded_4dp': f'{mean:.4f}',
                                'correct_recomputed': recomputed_correct, 'examples': len(samples),
                                'eos_recomputed': recomputed_eos, 'unexpected_special_ids': unexpected}
    st = variant['storage']
    assert st['tensor_bytes'] == st['parameter_tensor_bytes']+st['buffer_tensor_bytes']
    assert st['file_bytes']-st['tensor_bytes'] == st['file_overhead_bytes']
    summary[name]['storage'] = {k:st[k] for k in ('tensor_bytes','file_bytes','file_overhead_bytes','parameter_count')}
    summary[name]['example'] = next(sample for sample in variant['test']['generated_samples']
                                    if sample['question']=='color=red;shape=square;pitch=high;describe')
assert summary['matched_ptq4']['storage']['tensor_bytes'] == summary['qat_packed4']['storage']['tensor_bytes']
assert all(v['test']['eos_recomputed']==10 and not v['test']['unexpected_special_ids'] for v in summary.values())
observations['historical_results_recalculated'] = summary
observations['historical_environment'] = {k:record[k] for k in ('revision','device','seed','torch_version','python_version','gpu','step_scale','evidence_status')}
observations['historical_fake_deployed_max_logit_difference'] = results['fake_deployed_max_logit_difference']
cli_receipts = []
for arguments in ([sys.executable, '-m', 'scripts.course_experiments.run', '--help'],
                  [sys.executable, 'scripts/infer.py', '--help']):
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, check=True)
    cli_receipts.append({'command':arguments, 'exit_code':result.returncode,
                         'stdout':result.stdout, 'stderr':result.stderr})
observations['cli_help'] = cli_receipts

remote = {}
paper_data = [
    ('jacob', 'https://arxiv.org/html/1712.05877v1', '3 Training with simulated quantization', '4 Experiments',
     'all weights and biases are stored in floating point',
     '讀取訓練模擬量化、範圍估計及訓練/推論圖流程。式 (12) 在浮點計算裡套用 clamp/round/還原；Algorithm 1 分開浮點訓練圖與低位元推論圖。原文同時模擬權重及 activation，並非本專案只量化 Linear 權重的完整複製。'),
    ('bengio', 'https://arxiv.org/html/1308.3432v1', '4 Straight-Through Estimator', '5 Conditional Computation Experiments',
     'It is clearly a biased estimator',
     '讀取 Straight-Through Estimator 全節與式 (13)：把硬閾值當 identity 傳梯度；作者明言有偏，深層連梯度符號也不保證。研究對象是 stochastic binary neurons，本課採用相同的替代梯度想法，不能將其當 round 的真正導數或品質保證。'),
]
for key,url,start,end,excerpt,note in paper_data:
    raw, receipt = fetch(url)
    selected = section_text(raw,start,end)
    assert excerpt in selected
    remote[key] = {**receipt, 'url': url, 'accessed_on': '2026-10-20',
                   'locator': start, 'selected_text_sha256': digest(selected.encode()),
                   'selected_text_characters': len(selected), 'excerpt': excerpt, 'inspection_note': note}

official = [
    ('round','tools/autograd/derivatives.yaml','- name: round(Tensor self) -> Tensor\n  self: zeros_like(grad)',
     'round 與 round.decimals 的 backward 規則都是 zeros_like(grad)。'),
    ('detach','torch/_tensor.py','The result will never require gradient.',
     '讀取 detach 與 backward docstrings：detach 的結果不需梯度、共享 storage；backward 計算並累積葉 tensor 梯度。'),
    ('requires_grad','torch/_tensor_docs.py','Change if autograd should record operations on this tensor:',
     '讀取 requires_grad_ docstring：原地切換 autograd 是否記錄操作。'),
    ('clone','torch/_torch_docs.py','Returns a copy of :attr:`input`.',
     '讀取 clone docstring：複製輸入；操作本身可求導，因此先 detach 再 clone 才斷開來源計算關係。'),
    ('optimizer','torch/optim/optimizer.py','Perform a single optimization step to update parameter.',
     '讀取 Optimizer.step 契約：執行最佳化步驟以更新參數。'),
    ('cross_entropy','torch/nn/modules/loss.py','on an input, followed by :class:`~torch.nn.NLLLoss`.',
     '讀取 CrossEntropyLoss 的 class-index 公式、ignore_index 及 sum/mean reduction：未加權時是有效目標的負 log softmax 加總或平均，不等同 greedy 生成正確率。'),
]
for key,path,excerpt,note in official:
    url = 'https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/'+path
    raw, receipt = fetch(url)
    assert excerpt in raw.decode()
    remote[key] = {**receipt, 'url':url, 'version':'v2.14.1', 'accessed_on':'2026-10-20',
                   'locator':path, 'excerpt':excerpt, 'inspection_note':note}

path = 'scripts/course_experiments/compression.py'
url = 'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/'+record['revision']+'/'+path
raw, receipt = fetch(url)
assert digest(raw) == record['code_sha256'][path]
names = {'_teacher','_dataset','_example','_prompt','_answer','_fit_text','_QATLinear','_qat_layers','_packed','_evaluate','run_qat','_storage'}
current = (ROOT/path).read_bytes()
old_nodes = function_dumps(raw,names)
new_nodes = function_dumps(current,names)
matches = {key: old_nodes[key]==new_nodes[key] for key in sorted(names)}
assert all(matches.values())
remote['experiment_code'] = {**receipt, 'url':url, 'version':record['revision'], 'accessed_on':'2026-10-20',
                              'receipt_hash_matches_record':True, 'current_sha256':digest(current),
                              'function_ast_matches':matches,
                              'inspection_note':'現行檔案與實驗版本的差異位於其他多模態蒸餾流程；列出的 QAT 與評估函式 AST 相同。'}
observations['original_sources'] = remote
observations['repository_source_hashes'] = {path: digest((ROOT/path).read_bytes()) for path in [
    'tiny_perceptron/quantization.py','tiny_perceptron/data.py','tiny_perceptron/model.py',
    'scripts/course_experiments/compression.py','scripts/course_experiments/common.py',
    'scripts/course_experiments/run.py','scripts/course_experiments/text.py','scripts/prepare_data.py',
    'scripts/infer.py','tiny_perceptron/training.py','docs/course-experiments/plan.json']}
observations['checks_passed'] = True
print(json.dumps(observations, ensure_ascii=False, indent=2, allow_nan=False))
