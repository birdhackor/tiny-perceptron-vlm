"""9.15 independent bounded CPU check; no training or historical model reload."""

import contextlib
import hashlib
import io
import json
import math
import platform
from pathlib import Path

import torch

from tiny_perceptron.lesson_products.data import CANDIDATES
from tiny_perceptron.lesson_products.inference import Product
from tiny_perceptron.lesson_products.model import CandidateScorer, Config, Tokenizer


def concentration(p):
    n = len(p)
    return (max(p) - 1 / n) / (1 - 1 / n)


root = Path(__file__).resolve().parents[4]
frozen = Path('/workspace/work/textbook-repair-20261017/input-freeze02/pages/9.15.md')
body = frozen.read_text()
source = (root / 'course/chapters/09.md').read_text()
current = source[source.index('## 9.15 '):source.index('## 9.16 ')]
assert current == body, 'Current section differs from assigned freeze02 input'
snippet = body.split('```python\n', 1)[1].split('\n```', 1)[0]
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(snippet, 'frozen-9.15-snippet', 'exec'), {})
assert stdout.getvalue() == '最大機率 0.6 集中度 0.4\n'

cases = []
for name, p, expected in [
    ('three_choices', [0.6, 0.2, 0.2], 0.4),
    ('permuted', [0.2, 0.6, 0.2], 0.4),
    ('two_choices', [0.6, 0.4], 0.2),
    ('uniform_three', [1 / 3] * 3, 0.0),
    ('point_mass', [1.0, 0.0, 0.0], 1.0),
    ('confident_wrong_hypothetical', [0.99, 0.005, 0.005], 0.985),
]:
    observed = concentration(p)
    assert math.isclose(observed, expected, abs_tol=1e-12, rel_tol=0)
    cases.append({'name': name, 'probabilities': p, 'expected': expected,
                  'observed': observed, 'argmax_index': p.index(max(p))})
try:
    concentration([1.0])
except ZeroDivisionError as error:
    singleton_formula = type(error).__name__
else:
    raise AssertionError('n=1 should cause division by zero in the stated formula')

# Exercise the real Product.judge path with an explicitly untrained tiny scorer.
# This verifies returned fields only, not the historical product's learned quality.
torch.set_num_threads(1)
torch.manual_seed(915)
product = Product.__new__(Product)
product.product = 'c'
product.device = torch.device('cpu')
product.tokenizer = Tokenizer()
product.config = Config(vocab_size=len(product.tokenizer), width=8, heads=2,
                        ff=16, max_length=128, c_layers=1)
product.model = CandidateScorer(product.config).eval()
product.calibration = {'temperatures': {q: 1.0 for q in CANDIDATES}}
interface = {}
singletons = {}
for question, candidates in CANDIDATES.items():
    result = product.judge('卡片進度;申請編號=?', question)
    probabilities = result['probabilities']
    assert list(probabilities) == list(candidates)
    assert math.isclose(sum(probabilities.values()), 1, abs_tol=2e-6, rel_tol=0)
    assert all(0 <= p <= 1 for p in probabilities.values())
    expected = concentration(list(probabilities.values()))
    assert math.isclose(result['concentration'], expected, abs_tol=1e-6, rel_tol=0)
    assert result['concentration_scope'] == '候選間集中程度；不是答對機率。候選集合改變時需重新計算。'
    interface[question] = {key: value for key, value in result.items() if key != 'seconds'}
    try:
        product.judge('卡片進度;申請編號=?', question,
                      ['none'] if question == 'choice' else [candidates[0]])
    except ValueError as error:
        singletons[question] = {'error': type(error).__name__, 'message': str(error)}
        assert str(error) == '至少兩個不重複的候選意義'
    else:
        raise AssertionError('The API should reject singleton candidates')

record = {
    'reviewer_task': '/root/repair_tech_9_15',
    'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                    'device': 'cpu', 'torch_threads': str(torch.get_num_threads())},
    'source_sha256': hashlib.sha256(body.encode()).hexdigest(),
    'snippet_stdout': stdout.getvalue(),
    'numeric_cases': cases,
    'singleton_formula': singleton_formula,
    'interface_setup': {'scorer': 'fresh untrained CandidateScorer; no checkpoint loaded',
                        'seed': 915, 'width': 8, 'heads': 2, 'ff': 16, 'c_layers': 1,
                        'max_length': 128, 'temperature': 1.0, 'state': '卡片進度;申請編號=?'},
    'interface_results': interface,
    'singleton_interface': singletons,
    'limits': 'No optimizer updates, no training, no historical experiment reproduction. '
              'Random predictions establish API fields only; numeric cases are hypothetical distributions.',
    'result': 'All numeric and interface assertions passed.'
}
print(json.dumps(record, ensure_ascii=False, indent=2))
