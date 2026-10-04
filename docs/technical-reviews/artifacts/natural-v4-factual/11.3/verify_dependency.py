"""Short CPU recheck of newly required Adam/AdamW freeze qualifications."""
import hashlib
import json
import math
import platform
from pathlib import Path

import torch
from torch import nn

ART = Path(__file__).resolve().parent
ROOT = ART.parents[4]
RESEARCH = ROOT / 'outputs/natural-v4/factual-research/11.3'
torch.set_num_threads(2)
result = {'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                           'torch_git': torch.version.git_version, 'device': 'cpu'},
          'scope': 'Own new bounded dependency checks only. Earlier main CPU probes were not rerun.'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

result['official_source_matches'] = {}
for file in ['adam.py', 'adamw.py']:
    installed = Path(torch.__file__).parent / 'optim' / file
    retrieved = RESEARCH / ('torch-' + file)
    assert installed.read_bytes() == retrieved.read_bytes()
    result['official_source_matches'][file] = {'installed_sha256': digest(installed),
                                              'retrieved_sha256': digest(retrieved), 'byte_equal': True}

# Actual 5.4 input and exercise, in the lesson's float32 then independent Python arithmetic.
rows = []
for second in [100.0, -100.0]:
    g = torch.tensor([1.0, second])
    m = 0.9 * torch.zeros(2) + 0.1 * g
    v = 0.999 * torch.zeros(2) + 0.001 * g.square()
    mhat = m / (1 - 0.9)
    vhat = v / (1 - 0.999)
    update = 0.001 * mhat / (vhat.sqrt() + 1e-8)
    expected = [0.001 * x / (abs(x) + 1e-8) for x in [1.0, second]]
    assert torch.allclose(update, torch.tensor(expected), atol=1e-9, rtol=1e-6)
    rows.append({'grad': g.tolist(), 'm': m.tolist(), 'v': v.tolist(), 'mhat': mhat.tolist(),
                 'vhat': vhat.tolist(), 'update': update.tolist(), 'independent_expected': expected})
result['adam_first_step'] = rows

# Actual 5.5 example and both exercises: zero tensor participates, None is skipped.
cases = []
for decay, gradient in [(0.2, 'zero'), (0.0, 'zero'), (0.2, 'none')]:
    w = nn.Parameter(torch.tensor([2.0]))
    opt = torch.optim.AdamW([w], lr=0.1, weight_decay=decay)
    w.grad = torch.zeros_like(w) if gradient == 'zero' else None
    opt.step()
    expected = 2 * (1 - 0.1 * decay) if gradient == 'zero' else 2.0
    observed = float(w.detach())
    assert math.isclose(observed, expected, rel_tol=1e-6, abs_tol=2e-7)
    cases.append({'lr': 0.1, 'weight_decay': decay, 'gradient': gradient,
                  'expected': expected, 'observed': observed})
result['zero_vs_none'] = cases

# Changing requires_grad does not clear the existing leaf gradient or optimizer list.
w = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
opt = torch.optim.AdamW([w], lr=0.1, weight_decay=0.2)
w.sum().backward()
original_gradient = w.grad
w.requires_grad_(False)
assert w.grad is original_gradient and float(w.grad) == 1.0
assert any(p is w for group in opt.param_groups for p in group['params'])
opt.step()
expected = 2 * 0.98 - 0.1 / (1 + 1e-8)
assert math.isclose(float(w), expected, rel_tol=1e-12)
after_stale = float(w)
w.grad = None
opt.step()
assert float(w) == after_stale
result['stale_gradient'] = {'requires_grad': w.requires_grad, 'old_gradient_remains_after_flag': True,
                            'optimizer_still_contains_parameter': True, 'value_before': 2.0,
                            'expected_after_stale_step': expected, 'observed_after_stale_step': after_stale,
                            'after_none_step': float(w), 'none_step_skipped': True}

# Historical first moment can move a parameter even with current zero task gradient and no decay.
w = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
opt = torch.optim.AdamW([w], lr=0.1, weight_decay=0.0)
w.grad = torch.ones_like(w)
opt.step()
first = float(w.detach())
w.requires_grad_(False)
w.grad = torch.zeros_like(w)
opt.step()
m2, v2 = 0.9 * 0.1, 0.999 * 0.001
expected = first - 0.1 * (m2 / (1 - 0.9**2)) / (math.sqrt(v2 / (1 - 0.999**2)) + 1e-8)
assert math.isclose(float(w), expected, rel_tol=1e-12)
assert float(w) < first
result['history_with_zero_gradient'] = {'weight_decay': 0.0, 'after_first_gradient': first,
                                       'expected_after_zero': expected, 'observed_after_zero': float(w),
                                       'exp_avg': float(opt.state[w]['exp_avg']),
                                       'exp_avg_sq': float(opt.state[w]['exp_avg_sq']), 'step': float(opt.state[w]['step'])}

# Honest reuse: check every prior main CPU input/artifact binding rather than rerunning it.
prior = json.loads((ART / 'cpu-results.json').read_text())
result['main_evidence_reuse'] = {'code': {}, 'checkpoints': {}}
for path, binding in prior['code_bindings'].items():
    current = digest(ROOT / path)
    assert current == binding['sha256']
    result['main_evidence_reuse']['code'][path] = current
for key, binding in prior['checkpoint_bindings'].items():
    assert digest(ROOT / binding['path']) == binding['sha256']
    assert digest(ROOT / binding['release_manifest']) == binding['manifest_sha256']
    result['main_evidence_reuse']['checkpoints'][key] = binding['sha256']
for row in json.loads((ART/'render-receipt.json').read_text()):
    assert digest(ROOT/row['svg']) == row['svg_sha256']
    assert digest(ROOT/row['render']) == row['render_sha256']
result['main_evidence_reuse']['prior_cpu_results_sha256'] = digest(ART/'cpu-results.json')
result['main_evidence_reuse']['reran_main_probes'] = False
result['main_evidence_reuse']['rerendered_or_reviewed_unchanged_svg_on_resume'] = False
(ART/'dependency-cpu-results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
