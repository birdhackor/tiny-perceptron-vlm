"""Independent bounded CPU check of freeze-02 lesson 7.5; no optimizer or training."""
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
import warnings
from pathlib import Path

import torch

from tiny_perceptron.data import render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss

ROOT = Path(__file__).resolve().parents[4]
torch.set_num_threads(1)
source = ROOT / 'course/chapters/07.md'
body = re.search(r'^## 7\.5 .*?(?=^## |\Z)', source.read_text(), re.M | re.S).group()
canonical = re.findall(r'```python\n(.*?)```', body, re.S)
assert len(canonical) == 1
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(canonical[0], str(source) + '#7.5', 'exec'), {})

rows = []
for question, mode in [('Q', 'ordinary'), ('Z', 'ordinary'), ('Q', 'detach_inputs'), ('Q', 'block_question_key')]:
    torch.manual_seed(42)
    x, y = render_chat([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': 'A'}])
    model = TinyLM(ModelConfig(width=8))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    inputs = model.embedding(x[None])
    inputs.retain_grad()
    supplied = inputs.detach() if mode == 'detach_inputs' else inputs
    valid = None
    if mode == 'block_question_key':
        valid = torch.ones_like(x[None], dtype=torch.bool)
        valid[0, 2] = False
    logits = model(embeddings=supplied, valid=valid)['logits']
    is_leaf = logits.is_leaf
    retained_before = logits.retains_grad
    logits.retain_grad()
    loss = masked_loss(logits, y[None])
    loss.backward()
    emb_grad = model.embedding.weight.grad
    q_row = None if emb_grad is None else emb_grad[x[2]]
    input_q_grad = None if inputs.grad is None else inputs.grad[0, 2]
    changed = [n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])]
    ignored_grad = logits.grad[y[None] == -100]
    assert torch.equal(ignored_grad, torch.zeros_like(ignored_grad))
    assert not changed
    if mode == 'ordinary':
        assert q_row is not None and q_row.norm() > 0
        assert torch.equal(q_row, input_q_grad)  # question byte occurs once; output weights untied.
    if mode == 'detach_inputs':
        assert emb_grad is None and inputs.grad is None
    if mode == 'block_question_key':
        assert q_row is not None and torch.equal(q_row, torch.zeros_like(q_row))
    norm_manual = None if q_row is None else q_row.square().sum().sqrt().item()
    rows.append({
        'question': question, 'mode': mode, 'seed': 42,
        'x': x.tolist(), 'y': y.tolist(), 'x_dtype': str(x.dtype), 'y_dtype': str(y.dtype),
        'question_index': 2, 'question_id': x[2].item(), 'logits_shape': list(logits.shape),
        'logits_dtype': str(logits.dtype), 'logits_is_leaf': is_leaf,
        'retains_grad_before': retained_before, 'retains_grad_after': logits.retains_grad,
        'loss': loss.item(), 'supervised_tokens': int((y != -100).sum()),
        'supervised_positions': (y != -100).nonzero().flatten().tolist(),
        'logits_grad_norm_by_position': logits.grad[0].norm(dim=-1).tolist(),
        'question_embedding_gradient': None if q_row is None else q_row.tolist(),
        'question_embedding_grad_norm': None if q_row is None else q_row.norm().item(),
        'question_embedding_norm_sqrt_sum_square': norm_manual,
        'embedding_grad_is_none': emb_grad is None,
        'unused_id_90_grad_norm': None if emb_grad is None else emb_grad[90].norm().item(),
        'input_question_grad_norm': None if input_q_grad is None else input_q_grad.norm().item(),
        'input_equals_embedding_row_gradient': None if input_q_grad is None else torch.equal(q_row, input_q_grad),
        'embedding_and_output_weights_tied': model.embedding.weight is model.output.weight,
        'valid_mask': None if valid is None else valid.tolist(),
        'changed_parameters': changed, 'optimizer_steps': 0,
    })

# A finite-difference derivative of the question row checks an actual loss dependency.
torch.manual_seed(42)
x, y = render_chat([{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': 'A'}])
model = TinyLM(ModelConfig(width=8)).double()
loss = masked_loss(model(x[None])['logits'], y[None])
loss.backward()
gradient = model.embedding.weight.grad[89, 0].item()
eps = 1e-5
with torch.no_grad():
    original = model.embedding.weight[89, 0].item()
    model.embedding.weight[89, 0] = original + eps
    plus = masked_loss(model(x[None])['logits'], y[None]).item()
    model.embedding.weight[89, 0] = original - eps
    minus = masked_loss(model(x[None])['logits'], y[None]).item()
    model.embedding.weight[89, 0] = original
finite_difference = (plus - minus) / (2 * eps)
assert abs(gradient - finite_difference) < 1e-8
result = {
    'purpose': 'Micro CPU gradient/dependency checks only; no historical experiment reproduction or parameter training.',
    'command': 'uv run --frozen --extra cpu --no-sync python docs/technical-reviews/artifacts/repair-20261017-7-5/verify.py',
    'environment': {'python': sys.version, 'torch': torch.__version__, 'torch_git_version': torch.version.git_version,
                    'device': 'cpu', 'platform': platform.platform(), 'torch_threads': str(torch.get_num_threads())},
    'source_sha256': hashlib.sha256(body.encode()).hexdigest(),
    'canonical_snippet_sha256': hashlib.sha256(canonical[0].encode()).hexdigest(),
    'canonical_stdout': stdout.getvalue(), 'cases': rows,
    'finite_difference': {'dtype': 'torch.float64', 'question_id': 89, 'feature': 0, 'epsilon': eps,
                          'autograd': gradient, 'central_difference': finite_difference,
                          'absolute_error': abs(gradient - finite_difference), 'tolerance': 1e-8},
    'assertions': 'All passed. No optimizer instantiated; zero optimizer updates.',
}
print(json.dumps(result, ensure_ascii=False, indent=2))
