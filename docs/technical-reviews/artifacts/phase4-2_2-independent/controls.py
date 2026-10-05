"""Bounded independent 2.2 controls; exact fence is run first, CPU only."""
from pathlib import Path
import hashlib
import json
import sys
import torch
from torch import nn

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
from tiny_perceptron.simple import ContextMLP

torch.set_num_threads(1)
torch.set_default_device('cpu')
torch.manual_seed(20261005)
base = Path(__file__).resolve().parent
scope = {'__name__': '__main__'}
original = (base/'original/fence-1.py').read_bytes()
exec(compile(original, 'original/fence-1.py', 'exec'), scope)
embedding, ids, x, context = (scope[name] for name in ('embedding', 'ids', 'x', 'context'))
weight_before = embedding.weight.detach().clone()
assert ids.shape == (2, 3) and x.shape == (2, 3, 2)
assert torch.equal(context, torch.tensor([[1.,10.,2.,20.,3.,30.],[3.,30.,2.,20.,1.,10.]]))
assert torch.equal(context.reshape(2, 3, 2), x)
assert context.numel() == x.numel() == 12
assert torch.equal(context.sum(1), torch.tensor([66.,66.]))
assert not torch.equal(context[0], context[1])
# The section itself has not called backward or updated any parameter.
assert embedding.weight.grad is None
assert context.requires_grad and x.requires_grad
assert type(context.grad_fn).__name__ == 'ViewBackward0'
assert torch.equal(weight_before, embedding.weight.detach())

changed_ids = ids.clone()
changed_ids[0] = torch.tensor([2,1,3])
changed = embedding(changed_ids).reshape(2, -1)
assert torch.equal(changed[0], torch.tensor([2.,20.,1.,10.,3.,30.]))
assert torch.equal(changed[1], context[1])
assert torch.equal(changed.sum(1), context.sum(1))

position_recipe = nn.Linear(6, 1, bias=False)
with torch.no_grad():
    position_recipe.weight.copy_(torch.tensor([[1.,0.,0.,0.,-1.,0.]]))
position_scores = position_recipe(context)
assert torch.equal(position_scores.detach(), torch.tensor([[-2.],[2.]]))

shape_cases=[]
for batch, positions, features in [(1,3,2), (2,5,2), (3,2,4)]:
    labels = torch.arange(batch*positions*features).reshape(batch,positions,features)
    flattened = labels.reshape(batch,-1)
    assert flattened.shape == (batch, positions*features)
    assert torch.equal(flattened.reshape(batch,positions,features), labels)
    shape_cases.append({'BCD':[batch,positions,features],'out':list(flattened.shape),'elements':flattened.numel()})

five = embedding(torch.tensor([[1,2,3,1,2]])).reshape(1,-1)
try:
    position_recipe(five)
except RuntimeError as error:
    fixed_linear_error = str(error)
else:
    raise AssertionError('Six-feature Linear wrongly accepted ten features')
model = ContextMLP(vocab_size=4, context=3, width=2)
assert model(ids).shape == (2,4)
try:
    model(torch.tensor([[1,2,3,1,2]]))
except ValueError as error:
    fixed_model_error = str(error)
else:
    raise AssertionError('ContextMLP wrongly accepted five positions')

# A version-independent autograd contract, independent from the internal node name.
context.sum().backward()
expected_gradient = torch.tensor([[0.,0.],[2.,2.],[2.,2.],[2.,2.]])
assert torch.equal(embedding.weight.grad, expected_gradient)
assert torch.equal(weight_before, embedding.weight.detach())
with torch.no_grad():
    untracked = embedding(ids).reshape(2,-1)
assert not untracked.requires_grad and untracked.grad_fn is None

# reshape preserves logical ordering even when input is noncontiguous; it may copy.
noncontiguous = x.transpose(1,2)
reshaped = noncontiguous.reshape(2,-1)
expected_noncontiguous = torch.tensor([[1.,2.,3.,10.,20.,30.],[3.,2.,1.,30.,20.,10.]])
assert not noncontiguous.is_contiguous()
assert torch.equal(reshaped, expected_noncontiguous)

print(json.dumps({
 'environment': {'python':sys.version,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),'cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'device':str(context.device),'num_threads':str(torch.get_num_threads())},
 'original_code_sha256':hashlib.sha256(original).hexdigest(),
 'original_ids_shape':list(ids.shape),'embedding_shape':list(x.shape),'context_shape':list(context.shape),
 'original_values':context.detach().tolist(),'original_rows_sum':context.sum(1).detach().tolist(),
 'original_grad_node':type(context.grad_fn).__name__,
 'exercise_values':changed.detach().tolist(),'exercise_rows_sum':changed.sum(1).detach().tolist(),
 'position_recipe_weights':position_recipe.weight.detach().tolist(),'position_scores':position_scores.detach().tolist(),
 'shape_cases':shape_cases,'five_position_shape':list(five.shape),'fixed_linear_error':fixed_linear_error,'fixed_model_error':fixed_model_error,
 'gradient_after_control_backward':embedding.weight.grad.tolist(),'weight_unchanged_after_backward':True,
 'all_lookup_in_no_grad_requires_grad':untracked.requires_grad,'noncontiguous_reshape_values':reshaped.detach().tolist(),
 'tolerance':'exact equality: all values and gradient counts are exactly representable integers in float32; shapes and element counts are integers',
 'scope':'No optimizer, no training, no inference quality measurement. One isolated backward verifies graph continuity after original demonstration.'
},ensure_ascii=False,indent=2))
