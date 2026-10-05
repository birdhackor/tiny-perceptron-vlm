from pathlib import Path
import hashlib
import json
import platform
import importlib.util
import torch
from torch import nn

torch.set_num_threads(1)
torch.set_default_device('cpu')
torch.manual_seed(42)
dest = Path(__file__).resolve().parent
root = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('section_2_1_simple', root / 'tiny_perceptron/simple.py')
simple = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simple)
bigram = simple.BigramLM(5)
bigram_contexts = torch.tensor([[0, 1, 2], [3, 4, 2]])
bigram_scores = bigram(bigram_contexts)
assert torch.equal(bigram_scores[0], bigram_scores[1])
assert bigram.table.weight.shape == (5, 5)
table = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
ids = torch.tensor([[1, 2, 1]])
embedding = nn.Embedding(5, 3)
with torch.no_grad():
    embedding.weight.copy_(table)
before = embedding.weight.detach().clone()
x = embedding(ids)
assert torch.equal(x, torch.tensor([[[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]]]))
assert list(x.shape) == [1, 3, 3]
assert torch.equal(x[0, 0], x[0, 2])
exercise = embedding(torch.tensor([[1, 1, 1]]))
assert torch.equal(exercise, torch.tensor([[[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]]))
assert exercise.shape == x.shape
assert embedding.weight.requires_grad and x.requires_grad
assert sum(p.numel() for p in embedding.parameters()) == 5 * 3 == 15

# A pure sum makes the contribution count independently predictable.
x.sum().backward()
expected_grad = torch.tensor([[0., 0., 0.], [2., 2., 2.], [1., 1., 1.], [0., 0., 0.], [0., 0., 0.]])
assert torch.equal(embedding.weight.grad, expected_grad)
assert torch.equal(embedding.weight.detach(), before)
gradient = embedding.weight.grad.tolist()

# The candidate recipe is an added reviewer check, absent from the textbook fence.
recipe = torch.tensor([2., 0., 1.])
original_scores = (x.detach() @ recipe).tolist()
assert original_scores == [[2., 0., 2.]]
with torch.no_grad():
    embedding.weight[1, 0] = 1.5
modified_scores = (embedding(ids).detach() @ recipe).tolist()
assert modified_scores == [[3., 0., 3.]]
changed_recipe = torch.tensor([3., 0., 1.])
assert (table[1] @ changed_recipe).item() == 3.

# Re-labeling integer IDs while moving matching rows preserves the represented characters.
permutation = torch.tensor([2, 4, 0, 1, 3])
inverse = torch.argsort(permutation)
relabeled_table = table[permutation]
assert torch.equal(relabeled_table[inverse[ids]], table[ids])

# Constructors with different seeds give different parameters; no learning has occurred.
torch.manual_seed(111)
random_a = nn.Embedding(5, 3)
torch.manual_seed(222)
random_b = nn.Embedding(5, 3)
assert not torch.equal(random_a.weight, random_b.weight)

# One synthetic, five-class CPU update illustrates the stated chain, not learned language quality.
train_embedding = nn.Embedding(5, 3)
head = nn.Linear(3, 5, bias=False)
with torch.no_grad():
    train_embedding.weight.copy_(table)
    head.weight.zero_()
    head.weight[3].copy_(recipe)
optimizer = torch.optim.SGD([*train_embedding.parameters(), *head.parameters()], lr=0.1)
target = torch.tensor([3])
logits_before = head(train_embedding(torch.tensor([1])))
probability_before = logits_before.softmax(-1)[0, 3].item()
loss_before = nn.functional.cross_entropy(logits_before, target)
weight_before = train_embedding.weight.detach().clone()
head_before = head.weight.detach().clone()
optimizer.zero_grad()
loss_before.backward()
assert torch.count_nonzero(train_embedding.weight.grad[[0, 2, 3, 4]]) == 0
assert torch.count_nonzero(train_embedding.weight.grad[1]) > 0
assert torch.equal(train_embedding.weight.detach(), weight_before)
optimizer.step()
logits_after = head(train_embedding(torch.tensor([1])))
probability_after = logits_after.softmax(-1)[0, 3].item()
loss_after = nn.functional.cross_entropy(logits_after, target)
assert probability_after > probability_before
assert loss_after < loss_before
assert not torch.equal(train_embedding.weight.detach(), weight_before)
assert not torch.equal(head.weight.detach(), head_before)
assert torch.equal(train_embedding.weight.detach()[[0, 2, 3, 4]], weight_before[[0, 2, 3, 4]])

result = {
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'torch_git_version': str(torch.version.git_version), 'device': str(x.device),
                    'cuda_build': str(torch.version.cuda), 'cuda_available': str(torch.cuda.is_available()),
                    'threads': str(torch.get_num_threads())},
    'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'axes': {'B': 1, 'C': 3, 'D': 3, 'V': 5, 'unit': 'dimensionless toy feature and candidate-score numbers'},
    'original_shape': list(x.shape), 'original_vectors': x.detach().tolist(),
    'exercise_shape': list(exercise.shape), 'exercise_vectors': exercise.detach().tolist(),
    'table_parameter_count': 15, 'sum_loss_gradient': gradient,
    'BigramLM_same_last_id_same_scores': True, 'BigramLM_parameter_shape': list(bigram.table.weight.shape),
    'BigramLM_source_sha256': hashlib.sha256((root / 'tiny_perceptron/simple.py').read_bytes()).hexdigest(),
    'forward_and_backward_without_step_preserve_table': True,
    'candidate_scores_original': original_scores, 'candidate_scores_after_cat_1_to_1_5': modified_scores,
    'cat_score_after_recipe_multiplier_2_to_3': 3.0,
    'ID_relabeling_preserves_lookup': True, 'different_constructor_seeds_produce_different_weights': True,
    'synthetic_one_step': {'samples': 1, 'positions': 1, 'classes': 5, 'SGD_steps': 1, 'lr': 0.1,
                           'target_id': 3, 'probability_before': probability_before,
                           'probability_after': probability_after, 'loss_before': loss_before.item(),
                           'loss_after': loss_after.item(),
                           'only_used_embedding_row_changed': True, 'head_changed': True,
                           'scope': 'Constructed numerical chain demonstration only; no language learning or accuracy claim.'},
    'tolerance': 'lookup, count and toy scores: exact equality; step checks: strict direction, not a promised training rate',
}
print(json.dumps(result, ensure_ascii=False, indent=2))
