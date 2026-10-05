"""Bounded initialization and arithmetic probes; no model-quality measurements."""
from pathlib import Path
import contextlib
from decimal import Decimal
import hashlib
import io
import json
import sys

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.model import ModelConfig, TinyLM
torch.set_num_threads(1)
torch.set_default_device('cpu')
assert torch.version.cuda is None and not torch.cuda.is_available()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
original=(BASE/'original/fence-1.py').read_bytes()
needle=b'torch.manual_seed(1)\n'
location=original.rfind(needle)
assert location>original.find(needle)
variant=original[:location]+original[location+len(needle):]
(BASE/'exercise-without-second-reset.py').write_bytes(variant)
namespace={'__name__':'__main__'}
out=io.StringIO()
with contextlib.redirect_stdout(out):
    exec(compile(variant,'exercise-without-second-reset.py','exec'),namespace)
assert not torch.equal(namespace['a'].embedding.weight,namespace['b'].embedding.weight)
print('EXACT EXERCISE MUTATION OUTPUT')
print(out.getvalue(),end='')

torch.manual_seed(1)
state_before=torch.get_rng_state().clone()
a=TinyLM(ModelConfig(width=8))
state_after_a=torch.get_rng_state().clone()
b=TinyLM(ModelConfig(width=8))
state_after_b=torch.get_rng_state().clone()
assert not torch.equal(state_before,state_after_a)
assert not torch.equal(state_after_a,state_after_b)
torch.manual_seed(1)
rebuilt=TinyLM(ModelConfig(width=8))
all_parameters_equal=all(torch.equal(p,q) for p,q in zip(a.parameters(),rebuilt.parameters(),strict=True))
assert all_parameters_equal
ids=torch.tensor([0,1,263])
assert torch.equal(a.embedding(ids),a.embedding.weight[ids])
assert a.embedding.weight.requires_grad and a.embedding.weight.grad is None
count=a.embedding.weight.numel()
standard_deviation_manual=((a.embedding.weight.double()-a.embedding.weight.double().mean()).square().sum()/(count-1)).sqrt().item()
standard_deviation_api=a.embedding.weight.std().item()
assert abs(standard_deviation_manual-standard_deviation_api)<1e-6
shape=list(a.embedding.weight.shape)
assert shape==[264,8]

def permutation(width,separate):
    torch.manual_seed(1)
    data_generator=torch.Generator(device='cpu').manual_seed(17) if separate else None
    model=TinyLM(ModelConfig(width=width))
    return torch.randperm(20,generator=data_generator).tolist()
shared8,shared16=permutation(8,False),permutation(16,False)
separate8,separate16=permutation(8,True),permutation(16,True)
assert shared8!=shared16
assert separate8==separate16

A=list(map(Decimal,['0.70','0.72','0.69']))
B=list(map(Decimal,['0.71','0.71','0.73']))
differences=[b-a for a,b in zip(A,B,strict=True)]
assert differences==list(map(Decimal,['0.01','-0.01','0.04']))
values={
    'hypothetical_arithmetic':{'A':[str(x) for x in A],'B':[str(x) for x in B],
        'paired_B_minus_A':[str(x) for x in differences],
        'A_mean':str(sum(A)/3),'B_mean':str(sum(B)/3),'mean_difference':str(sum(differences)/3),
        'A_range':[str(min(A)),str(max(A))],'B_range':[str(min(B)),str(max(B))],
        'best_B_minus_best_A':str(max(B)-max(A)),
        'denominators':{'paired_repeats':3,'scores_per_method':3,'positive_pairs':2,'negative_pairs':1},
        'provenance':'Invented values explicitly stated as a hypothetical example in 5.15; no model, task, dataset, or success-rate denominator is claimed.'},
    'embedding':{'shape':shape,'elements':count,'requires_grad':a.embedding.weight.requires_grad,
        'gradient_is_none':a.embedding.weight.grad is None,'ID_lookup_matches_rows':True,
        'standard_deviation_api':standard_deviation_api,'standard_deviation_hand_formula_N_minus_1':standard_deviation_manual,
        'absolute_difference':abs(standard_deviation_api-standard_deviation_manual)},
    'rng':{'state_advanced_by_first_constructor':True,'state_advanced_by_second_constructor':True,
        'same_seed_same_parameters':all_parameters_equal,
        'exercise_without_reset_equal':False,'shared_default_rng_width8_permutation':shared8,
        'shared_default_rng_width16_permutation':shared16,
        'independent_data_generator_width8_permutation':separate8,
        'independent_data_generator_width16_permutation':separate16,
        'separate_data_generator_matches_across_widths':True},
    'environment':{'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),
        'torch_git_version':torch.version.git_version,'cuda_build':str(torch.version.cuda),
        'cuda_available':str(torch.cuda.is_available()),'device':'cpu','threads':str(torch.get_num_threads())},
    'inputs':{'original_fence_sha256':sha(original),'exact_exercise_variant_sha256':sha(variant),
        'model_py_sha256':sha((ROOT/'tiny_perceptron/model.py').read_bytes())},
    'limits':'Only CPU constructors, three-value Decimal arithmetic, row lookup and length-20 sampling. No forward/backward, training, benchmark scoring, data/model download or empirical significance claim.'
}
(BASE/'cpu-probe-results.json').write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(values,ensure_ascii=False,indent=2))
