"""Independent bounded CPU verification of section 13.5; no model/checkpoint access."""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.data import ByteTokenizer, render_chat

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
out = Path(__file__).resolve().parent
rows = []
for chosen_value in [-4.0, -5.0]:
    chosen = torch.tensor([chosen_value], dtype=torch.float64, requires_grad=True)
    rejected = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    ref_chosen = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    ref_rejected = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
    loss.backward()
    margin = (chosen_value - (-3.0)) - (-4.0 - (-3.0))
    expected_loss = math.log1p(math.exp(-0.1 * margin))
    expected_chosen = -0.1 / (1 + math.exp(0.1 * margin))
    h = 1e-5
    numeric_grad = (math.log1p(math.exp(-0.1 * (margin + h))) - math.log1p(math.exp(-0.1 * (margin - h)))) / (2*h)
    assert abs(loss.item() - expected_loss) < 1e-12
    assert abs(chosen.grad.item() - expected_chosen) < 1e-12
    assert abs(chosen.grad.item() - numeric_grad) < 1e-10
    assert rejected.grad.item() == -chosen.grad.item()
    assert chosen.item() == chosen_value and rejected.item() == -3.0
    assert ref_chosen.grad is None and ref_rejected.grad is None
    rows.append(dict(chosen=chosen_value, rejected=-3, margin=margin,
                     loss=loss.item(), chosen_gradient=chosen.grad.item(),
                     rejected_gradient=rejected.grad.item(), finite_difference=numeric_grad,
                     unchanged_after_backward=True, reference_gradients_detached=True))

new_chosen = -4.0 - 0.1 * rows[0]['chosen_gradient']
new_rejected = -3.0 - 0.1 * rows[0]['rejected_gradient']
new_gap = new_chosen - new_rejected
assert abs(new_chosen - (-3.995)) < 1e-12
assert abs(new_rejected - (-3.005)) < 1e-12
assert abs(new_gap - (-0.99)) < 1e-12

batch = torch.tensor([-4.0, -4.0], dtype=torch.float64, requires_grad=True)
batch_loss = dpo_loss(batch, torch.full_like(batch,-3), torch.full_like(batch,-4), torch.full_like(batch,-3))
batch_loss.backward()
assert torch.allclose(batch.grad, torch.full_like(batch,-0.025), atol=1e-12, rtol=0)
bad_beta = {}
for beta in [0,-0.1]:
    try:
        dpo_loss(torch.tensor([-4.]),torch.tensor([-3.]),torch.tensor([-4.]),torch.tensor([-3.]),beta=beta)
    except ValueError as e:
        bad_beta[str(beta)] = str(e)
    else:
        raise AssertionError('nonpositive beta accepted')

raw_path = Path('docs/course-experiments/results/dpo.json')
raw = raw_path.read_bytes()
original = json.loads(raw)
model = original['results']['runs']['model']
preference = model['preference']['test']
arithmetic = model['arithmetic']['test']
assert preference['records'] == len(preference['samples']) == 7
assert arithmetic['records'] == len(arithmetic['samples']) == 7
chosen_higher = 0
for row in preference['samples']:
    computed_margin = row['policy_chosen_logp'] - row['policy_rejected_logp']
    assert abs(computed_margin - row['policy_margin']) < 1e-12
    assert abs(computed_margin - row['reference_margin'] - row['relative_margin']) < 1e-12
    chosen_higher += computed_margin > 0
assert chosen_higher == preference['chosen_higher_absolute_probability']
selected = [(i,s) for i,s in enumerate(preference['samples']) if s['prompt']=='4+2=?']
generated = [(i,s) for i,s in enumerate(arithmetic['samples']) if s['messages']==[{'role':'user','content':'4+2=?'}]]
assert len(selected) == len(generated) == 1
pi,p = selected[0]; gi,g = generated[0]
tok = ByteTokenizer()
assert model['beta'] == 0.1 and model['training']['steps'] == 250
assert p['chosen'] == g['expected'] == '6' and p['rejected'] == '7'
assert abs(p['policy_chosen_logp'] - (-24.88083)) < 5e-6
assert abs(p['policy_rejected_logp'] - (-28.11597)) < 5e-6
assert p['policy_chosen_logp'] > p['policy_rejected_logp']
assert g['generated_ids'] == tok.encode('8') + [tok.eos_id] == [64,2]
assert g['generated'] == '8' and g['eos'] is True and g['exact'] is False
assert sum(int(s['exact']) for s in arithmetic['samples']) == arithmetic['matches']
assert abs(sum(int(s['eos']) for s in arithmetic['samples'])/7 - arithmetic['eos_rate']) < 1e-12
token_counts={}
for name in ['chosen','rejected']:
    _, labels=render_chat([{'role':'user','content':p['prompt']},{'role':'assistant','content':p[name]}])
    token_counts[name]=int((labels!=-100).sum())
    assert token_counts[name] == p[name+'_answer_tokens'] == 2
measurement = {
    'original_file':str(raw_path),'original_file_sha256':hashlib.sha256(raw).hexdigest(),
    'pointers_read':['/schema_version','/experiment_id','/revision','/device','/seed','/torch_version','/python_version','/evidence_status',
        '/code_sha256','/results/data/test','/results/runs/model/beta','/results/runs/model/training/steps',
        '/results/runs/model/preference/test/records','/results/runs/model/preference/test/chosen_higher_absolute_probability',
        '/results/runs/model/preference/test/samples','/results/runs/model/arithmetic/test/records',
        '/results/runs/model/arithmetic/test/matches','/results/runs/model/arithmetic/test/eos_rate',
        '/results/runs/model/arithmetic/test/samples'],
    'selected_pointers':[f'/results/runs/model/preference/test/samples/{pi}', f'/results/runs/model/arithmetic/test/samples/{gi}'],
    'candidate_sample':p,'generated_sample':g,'checked_token_counts':token_counts,
    'denominators':{'selected_question':1,'test_records':7,'complete_answer_tokens_per_candidate':2,'training_updates':250},
    'checks':{'candidate_margin_recomputed':p['policy_chosen_logp']-p['policy_rejected_logp'],
              'candidate_probability_ratio':math.exp(p['policy_chosen_logp']-p['policy_rejected_logp']),
              'generated_raw_ids_decoded':tok.decode(g['generated_ids'][:-1])},
    'scope':'Existing raw measurements audited; no training, weight loading, or model re-evaluation.'
}
(out/'raw-measurement-audit.json').write_text(json.dumps(measurement,ensure_ascii=False,indent=2)+'\n')
result = {'environment':{'python':sys.version,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),
                         'cuda_build':str(torch.version.cuda),'cuda_available':torch.cuda.is_available(),'device':'cpu'},
          'scalar_cases':rows,'paper_update':{'learning_rate':0.1,'new_chosen':new_chosen,'new_rejected':new_rejected,
                                            'new_gap':new_gap,'relative_improvement':new_gap-(-1)},
          'mean_batch_gradient':batch.grad.tolist(),'nonpositive_beta':bad_beta,
          'raw_measurement_audit':'raw-measurement-audit.json','all_assertions_passed':True}
(out/'bounded-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
