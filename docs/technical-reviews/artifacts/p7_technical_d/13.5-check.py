import torch,json,math
from tiny_perceptron.alignment import dpo_loss
c=torch.tensor([-5.],requires_grad=True);r=torch.tensor([-3.],requires_grad=True);loss=dpo_loss(c,r,torch.tensor([-4.]),torch.tensor([-3.]),beta=.1);loss.backward();print('variation_minus5',loss.item(),c.grad.item(),r.grad.item())
print('paper_step',-4-.1*(-.05),-3-.1*.05,'margin_delta',(-3.995-(-3.005))-(-4-(-3)))
j=json.load(open('docs/course-experiments/results/dpo.json'))['results']['runs']['model']
p=next(r for r in j['preference']['test']['samples'] if r['prompt']=='4+2=?');g=next(r for r in j['arithmetic']['test']['samples'] if r['messages'][0]['content']=='4+2=?')
print('candidate',p);print('generation',g);assert p['policy_chosen_logp']>p['policy_rejected_logp'] and g['expected']=='6' and g['generated']=='8' and g['eos'] is True
print('torch',torch.__version__)
