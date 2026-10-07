import json,copy,torch,hashlib
from tiny_perceptron.model import TinyLM,ModelConfig
policy=TinyLM(ModelConfig(width=8));reference=policy.eval().requires_grad_(False);before=reference.embedding.weight.clone()
with torch.no_grad():policy.embedding.weight.add_(.1)
print('alias_reference_unchanged',torch.equal(before,reference.embedding.weight),'policy_all_frozen',all(not p.requires_grad for p in policy.parameters()))
pj='docs/course-experiments/results/dpo.json';sj='docs/course-experiments/results/style.json';j=json.load(open(pj));s=json.load(open(sj))
r=next(r for r in j['results']['runs']['model']['preference']['validation']['samples'] if r['prompt']=='3+5=?')
pm=r['policy_chosen_logp']-r['policy_rejected_logp'];delta=pm-r['reference_margin'];print('3+5',r,'computed_policy_margin',pm,'computed_relative',delta);assert abs(pm-r['policy_margin'])<1e-10 and abs(delta-r['relative_margin'])<1e-10 and pm<0
base=s['results']['content_evaluation']['test'];n=len(base['samples']);matches=sum(r['generated']==r['expected'] for r in base['samples']);print('content.pt baseline',matches,n,'reference_unchanged_raw',j['results']['reference_unchanged']);assert n==7 and matches==base['matches']==0
print('toy relative',(-3-(-3))-(-4-(-3)))
for p in (pj,sj):print('raw_sha',p,hashlib.sha256(open(p,'rb').read()).hexdigest())
print('torch',torch.__version__)
