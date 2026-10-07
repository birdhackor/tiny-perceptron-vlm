import json,copy,torch
from tiny_perceptron.posttraining import FiniteResponsePolicy
from tiny_perceptron.alignment import dpo_loss
r=json.load(open('docs/course-experiments/results/posttraining.json'))['results'];c=r['config']
print('historical_initial_hashes',r['ppo']['initial_state_sha256']==r['dpo']['initial_state_sha256']==r['reference_state_sha256_before']==r['reference_state_sha256_after'])
p=FiniteResponsePolicy();p1,p2=copy.deepcopy(p),copy.deepcopy(p);print('same_each_weight',all(torch.equal(a,b) for a,b in zip(p1.parameters(),p2.parameters())))
print('updates',r['ppo']['policy_updates'],r['dpo']['policy_updates'],'draws',c['dpo_steps']*c['dpo_batch_size'],r['dpo']['processed_pair_draws'],'extra',r['reward']['history'][-1]['step'],r['ppo']['value_updates'],r['ppo']['sampled_actions'])
assert c['dpo_steps']*c['dpo_batch_size']==r['dpo']['processed_pair_draws']==23040
policy=torch.tensor([.3,.7]).log();ref=torch.tensor([.5,.5]).log();print('variation',dpo_loss(policy[:1],policy[1:],ref[:1],ref[1:],beta=1).item());print('torch',torch.__version__)
