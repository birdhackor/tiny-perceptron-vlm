"""Independent bounded CPU verification; no training schedule or weight files."""
import ast
import copy
import hashlib
import json
import math
import platform
import random
import re
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

BASE = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()

def load_defs(path, names, namespace):
    tree = ast.parse(path.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)

ns = dict(torch=torch, nn=nn, random=random, hashlib=hashlib, json=json, re=re)
load_defs(BASE/'inputs/applications.py', ['_reasoning_records', '_FinitePolicy', '_policy_features', '_policy_reward'], ns)
load_defs(BASE/'inputs/common.py', ['split_records', 'records_sha256'], ns)
actions = [str(i) for i in range(16)] + [' '.join(map(str, range(16)))]
ns['_POLICY_ACTIONS'] = actions
splits = ns['split_records'](ns['_reasoning_records'](), 42)
raw = json.loads((BASE/'inputs/reasoning.raw.json').read_bytes())
inspected = ['/revision', '/device', '/seed', '/torch_version', '/python_version', '/gpu', '/step_scale',
             '/code_sha256/scripts~1course_experiments~1applications.py',
             '/code_sha256/scripts~1course_experiments~1common.py', '/results/split']
for name in ['applications.py', 'common.py', 'run.py']:
    key = 'scripts/course_experiments/' + name
    assert hashlib.sha256((BASE/'inputs'/name).read_bytes()).hexdigest() == raw['code_sha256'][key]
families = [{r['family'] for r in rows} for rows in splits.values()]
assert all(not families[i] & families[j] for i in range(3) for j in range(i+1, 3))
split_checks = {}
for name, rows in splits.items():
    observed = {'records':len(rows), 'families':len({r['family'] for r in rows}), 'sha256':ns['records_sha256'](rows)}
    assert observed == raw['results']['split'][name]
    split_checks[name] = observed

def independent_rewards(text, truth):
    strict = float(bool(re.fullmatch(r'-?[0-9]+', text)) and int(text) == truth)
    proxy = float(str(truth) in text.split())
    return strict, proxy

evaluations = {}
for branch in ['strict', 'weak_proxy']:
    entry = raw['results']['reinforce'][branch]
    prefix = '/results/reinforce/' + branch
    inspected.extend([prefix+'/training/steps', prefix+'/training/planned_steps', prefix+'/training/step_scale',
                      prefix+'/training/effective_tokens', prefix+'/training/sampled_actions',
                      prefix+'/training/reward_history'])
    training = entry['training']
    assert training['steps'] == training['planned_steps'] == 1200
    assert training['sampled_actions'] == training['steps'] * 64 == 76800
    assert training['effective_tokens'] == 0 and training['step_scale'] == 1
    for phase in ['before', 'after']:
        inspected.append(prefix+'/'+phase+'/samples')
        for key in ['greedy_accuracy','sample_accuracy','sample_proxy_reward','enumeration_action_rate']:
            inspected.append(prefix+'/'+phase+'/'+key)
        evaluation = entry[phase]
        traces = evaluation['samples']
        assert len(traces) == len(splits['test']) == 24
        totals = dict(greedy_accuracy=0, sample_accuracy=0, sample_proxy_reward=0, enumeration_action_rate=0)
        max_sum_error = 0
        for trace, row in zip(traces, splits['test'], strict=True):
            assert {k:trace[k] for k in row} == row
            assert trace['truth'] == trace['a'] + trace['b'] + trace['c']
            probs = trace['probabilities']
            assert len(probs) == 17 and all(0 <= p <= 1 for p in probs)
            max_sum_error = max(max_sum_error, abs(sum(probs)-1))
            assert max_sum_error < 2e-6
            greedy_id = max(range(17), key=probs.__getitem__)
            assert trace['greedy_action'] == actions[greedy_id]
            strict, proxy = independent_rewards(trace['greedy_action'], row['truth'])
            assert bool(strict) == trace['greedy_strict_correct']
            totals['greedy_accuracy'] += int(strict)
            assert len(trace['samples']) == 16
            for candidate in trace['samples']:
                action = candidate['action_id']
                assert candidate['generated'] == actions[action]
                strict, proxy = independent_rewards(candidate['generated'], row['truth'])
                assert strict == candidate['strict_reward'] and proxy == candidate['proxy_reward']
                assert strict == ns['_policy_reward'](action, row['truth'])
                assert proxy == ns['_policy_reward'](action, row['truth'], True)
                totals['sample_accuracy'] += int(strict)
                totals['sample_proxy_reward'] += int(proxy)
                totals['enumeration_action_rate'] += int(action == 16)
        for key, value in totals.items():
            denominator = 24 if key == 'greedy_accuracy' else 384
            assert evaluation[key] == {'numerator':value, 'denominator':denominator, 'rate':value/denominator}
        evaluations[branch+'/'+phase] = dict(totals=totals, questions=24, draws_per_question=16,
                                             sampled_actions=384, max_probability_sum_error=max_sum_error)
    first = entry['after']['samples'][0]
    assert first['question'] == '(0+3)+3=?' and first['truth'] == 6
    assert first['greedy_action'] == ('7' if branch == 'strict' else actions[16])
assert evaluations['strict/after']['totals']['greedy_accuracy'] == 0
assert evaluations['weak_proxy/after']['totals'] == dict(greedy_accuracy=0,sample_accuracy=0,sample_proxy_reward=384,enumeration_action_rate=384)

# Analytic d[-A log softmax(z)_a]/dz_i = A(p_i - 1[i=a]).
variants = []
for reward, baseline, action in [(1.,.5,1),(0.,.5,1),(1.,.5,0),(.5,.5,1)]:
    z = torch.tensor([0.,0.], dtype=torch.float64, requires_grad=True)
    before = z.detach().softmax(0).clone()
    advantage = reward-baseline
    loss = -advantage*z.log_softmax(0)[action]
    loss.backward()
    analytic = advantage*(before-torch.nn.functional.one_hot(torch.tensor(action),2))
    assert torch.allclose(z.grad, analytic, atol=1e-14, rtol=0)
    finite = []
    for i in range(2):
        delta = torch.zeros(2,dtype=torch.float64);delta[i]=1e-6
        finite.append(float((-advantage*(z.detach()+delta).log_softmax(0)[action]
                             +advantage*(z.detach()-delta).log_softmax(0)[action])/(2e-6)))
    assert torch.allclose(z.grad,torch.tensor(finite,dtype=torch.float64),atol=1e-10,rtol=0)
    with torch.no_grad():
        updated = z-.1*z.grad
        after = updated.softmax(0)
    assert z.tolist() == [0.,0.] and not updated.requires_grad
    if advantage: assert (float(after[action]-before[action]) > 0) == (advantage > 0)
    else: assert torch.equal(after,before)
    variants.append(dict(reward=reward,baseline=baseline,action=action,gradient=z.grad.tolist(),
                         updated=updated.tolist(),after=after.tolist(),finite_difference=finite))
assert [round(p,6) for p in variants[0]['after']] == [.487503,.512497]
assert [round(p,6) for p in variants[1]['after']] == [.512497,.487503]

# Bounded mechanism check of finite policy. This creates no model scores for the lesson.
torch.manual_seed(42)
base = ns['_FinitePolicy'](); initial=copy.deepcopy(base.state_dict())
left=ns['_FinitePolicy']();right=ns['_FinitePolicy']()
left.load_state_dict(initial);right.load_state_dict(initial)
features=ns['_policy_features'](splits['test'][:2],SimpleNamespace(device='cpu'))
assert features.shape==(2,18) and torch.equal(features.sum(-1),torch.tensor([3.,3.]))
assert torch.equal(left(features),right(features)) and left(features).shape==(2,17)
old={k:v.detach().clone() for k,v in left.state_dict().items()}
distribution=torch.distributions.Categorical(logits=left(features))
sampled=distribution.sample(); assert sampled.shape==(2,) and not sampled.requires_grad
loss=-(distribution.log_prob(sampled)*torch.tensor([.5,-.5])).mean()-.005*distribution.entropy().mean()
optimizer=torch.optim.AdamW(left.parameters(),lr=.003)
optimizer.zero_grad(set_to_none=True);loss.backward()
torch.nn.utils.clip_grad_norm_(left.parameters(),1.,error_if_nonfinite=True)
optimizer.step()
assert any(not torch.equal(old[k],v) for k,v in left.state_dict().items())

result=dict(environment=dict(python=platform.python_version(),torch=str(torch.__version__),
                             torch_git_version=torch.version.git_version,device='cpu',cuda_build=str(torch.version.cuda)),
            source_sha256=hashlib.sha256((BASE/'inputs/C.7.raw.md').read_bytes()).hexdigest(),
            inspected_json_pointers=inspected,split_checks=split_checks,evaluations=evaluations,variants=variants,
            mechanism_checks={'features_shape':[2,18],'ones_per_row':3,'logits_shape':[2,17],
                              'identical_initial_logits':True,'categorical_sample_and_log_prob_gradient':True,
                              'single_adamw_step_changes_parameters':True,'retained_weight_files':0},
            original_provenance={k:raw[k] for k in ['revision','device','seed','torch_version','python_version','gpu','step_scale']},
            scope='Recomputed existing raw records; one synthetic optimizer step checks update mechanics only. No retraining, GPU, downloaded training data, generated model score, or saved weights.')
(BASE/'checks/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
