"""Independent bounded CPU checks; never calls run_posttraining or loads weights."""
import copy
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.posttraining import FiniteResponsePolicy
from scripts.course_experiments.posttraining import CONFIG, build_records, split_records, rule_best_action

torch.set_num_threads(1)
assert not torch.cuda.is_available() and torch.version.cuda is None
ART = Path(__file__).resolve().parents[1]
raw = ART / "inputs/posttraining.raw.json"
record = json.loads(raw.read_text())
result = record["results"]
checked = []
def pointer(p):
    v = record
    for part in p.strip("/").split("/"):
        v = v[int(part)] if isinstance(v, list) else v[part]
    checked.append(p)
    return v

out = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version), "device": "cpu", "threads": str(torch.get_num_threads())},
       "scope": "Original fence ran separately; this script checks variants and existing raw measurements, not retraining.",
       "raw_result_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
reference = torch.tensor([0.5, 0.5], dtype=torch.float64).log()
examples = []
for probs in ([0.5, 0.5], [0.7, 0.3], [0.3, 0.7]):
    logp = torch.tensor(probs, dtype=torch.float64).log().requires_grad_()
    ref = reference.clone().requires_grad_()
    loss = dpo_loss(logp[:1], logp[1:], ref[:1], ref[1:], beta=1.0)
    loss.backward()
    expected = -math.log(probs[0])
    assert abs(loss.item() - expected) < 1e-12
    assert ref.grad is None
    assert logp.grad[0] < 0 < logp.grad[1]
    examples.append({"probabilities": probs, "relative_margin": (logp[0]-logp[1]).item(),
                     "expected_negative_log_chosen": expected, "observed": loss.item(),
                     "rounded_4dp": round(loss.item(), 4), "policy_logscore_gradients": logp.grad.tolist(),
                     "reference_gradient": None})
out["dpo_variants"] = examples
chosen = torch.tensor([math.log(0.5), math.log(0.7), math.log(0.3)], dtype=torch.float64)
rejected = torch.tensor([math.log(0.5), math.log(0.3), math.log(0.7)], dtype=torch.float64)
refs = torch.full((3,), math.log(0.5), dtype=torch.float64)
batch = dpo_loss(chosen, rejected, refs, refs, beta=1)
assert abs(batch.item() - sum(x["observed"] for x in examples)/3) < 1e-12
shifted = dpo_loss(chosen, rejected, refs+3, refs+3, beta=1)
assert abs(shifted.item() - batch.item()) < 1e-12
errors=[]
for beta in [0, -1]:
    try: dpo_loss(chosen, rejected, refs, refs, beta=beta)
    except ValueError as e: errors.append({"beta": beta, "error_type": type(e).__name__, "message": str(e)})
    else: raise AssertionError("nonpositive beta must be rejected")
out["api_contract"]={"batch_axis": "3 preference pairs, arithmetic mean", "batch_loss": batch.item(),
                     "common_reference_logscore_shift_invariant": True, "nonpositive_beta": errors,
                     "updates_or_optimizer_in_original_fence": False}

generated = split_records(build_records(), seed_value=pointer('/seed'))
split_info = {}
for split, rows in generated.items():
    families = sorted({r['family'] for r in rows})
    observed_families = pointer('/results/splits/'+split+'/families')
    assert families == observed_families
    assert len(rows) == pointer('/results/splits/'+split+'/contexts')
    pair_count = sum(len(r['preference_pairs']) for r in rows)
    assert pair_count == pointer('/results/splits/'+split+'/preference_pairs')
    regenerated_hash = hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    assert regenerated_hash == pointer('/results/splits/'+split+'/sha256')
    split_info[split]={'families':len(families),'contexts':len(rows),'preference_pairs':pair_count,
                       'mode_counts':dict(Counter(r['mode'] for r in rows)), 'regenerated_record_sha256':regenerated_hash}
    for r in rows:
        a,b=r['operands']; assert a <= b
        assert r['family']==f'pair:{min(a,b)}:{max(a,b)}'
        assert r['expected_action']==rule_best_action(r['mode'])
assert not any(set(pointer('/results/splits/'+a+'/families')) & set(pointer('/results/splits/'+b+'/families'))
               for a,b in [('train','validation'),('train','test'),('validation','test')])
out['split_recalculation']=split_info

saved_rows=pointer('/results/evaluations/test/rows')
assert len(saved_rows)==18
assert Counter(r['mode'] for r in saved_rows)=={'number':6,'explain':6,'missing':6}
for i, saved in enumerate(saved_rows):
    original=generated['test'][i]
    for key in ['family','operands','mode','prompt','candidates','features','preference_pairs','expected_action',
                'candidate_content_or_clarification_correct','candidate_meets_full_request']:
        assert pointer(f'/results/evaluations/test/rows/{i}/{key}')==original[key]
raw_success={}
sample_audit=[]
for name in ['sft','ppo','dpo']:
    wins=Counter();den=Counter()
    for i,r in enumerate(saved_rows):
        # Only raw sample/label/measurement fields; no recursive notes or narrative values.
        mode=pointer(f'/results/evaluations/test/rows/{i}/mode')
        expected=pointer(f'/results/evaluations/test/rows/{i}/expected_action')
        probs=pointer(f'/results/evaluations/test/rows/{i}/policies/{name}/probabilities')
        action=pointer(f'/results/evaluations/test/rows/{i}/policies/{name}/chosen_action')
        chosen_response=pointer(f'/results/evaluations/test/rows/{i}/policies/{name}/chosen_response')
        candidates=pointer(f'/results/evaluations/test/rows/{i}/candidates')
        reported=pointer(f'/results/evaluations/test/rows/{i}/policies/{name}/full_request_success')
        assert abs(sum(probs)-1)<2e-6 and len(probs)==4
        assert action==max(range(4),key=lambda j:probs[j])
        assert chosen_response==candidates[action]
        assert expected==rule_best_action(mode) and reported==(action==expected)
        den[mode]+=1; wins[mode]+=int(action==expected)
        sample_audit.append({'row':i,'policy':name,'family':r['family'],'mode':mode,'expected':expected,
                             'action':action,'response':chosen_response,'success':action==expected})
    total=sum(wins.values());assert total==pointer(f'/results/evaluations/test/policies/{name}/greedy_full_request_success/numerator')
    assert len(saved_rows)==pointer(f'/results/evaluations/test/policies/{name}/greedy_full_request_success/denominator')
    for mode in den:
        assert wins[mode]==pointer(f'/results/evaluations/test/policies/{name}/by_mode/{mode}/numerator')
        assert den[mode]==pointer(f'/results/evaluations/test/policies/{name}/by_mode/{mode}/denominator')
    raw_success[name]={'numerator':total,'denominator':len(saved_rows),'by_mode':{m:{'numerator':wins[m],'denominator':den[m]} for m in den}}
assert raw_success['sft']['numerator']==6
assert raw_success['ppo']==raw_success['dpo'] and raw_success['ppo']['numerator']==12
out['independent_test_aggregation']=raw_success
out['raw_sample_audit']=sample_audit

config=pointer('/results/config');assert config==CONFIG
counts={'ppo_policy_updates':config['ppo_rollout_batches']*config['ppo_epochs_per_rollout'],
        'ppo_value_updates':config['ppo_rollout_batches']*config['ppo_epochs_per_rollout'],
        'new_action_samples':config['ppo_rollout_batches']*config['ppo_rollout_size'],
        'dpo_policy_updates':config['dpo_steps'],
        'dpo_pair_draws':config['dpo_steps']*config['dpo_batch_size'],
        'reward_updates':config['reward_steps']}
for key,p in [('ppo_policy_updates','/results/ppo/policy_updates'),('ppo_value_updates','/results/ppo/value_updates'),
              ('new_action_samples','/results/ppo/sampled_actions'),('dpo_policy_updates','/results/dpo/policy_updates'),
              ('dpo_pair_draws','/results/dpo/processed_pair_draws')]: assert counts[key]==pointer(p)
assert counts=={'ppo_policy_updates':360,'ppo_value_updates':360,'new_action_samples':7680,'dpo_policy_updates':360,'dpo_pair_draws':23040,'reward_updates':300}
out['update_and_draw_counts']=counts
timings={name:pointer(f'/results/{name}/seconds') for name in ['ppo','dpo','reward']}
assert {k:round(v,2) for k,v in timings.items()}=={'ppo':0.66,'dpo':0.34,'reward':0.27}
out['historical_segment_times_seconds']=timings
out['recorded_measurement_environment']={k:pointer('/'+k) for k in ['device','seed','torch_version','python_version']}
provenance=pointer('/code_sha256')
for file,expected in provenance.items(): assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==expected
out['source_provenance_matches_current_named_originals']=provenance
hashes={k:pointer(p) for k,p in [('ppo','/results/ppo/initial_state_sha256'),('dpo','/results/dpo/initial_state_sha256'),
                               ('reference_before','/results/reference_state_sha256_before'),('reference_after','/results/reference_state_sha256_after')]}
assert len(set(hashes.values()))==1
torch.manual_seed(932)
model=FiniteResponsePolicy();reference_model=copy.deepcopy(model).requires_grad_(False).eval()
ppo=copy.deepcopy(reference_model).requires_grad_(True);dpo=copy.deepcopy(reference_model).requires_grad_(True)
assert all(torch.equal(ppo.state_dict()[k],dpo.state_dict()[k]) and torch.equal(ppo.state_dict()[k],reference_model.state_dict()[k]) for k in model.state_dict())
count=sum(p.numel() for p in model.parameters());assert count==pointer('/results/parameters/policy')==148
assert pointer('/results/sft/demonstrations')==44
out['initial_state_contract']={'historical_hashes':hashes,'bounded_untrained_copy_test_all_tensors_equal':True,
                               'policy_parameters':count,'formula':'(4*16+16)+(16*4+4)=148',
                               'demonstration_modes_in_original_method':['number'],'no_same_preference_data_sft_control':True}
out['checked_json_pointers']=sorted(set(checked))
print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
