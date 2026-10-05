"""Bounded CPU verification of C.3; replays existing raw samples, never trains/generates a model."""
import ast, hashlib, itertools, json, math, platform, random, re, sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
RAW=OUT/'originals/docs/course-experiments/results/reasoning.json'
data=json.loads(RAW.read_text())
result={'environment':{'python':sys.version,'torch':str(torch.__version__),'device':'cpu','cuda_available':str(torch.cuda.is_available()),'cuda_build':str(torch.version.cuda)},'raw_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'checks':{}}
assert torch.version.cuda is None and not torch.cuda.is_available()

def exact_function(path,name,namespace):
    p=OUT/'originals'/path
    text=p.read_text(); tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    # Preserve the original source bytes for the selected implementation.
    body=b''.join(p.read_bytes().splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    dest=OUT/'execution'/('method-'+name+'.py'); dest.write_bytes(body)
    exec(compile(body,str(p)+':'+name,'exec'),namespace)
    return {'path':path,'lines':[node.lineno,node.end_lineno],'sha256':hashlib.sha256(body).hexdigest()}

# Mathematical checks use exact rational arithmetic and all outcome patterns, not sampled scores.
p=Fraction(3,10)
values={}
for n in (1,2,4,8):
    mass=sum((p**sum(bits))*((1-p)**(n-sum(bits))) for bits in itertools.product((0,1),repeat=n) if any(bits))
    assert mass==1-(1-p)**n
    values[str(n)]={'exact':str(mass),'six_decimals':round(float(mass),6)}
assert values['8']['six_decimals']==0.942352
estimates=[]
for n in (1,2,4,8):
    for c in range(n+1):
        estimate=Fraction(1)-Fraction(math.comb(n-c,n),math.comb(n,n))
        assert estimate==int(c>0)
        estimates.append({'n':n,'k':n,'c':c,'estimate':int(estimate)})
heterogeneous=(1-(1-Fraction(0))**2 + 1-(1-Fraction(6,10))**2)/2
pooled=1-(1-p)**2
assert heterogeneous==Fraction(21,50) and pooled==Fraction(51,100)
original_candidates=[3,4,3,5]; variant=[3,3,3,3]
assert 4 in original_candidates and original_candidates[0]!=4 and 4 not in variant
result['checks']['math_and_original_exercise']={'coverage':values,'n_equals_k_all_c_cases':estimates,'heterogeneous_two_question_coverage':str(heterogeneous),'incorrect_pooled_prediction':str(pooled),'two_wrong_iid_probability':str((1-p)**2),'original_contains_truth':True,'first_selection_correct':False,'all_three_variant_contains_truth':False,'variant_does_not_assign_p':True}

ns={'torch':torch,'re':re,'Counter':Counter,'random':random,'json':json,'hashlib':hashlib,'IGNORE':-100,'SPECIALS':('<pad>','<bos>','<eos>','<user>','<assistant>','<image>','<audio>','<system>')}
methods=[]
for name in ['ByteTokenizer','render_chat']:
    methods.append(exact_function('tiny_perceptron/data.py',name,ns))
for name in ['split_records','records_sha256']:
    methods.append(exact_function('scripts/course_experiments/common.py',name,ns))
for name in ['_conversation','_reasoning_records','_reasoning_sft','_verify_reasoning','_prompt_ids','_rate','_reasoning_samples']:
    methods.append(exact_function('scripts/course_experiments/applications.py',name,ns))
result['inspected_method_slices']=methods
splits=ns['split_records'](ns['_reasoning_records'](),data['seed'])
family_sets={key:{r['family'] for r in rows} for key,rows in splits.items()}
for a,b in itertools.combinations(family_sets,2): assert not family_sets[a]&family_sets[b]
for key,rows in splits.items():
    expected=data['results']['split'][key]
    assert len(rows)==expected['records'] and len(family_sets[key])==expected['families']
    assert ns['records_sha256'](rows)==expected['sha256']
test=splits['test']; assert len(test)==24 and len({r['question'] for r in test})==24
result['checks']['splits']={'sizes':{k:len(v) for k,v in splits.items()},'families':{k:len(v) for k,v in family_sets.items()},'family_overlap':False,'test_unique_questions':24}

# Independent parser and independent byte decoding for each recorded candidate.
def independent_final(candidate,row,mode):
    ids=candidate['generated_ids']; eos=bool(ids and ids[-1]==2)
    raw=ids[:-1] if eos else ids
    text=bytes(i-8 for i in raw if i>=8).decode('utf-8',errors='replace')
    invalid=[i for i in raw if i<8]
    assert text==candidate['generated'] and eos==candidate['eos']
    assert invalid==candidate['invalid_special_tokens'] and len(ids)==candidate['generated_tokens']
    stripped=text.strip()
    final=None
    if not invalid:
        if mode=='direct' and re.fullmatch(r'-?[0-9]+',stripped): final=int(stripped)
        if mode=='steps':
            m=re.search(r';answer=(-?[0-9]+)$',stripped)
            if m: final=int(m[1])
    correct=final==row['a']+row['b']+row['c']
    assert final==candidate['final_answer'] and correct==candidate['final_correct']
    original=ns['_verify_reasoning'](candidate,row,mode)
    for key,value in original.items(): assert candidate[key]==value
    return correct

mode_results={}; training_results={}; pointer_log=[]
for mode in ['direct','steps']:
    comp=data['results']['comparison'][mode]; training=comp['training']
    assert training['steps']==training['planned_steps']==900 and training['step_scale']==1.0
    sft=ns['_reasoning_sft'](splits['train'],mode)
    assert ns['records_sha256'](sft)==training['records_sha256']
    lengths=[int((ns['render_chat'](r['messages'])[1]!=-100).sum()) for r in sft]
    rng=random.Random(data['seed'])
    effective=sum(sum(rng.choices(lengths,k=24)) for _ in range(900))
    assert effective==training['effective_tokens'] and training['records']==167
    target=ns['_reasoning_sft']([{'a':1,'b':2,'c':3,'truth':6,'question':'(1+2)+3=?'}],mode)[0]['messages'][-1]['content']
    assert target==('6' if mode=='direct' else '1+2=3;3+3=6;answer=6')
    training_results[mode]={'steps':900,'sampled_examples':900*24,'effective_answer_tokens_including_eos':effective,'example_target':target,'base_state_sha256':comp['base_state_sha256']}
    budgets=comp['budgets']; assert [b['candidate_count'] for b in budgets]==[1,2,4,8]
    summaries=[]; total=0
    for bi,budget in enumerate(budgets):
        k=budget['candidate_count']; samples=budget['samples']; assert len(samples)==24
        total+=sum(len(s['candidates']) for s in samples)
        covered=0
        for row,expected in zip(samples,test,strict=True):
            for key in ['family','a','b','c','question','truth']: assert row[key]==expected[key]
            assert row['truth']==row['a']+row['b']+row['c']
            candidates=row['candidates']; assert len(candidates)==k
            flags=[independent_final(c,row,mode) for c in candidates]
            hit=any(flags); covered+=hit; assert row['oracle_coverage']==hit
            g=row['generation']; assert g['candidate_count']==k and g['temperature']==0.7
            assert g['max_new_tokens']==(8 if mode=='direct' else 48)
            assert g['messages']==[{'role':'user','content':row['question']}]
            assert g['input_ids']==ns['_prompt_ids'](g['messages'],ns['ByteTokenizer']())
            assert g['input_tokens']==len(g['input_ids'])
            assert g['generated_tokens']==sum(len(c['generated_ids']) for c in candidates)
            assert all(len(c['generated_ids'])<=g['max_new_tokens'] for c in candidates)
        assert budget['oracle_coverage']=={'numerator':covered,'denominator':24,'rate':covered/24}
        summaries.append({'k':k,'covered':covered,'questions':24,'candidates':24*k,'temperature':0.7,'max_new_tokens':8 if mode=='direct' else 48})
        pointer_log.append('/results/comparison/'+mode+'/budgets/'+str(bi)+'/samples/*/{family,a,b,c,question,truth,generation,candidates,oracle_coverage}')
    assert total==360
    # Replay original sampling coordinator against existing raw fixtures. No logits, weights, new samples or training.
    calls=[]; model=object(); fixture_index=0
    def recorded_sample(passed_model,messages,ctx,count,tokens,temperature):
        nonlocal_placeholder=None
        index=len(calls); bi,ri=divmod(index,24)
        row=budgets[bi]['samples'][ri]
        assert passed_model is model and messages==row['generation']['messages']
        assert count==budgets[bi]['candidate_count'] and tokens==(8 if mode=='direct' else 48) and temperature==0.7
        calls.append((bi,ri))
        return {**row['generation'],'samples':[{key:c[key] for key in ['generated','generated_ids','eos','invalid_special_tokens','generated_tokens']} for c in row['candidates']]}
    ns['_sample']=recorded_sample
    replay=ns['_reasoning_samples'](model,test,None,mode)
    assert replay==budgets and len(calls)==96
    non_nested=any([c['generated_ids'] for c in small['candidates']]!=[c['generated_ids'] for c in big['candidates'][:len(small['candidates'])]] for b1,b2 in zip(budgets,budgets[1:]) for small,big in zip(b1['samples'],b2['samples']))
    assert non_nested
    mode_results[mode]={'budgets':summaries,'retained_candidates':total,'raw_ids_decoded_and_independently_rescored':total,'replay_calls':len(calls),'same_model_object_all_calls':True,'at_least_one_non_nested_candidate_prefix':True}
    pointer_log.extend(['/results/comparison/'+mode+'/training/{steps,planned_steps,step_scale,effective_tokens,records,records_sha256,checkpoint}','/results/comparison/'+mode+'/base_state_sha256'])
assert training_results['direct']['base_state_sha256']==training_results['steps']['base_state_sha256']
assert training_results['direct']['effective_answer_tokens_including_eos']<training_results['steps']['effective_answer_tokens_including_eos']
assert [b['covered'] for b in mode_results['direct']['budgets']]==[1,1,2,1]
assert [b['covered'] for b in mode_results['steps']['budgets']]==[11,13,13,14]
result['checks']['raw_candidate_recount']=mode_results
result['checks']['training_provenance']=training_results
result['read_pointers']=['/revision','/seed','/device','/torch_version','/python_version','/code_sha256','/artifacts','/assets (provenance only)','/results/split']+pointer_log
result['source_scope']='Read original mathematical/code/raw fields; never read prior review contents, result limitations/correction notes, policy/GSM8K results, model checkpoints, or training data. All candidate checks replay existing raw data.'
(OUT/'execution/independent-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'all_assertions_passed':True,'exact_math':values,'coverage_counts':{m:[b['covered'] for b in v['budgets']] for m,v in mode_results.items()},'question_denominator':24,'retained_candidates_per_branch':360,'existing_candidates_checked':720,'training_answer_tokens':{m:v['effective_answer_tokens_including_eos'] for m,v in training_results.items()},'model_generation_or_training_executed':False},ensure_ascii=False))
