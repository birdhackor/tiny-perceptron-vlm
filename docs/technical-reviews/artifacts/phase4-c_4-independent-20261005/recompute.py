"""Bounded, offline audit of existing arithmetic candidates; no training or model load."""
from pathlib import Path
import ast
from collections import Counter
import hashlib
import json
import re
import sys
import platform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
raw_path = HERE / 'frozen-input/docs/course-experiments/results/reasoning.json'
code_path = HERE / 'frozen-input/scripts/course_experiments/applications.py'
data = json.loads(raw_path.read_text())
tree = ast.parse(code_path.read_bytes())
names = {'_rate', '_verify_reasoning', '_reasoning_samples'}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
ns = {'re': re, 'Counter': Counter}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(code_path), 'exec'), ns)
verify = ns['_verify_reasoning']

def own_parse(c, row, mode):
    text = c['generated'].strip()
    if c['invalid_special_tokens']:
        return None, False
    if mode == 'direct':
        final = int(text) if re.fullmatch(r'-?[0-9]+', text) else None
        return final, final == row['a'] + row['b'] + row['c']
    full = re.fullmatch(r'(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);answer=(-?[0-9]+)', text)
    if full:
        a,b,u,v,c2,w,f = map(int, full.groups())
        return f, (a,b,c2)==(row['a'],row['b'],row['c']) and a+b==u and v+c2==w and v==u and w==f and f==row['a']+row['b']+row['c']
    final = re.search(r';answer=(-?[0-9]+)$', text)
    return (int(final[1]) if final else None), False

out = {'environment': {'python':sys.version,'platform':platform.platform(),'device':'CPU; no model loaded'},
       'raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
       'code_sha256':hashlib.sha256(code_path.read_bytes()).hexdigest(),
       'raw_pointers':['/revision','/seed','/torch_version','/python_version','/device','/code_sha256/scripts~1course_experiments~1applications.py','/results/split/test','/results/comparison/direct/budgets/*','/results/comparison/steps/budgets/*'],
       'pointer_scope':'Budget numeric measurement fields; samples a,b,c,question,truth,family,generation,candidates and stored flags. No training results, notes, author interpretations or other results branches read.',
       'budgets':[], 'focused_samples':[], 'variants':[]}
for mode in ['direct','steps']:
    budgets = data['results']['comparison'][mode]['budgets']
    replay_queue = iter(s for b in budgets for s in b['samples'])
    def replay_sample(model,messages,ctx,count,tokens,temperature):
        row=next(replay_queue)
        assert messages[0]['content']==row['question']
        assert row['generation']['candidate_count']==count
        assert row['generation']['temperature']==temperature==0.7
        assert row['generation']['max_new_tokens']==tokens
        return {**row['generation'], 'samples':[{k:c[k] for k in ['generated','generated_ids','eos','invalid_special_tokens','generated_tokens']} for c in row['candidates']]}
    ns['_sample']=replay_sample
    replayed=ns['_reasoning_samples'](None,[{k:s[k] for k in ['a','b','c','family','question','truth']} for s in budgets[0]['samples']],None,mode)
    for bi,(b,replayed_b) in enumerate(zip(budgets,replayed,strict=True)):
        k=b['candidate_count']; rows=b['samples']; assert len(rows)==24
        counts={'covered':0,'majority_correct':0,'verified_correct':0,'selection_failed':0,'correct_final_invalid_steps':0,'covered_but_no_fully_verified':0}
        for ri,row in enumerate(rows):
            assert row['truth']==row['a']+row['b']+row['c']
            assert row['question']==f"({row['a']}+{row['b']})+{row['c']}=?"
            assert len(row['candidates'])==k
            parsed=[own_parse(c,row,mode) for c in row['candidates']]
            for ci,(c,(final,fully)) in enumerate(zip(row['candidates'],parsed,strict=True)):
                v=verify(c,row,mode)
                assert v['final_answer']==final==c['final_answer']
                assert v['fully_verified']==fully==c['fully_verified']
                assert v['final_correct']==(final==row['truth'])==c['final_correct']
                for key,value in v.items(): assert c[key]==value,(mode,k,ri,ci,key)
            answers=[f for f,full in parsed if f is not None]
            majority=Counter(answers).most_common(1)[0][0] if answers else None
            verified=next((f for f,full in parsed if full),None)
            covered=any(f==row['truth'] for f,full in parsed)
            mc=majority==row['truth']; vc=verified==row['truth']; failed=covered and not mc
            for key,value in {'oracle_coverage':covered,'majority_answer':majority,'majority_correct':mc,'verified_answer':verified,'verifier_correct':vc,'selection_failed_despite_coverage':failed}.items(): assert row[key]==value
            assert replayed_b['samples'][ri]==row
            counts['covered']+=covered;counts['majority_correct']+=mc;counts['verified_correct']+=vc;counts['selection_failed']+=failed
            counts['correct_final_invalid_steps']+=sum(f==row['truth'] and not full for f,full in parsed)
            counts['covered_but_no_fully_verified']+=covered and not vc
            if (mode,k,row['question']) in [('direct',4,'(3+3)+0=?'),('steps',2,'(5+0)+3=?')]:
                out['focused_samples'].append({'pointer':f'/results/comparison/{mode}/budgets/{bi}/samples/{ri}','question':row['question'],'truth':row['truth'],'final_answers':[f for f,full in parsed],'generated':[c['generated'] for c in row['candidates']],'fully_verified':[full for f,full in parsed],'majority':majority,'verified':verified})
        for key in ['oracle_coverage','majority_accuracy','verifier_accuracy','majority_accuracy_given_coverage','parsed_candidates','candidate_final_accuracy','fully_verified_candidates','final_correct_but_steps_invalid','generated_tokens','forward_input_tokens','generation_seconds']:
            assert replayed_b[key]==b[key],(mode,k,key)
        out['budgets'].append({'mode':mode,'k':k,'questions':len(rows),'candidate_denominator':len(rows)*k,**counts,'conditional_selection':f"{counts['majority_correct']}/{counts['covered']}",'final_selection':f"{counts['majority_correct']}/{len(rows)}"})

for candidates in ([3,3,4],[3,3,3],[],[3,4],[4,3],[3,3,4,5]):
    majority=Counter(candidates).most_common(1)[0][0] if candidates else None
    verified=next((x for x in candidates if x==2+2),None)
    out['variants'].append({'candidates':candidates,'covered':4 in candidates,'majority':majority,'verified':verified})
row={'a':2,'b':2,'c':0,'truth':4}
variants=['2+2=4;4+0=4;answer=4','2+2=3;3+0=4;answer=4','3+1=4;4+0=4;answer=4','2+2=4;9+0=9;answer=4','garbage;answer=4','2+2=4;4+0=4;answer=3']
for text in variants:
    c={'generated':text,'invalid_special_tokens':[]}
    v=verify(c,row,'steps'); assert own_parse(c,row,'steps')==(v['final_answer'],v['fully_verified'])
    out['variants'].append({'text':text,**v})
bad={'generated':'2+2=3;3+0=4;answer=4','invalid_special_tokens':[]}
v=verify(bad,row,'steps')
assert v['final_correct'] and not v['fully_verified']
out['equality_counterexample']={'criterion':'C.3 coverage checks final_correct; production verifier requires fully_verified','sample':bad['generated'],'oracle_coverage':True,'verifier_selected':None,'verifier_correct':False}
print(json.dumps(out,ensure_ascii=False,indent=2))
