"""Independent bounded CPU audit of C.5. No training/model/data download."""
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import sys
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def exact_functions(path, names, namespace, destination):
    raw = path.read_bytes()
    tree = ast.parse(raw)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    lines = raw.decode().splitlines(keepends=True)
    snapshot = []
    locations = {}
    for n in selected:
        start = min([n.lineno] + [d.lineno for d in n.decorator_list])
        locations[n.name] = [start, n.end_lineno]
        snapshot.append(f"# Original {path.relative_to(ROOT)}:{start}-{n.end_lineno}\n")
        snapshot.extend(lines[start-1:n.end_lineno])
        snapshot.append("\n")
    destination.write_text("".join(snapshot))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return locations

namespace = dict(torch=torch, ByteTokenizer=ByteTokenizer, time=time, re=re, Counter=Counter)
app = ROOT / 'scripts/course_experiments/applications.py'
locations = exact_functions(app, ['_sync', '_prompt_ids', '_sample', '_rate', '_reasoning_records',
                                  '_verify_reasoning', '_reasoning_samples'], namespace,
                            OUT.parent/'code/exact-method-functions.py')
original_sample = namespace['_sample']
split_ns = dict(json=json, hashlib=hashlib, random=random)
split_locations = exact_functions(ROOT/'scripts/course_experiments/common.py', ['split_records'],
                                  split_ns, OUT.parent/'code/exact-split-function.py')
raw_path = ROOT/'docs/course-experiments/results/reasoning.json'
data = json.loads(raw_path.read_bytes())
# Read only named raw/provenance pointers, never /results/limits, /reinforce or author commentary.
pointers = ['/revision','/device','/seed','/torch_version','/python_version','/gpu','/step_scale',
            '/code_sha256/scripts~1course_experiments~1applications.py', '/results/split',
            '/results/comparison/direct/base_state_sha256', '/results/comparison/steps/base_state_sha256']
splits = split_ns['split_records'](namespace['_reasoning_records'](), data['seed'])
assert len(splits['test']) == 24
assert not set(r['family'] for r in splits['train']) & set(r['family'] for r in splits['test'])
rows_by_question = {r['question']:r for r in splits['test']}
summaries = {}
replayed = {}
total_candidates = 0
for mode in ('direct', 'steps'):
    branch = data['results']['comparison'][mode]
    summaries[mode] = []
    pointers.append(f'/results/comparison/{mode}/budgets/*/candidate_count')
    pointers.extend(f'/results/comparison/{mode}/budgets/*/{f}' for f in (
        'generated_tokens','forward_input_tokens','generation_seconds','oracle_coverage',
        'majority_accuracy','verifier_accuracy','majority_accuracy_given_coverage'))
    pointers.extend(f'/results/comparison/{mode}/budgets/*/samples/*/{f}' for f in (
        'a','b','c','family','question','truth','generation','candidates','oracle_coverage',
        'majority_answer','majority_correct','verified_answer','verifier_correct',
        'selection_failed_despite_coverage'))
    lookup = {}
    for b in branch['budgets']:
        k = b['candidate_count']
        assert len(b['samples']) == 24
        assert {r['question'] for r in b['samples']} == set(rows_by_question)
        tok_total = forward_total = coverage = majority_ok = verifier_ok = selected_wrong = eos = 0
        seconds = 0.0
        for row in b['samples']:
            assert all(row[f] == rows_by_question[row['question']][f] for f in ('a','b','c','family','truth'))
            assert row['truth'] == row['a']+row['b']+row['c']
            g = row['generation']
            assert g['candidate_count'] == k and len(row['candidates']) == k
            assert g['temperature'] == 0.7 and g['max_new_tokens'] == (8 if mode=='direct' else 48)
            current_tokens = 0
            finals = []
            fully = []
            for c in row['candidates']:
                ids = c['generated_ids']
                assert len(ids) == c['generated_tokens'] <= g['max_new_tokens']
                ended = bool(ids and ids[-1] == 2)
                assert c['eos'] == ended
                eos += ended
                payload = ids[:-1] if ended else ids
                assert ByteTokenizer().decode(payload) == c['generated']
                assert [x for x in payload if x<8] == c['invalid_special_tokens']
                text = c['generated'].strip()
                clean = not c['invalid_special_tokens']
                # Independently parse final answers and complete proof criteria from raw text.
                m = re.fullmatch(r'-?[0-9]+',text) if mode=='direct' else re.search(r';answer=(-?[0-9]+)$',text)
                final = (int(m[0]) if mode=='direct' else int(m[1])) if clean and m else None
                good = final == row['truth']
                full = good
                if mode=='steps':
                    mm = re.fullmatch(r'(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);answer=(-?[0-9]+)', text)
                    if clean and mm:
                        a, bb, subtotal, previous, cc, total, ff = map(int,mm.groups())
                        full = a+bb==subtotal and previous+cc==total and previous==subtotal and total==ff and (a,bb,cc)==(row['a'],row['b'],row['c']) and ff==row['truth']
                    else: full = False
                assert final == c['final_answer'] and good == c['final_correct'] and full == c['fully_verified']
                original_verification = namespace['_verify_reasoning'](c,row,mode)
                assert all(c[f] == v for f,v in original_verification.items())
                finals.append(final); fully.append(full); current_tokens += len(ids)
            votes = [f for f in finals if f is not None]
            vote = Counter(votes).most_common(1)[0][0] if votes else None
            first_verified = next((f for f,v in zip(finals,fully) if v),None)
            hit = row['truth'] in finals
            assert row['oracle_coverage']==hit and row['majority_answer']==vote
            assert row['majority_correct']==(vote==row['truth'])
            assert row['verified_answer']==first_verified and row['verifier_correct']==(first_verified==row['truth'])
            assert row['selection_failed_despite_coverage']==(hit and vote!=row['truth'])
            assert current_tokens == g['generated_tokens']
            # The same-sized batch persists to its longest active output, including inactive PAD work.
            expected_forward = k*(g['input_tokens'] + max(len(c['generated_ids']) for c in row['candidates'])-1)
            assert expected_forward == g['forward_input_tokens']
            tok_total += current_tokens; forward_total += expected_forward; seconds += g['seconds']
            coverage += hit; majority_ok += vote==row['truth']; verifier_ok += first_verified==row['truth']
            selected_wrong += hit and vote!=row['truth']
            lookup[k,row['question']] = {**g,'samples':copy.deepcopy(row['candidates'])}
        assert tok_total == b['generated_tokens'] and forward_total == b['forward_input_tokens']
        assert abs(seconds-b['generation_seconds']) < 1e-12
        for f,n,d in [('oracle_coverage',coverage,24),('majority_accuracy',majority_ok,24),
                      ('verifier_accuracy',verifier_ok,24),('majority_accuracy_given_coverage',majority_ok,coverage)]:
            assert b[f] == namespace['_rate'](n,d)
        total_candidates += 24*k
        summaries[mode].append(dict(k=k,questions=24,candidates=24*k,generated_tokens=tok_total,
            eos_tokens=eos,forward_input_tokens=forward_total,generation_seconds=seconds,
            generation_seconds_rounded_3=round(seconds,3),coverage=coverage,
            majority_correct=majority_ok,verifier_correct=verifier_ok,selected_wrong=selected_wrong))
    namespace['_sample'] = lambda model,messages,ctx,count,tokens,temperature: copy.deepcopy(lookup[count,messages[0]['content']])
    replay = namespace['_reasoning_samples'](None,splits['test'],None,mode)
    assert replay == branch['budgets']
    replayed[mode] = 'exact original _reasoning_samples replayed existing raw candidates; no model generation'

# Bounded arithmetic variants; verify the total budget includes per-candidate verifier allowance.
plans = []
for n,total,verify in ((1,80,5),(4,80,5),(4,80,0),(5,75,5)):
    each = total//n
    computed = n*each+n*verify
    plans.append(dict(candidates=n,answer_each=each,generated=n*each,verifier=n*verify,total=computed))
assert [x['total'] for x in plans] == [85,100,80,100]
one = summaries['steps'][0]; eight = summaries['steps'][3]
assert eight['coverage']-one['coverage']==3 and eight['majority_correct']-one['majority_correct']==1
ratio = eight['generated_tokens']/one['generated_tokens']
assert ratio == 4132/512

# Execute exact _sample against scripted logits to check batch, EOS, cache and timer boundaries.
# This is an API/contract check with no learned weights or quality/speed result.
events = []
class SpyTokenizer(ByteTokenizer):
    def encode(self,text):
        events.append('encode')
        return super().encode(text)
    def decode(self,ids):
        events.append('decode')
        return super().decode(ids)
class Clock:
    def perf_counter(self):
        events.append('clock')
        return time.perf_counter()
class ScriptedModel:
    config = SimpleNamespace(max_length=128)
    training = True
    def __init__(self):
        self.shapes = []
        self.cache_states = []
    def eval(self):
        self.training = False
        events.append('eval')
    def train(self,was_training):
        self.training = was_training
        events.append('restore_train_mode')
    def __call__(self,current,cache=None):
        events.append('forward')
        i = len(self.shapes)
        self.shapes.append(list(current.shape))
        self.cache_states.append(cache)
        chosen = ([73,73,73,73], [2,2,74,74], [2,2,2,75], [2,2,2,2])[i]
        logits = torch.full((4,current.shape[1],264),float('-inf'))
        for row,token in enumerate(chosen):
            logits[row,-1,token] = 0
        return {'logits':logits,'cache':i+1}
namespace['ByteTokenizer'] = SpyTokenizer
namespace['time'] = Clock()
scripted = ScriptedModel()
bounded = original_sample(scripted,[{'role':'user','content':'2+2=?'}],SimpleNamespace(device='cpu'),
                          count=4,tokens=4,temperature=0.7)
assert [x['generated_tokens'] for x in bounded['samples']] == [2,2,3,4]
assert bounded['generated_tokens']==11 and all(x['eos'] for x in bounded['samples'])
assert scripted.shapes[1:]==[[4,1]]*3 and scripted.cache_states==[None,1,2,3]
assert bounded['forward_input_tokens']==4*(bounded['input_tokens']+3)
clock_positions = [i for i,x in enumerate(events) if x=='clock']
assert len(clock_positions)==2
assert events.index('encode')<clock_positions[0]<events.index('forward')<clock_positions[1]<events.index('decode')
assert events.index('restore_train_mode')<clock_positions[1] and scripted.training is True
timer_contract = dict(events=events,forward_shapes=scripted.shapes,cache_inputs=scripted.cache_states,
                      per_candidate_generated_tokens=[2,2,3,4],total_generated_tokens=11,
                      statement='Original helper executed with scripted logits; confirms EOS, batching, cache and timer ordering. No model quality or latency estimate.')
for mode in ('direct','steps'):
    for k,expected_tokens,expected_seconds in ((1,51,'0.239'),(8,400,'0.120')) if mode=='direct' else ((1,512,'1.160'),(8,4132,'1.190')):
        measured=next(x for x in summaries[mode] if x['k']==k)
        assert measured['generated_tokens']==expected_tokens
        assert f"{measured['generation_seconds']:.3f}"==expected_seconds
result = dict(raw_results_sha256=sha(raw_path),original_method_sha256=sha(app),
    original_method_recorded_sha256=data['code_sha256']['scripts/course_experiments/applications.py'],
    original_method_locations=locations,split_method_locations=split_locations,
    provenance={k:data[k] for k in ('revision','device','seed','torch_version','python_version','gpu','step_scale')},
    equal_initial_state= data['results']['comparison']['direct']['base_state_sha256']==data['results']['comparison']['steps']['base_state_sha256'],
    current_execution_environment=dict(python=platform.python_version(),torch=torch.__version__,device='cpu',cuda_available=torch.cuda.is_available()),
    total_raw_candidates_verified=total_candidates,summaries=summaries,replayed=replayed,
    arithmetic_variants=plans,steps_token_ratio_k8_over_k1=ratio,
    bounded_original_sample_contract=timer_contract,
    scope='Existing raw measurement recomputation only. No training, weights, speed rerun, or new model scores.',
    all_assertions_passed=True)
(OUT/'inspected-pointers.json').write_text(json.dumps(pointers,indent=2)+'\n')
(OUT/'audit-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
