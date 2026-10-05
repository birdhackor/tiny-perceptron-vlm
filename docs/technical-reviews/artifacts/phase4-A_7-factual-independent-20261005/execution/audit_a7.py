"""Bounded independent checks; no weight loading, training, or model scoring."""
import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer
from scripts.course_experiments.applications import _grounding, _prompt_ids, _rag_question, _sample

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent/'inputs/docs/course-experiments/results/rag.json'
report = json.loads(INPUT.read_text())
tok = ByteTokenizer()
rows = report['results']['samples']
config = report['results']['model']['config']
capacity = config['max_length']
records = []
pointers = ['/schema_version','/experiment_id','/revision','/device','/seed','/torch_version',
            '/python_version','/step_scale','/results/model/config','/results/training/steps',
            '/results/training/planned_steps','/modal/run_id','/modal/batch_id']
fields = ('family','mode','fact','context_fact','documents','expected')
generation_fields = ('messages','input_ids','input_tokens','max_new_tokens','candidate_count',
                     'temperature','generated_tokens','forward_input_tokens')
sample_fields = ('generated','generated_ids','eos','invalid_special_tokens','generated_tokens')
for index,row in enumerate(rows):
    base = f'/results/samples/{index}'
    pointers.extend(base+'/'+field for field in fields)
    generation = row['generation']
    pointers.extend(base+'/generation/'+field for field in generation_fields)
    assert generation['candidate_count'] == 1
    assert len(generation['samples']) == 1
    sample = generation['samples'][0]
    pointers.extend(base+'/generation/samples/0/'+field for field in sample_fields)
    question = _rag_question(row['family'],row['documents'])
    assert generation['messages'] == [{'role':'user','content':question}]
    full_prompt = _prompt_ids(generation['messages'],tok)
    assert generation['input_ids'] == full_prompt
    prompt_length = len(full_prompt)
    assert prompt_length == generation['input_tokens']
    ids = sample['generated_ids']
    eos = bool(ids and ids[-1] == tok.eos_id)
    payload = ids[:-1] if eos else ids
    controls = [value for value in payload if value < len(range(8))]
    assert sample['eos'] == eos
    assert sample['invalid_special_tokens'] == controls
    assert sample['generated'] == tok.decode(payload)
    assert sample['generated_tokens'] == len(ids) == generation['generated_tokens']
    assert generation['forward_input_tokens'] == prompt_length + len(ids) - 1
    assert all(0 <= value < tok.vocab_size for value in ids)
    assert tok.eos_id not in payload
    reserve = generation['max_new_tokens']
    assert len(ids) <= reserve
    assert prompt_length + reserve <= capacity
    fact = row['context_fact']
    expected = ('UNKNOWN' if row['mode'] in ('without_context','distractor_only')
                or (row['mode']=='retrieved_context' and not row['documents'])
                else f"{fact['address']}[{fact['source']}]")
    assert row['expected'] == expected
    if expected == 'UNKNOWN':
        assert not any(document['text'].startswith(row['family']+' address=')
                       for document in row['documents'])
    else:
        assert any(document['id'] == fact['source']
                   and document['text'] == f"{row['family']} address={fact['address']}"
                   for document in row['documents'])
    clean_match = not controls and tok.decode(payload) == expected
    records.append({'index':index,'family':row['family'],'mode':row['mode'],
                    'input_tokens':prompt_length,'reserve':reserve,'generated_tokens':len(ids),
                    'eos':eos,'invalid_controls':controls,'content_correct':clean_match})

assert len(rows) == 84
mode_counts=Counter(record['mode'] for record in records)
family_counts=Counter(record['family'] for record in records)
assert len(mode_counts) == 7 and set(mode_counts.values()) == {12}
assert len(family_counts) == 12 and set(family_counts.values()) == {7}
assert len({(r['family'],r['mode']) for r in records}) == len(rows)
assert min(r['input_tokens'] for r in records) == 58
assert max(r['input_tokens'] for r in records) == 94
assert {r['reserve'] for r in records} == {16}
assert sum(r['eos'] for r in records) == 84
assert not any(r['invalid_controls'] for r in records)
assert any(not r['content_correct'] for r in records)
hash_matches={}
for name in ('tiny_perceptron/data.py','tiny_perceptron/model.py','scripts/course_experiments/applications.py',
             'scripts/course_experiments/common.py'):
    pointers.append('/code_sha256/'+name.replace('~','~0').replace('/','~1'))
    observed=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    assert observed == report['code_sha256'][name]
    assert observed == hashlib.sha256((HERE.parent/'inputs'/name).read_bytes()).hexdigest()
    hash_matches[name]=observed

# Original fence and purposeful arithmetic variants.
namespace={}
exec(compile((HERE/'fence-1.py').read_bytes(),'A.7:original-fence','exec'),namespace)
assert namespace['used']==130
budget=dict(namespace['budget']);budget['retrieval']=10
assert sum(budget.values())==100
boundary={'capacity':100,'prompt':75,'reserve':25,'fits':75+25<=100,
          'one_more_reserve_fits':75+26<=100,'after_history_30':130,'after_retrieval_minus_30':100}
assert boundary['fits'] and not boundary['one_more_reserve_fits']

# Controlled logits exercise the real original generation/validation branches.
# They are token-stream fixtures, not a language model accuracy experiment.
class ScriptedTokenSource:
    def __init__(self,sequences,capacity=160):
        self.sequences=sequences;self.config=SimpleNamespace(max_length=capacity)
        self.training=True;self.step=0;self.seen_inputs=[]
    def eval(self):self.training=False
    def train(self,value):self.training=value
    def __call__(self,current,cache=None):
        self.seen_inputs.append(current.tolist())
        logits=torch.full((len(self.sequences),current.shape[1],264),-100.0)
        for row,sequence in enumerate(self.sequences):
            token=sequence[min(self.step,len(sequence)-1)]
            logits[row,-1,token]=100.0
        self.step+=1
        return {'logits':logits,'cache':None}

ctx=SimpleNamespace(device='cpu')
messages=[{'role':'system','content':'店名要保留'}, {'role':'user','content':'青街8號？'},
          {'role':'assistant','content':'請看公告。'},{'role':'user','content':'只答店名。'}]
multiturn_ids=_prompt_ids(messages,tok)
assert len(multiturn_ids)==1+sum(2+len(tok.encode(m['content'])) for m in messages)+1
assert len(tok.encode('店名'))==6 and len('店名')==2
script=ScriptedTokenSource([[tok.eos_id]])
early=_sample(script,messages,ctx,tokens=4)['samples'][0]
assert early['generated_ids']==[tok.eos_id] and early['generated_tokens']==1 and early['eos']
assert script.training is True
script=ScriptedTokenSource([tok.encode('AB')+[tok.eos_id]])
limited=_sample(script,messages,ctx,tokens=2)['samples'][0]
assert limited['generated']=='AB' and not limited['eos'] and limited['generated_tokens']==2
script=ScriptedTokenSource([[tok.assistant_id]+tok.encode('C2[D1]')+[tok.eos_id]])
dirty=_sample(script,[{'role':'user','content':'x'}],ctx,tokens=16)['samples'][0]
assert dirty['generated']=='C2[D1]' and dirty['invalid_special_tokens']==[tok.assistant_id]
assert not _grounding(dirty,[{'id':'D1','text':'K017 address=C2'}],'C2[D1]')['exact_match']
script=ScriptedTokenSource([tok.encode('A')+[tok.eos_id],tok.encode('XYZ')+[tok.eos_id]])
batched=_sample(script,[{'role':'user','content':'x'}],ctx,count=2,tokens=4)
assert [s['generated_tokens'] for s in batched['samples']]==[2,4]
assert batched['generated_tokens']==6 and all(s['eos'] for s in batched['samples'])
script=ScriptedTokenSource([[tok.eos_id]],capacity=len(multiturn_ids)+3)
try:
    _sample(script,messages,ctx,tokens=4)
except ValueError as error:
    reject_message=str(error)
else:
    raise AssertionError('incompatible full-history budget was not rejected')
assert not script.seen_inputs

# Information-preservation counterexample: same length budget, absent necessary field.
announcement={'store':'小小書店','address':'青街8號','date':'2027年3月1日','source':'公告1'}
summary={key:announcement[key] for key in ('store','date')}
assert 'address' not in summary and 'source' not in summary

result={'raw_result_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
 'inspected_json_pointers':pointers,'original_provenance':{key:report[key] for key in
 ['schema_version','experiment_id','revision','device','seed','torch_version','python_version','step_scale']},
 'run_id':report['modal']['run_id'],'batch_id':report['modal']['batch_id'],
 'training_steps':report['results']['training']['steps'],'original_code_hash_matches':hash_matches,
 'sample_denominator':len(records),'unique_family_denominator':len(family_counts),
 'mode_denominator':len(mode_counts),'mode_sample_counts':dict(mode_counts),
 'capacity':capacity,'input_min':min(r['input_tokens'] for r in records),
 'input_max':max(r['input_tokens'] for r in records),'reserve_values':[16],
 'max_reserved_total':max(r['input_tokens']+r['reserve'] for r in records),
 'eos_numerator':sum(r['eos'] for r in records),
 'invalid_control_numerator':sum(bool(r['invalid_controls']) for r in records),
 'target_quality_checks':'Each answer gold has an exact key/address and citation in the input documents; UNKNOWN has no target-key document. Expected answers independently rebuilt from raw context facts and actual documents.',
 'forward_input_token_count_checks':'Single-candidate forward_input_tokens equals complete prompt length + generated ID count - 1 for every sample.',
 'content_failure_numerator':sum(not r['content_correct'] for r in records),
 'samples_recomputed':records,'arithmetic_variant':boundary,
 'controlled_stream_variants':{'early_eos':early,'token_limit':limited,'hidden_control':dirty,
   'batch_generated_token_counts':[2,4],'batch_total_generated_tokens':6,
   'budget_rejection':reject_message,'multiturn_prompt_length':len(multiturn_ids),
   'chinese_char_count':2,'byte_token_count':6},
 'summary_counterexample':{'original':announcement,'short_summary':summary},
 'environment':{'python':sys.version,'torch':str(torch.__version__),'device':'cpu',
                'cuda_available':str(torch.cuda.is_available()),'cuda_build':str(torch.version.cuda)}}
(HERE/'independent-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({key:result[key] for key in ['sample_denominator','unique_family_denominator','mode_denominator',
 'mode_sample_counts','capacity','input_min','input_max','max_reserved_total','eos_numerator',
 'invalid_control_numerator','content_failure_numerator','arithmetic_variant','controlled_stream_variants',
 'environment']},ensure_ascii=False,indent=2))
