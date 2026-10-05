"""Independent, bounded CPU arithmetic and raw historical measurement audit.

No checkpoint load, model inference, optimizer, download, or weight writes.
The original scoring function is run with controlled generated token sequences.
"""
from pathlib import Path
import ast
import hashlib
import json
import platform
from collections import defaultdict
import torch
from tiny_perceptron.data import ByteTokenizer

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device('cpu')
sha = lambda b: hashlib.sha256(b).hexdigest()
def output(label, value):
    print(label, json.dumps(value, ensure_ascii=False, sort_keys=True))

output('environment', {'python': platform.python_version(), 'torch': str(torch.__version__),
    'device':'cpu', 'cuda_build':torch.version.cuda, 'threads':torch.get_num_threads()})
assert torch.version.cuda is None
def means(groups):
    return (sum(c for c,n in groups)/sum(n for c,n in groups),
            sum(c/n for c,n in groups)/len(groups))
cases = {'original': [(90,90),(0,10)], 'rare_count_20': [(90,90),(0,20)],
    'same_total_other_model': [(81,90),(9,10)], 'equal_sized_question_groups':[(6,6),(3,6)],
    'shape_target_groups':[(3,3),(0,3)]}
for label, groups in cases.items():
    micro,macro=means(groups)
    output(label, {'groups_correct_count':groups, 'micro':micro,'macro':macro,
      'micro_axis':'all distinct questions', 'micro_unit':'correct questions / questions',
      'macro_axis':'groups', 'macro_unit':'mean of dimensionless group accuracy',
      'micro_denominator':sum(n for c,n in groups),'macro_denominator':len(groups)})
assert means(cases['original']) == (0.9,0.5)
assert abs(means(cases['rare_count_20'])[0]-90/110)<1e-15
assert means(cases['rare_count_20'])[1]==0.5
assert means(cases['same_total_other_model']) == (0.9,0.9)
assert means(cases['equal_sized_question_groups']) == (0.75,0.75)
output('overlap_counterexample', {'unique_correct_count':[1,2], 'unique_micro':1/2,
    'overlapping_groups_correct_count':[[1,1],[1,2]],'naive_repeated_micro':2/3})
output('one_success_uncertainty', {'n':1,'k':1,'assumed_true_success_probability':0.5,
    'probability_of_one_success':0.5,'two_sided_95pct_clopper_pearson_interval':[0.025,1.0],
    'derivation':'For n=k=1, P(X>=1)=p; lower bound p=alpha/2=.025, upper bound=1.'})

raw=(ART/'original-vision_ablation.json').read_bytes()
result=json.loads(raw)
output('original_json_sha256',sha(raw))
chosen = ['/experiment_id','/revision','/device','/seed','/torch_version','/python_version','/step_scale',
 '/code_sha256/scripts~1course_experiments~1modalities.py',
 '/results/interventions/none/samples','/results/interventions/none/groups',
 '/results/interventions/none/examples','/results/interventions/none/correct',
 '/results/interventions/none/exact_match','/results/interventions/none/macro_accuracy',
 '/results/data/seed','/results/data/split_policy',
 '/results/data/splits/train','/results/data/splits/validation','/results/data/splits/test',
 '/results/training/config','/results/training/modal_config','/results/training/steps',
 '/results/training/weights_changed','/results/training/nonzero_gradient_seen']
selected={}
for pointer in chosen:
    v=result
    for k in pointer.strip('/').split('/'):
        v=v[k.replace('~1','/').replace('~0','~')]
    selected[pointer]=v
(ART/'inspected-json-pointers.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
samples=selected['/results/interventions/none/samples']
tok=ByteTokenizer()
groups=defaultdict(lambda:{'correct':0,'count':0})
by_shape=defaultdict(lambda:{'correct':0,'count':0})
records=result['results']['data']['splits']['test']['records']
assert len(records)==len(samples)==12
for row,sample in zip(records,samples,strict=True):
    assert row['family']==sample['family'] and row['question']==sample['question']
    assert row['answer']==sample['target']
    ids=sample['generated_ids']
    raw_ids=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
    correct=raw_ids==tok.encode(row['answer'])
    assert correct==sample['exact_match']
    assert tok.decode(raw_ids)==sample['generated']
    g=groups[sample['question']];g['count']+=1;g['correct']+=int(correct)
    if sample['question']=='shape?':
        s=by_shape[sample['target']];s['count']+=1;s['correct']+=int(correct)
        assert sample['generated']=='circle'
assert dict(groups)==result['results']['interventions']['none']['groups']
output('historical_recalculation', {'question_groups':groups,'shape_target_groups':by_shape,
    'distinct_questions':len({x['row'] for x in samples}),
    'correct':sum(x['exact_match'] for x in samples),'all_eos':all(x['eos'] for x in samples),
    'micro':means([(g['correct'],g['count']) for g in groups.values()])[0],
    'macro':means([(g['correct'],g['count']) for g in groups.values()])[1]})
tree=ast.parse((ART/'original-modalities.py').read_bytes())
needed={'_vision_records','_sequence','_evaluate'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in needed]
class Context:
    device='cpu'
ctx=Context()
def media(row,ctx):
    return row,None
def controlled_generate(model,prefix,image,waveform,tokens):
    if image.get('probe')=='generation_error':
        raise ValueError('placeholder 缺少配對: controlled probe')
    ids=tok.encode('circle' if image['question']=='shape?' else image['answer'])
    if image.get('probe')=='invalid_special': ids=[tok.image_id]+ids
    if image.get('probe')!='no_eos': ids=ids+[tok.eos_id]
    return torch.cat((prefix,torch.tensor(ids,dtype=torch.long)))
class DummyModel:
    def eval(self): pass
    def __call__(self,ids,labels,**kwargs): return {'logits':torch.zeros(1),'labels':labels}
namespace={'torch':torch,'ByteTokenizer':ByteTokenizer,'_media':media,
 'generate_modal':controlled_generate,'masked_loss':lambda logits,labels:torch.tensor(0.)}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ART/'original-modalities.py'),'exec'),namespace)
splits=namespace['_vision_records'](('shape?','color?'))
for split,rows in splits.items():
    assert rows==result['results']['data']['splits'][split]['records']
family_sets={k:{r['family'] for r in rows} for k,rows in splits.items()}
assert not any(family_sets[a]&family_sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
for split,v in result['results']['data']['splits'].items():
    assert sha(json.dumps(v['records'],ensure_ascii=False,sort_keys=True).encode())==v['sha256']
output('original_manifest_digest_verification', 'All three raw split-record SHA256 digests exactly match original JSON.')
output('original_record_builder', {'counts':{k:len(v) for k,v in splits.items()},
    'family_intersections_empty':True,'same_records_as_original_json':True,
    'held_out_offsets':{k:sorted({r['offset'] for r in v}) for k,v in splits.items()}})
score=namespace['_evaluate'](DummyModel(),records,ctx)
assert score['correct']==9 and score['exact_match']==0.75 and score['macro_accuracy']==0.75
probe_rows=[dict(records[0],probe='no_eos'),dict(records[0],probe='invalid_special'),
    dict(records[0],probe='generation_error')]
scored=namespace['_evaluate'](DummyModel(),probe_rows,ctx)
assert scored['examples']==3 and scored['correct']==1 and scored['generation_errors']==1
assert scored['samples'][0]['exact_match'] and not scored['samples'][0]['eos']
assert not scored['samples'][1]['exact_match'] and scored['samples'][1]['invalid_special_tokens']==1
assert not scored['samples'][2]['exact_match']
output('original_scoring_contract_controlled_tokens',{'baseline':{k:score[k] for k in ['examples','correct','exact_match','macro_accuracy','groups']},
    'probes':{k:scored[k] for k in ['examples','correct','exact_match','generation_errors','eos_rate','groups']},
    'no_eos_content_exact_match':scored['samples'][0]['exact_match'],
    'invalid_special_decode_is_not_exact_match':not scored['samples'][1]['exact_match'],
    'generation_failure_kept_in_denominator':scored['examples']==3,
    'scope':'Actual original scoring and record construction, controlled token fixtures; no model capability measurement.'})
output('PASS','All bounded arithmetic, raw-measurement, record and scoring-contract assertions passed.')
