"""Independent bounded CPU arithmetic and existing-result inspection; no training or inference."""
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer
from scripts.course_experiments.modalities import _vision_records

torch.set_num_threads(1)
assert torch.version.cuda is None
OUT = Path(__file__).resolve().parent

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def average(groups):
    micro = Fraction(sum(c for c, n in groups), sum(n for c, n in groups))
    macro = sum((Fraction(c,n) for c,n in groups), Fraction(0)) / len(groups)
    return {'groups': groups, 'micro_fraction': str(micro), 'micro': float(micro),
            'macro_fraction': str(macro), 'macro': float(macro),
            'denominator_examples': sum(n for c,n in groups), 'denominator_groups': len(groups)}

numeric = {name: average(groups) for name,groups in {
    'initial_common_rare': [(90,90),(0,10)],
    'rare_count_20': [(90,90),(0,20)],
    'same_total_second_model': [(81,90),(9,10)],
    'color_shape': [(6,6),(3,6)],
    'shape_circle_square': [(3,3),(0,3)],
}.items()}
assert numeric['initial_common_rare']['micro'] == .9
assert numeric['initial_common_rare']['macro'] == .5
assert numeric['rare_count_20']['micro_fraction'] == '9/11'
assert round(numeric['rare_count_20']['micro'],4) == .8182
assert numeric['rare_count_20']['macro'] == .5
assert numeric['same_total_second_model']['micro'] == .9
assert numeric['same_total_second_model']['macro'] == .9
assert numeric['color_shape']['micro'] == numeric['color_shape']['macro'] == .75
# Ordinary example-level accuracy counts a sample once even when tags overlap.
rows = {'a': True, 'b': False, 'c': False}
subgroups = [['a','b'], ['a','c']]
naive = Fraction(sum(rows[i] for g in subgroups for i in g),sum(map(len,subgroups)))
unique = Fraction(sum(rows[i] for i in set().union(*map(set,subgroups))),len(rows))
assert naive == Fraction(1,2) and unique == Fraction(1,3)
# For 1/1 successes and iid Bernoulli trials: exact two-sided 95% lower bound
# solves P_p(X>=1)=p=alpha/2=.025; the upper bound is 1 (k=n).
ci = [(1-.95)/2,1.0]
assert abs(ci[0]-.025)<1e-15

original = ROOT/'docs/course-experiments/results/vision_ablation.json'
raw = json.loads(original.read_bytes())
# Only these named raw-provenance/measurement pointers are inspected. No notes,
# review, title, status, or *_scope_correction values are accessed.
pointers = ['/schema_version','/experiment_id','/revision','/device','/seed',
 '/torch_version','/python_version','/step_scale','/modal/run_id','/modal/batch_id',
 '/code_sha256/scripts~1course_experiments~1modalities.py','/code_sha256/tiny_perceptron~1data.py',
 '/results/data/seed','/results/data/split_policy',
 '/results/data/splits/train','/results/data/splits/validation','/results/data/splits/test',
 '/results/training/config','/results/training/modal_config','/results/training/steps',
 '/results/training/weights_changed','/results/training/nonzero_gradient_seen',
 '/results/training/effective_tokens','/results/training/effective_targets',
 '/results/interventions/none/examples','/results/interventions/none/correct',
 '/results/interventions/none/exact_match','/results/interventions/none/macro_accuracy',
 '/results/interventions/none/groups','/results/interventions/none/samples',
 '/results/interventions/none/eos_rate','/results/interventions/none/generation_errors',
 '/results/interventions/none/invalid_special_tokens','/results/interventions/none/ablation',
 '/results/interventions/none/skipped']

def pointer(ptr):
    value=raw
    for k in ptr[1:].split('/'):
        value=value[k.replace('~1','/').replace('~0','~')]
    return value

seen = {ptr:pointer(ptr) for ptr in pointers}
tok=ByteTokenizer();stats=defaultdict(lambda:{'correct':0,'count':0})
shape=defaultdict(lambda:{'correct':0,'count':0}); sample_checks=[]
records=seen['/results/data/splits/test']['records']
samples=seen['/results/interventions/none/samples']
assert len(records)==len(samples)==12
for i,(r,s) in enumerate(zip(records,samples,strict=True)):
    assert s['row']==i and (s['family'],s['question'],s['target'])==(r['family'],r['question'],r['answer'])
    generated_ids=s['generated_ids']
    cut=generated_ids.index(tok.eos_id) if tok.eos_id in generated_ids else len(generated_ids)
    token_body=generated_ids[:cut]
    exact=(token_body==tok.encode(r['answer']))
    assert exact==s['exact_match']
    assert tok.decode(token_body)==s['generated']
    assert s['eos']==(tok.eos_id in generated_ids)
    assert s['generation_error'] is None and s['invalid_special_tokens']==0
    g=stats[r['question']];g['count']+=1;g['correct']+=exact
    if r['question']=='shape?':
        h=shape[r['shape']];h['count']+=1;h['correct']+=exact
        assert s['generated']=='circle'
    sample_checks.append({'row':i,'family':r['family'],'question':r['question'],
                          'target':r['answer'],'generated':s['generated'],'raw_token_exact_match':exact})
assert dict(stats)==seen['/results/interventions/none/groups']
assert dict(shape)=={'circle':{'correct':3,'count':3},'square':{'correct':0,'count':3}}
assert sum(s['exact_match'] for s in samples)==seen['/results/interventions/none/correct']==9
assert seen['/results/interventions/none/examples']==12
micro=Fraction(9,12);macro=sum(Fraction(g['correct'],g['count']) for g in stats.values())/len(stats)
assert float(micro)==seen['/results/interventions/none/exact_match']==.75
assert float(macro)==seen['/results/interventions/none/macro_accuracy']==.75
# Rebuild only the deterministic records, without calling a model or _manifest.
fresh=_vision_records(('shape?','color?'))
split_checks={}
for name,rows2 in fresh.items():
    stored=seen['/results/data/splits/'+name]
    rebuilt_hash=hashlib.sha256(json.dumps(rows2,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    assert rows2==stored['records'] and len(rows2)==stored['count'] and rebuilt_hash==stored['sha256']
    split_checks[name]={'count':len(rows2),'sha256':rebuilt_hash,'offsets':sorted({r['offset'] for r in rows2}),
                        'families':sorted({r['family'] for r in rows2})}
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert set(split_checks[a]['families']).isdisjoint(split_checks[b]['families'])
code_hashes={}
for p in ['scripts/course_experiments/modalities.py','tiny_perceptron/data.py']:
    actual=digest(ROOT/p);assert actual==raw['code_sha256'][p]
    code_hashes[p]={'sha256':actual,'equals_historical_provenance':True}
result={
 'environment':{'python':platform.python_version(),'torch':str(torch.__version__),
                'numpy':__import__('numpy').__version__,'device':'cpu','cuda_build':str(torch.version.cuda),
                'threads':str(torch.get_num_threads())},
 'numeric':numeric,'overlap_check':{'naive_duplicated':'1/2','unique_examples':'1/3'},
 'single_success_exact_95pct_iid_binomial_interval':ci,
 'raw_source_path':original.relative_to(ROOT).as_posix(),'raw_source_sha256':digest(original),
 'inspected_json_pointers':pointers,'provenance':{k:seen['/'+k] for k in ['revision','device','seed','torch_version','python_version','step_scale']},
 'historical_run':{'run_id':seen['/modal/run_id'],'batch_id':seen['/modal/batch_id']},
 'training_record_only':{k:seen['/results/training/'+k] for k in ['config','modal_config','steps','weights_changed','nonzero_gradient_seen','effective_tokens','effective_targets']},
 'splits':split_checks,'code_hashes':code_hashes,'task_groups':dict(stats),'shape_subgroups':dict(shape),
 'sample_checks':sample_checks,'micro_fraction':str(micro),'macro_fraction':str(macro),
 'historical_eos_rate':seen['/results/interventions/none/eos_rate'],
 'scope':'Arithmetic and deterministic record reconstruction plus recomputation from historical generated IDs. No training, model inference, checkpoint load, GPU, dataset/model download, or neural .pt write.'}
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
