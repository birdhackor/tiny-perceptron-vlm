"""Reviewer-owned, read-only course checks; no model or teacher forward and no training."""
from pathlib import Path
from datetime import datetime, UTC
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter

import torch
from bs4 import BeautifulSoup

ROOT = Path('/workspace/work/omni-book-integrated-v1')
CORE = Path('/workspace/work/omni-core-completed-40988311100/verified')
FINAL = Path('/workspace/work/omni-final-completed-41004229262/verified')
OUT = ROOT / 'docs/technical-reviews/artifacts/omni-p6-wave04-course/probe-result.json'
files = {}

def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    data = path.read_bytes()
    files[str(path)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return data

def load(path):
    return json.loads(read(path))

def digest(path):
    read(path)
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return files[str(path)]['sha256']

def cells(nb):
    return [(x['cell_type'], ''.join(x['source'])) for x in nb['cells']]

result = {'created_at': datetime.now(UTC).isoformat(),
          'reviewer_task': '/root/omni_p6_technical_course',
          'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'CPU'},
          'scope': 'Only string/dictionary/math checks, hash reads and preserved raw recomputation. No trained-model/teacher forward, no NN training, no final rerun, no network or publishing.'}
freeze = load('docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18.json')
page = next(x for x in freeze['pages'] if x['page_id'] == 'course')
body = read(page['snapshot']).decode()
assert hashlib.sha256(body.encode()).hexdigest() == page['source_sha256']
assert read('course/README.md') == body.encode()
result['whole_body'] = {'snapshot': page['snapshot'], 'source_sha256': page['source_sha256'],
    'headings': re.findall(r'^## (.+)$', body, re.M), 'details_summaries': re.findall(r'<summary>(.*?)</summary>',body),
    'inline_code': re.findall(r'`([^`\n]+)`', body), 'fenced_blocks': body.count('```'), 'figures': re.findall(r'!\[[^\]]*\]\((.*?)\)',body)}
links = []
for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', body):
    if target.startswith('https://'):
        links.append({'label': label, 'target': target, 'external': True})
        continue
    path,_,anchor = target.partition('#')
    dest = (ROOT/'course'/path).resolve() if path else ROOT/'course/README.md'
    text = read(dest).decode()
    heading = next((x for x in text.splitlines() if re.match(r'^#{1,6} '+re.escape(anchor)+r'(?: |$)',x)),None) if anchor else None
    assert not anchor or heading is not None, (target, 'missing anchor')
    links.append({'label':label,'target':target,'exists':True,'heading':heading})
result['source_links'] = links

# The published guide has no code blocks; these are its transparent introductory mechanisms.
text = '貓看狗，狗看貓。'
chars = sorted(set(text)); table = {x:i for i,x in enumerate(chars)}
encoded = [table[x] for x in text]
assert ''.join(chars[i] for i in encoded) == text
assert encoded[0] == encoded[6]
logits = {'看': [0.0,1.0,-1.0]}
def next_distribution(prefix):
    values = logits[prefix[-1]]
    z = sum(math.exp(x) for x in values)
    return [math.exp(x)/z for x in values]
assert next_distribution('貓看') == next_distribution('狗看')
result['transparent_mechanisms'] = {'text':text,'chars':chars,'encoded':encoded,'roundtrip':text,
    'same_cat_id':encoded[0], 'last_character_only_equal_distributions':next_distribution('貓看'),
    'weighted_sum_example': 'For weights [0.25,0.75], values [4,8], sum(w_i*x_i)=1+6=7; no learned-quality claim',
    'scope':'Hand-defined reversible encoding and same-input function equality, not training or semantic understanding.'}

joined_receipt = load('outputs/omni-P5-current-kernel-bundle-v10-receipt.json')
nb_rows = []
for row in joined_receipt['rows']:
    src = load(row['source']['path']); joined = load(row['joined']['path'])
    assert digest(row['source']['path']) == row['source']['sha256']
    assert digest(row['joined']['path']) == row['joined']['sha256']
    assert cells(src) == cells(joined), row['id']
    code = [x for x in joined['cells'] if x['cell_type']=='code']
    errors = [o for x in code for o in x.get('outputs',[]) if o.get('output_type')=='error']
    assert not errors
    nb_rows.append({'id':row['id'],'code_cells':len(code),
        'executed_cells':sum(x.get('execution_count') is not None for x in code),
        'output_cells':sum(bool(x.get('outputs')) for x in code)})
result['notebooks'] = {'total':len(nb_rows),'cell_type_and_source_exact':True,'error_outputs':0,
    'provenance_scope':'Existing exact code outputs joined from historical executions and later replacement executions; not 331 fresh kernels in this review, not evidence of Colab execution.',
    'rows':nb_rows,
    'old_kernel_receipt':load('/workspace/tiny-perceptron-vlm/outputs/omni-notebooks-review-v1/validation-kernel.json'),
    'new_917_receipt':load('outputs/omni-P5-coherence-9.17-kernel-v1/validation-kernel.json')}
preview = ROOT/'outputs/omni-P6-preview-source18-v2'
html = BeautifulSoup(read(preview/'course.html'),'html.parser')
article = html.find('article')
assert article is not None
details = article.find_all('details')
first = BeautifulSoup(read(preview/'1.1.html'),'html.parser')
result['preview'] = {'course':str(preview/'course.html'),'article_headings':[h.get_text(' ',strip=True) for h in article.find_all(['h1','h2'])],
    'details':[x.get_text(' ',strip=True) for x in details],
    'first_lesson_has_CPU_label':'CPU' in first.get_text(' ',strip=True),
    'first_lesson_has_expected_output': all(x in first.get_text() for x in ['貓看狗','狗看貓']),
    'scope':'Static generated HTML and exact notebook bytes checked. No committedGit, hosted endpoint or Colab acceptance; no desktop/mobile visual claim.'}

from tiny_perceptron.lesson_products.inference import auto_solve
policy = load('docs/lesson-products/results/b-auto-policy.json')
class FakeB:
    auto_policy = policy
    def synchronize(self): pass
    def solve(self,state,loops): return {'loops':loops,'scope':'stub, no NN forward'}
class FakeC:
    def __init__(self,choice,info): self.choice,self.info=choice,info
    def synchronize(self): pass
    def judge(self,state,kind): return {'selected':self.choice if kind=='choice' else self.info,'seconds':0.0}
state = '兩盒追蹤;A=1;B=2;事件=交換;查詢=A'
auto_rows = []
for choice,info,expected in [('boxes','yes','answered'),('boxes','no','clarification_needed'),('none','yes','unsupported')]:
    value = auto_solve(FakeB(),FakeC(choice,info),state)
    assert value['status'] == expected
    assert value['b_loops'] == (3 if expected=='answered' else 0)
    auto_rows.append(value)
result['auto_stub_mechanism'] = {'policy':{k:policy[k] for k in ['version','low_max_events','medium_max_events','sum_depth']},'rows':auto_rows,'scope':'Control flow only with explicitly artificial judges/solver; no product quality claim.'}

legacy = {}
for name in ['b','c']:
    receipt = load(Path('/workspace/tiny-perceptron-vlm/outputs/lesson-products')/name/'training-receipt.json')
    assert receipt['random_initialization'] and receipt['resume_source'] is None and receipt['weights_changed']
    legacy[name] = {k:receipt[k] for k in ['run_identity','product','random_initialization','resume_source','seed','last_step','selected_step','initial_weights_sha256','selected_weights_sha256','weights_changed','data_manifest_sha256','environment']}
    frozen_manifest = load(f'docs/lesson-products/results/{name}-artifact-manifest.json')
    local_manifest = load(Path('/workspace/tiny-perceptron-vlm/outputs/lesson-products')/name/'artifact-manifest.json')
    legacy[name]['local_manifest_equals_frozen'] = local_manifest == frozen_manifest
    assert legacy[name]['local_manifest_equals_frozen']
arows=[]
for stage in ['moe-pretrain','moe-vision','moe-audio','moe-ocr','moe-joint','moe-weighted','moe-native']:
    p=ROOT/f'docs/selftrained/results/training-raw/{stage}/raw'
    if not p.exists():continue
    tr=load(p/'train-receipt.json');ex=load(p/'execution.json')
    assert tr['origin']['kind']=='all-neural-weights-random'
    assert ex['status']=='completed' and ex['returncode']==0
    arows.append({'stage':stage,'training_stage':tr['stage'],'steps':tr['steps'],'origin_kind':tr['origin']['kind'],
        'seed':tr['origin']['seed'],'completed_requested_steps':tr['completed_requested_steps'],
        'selected':tr['selection'],'test_used_for_selection':tr['test_used_for_selection'],
        'command':ex['command'],'revision':ex['revision']})
legacy['a_raw_stages']=arows
result['legacy']=legacy

recipe=load(CORE/'frozen/formal-recipes.json');selection=load(CORE/'frozen/model-selection.json');metrics=load(CORE/'frozen/metrics.json')
jobs=[]
for job in recipe['job_order']:
    d=CORE/'core'/job
    origin=load(d/'job-origin.json');done=load(d/'job-result.json');tr=load(d/'training/training-result.json');log=load(d/'training/training-log.json')
    assert done['success'] and tr['complete'] and tr['optimizer_updates_observed'] and not done['final_opened'] and not tr['final_opened']
    assert done['job_origin_sha256']==digest(d/'job-origin.json')
    assert done['training_result_sha256']==digest(d/'training/training-result.json')
    assert tr['files']['training-log.json']['sha256']==digest(d/'training/training-log.json')
    assert tr['initial_parameter_sha256']!=tr['latest_parameter_sha256']
    assert tr['best'] in log
    assert digest(d/'training/weights.safetensors')==tr['files']['weights.safetensors']['sha256']
    jobs.append({'job':job,'success':done['success'],'complete':tr['complete'],'updates':tr['updates'],
        'best_update':tr['best']['update'],'initial_parameter_sha256':tr['initial_parameter_sha256'],
        'latest_parameter_sha256':tr['latest_parameter_sha256'],'selected_weights_sha256':tr['files']['weights.safetensors']['sha256'],
        'final_opened':done['final_opened'],'runtime':done['runtime'],'execution_host':done['execution_host']})
assert len(jobs)==10
result['formal_jobs']={'count':len(jobs),'selected':selection['selected'],'metrics_created_at':metrics['created_at'],
    'selection_created_at':selection['created_at'],'selection_before_final':selection['final_opened'] is False,
    'rows':jobs}
measurements={}
for part in ['av','sound','choice','integration']:
    d=FINAL/'measurements'/part
    origin=load(d/'part-origin.json');completion=load(d/'part-completion.json');raw=load(d/'results.json')
    assert completion['measurement_completed'] and completion['results_sha256']==digest(d/'results.json')
    assert completion['part_origin_sha256']==digest(d/'part-origin.json')
    assert origin['model_selection_sha256']==digest(CORE/'frozen/model-selection.json')
    assert origin['metrics_sha256']==digest(CORE/'frozen/metrics.json')
    measurements[part]=raw
av=[]
for name,raw in measurements['av']['models'].items():
    rows=raw['records'];assert len({r['family'] for r in rows})==200
    correct=sum(r['answers']==r['gold'] for r in rows)
    assert correct==raw['primary']['correct']
    panels={str(i):sum(r['answers']==r['gold'] for r in rows if r['panel']==i) for i in range(4)}
    assert all(sum(r['panel']==i for r in rows)==50 for i in range(4))
    swapped=sum(r['waveform_swap_answers']==r['waveform_swap_gold'] for r in rows)
    assert swapped==raw['same_metadata_waveform_swap']['paired']['correct']
    removed={m:sum(r[f'remove_{m}_answer']==2 for r in rows) for m in ['image','audio']}
    av.append({'job':name,'paired_correct':correct,'family_denominator':200,'panel_correct':panels,
        'removed_correct':removed,'swap_pair_correct':swapped,'swap_pair_failures':200-swapped,
        'primary_gate':correct>=metrics['AV']['paired_minimum_correct'] and all(v/50>=metrics['AV']['panel_minimum_accuracy'] for v in panels.values()) and all(v/200>=0.9 for v in removed.values())})
sound=measurements['sound']['codec_and_gold_program'];rows=sound['rows']
snr=sum(r['active_reference_snr_db'] for r in rows)/len(rows)
gold=sum(r.get('generation',{}).get('program_success',False) for r in rows)
legal=sum(r.get('generation',{}).get('legal',False) for r in rows)
assert len(rows)==200 and math.isclose(snr,sound['mean_active_reference_snr_db'],abs_tol=1e-12)
chain=measurements['sound']['actual_A_chain']['rows'];chain_correct=sum(r['chain_correct'] for r in chain)
assert len(chain)==200 and len({r['family'] for r in chain})==200
sound_summary={'families':200,'mean_active_reference_snr_db':snr,'gold_program_correct':gold,'legal_correct':legal,
    'actual_A_chain_correct':chain_correct,'chain_families':200,'chunk_controls':sound['chunk_controls'],
    'primary_gates':snr>=10 and gold/200>=0.85 and legal==200 and chain_correct/200>=0.8,
    'scope':'8kHz <=2s finite tones/rhythms; actual chain has only two semantic programs, not general speech.'}
choice=measurements['choice']['primary'];rows=choice['rows']
assert len(rows)==60 and len({r['family'] for r in rows})==60
counts={t:sum(r['tasks'][t]['answer']==r['tasks'][t]['gold'] for r in rows) for t in ['choice','info','rubric']}
policy_rows=[]
for r in rows:
    c=r['tasks']['choice'];accepted=c['p_max']>=selection['threshold'] and c['answer']!='none' and r['choice_info']['answer']=='yes'
    error=c['answer']!=c['gold'] or not r['world_supported'] or not r['choice_information_sufficient']
    policy_rows.append((r['family'],accepted,error))
accepted=sum(a for _,a,e in policy_rows);errors=sum(a and e for _,a,e in policy_rows)
assert accepted==choice['policy']['accepted'] and errors==choice['policy']['errors']
inv=measurements['choice']['development_invariants'];assert len(inv['rows'])==8
invariant_max={k:max(r[k] for r in inv['rows']) for k in inv['rows'][0] if k.endswith('max_abs')}
assert all(v<=1e-5 for v in invariant_max.values()) and all(r['permutation_answer_same'] for r in inv['rows'])
choice_summary={'families':60,'counts':counts,'threshold':selection['threshold'],'temperature':selection['temperature'],
    'accepted':accepted,'errors':errors,'accepted_empirical_risk':errors/accepted,'minimum_accepted':30,'maximum_risk':0.05,
    'gate_counts':{'choice':54,'info':51,'rubric':51},'passes_all_quality_gates':False,
    'invariant_families':8,'invariant_max':invariant_max,'permutation_answer_same':True,
    'scope':'Recomputed from preserved raw predictions/flags; no fresh trained forward. Three types share 60 families, not 180 independent trials.'}
integ=measurements['integration'];b=integ['B_retention'];br=b['rows']
assert len(br)==84 and sum(r['old_correct'] for r in br)==60 and sum(r['wrapper_correct'] for r in br)==60
assert all(r['old']['loops']==r['wrapper']['loops']==1 for r in br)
result['raw_recomputation']={'av':av,'sound':sound_summary,'choice':choice_summary,
    'B':{'population_families':84,'old_correct':60,'wrapper_correct':60,'fixed_depth':1},
    'limits':'Recounted retained family-level raw outputs, independent of final-report booleans. No ability outside finite tasks established.'}
from scripts.omni_revision.public_download import descriptor, verify_installed
published=descriptor(ROOT/'docs/omni-products/v1/public-release.json')
for kind in ['av','choice','codec','talker']:
    verify_installed(CORE,published['_model_selection'],kind)
help_rows=[]
for module in ['scripts.omni_revision.public_demo','scripts.omni_revision.public_download']:
    command=[sys.executable,'-m',module,'--help']
    completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=20)
    assert completed.returncode==0
    help_rows.append({'command':command,'returncode':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr})
result['download_entrypoint_checks']={'descriptor_matches_selected':published['_model_selection']==selection,
    'four_installed_inference_components_hash_verified':True,'help':help_rows,
    'scope':'Local package hashes and CLI help only. No download, neural forward, external endpoint, committedGit or release acceptance performed.'}
assert result['download_entrypoint_checks']['descriptor_matches_selected']
result['file_receipts']=files
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(OUT),'files_hashed':len(files),'links':len(links),'notebooks':len(nb_rows),
    'jobs':len(jobs),'selected_av':selection['selected']['av'],'choice':choice_summary,'sound':sound_summary},ensure_ascii=False))
