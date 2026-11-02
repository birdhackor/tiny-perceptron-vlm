"""Independent README_en evidence checks; no model import, forward, training or publication."""

import ast
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RAW = Path('/workspace/work/omni-P5-readme-en-raw-v1')
FINAL = Path('/workspace/work/omni-final-completed-41004229262/verified')
CORE = Path('/workspace/work/omni-core-completed-40988311100/verified')
PYTHON = '/workspace/tiny-perceptron-vlm/.venv/bin/python'
INPUTS = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    raw = p.read_bytes()
    INPUTS[str(p)] = {'bytes': len(raw), 'sha256': sha(raw)}
    return raw


def js(path):
    return json.loads(read(path))


def save(name, obj):
    name = name.replace('-v1.json', '-v2.json')
    p = OUT / name
    if p.exists():
        raise RuntimeError('Evidence is immutable: ' + str(p))
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


started = now()
freeze = js('docs/course-revision-20261029-omni/reviews/P5-actual/auxiliary-source-freeze-v2.json')
en_page = next(p for p in freeze['pages'] if p['page_id'] == 'readme-en')
body = read(en_page['snapshot']).decode()
assert sha(body.encode()) == en_page['source_sha256']
canonical = js('docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v10.json')
zh_page = next(p for p in canonical['pages'] if p['page_id'] == 'readme')
zh = read(zh_page['snapshot']).decode()
assert sha(zh.encode()) == zh_page['source_sha256']
links = []
for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', body):
    if '://' in target:
        continue
    path, _, anchor = target.partition('#')
    text = read(path).decode('utf-8')
    anchor_ok = not anchor or any(line.startswith('## ' + anchor + ' ') for line in text.splitlines())
    links.append({'target': target, 'exists': True, 'anchor_ok': anchor_ok})
assert all(r['anchor_ok'] for r in links)
lesson_pages = [p for p in canonical['pages'] if p['kind'] == 'lesson']
numbered = {}
for p in ['course/README.md', 'course/first-steps.md', 'course/training.md', 'course/glossary.md']:
    numbered[p] = len(re.findall(r'^## [A-Z]\.\d+ ', read(p).decode(), re.M))
notebooks = list((ROOT / 'notebooks').rglob('*.ipynb'))
assert len(lesson_pages) == len(notebooks) == 331
assert sum(numbered.values()) == 26
new_ids = ['7.26','9.17','9.18','10.12','12.17','12.18','14.19','16.15','16.16','17.16','17.17','19.21','19.22','19.23','19.24','20.14','20.15','20.16']
assert len(new_ids) == 18 and set(new_ids) <= {p['page_id'] for p in lesson_pages}

# Parse syntax and argument declarations without importing executable modules.
code_files = ['scripts/check_env.py','scripts/prepare_data.py','scripts/train.py',
              'scripts/fetch_training_assets.py','scripts/fetch_course_models.py',
              'scripts/check_course_models.py','scripts/build_visuals.py',
              'scripts/format_course_code.py','scripts/build_course.py',
              'scripts/check_course_reviews.py','scripts/check_technical_reviews.py',
              'docs/review-tools/check_review_round.py','scripts/check_notebooks.py',
              'scripts/export_course.py','scripts/stage_mathjax.py',
              'tiny_perceptron/model.py','tiny_perceptron/training.py',
              'tiny_perceptron/omni_products/reliability.py',
              'tiny_perceptron/omni_products/signal_metrics.py',
              'scripts/omni_revision/core_evaluation.py']
syntax = []
for p in code_files:
    raw = read(p)
    tree = ast.parse(raw, filename=p)
    flags = [n.args[0].value for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'add_argument' and n.args
             and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)]
    syntax.append({'path':p,'sha256':sha(raw),'syntax':'valid','declared_arguments':flags})
safe_commands = []
for args in [['scripts/fetch_course_models.py','--list'], ['scripts/fetch_training_assets.py','--list']]:
    result = subprocess.run([PYTHON,*args],cwd=ROOT,text=True,capture_output=True,check=False)
    safe_commands.append({'command':[PYTHON,*args],'returncode':result.returncode,
                          'stdout':result.stdout,'stderr':result.stderr})
    assert result.returncode == 0
save('safe-cli-v1.json', {'started_at':started,'completed_at':now(),'commands':safe_commands,
                        'scope':'Only read-only --list commands executed; no download or NN execution.'})

# Actual historical command stdout, kept distinct from this review's own execution.
old = js('docs/technical-reviews/artifacts/omni-technical-readme-v1/commands.json')
historical = []
for row in old['records']:
    if row['command'] in ['python scripts/check_env.py',
                          'python scripts/prepare_data.py --kind toy-text',
                          'python scripts/train.py --task text --data data/generated/toy-text/train.jsonl']:
        raw = read(row['raw_path'])
        assert sha(raw) == row['raw_sha256'] and len(raw) == row['raw_bytes']
        assert raw.decode() == row['output']
        historical.append({k:row[k] for k in ['command','actual_invocation','cwd','started_at','exit_code','raw_path','raw_sha256','raw_bytes','output']})
env_stdout = next(r['output'] for r in historical if r['command']=='python scripts/check_env.py')
assert 'CPU（5 執行緒）' in env_stdout and '2.14.1+cpu' in env_stdout and 'OK（cpu）' in env_stdout
dry = json.loads(next(r['output'] for r in historical if r['command'].startswith('python scripts/train.py')).splitlines()[-1])
assert dry['mode'] == 'dry-run-no-weight-update' and dry['device'] == 'cpu'
assert dry['history'][0]['loss'] == 5.770333290100098
assert dry['history'][0]['grad_norm'] == 0.9595053195953369
clean = js('docs/technical-reviews/artifacts/revision-20261016/engineering-readme-clean/checks-result.json')
state_observation = [{k:r[k] for k in ['changed_state_tensors','optimizer_step_calls','checkpoint_save_calls']} for r in clean['dry_runs']]
assert all(not r['changed_state_tensors'] and r['optimizer_step_calls']==r['checkpoint_save_calls']==0 for r in state_observation)
public_raw = js('outputs/omni-P5-readme-old-public-text-foundation-generated-record-v1.json')
infer = public_raw['commands'][-1]
assert infer['output']['answer'] == 'side=right.' and infer['returncode'] == 0
assert sha(infer['stdout'].encode()) == infer['stdout_sha256']
matching_source_hashes = {p: sha(read(p))==h for p,h in public_raw['source_hashes'].items()}
assert all(matching_source_hashes.values())
save('historical-command-extracts-v1.json', {
    'created_at':now(),'historical_commands':historical,'dry_run_state_observations':state_observation,
    'public_model_check':{'checked_at':public_raw['checked_at'],'public_source':public_raw['public_source'],
        'anonymous_download':public_raw['anonymous_download'],'environment':public_raw['environment'],
        'files':len(public_raw['verified_files']),'probe_count':len(public_raw['commands'])-1,
        'inference':infer,'matching_current_source_hashes':matching_source_hashes},
    'scope':'Read original raw stdout and inputs; did not repeat these forward/backward or model inference commands.'})

# Recompute fixed final results from saved rows/logits; never call model forward.
selection = js(FINAL / 'frozen/model-selection.json')
opening = js(FINAL / 'frozen/final-opening.json')
assert opening['model_selection_sha256'] == sha(read(FINAL / 'frozen/model-selection.json'))
assert selection['final_opened'] is False and opening['final_opened'] is True
jobs = []
for p in sorted((CORE/'core').glob('*/training/training-result.json')):
    j = js(p)
    assert j['complete'] and j['optimizer_updates_observed'] > 0
    assert j['initial_parameter_sha256'] != j['latest_parameter_sha256']
    jobs.append({'job':p.parent.parent.name,'complete':j['complete'],'updates':j['updates'],
                 'optimizer_updates_observed':j['optimizer_updates_observed'],'runtime':j['runtime'],
                 'initial_parameter_sha256':j['initial_parameter_sha256'],'latest_parameter_sha256':j['latest_parameter_sha256']})
assert len(jobs)==10
av = js(FINAL/'measurements/av/results.json')
selected = av['models'][selection['selected']['av']]
av_rows = [{k:r[k] for k in ['family','panel','answers','gold','waveform_swap_answers','waveform_swap_gold']} for r in selected['records']]
av_success = sum(r['answers']==r['gold'] for r in av_rows)
swap_success = sum(r['waveform_swap_answers']==r['waveform_swap_gold'] for r in av_rows)
assert len(av_rows)==200 and len({r['family'] for r in av_rows})==200 and av_success==200 and swap_success==148
choice = js(FINAL/'measurements/choice/results.json')['primary']
crows = []
T,tau = selection['temperature'],selection['threshold']
for r in choice['rows']:
    c,info = r['tasks']['choice'],r['choice_info']
    choose = lambda x:min(range(len(x['ids'])),key=lambda i:(-x['logits'][i],x['ids'][i]))
    answer = c['ids'][choose(c)]
    scaled = [x/T for x in c['logits']]; pivot=max(scaled)
    exps=[math.exp(x-pivot) for x in scaled];pmax=max(exps)/sum(exps)
    accept = pmax>=tau and answer!='none' and info['ids'][choose(info)]=='yes'
    error = accept and (answer!=c['gold'] or not r['world_supported'] or not r['choice_information_sufficient'])
    crows.append({'family':r['family'],'ids':c['ids'],'logits':c['logits'],'gold':c['gold'],
        'choice_info':{'ids':info['ids'],'logits':info['logits']},
        'world_supported':r['world_supported'],'choice_information_sufficient':r['choice_information_sufficient'],
        'answer':answer,'p_max_recomputed':pmax,'accepted_recomputed':accept,'error_recomputed':error})
choice_correct=sum(r['answer']==r['gold'] for r in crows)
accepted=sum(r['accepted_recomputed'] for r in crows); errors=sum(r['error_recomputed'] for r in crows)
assert (len(crows),choice_correct,accepted,errors)==(60,39,53,19)
sound=js(FINAL/'measurements/sound/results.json')['codec_and_gold_program']
srows=[]
for r in sound['rows']:
    g=r['generation']; a=g['analysis'];events=a['measured']['events'];expected=a['expected']
    timing_ok=len(events)==len(expected) and all(
        abs(x['frequency']-y['frequency'])/y['frequency']<=0.05
        and abs(x['onset_ms']-y['onset_ms'])<=30
        and abs(x['offset_ms']-y['offset_ms'])<=30 for x,y in zip(events,expected))
    ok=g['eos'] and not g['capped'] and a['measured']['legal'] and timing_ok
    assert ok==g['program_success']
    srows.append({'family_index':r['family_index'],'program':r['program'],'eos':g['eos'],'capped':g['capped'],
        'measured':events,'expected':expected,'legal':a['measured']['legal'],'success_recomputed':ok})
assert len(srows)==200 and sum(r['success_recomputed'] for r in srows)==200
save('new-results-recomputed-v1.json',{'created_at':now(),'selection':selection['selected'],
    'selection_created_at':selection['created_at'],'final_opening_created_at':opening['created_at'],
    'training_jobs':jobs,'av':{'families':len(av_rows),'paired_correct':av_success,'swap_paired_correct':swap_success,'rows':av_rows},
    'Choice':{'families':60,'correct':choice_correct,'accepted':accepted,'accepted_errors':errors,'T':T,'tau':tau,'rows':crows},
    'sound':{'programs':200,'successful':200,'tolerances':{'frequency_relative':0.05,'time_ms':30},'rows':srows},
    'scope':'Pure arithmetic on preserved predictions/logits/event measurements. No final/teacher neural forward, calibration search, policy selection or training.'})

# Original v2 archive: inspect metadata and JSONL only, without extracting weights.
manifest=js('docs/selftrained/v2-manifest.json')
package=ROOT/manifest['package']['path']; archive=read(package)
assert len(archive)==manifest['package']['bytes']==67862431 and sha(archive)==manifest['package']['sha256']
expected={r['path']:r for r in manifest['records']+manifest['assets']}
counts=Counter();nfiles=0
with tarfile.open(package,'r:gz') as tar:
    for member in tar:
        if not member.isfile():continue
        nfiles+=1
        name=member.name.removeprefix('./')
        raw=tar.extractfile(member).read()
        spec=expected[name]
        assert len(raw)==spec['bytes'] and sha(raw)==spec['sha256']
        if name.endswith('.jsonl'):
            split=name.rsplit('-',1)[-1].removesuffix('.jsonl')
            counts[split]+=sum(bool(line.strip()) for line in raw.splitlines())
assert nfiles==len(expected)==8962 and dict(counts)=={'test':3734,'train':28876,'validation':2435}
public=js(RAW/'hf-v2-api.json')
v2files=[r for r in public['siblings'] if r['rfilename'].startswith('selftrained/v2/')]
assert public['sha']=='979cdfacc588ad0536f1c64fff96f264571cf054' and len(v2files)==16
training={}
for name in ['moe-native','dense-weighted']:
    j=js(f'docs/selftrained/results/training-raw/{name}/raw/train-receipt.json')
    training[name]={k:j[k] for k in ['steps','total_parameters','selection','completed_requested_steps']}
test_metrics={}
for architecture in ['moe','dense']:
    j=js(f'docs/selftrained/results/public-raw/{architecture}/test/metrics.json')
    test_metrics[architecture]={'count':j['count'],'evaluation_complete':j['evaluation_complete'],
        'tool_roundtrip':j['per_task_final_reply']['tool_call']['tool_roundtrip']}
    assert j['count']==3734 and j['evaluation_complete']
sel=js('docs/selftrained/results/v2-final-training-source-selection.json')
selection_steps={
    'moe_joint': {'selected_step':sel['moe_joint']['selected_step'],'completed_steps':sel['moe_joint']['completed_requested_steps']},
    'dense_joint': {'selected_step':sel['dense_joint']['selected_step'],'completed_steps':sel['dense_joint']['completed_steps']},
}
experiments=js('docs/course-experiments/public-models.json')['models']
weight_files=[f for m in experiments for f in m['files'] if Path(f['output']).suffix in ['.pt','.pth','.ckpt','.safetensors']]
capstone=js('docs/course-experiments/capstone-public.json')['models']
assert (len(experiments),len(weight_files),len(capstone))==(30,120,11)
legacy=js('docs/lesson-products/results/final-report-v1.json')
legacy_values={'B_5_loops':{k:legacy['b']['5'][k] for k in ['boxes','sum']},'C':legacy['c'],'C_wording':legacy['shifts']['c_wording'],
               'recorded_at':legacy['recorded_at'],'settings_freeze_sha256':legacy['settings_freeze_sha256']}
assert legacy_values['B_5_loops']['boxes']['correct']==2928 and legacy_values['B_5_loops']['sum']['correct']==2
gpu_results=[]
for p in sorted((ROOT/'docs/course-experiments/results').glob('*.json')):
    j=js(p)
    if j.get('gpu')=='NVIDIA L4':gpu_results.append({'path':str(p.relative_to(ROOT)),'device':j.get('device'),'torch':j.get('torch_version'),'gpu':j.get('gpu')})
ci=js('docs/validation-artifacts/release-compatibility-ci.json')
save('local-inventory-and-historical-results-v1.json',{
    'created_at':now(),'source':en_page,'canonical_prerequisite':zh_page,
    'counts':{'lesson_sections':len(lesson_pages),'notebooks':len(notebooks),'other_numbered':numbered,'numbered_total':331+26,'new_lessons':new_ids},
    'local_links':links,'syntax_and_cli_arguments':syntax,'v2_archive':{'bytes':len(archive),'sha256':sha(archive),'verified_files':nfiles,'records':dict(counts)},
    'v2_public_files':v2files,'v2_training':training,'v2_tests':test_metrics,'v2_selection':selection_steps,
    'public_counts':{'experiment_groups':len(experiments),'weight_files':len(weight_files),'capstone_files':len(capstone)},
    'legacy_B_C':legacy_values,'historical_L4_result_environments':gpu_results,
    'historical_CI':{'run_id':ci['run_id'],'revision':ci['revision'],'scope':ci['scope'],'jobs':ci['jobs']},
    'environment_wording_check':{'english':'Chapter 20 uses a separate Python 3.12 and PyTorch 2.8.0 environment',
        'chinese_scope':'existing Qwen/Whisper route','new_Kev_actual_runtime':js(Path('/workspace/work/omni-kev-recovery-completed-41011677439/verified/runtime-before.json')),
        'decision':'English is less specific; no contradictory runtime found. It does not assert every inline CPU example requires this environment.'}})
save('execution-inputs-v1.json',{'started_at':started,'completed_at':now(),
    'command':f'{PYTHON} docs/technical-reviews/artifacts/omni-P5-readme-en/verify_readme_en.py',
    'environment':{'python':platform.python_version(),'platform':platform.platform(),'device':'CPU; Python arithmetic and file reads only'},
    'input_digests':INPUTS,'result':'All explicit assertions passed. No NN import, forward, training, retune, workflow, Modal or publication.'})
print(json.dumps({'started_at':started,'completed_at':now(),'result':'all assertions passed','inputs':len(INPUTS)},ensure_ascii=False))
