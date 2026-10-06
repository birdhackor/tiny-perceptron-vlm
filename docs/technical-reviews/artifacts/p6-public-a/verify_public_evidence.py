"""Bounded, read-only factual checks; no model generation, downloads or training."""
from pathlib import Path
import ast, collections, hashlib, importlib.metadata, json, platform, re, subprocess, tarfile
ROOT = Path.cwd()
OUT = ROOT / 'docs/technical-reviews/artifacts/p6-public-a'
ORIG = OUT / 'originals'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_bytes())
def materialized(p,oid,size):
    p=Path(p)
    if p.stat().st_size==size: return p
    pointer=p.read_text();assert ('oid sha256:'+oid) in pointer and ('size '+str(size)) in pointer
    git=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip())
    return git/'lfs/objects'/oid[:2]/oid[2:4]/oid
def bsha(b): return hashlib.sha256(b).hexdigest()
def archive_check(p, expected):
    actual={}; extensions=collections.Counter()
    with tarfile.open(p,'r:gz') as tar:
        for m in tar:
            assert m.isfile(), m.name
            b=tar.extractfile(m).read()
            actual[m.name]={'bytes':len(b),'sha256':bsha(b)}
            extensions[Path(m.name).suffix]+=1
    exp={x['path']:{'bytes':x['bytes'],'sha256':x['sha256']} for x in expected}
    assert actual==exp, str(p)
    return {'files':len(actual),'unpacked_bytes':sum(x['bytes'] for x in actual.values()),'extension_counts':dict(extensions),'all_manifest_entries_exact':True}
result={'environment':{'python':platform.python_version(),'torch':importlib.metadata.version('torch'),'device':'cpu'},'source_hashes':{},'pointers':{}}
manifest=read('docs/selftrained/v2-manifest.json'); package=manifest['package']
assert sha(package['path'])==package['sha256']
assert Path(package['path']).stat().st_size==package['bytes']
result['v2_archive']={'bytes':package['bytes'],'sha256':package['sha256'],**archive_check(package['path'],manifest['records']+manifest['assets'])}
counts=collections.Counter(); voices={s:set() for s in ['train','validation','test']}; voice_native={s:set() for s in voices}; ocr_fonts=set(); ocr_chars=set(); vision_classes=set(); tasks=collections.Counter(); selected_rows={}
with tarfile.open(package['path'],'r:gz') as tar:
    for m in manifest['records']:
        rows=[json.loads(l) for l in tar.extractfile(m['path']).read().splitlines() if l.strip()]
        for r in rows:
            s=r['split']; counts[s]+=1;tasks[(s,r['task'])]+=1
            if r.get('audio'):
                voices[s].add(r['audio'])
                if 'augmentation' not in r: voice_native[s].add(r['audio'])
            if r['task']=='ocr':
                ocr_fonts.add(r['supervision']['font_family']);ocr_chars.update(r['supervision']['ocr_text'])
            if r['task'].startswith('vision_'):vision_classes.update(x for x in r['supervision']['vision_labels'] if x>=0)
        selected_rows[m['path']]={'count':len(rows),'sha256':m['sha256']}
result['v2_records']={'counts':dict(counts),'files':selected_rows,'task_counts':{f'{s}/{t}':n for (s,t),n in tasks.items()},'audio_files_by_split':{s:len(x) for s,x in voices.items()},'original_recording_files_by_split':{s:len(x) for s,x in voice_native.items()},'ocr_fonts':sorted(ocr_fonts),'ocr_chars':sorted(ocr_chars),'vision_classes':sorted(vision_classes)}
assert dict(counts)=={'test':3734,'train':28876,'validation':2435}
assert result['v2_records']['original_recording_files_by_split']=={'train':62,'validation':15,'test':30}
basic=read('assets/training/manifest.json'); base_checks=[]
for a in basic['assets']:
    p=materialized(a['archive'],a['archive_sha256'],a['archive_bytes']); assert p.stat().st_size==a['archive_bytes'] and sha(p)==a['archive_sha256']
    base_checks.append({'id':a['id'],'archive_sha256':a['archive_sha256'],'archive_bytes':a['archive_bytes'],'training_records':a['training_records'],'license':a['license'],**archive_check(p,a['files'])})
result['baseline_assets']={'count':len(base_checks),'archive_bytes':sum(x['archive_bytes'] for x in base_checks),'MiB':sum(x['archive_bytes'] for x in base_checks)/2**20,'training_records':sum(x['training_records'] for x in base_checks),'per_asset':base_checks}
natural=read('docs/natural-assistant/v4/manifest.json')
result['natural_archives']={'total_bytes':sum(x['bytes'] for x in natural['archives']),'archives':[]}
for a in natural['archives']:
    ap=materialized(a['path'],a['sha256'],a['bytes']);assert ap.stat().st_size==a['bytes'] and sha(ap)==a['sha256']
    result['natural_archives']['archives'].append({k:a[k] for k in ['path','bytes','sha256']})
# Read only the named measurement/method/provenance pointers; preserve original full-file hashes.
training=[]
for p in sorted(Path('docs/selftrained/results/training-raw').glob('*/raw/train-receipt.json')):
    d=read(p); execution=read(p.with_name('execution.json'))
    fields=['stage','architecture','steps','tokens','target_tokens','total_parameters','trainable_parameters','train_records','validation_records','test_used_for_selection','completed_requested_steps','selection','seed']
    training.append({'path':str(p),'sha256':sha(p),**{k:d[k] for k in fields},'origin_kind':d['origin']['kind'],'stage_history':[{k:h[k] for k in ['stage','step','checkpoint_sha256','selection'] if k in h} for h in d['stage_history']],'execution':{k:execution[k] for k in ['status','returncode','command','resource_spec','revision']}})
    assert d['completed_requested_steps'] and not d['test_used_for_selection'] and d['origin']['kind']=='all-neural-weights-random'
    assert execution['returncode']==0
    result['pointers'][str(p)]=['/'+k for k in fields]+['/origin/kind','/stage_history/*/{stage,step,checkpoint_sha256,selection}']
result['training']=training; assert len(training)==15
hf_tree=read(ORIG/'huggingface.co__api__models__birdhackor__tiny-perceptron-course-models__tree__979cdfacc588ad0536f1c64fff96f264571cf054__selftrained__v2_recursive_true')
files=[x for x in hf_tree if x['type']=='file'];assert len(files)==16
hf_manifests={}
for n in ['moe-pretrain','moe-sft','moe-joint','dense-joint']:
    f=next(ORIG.glob('*__'+n+'__inference-manifest.json'));d=read(f)
    for name in ['model-config.json','tokenizer.json']:
        assert sha(next(ORIG.glob('*__'+n+'__'+name)))==d['files'][name]
    safe=next(x for x in files if x['path']==f'selftrained/v2/{n}/model.safetensors')
    assert safe['lfs']['oid']==d['files']['model.safetensors']
    hf_manifests[n]={'path':str(f.relative_to(ROOT)),'sha256':sha(f),'selected_step':d['selected_step'],'selected_checkpoint_sha256':d['selected_checkpoint_sha256'],'files':d['files'],'selection':d['selection']}
result['hf_exports']={'revision':'979cdfacc588ad0536f1c64fff96f264571cf054','file_count':len(files),'manifests':hf_manifests}
final={}
for arch in ['moe','dense']:
    folder=Path('docs/selftrained/results/public-raw')/arch
    d=read(folder/'test/metrics.json');frozen=read(folder/'freeze/frozen.json');ev=read(folder/'test/evaluation-receipt.json')
    assert d['count']==d['expected_count']==ev['completed_count']==3734
    assert d['evaluation_complete'] and not d['interrupted'] and ev['resumed_from'] is None
    assert d['safe_weights_sha256']==frozen['safe_weights_sha256']==hf_manifests[arch+'-joint']['files']['model.safetensors']
    assert d['selected_checkpoint_sha256']==frozen['selected_checkpoint_sha256']==hf_manifests[arch+'-joint']['selected_checkpoint_sha256']
    assert frozen['validation_metrics_sha256']==sha(folder/'freeze/metrics.json')
    tasks=d['per_task_final_reply'];assert sum(x['count'] for x in tasks.values())==3734
    tool=tasks['tool_call']['tool_roundtrip'];assert tool['numerator']/tool['denominator']<d['thresholds']['tool_roundtrip']
    final[arch]={'test_metrics_sha256':sha(folder/'test/metrics.json'),'test_count':d['count'],'generation_uses_teacher_forcing':d['teacher_forcing_used_for_generation'],'tool_roundtrip':tool,'thresholds':d['thresholds'],'per_task':{k:{z:v[z] for z in ['count','source_groups','recording_assets','image_assets','semantic','exact','final_cer'] if z in v} for k,v in tasks.items()},'voice_topic_continuation':d['voice_topic_continuation'],'selected_step':hf_manifests[arch+'-joint']['selected_step'],'validation_only':frozen['selection'],'test_once':frozen['test_once']}
    for fn in ['test/metrics.json','freeze/frozen.json','test/evaluation-receipt.json','freeze/metrics.json']:
        p=folder/fn; result['source_hashes'][str(p)]=sha(p)
    result['pointers'][str(folder/'test/metrics.json')]=['/count','/expected_count','/evaluation_complete','/interrupted','/safe_weights_sha256','/selected_checkpoint_sha256','/thresholds','/per_task_final_reply','/voice_topic_continuation','/teacher_forcing_used_for_generation']
result['final_test']=final
cpu=[]
for p in sorted(Path('docs/selftrained/results/public-cpu-raw').glob('*-result.json')):
    d=read(p)
    for stream in ['stdout','stderr']:
        original=p.parent/Path(d[stream+'_path']).name
        assert sha(original)==d[stream+'_sha256']
    assert d['returncode']==0
    cpu.append({'path':str(p),'sha256':sha(p),**{k:d[k] for k in ['chat','argv','returncode','actual_answer','public_source','CPU_only_argv_verified'] if k in d}})
result['cpu_original_receipts']={'chat_calls':sum(x['chat'] for x in cpu),'nonchat_calls':sum(not x['chat'] for x in cpu),'commands':cpu};assert result['cpu_original_receipts']['chat_calls']==8
index=read('course/lesson-index.json');numbered=list(Path('course/chapters').glob('*.md'))+[Path(p) for p in ['course/README.md','course/first-steps.md','course/training.md','course/glossary.md']]
result['course_inventory']={'notebooks':len(index),'actual_notebooks':len(list(Path('notebooks').rglob('*.ipynb'))),'numbered_sections':sum(len(re.findall(r'^## [\dA-Z]+\.\d+ ',p.read_text(),re.M)) for p in numbered),'chapters':len(list(Path('course/chapters').glob('*.md')))}
plan=read('docs/course-experiments/plan.json');exports=read('docs/course-experiments/public-models.json');capstone=read('docs/course-experiments/capstone-public.json')
result['course_experiment_inventory']={'experiments':len(plan['sequence']),'public_models':len(exports['models']),'public_checkpoints':sum(Path(x['path']).suffix=='.pt' for m in exports['models'] for x in m['files']),'capstone_checkpoints':sum(Path(x['path']).suffix=='.pt' for m in capstone['models'] for x in m['files']),'mapped_entrypoints':{e['id']:{k:e[k] for k in ['module','function','assets']} for e in plan['sequence'] if e['id'] in ['real_text','sft','dpo','safety','real_modal','distillation','reasoning']}}
for e in plan['sequence']:
    p=Path('scripts/course_experiments')/(e['module']+'.py');t=ast.parse(p.read_text());assert e['function'] in {x.name for x in t.body if isinstance(x,ast.FunctionDef)}
result['course_experiment_inventory']['entrypoints_all_exist']=True
result['gpu_smoke']={k:read('docs/gpu-smoke-result.json')[k] for k in ['parameters','device','gpu','torch','checks','resume']}
# Shape/parameter verification only: allocate random model structures; no forward/generation/update.
import torch
from tiny_perceptron.selftrained.model import SelftrainedConfig,LimitedAssistant
parameter_counts={}
with torch.device('meta'):
    for arch in ['moe','dense']:
        cfg=read(next(ORIG.glob('*__'+arch+'-joint__model-config.json')))
        model=LimitedAssistant(SelftrainedConfig(**cfg));parameter_counts[arch]=sum(p.numel() for p in model.parameters())
result['parameter_shape_check']=parameter_counts;assert parameter_counts=={'moe':5447107,'dense':2288067}
result['legacy_capstone']={}
d=read('docs/course-experiments/results/capstone_deployment.json');s=read('docs/course-experiments/results/capstone_student.json')
for k in ['joint','dpo']:result['legacy_capstone'][k]={z:d['results']['stages'][k][z] for z in ['count','end_to_end_correct']}
for k in ['test_ce','test_kd']:result['legacy_capstone'][k]={z:s['results']['evaluations'][k][z] for z in ['count','end_to_end_correct']}
ui=read('docs/natural-assistant/evidence/v4-research/student-selected-public-ui/actual-ui/report.json')
result['natural_cpu_observation']={k:ui[k] for k in ['mode','interaction','torch_threads','torch_interop_threads','status','headless_browser_interaction_complete','server_cpu_resources','model_load_counts']}
score_path=Path('docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json');scores=read(score_path)
result['natural_validation_original']={'file':str(score_path),'sha256':sha(score_path),'original_counts':{k:{n:v[n] for n in ['correct_counts','primary_numerator','primary_denominator','incomplete_case_ids','all_generations_complete']} for k,v in scores['variants'].items()},'selection_method_inspected':'scripts/score_natural_v4_validation.py:352-476; counts, EOS and nonregression branching; no new grades or generations'}
(OUT/'verified-measurements.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['v2_archive','course_inventory','parameter_shape_check']},ensure_ascii=False,indent=2))
print('All original count/hash/provenance assertions passed; no training, generation or new answer grading executed.')
