from pathlib import Path
import collections, hashlib,json,re, shutil, sys
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=[]
def read(p,keys):
 p=ROOT/p; d=json.loads(p.read_text());audit.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'topkeys':{k:type(v).__name__ for k,v in d.items()},'pointers':['/'+k for k in keys]});return {k:d[k] for k in keys if k in d}
def write(n,d):(OUT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
official=OUT/'official';tree=json.loads((official/'hf-v2-tree.json').read_text()); files=[x for x in tree if x['type']=='file']
assert len(files)==16 and sum(x['size'] for x in files)==81501640
release={'file_count':len(files),'total_bytes':sum(x['size'] for x in files),'files':files,'exports':{}}
stages=json.loads((OUT/'training-raw-derived.json').read_text())
for public,stage in [('moe-pretrain','moe-pretrain'),('moe-sft','moe-sft'),('moe-joint','moe-native'),('dense-joint','dense-weighted')]:
 p=official/(public+'-inference-manifest.json');d=json.loads(p.read_text());audit.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'topkeys':{k:type(v).__name__ for k,v in d.items()},'pointers':['/selected_step','/selected_checkpoint_sha256','/files','/origin/kind','/tool_loss_weight','/numeric_run_loss_weight','/native_voice_loss_weight']})
 assert sha(p)==stages[stage]['inference_manifest_sha256']
 assert d['selected_checkpoint_sha256']==stages[stage]['best_sha256']
 release['exports'][public]={k:d[k] for k in ['selected_step','selected_checkpoint_sha256','files','stage','tool_loss_weight','numeric_run_loss_weight','native_voice_loss_weight'] if k in d}
write('hf-release-derived.json',release)

model_index=read(Path('docs/course-experiments/public-models.json'),['models'])['models']
old=(official/'hf-current-root-README.md').read_text();new=(ROOT/'docs/selftrained/model-cards/repository-README.md').read_text()
index=[]
for model in model_index:
 ident=model['id'];rev=model['revision'];prefix='course/course-v1/'+ident
 assert prefix in old and prefix in new and rev in old and rev in new
 index.append({'id':ident,'revision':rev,'files':len(model['files']),'unchanged_fixed_index':True})
cap=read(Path('docs/course-experiments/capstone-public.json'),['revision','models'])
legacy={'original_30_fixed_index':index,'total_model_archive_files':sum(len(m['files']) for m in model_index),'capstone_revision':cap['revision'],'capstone_models':[x['id'] for x in cap['models']],'tests':{}}
legacy['total_declared_files']=legacy.pop('total_model_archive_files')
legacy['total_model_archive_files']=sum(Path(x['path']).suffix=='.pt' for m in model_index for x in m['files'])
for label,p in [('joint','docs/course-experiments/capstone-evidence/deployment/test-joint.json'),('dpo','docs/course-experiments/capstone-evidence/deployment/test-dpo.json'),('ce','docs/course-experiments/capstone-evidence/student/test-ce.json'),('kd','docs/course-experiments/capstone-evidence/student/test-kd.json')]:
 d=read(Path(p),['count','action_correct','end_to_end_correct','by_task','records','protocol']);rows=d['records']; assert len(rows)==d['count']
 legacy['tests'][label]={k:v for k,v in d.items() if k!='records'}
 legacy['tests'][label]['record_keys']={k:type(v).__name__ for k,v in rows[0].items()}
 assert sum(x['end_to_end_correct'] for x in rows)==d['end_to_end_correct']
write('legacy-derived.json',legacy)

v4={}
for stage,base in [('validation','docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review'),('test','docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review')]:
 b=Path(base);result=read(b/('raw/result.json' if stage=='validation' else 'result.json'),['model','model_revision','asr_model','asr_revision','versions','total_parameters','split','status','variants','asr','execution','requested_visual_text_rows','requested_audio_rows','requested_audio_chat_rows'])
 scores=read(b/'scored/scores.json',['denominators','correct_counts','case_decisions','variants','selected_variant','integer_weights','all_generations_complete'])
 if stage=='test':
  counts=collections.Counter(x['group'] for x in scores['case_decisions']);passes=collections.Counter(x['group'] for x in scores['case_decisions'] if x['passed']);assert dict(passes)==scores['correct_counts']
  result['case_count_recomputed']=dict(counts);result['passed_recomputed']=dict(passes)
 result['scores']=scores;v4[stage]=result
write('v4-raw-derived.json',v4)

engineering={}
eng=Path('docs/selftrained/infrastructure/v2-local-stage/independent-review')
dirs=[eng/'final-runs'/x for x in ['pretrain','sft','latest-exact-resume']]+[eng/'author-final/actual-cpu-subprocesses'/x for x in ['baseline-joint','weighted44-joint','native414-joint','invalid-zero-step','weighted44-interrupted','weighted44-exact-resume']]
for b in dirs:
 e=read(b/'execution.json',['command','job','status','returncode','interrupted','received_signals','parent','started_at','finished_at','revision','manifest_sha256','code_sha256'])
 if (ROOT/b/'train-receipt.json').exists():
  t=read(b/'train-receipt.json',['stage','steps','tokens','target_tokens','completed_requested_steps','interrupted','config','origin','stage_history','sampling_mode','sampler_policy']);e['training']=t
 r=read(b/'receipt.json',['files','status','returncode'])
 e['recorded_files']=r['files'];engineering[str(b.relative_to(eng))]=e
write('engineering-raw-derived.json',engineering)

data_root=ROOT/'outputs/selftrained-v2/data';extra={'voice_original_recordings':{},'ocr_fonts':{},'examples':{}}
for split in ['train','validation','test']:
 voice=[json.loads(x) for x in (data_root/('voice-'+split+'.jsonl')).read_text().splitlines()]
 extra['voice_original_recordings'][split]=len({r['audio'] for r in voice if 'augmentation' not in r})
 ocr=[json.loads(x) for x in (data_root/('ocr-'+split+'.jsonl')).read_text().splitlines()]
 extra['ocr_fonts'][split]=sorted({r['supervision']['font_family'] for r in ocr})
for p in sorted((ROOT/'docs/selftrained/examples/v2').glob('*.json')):
 d=json.loads(p.read_text());extra['examples'][p.name]={'sha256':sha(p),'content':d}
write('data-extra-derived.json',extra)

gpu=ROOT/'outputs/selftrained-v2/operations/moe-public-final-freeze-attempt-1/monitor/artifact/selftrained-run-35eb7b90d1f3966fddddf37389b02e370c0e2257-37446402227-1/gha-37446402227-1/raw/outputs.jsonl'
lines=gpu.read_bytes().splitlines();gpu_rows=[json.loads(x) for x in lines]
cpu=json.loads((OUT/'cpu-raw-derived.json').read_text())['invocations']
saved=[];matches={};known_keys={}
for i,row in enumerate(gpu_rows):
 if i==0:known_keys={k:type(v).__name__ for k,v in row.items()}
 # Only measurement structures are inspected. Original schema discovered before fields below.
 for name,c in cpu.items():
  if c['chat'] and row.get('task')==c['argv'][c['argv'].index('--task')+1]:
   if row.get('generated')==c['answer'] or row.get('final_output')==c['answer'] or row.get('output')==c['answer']:
    saved.append({'line_number':i+1,'row':row});matches.setdefault(name,[]).append(i+1)
write('gpu-original-journal-locator.json',{'source':str(gpu.relative_to(ROOT)),'source_sha256':sha(gpu),'lines':len(lines),'first_topkeys':known_keys,'preliminary_matching_lines':matches})
write('extra-input-audit.json',audit)
print(json.dumps({'release':(release['file_count'],release['total_bytes']),'legacy_files':legacy['total_model_archive_files'],'voice_original':extra['voice_original_recordings'],'ocr_fonts':extra['ocr_fonts'],'gpu_topkeys':known_keys,'gpu_lines':len(lines),'gpu_preliminary_matches':matches},ensure_ascii=False))
