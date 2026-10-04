from pathlib import Path
import sys, json, hashlib, platform, math, struct, re
sys.path.insert(0,str(Path.cwd()))
import torch
from scripts.course_experiments.posttraining import build_records
from tiny_perceptron.capstone import CapstoneModel, default_config, load_capstone, generate_trace, build_dataset, parse_action, TOK, calculator_runtime, expected_final
A=Path('docs/technical-reviews/artifacts')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'},'scope':'Independent record recount, architecture/checkpoint inspection and one small-world CPU generation; no new training or GPU run'}
student=read('docs/course-experiments/results/capstone_student.json');r=student['results'];report['student_original_run']={k:student[k] for k in ['revision','device','gpu','seed','torch_version','python_version']};report['student_training']={name:{k:b[k] for k in ['steps','requested_steps','effective_tokens','objective','initialization']} for name,b in r['branches'].items()}
report['student_counts']={}
splits, original_manifest=build_dataset(42)
assert original_manifest['sha256']==r['data_manifest']['sha256']
lookup={row['id']:row for row in splits['test']}
for name,key in [('test-ce','test_ce'),('test-kd','test_kd'),('test-kd-ptq4','test_kd_ptq4')]:
 p=Path('docs/course-experiments/capstone-evidence/student',name+'.json');d=read(p);print(name,'keys',list(d))
 rows=d.get('records',d.get('samples',d.get('rows',d.get('generated_samples',[]))))
 if not rows:raise AssertionError('Missing per-example evidence')
 print(name,'row keys',list(rows[0]))
 key_correct='end_to_end_correct' if 'end_to_end_correct' in rows[0] else 'correct'
 corrected=[]
 for row in rows:
  source_row=lookup[row['id']]
  assert row['expected_action']==source_row['answer'] and row['expected_final']==expected_final(source_row)
  trace=row['action_trace']; ids=trace['generated_ids']
  assert trace['eos']==(bool(ids) and ids[-1]==TOK.eos_id)
  assert TOK.decode(ids[:-1] if trace['eos'] else ids)==trace['raw']
  action_ok=trace['eos'] and trace['raw']==row['expected_action']
  assert action_ok==row['action_correct']
  action=parse_action(trace);answer=None
  if action['status'] in ('direct','ask'): answer=action['content']
  if action['status']=='tool':assert row['runtime']==calculator_runtime(action, available=source_row['available'])
  if action['status']=='tool' and row['final_trace'] is not None:
   final=row['final_trace']; final_ids=final['generated_ids']
   assert final['eos']==(bool(final_ids) and final_ids[-1]==TOK.eos_id)
   assert TOK.decode(final_ids[:-1] if final['eos'] else final_ids)==final['raw']
   parsed_final=parse_action(final)
   if parsed_final['status']=='direct':answer=parsed_final['content']
  assert answer==row['answer']
  correct_row=bool(action_ok and answer==row['expected_final'])
  assert correct_row==row['end_to_end_correct']
  corrected.append(correct_row)
 correct=sum(corrected)
 assert len(rows)==90 and correct==r['evaluations'][key]['end_to_end_correct']==d['end_to_end_correct']
 report['student_counts'][name]={'count':len(rows),'recounted_end_to_end_correct':correct,'protocol':d['protocol'],'raw_sha256':sha(p),'tasks':d['by_task']}
assert report['student_counts']['test-ce']['recounted_end_to_end_correct']==62
assert report['student_counts']['test-kd']['recounted_end_to_end_correct']==61
assert report['student_counts']['test-kd-ptq4']['recounted_end_to_end_correct']==61
report['student_denominators']={'seed':student['seed'],'train_rows':r['data_manifest']['counts']['train'],'validation_rows':r['data_manifest']['counts']['validation'],'test_rows':90,'updates_per_branch':350,'effective_answer_targets_per_branch':145163,'family_split_policy':r['data_manifest']['split_policy']}
report['manifests']={}
for name in ['capstone-public','public-models']:
 d=read('docs/course-experiments/'+name+'.json');models=d['models'];pts=[(m,f) for m in models for f in m['files'] if f['path'].endswith('.pt')]
 repos=sorted({m.get('repo',d.get('repo')) for m in models})
 report['manifests'][name]={'groups':len(models),'checkpoint_files':len(pts),'repo_ids':repos,'ids':[m['id'] for m in models]}
assert report['manifests']['capstone-public']['checkpoint_files']==11
assert report['manifests']['public-models']['groups']==30 and report['manifests']['public-models']['checkpoint_files']==120
assert report['manifests']['capstone-public']['repo_ids']==report['manifests']['public-models']['repo_ids']
model=CapstoneModel(default_config());report['toy_architecture']=model.description();assert report['toy_architecture']['parameters']==328128 and model.config.experts==4 and model.config.top_k==2
path=Path('outputs/integration-runs/v2-joint/capstone-review/capstone_joint/gha-37168518451-1/model.pt')
model,payload=load_capstone(path);torch.set_num_threads(2)
report['loaded_checkpoint']={'path':str(path),'sha256':sha(path),'keys':sorted(payload),'inference_only':payload['inference_only'],'stage':payload['stage']}
assert payload['inference_only'] and all(k not in payload for k in ['optimizer','training_state','python_rng','torch_rng','cuda_rng'])
splits, data_manifest=build_dataset(42)
row=next(row for row in splits['test'] if row['task']=='missing')
try:
 output=generate_trace(model,row);report['one_cpu_generation']=output
except TypeError:
 import inspect
 report['generation_signature']=str(inspect.signature(generate_trace))
# Count pinned pretrained weight shapes without loading or inferring with the 2B model.
paths=list(Path('outputs/natural-extension/student-base-cache/hf/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203').glob('model.safetensors'))
assert len(paths)==1
with paths[0].open('rb') as f:
 header_length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(header_length))
(A/'natural-r4-qwen-safetensors-header.json').write_text(json.dumps(header,indent=2)+'\n')
count=sum(math.prod(v['shape']) for k,v in header.items() if k!='__metadata__')
report['qwen_header']={'path':str(paths[0]),'header_length':header_length,'tensor_count':len(header)-('__metadata__' in header),'parameters_from_shapes':count,'rounded_billion':round(count/1e9,3),'dtype_counts':{t:sum(1 for k,v in header.items() if k!='__metadata__' and v['dtype']==t) for t in sorted({v['dtype'] for k,v in header.items() if k!='__metadata__'})}}
assert count==2127532032
routes=read('docs/natural-assistant/evidence/usage/cpu-base-input-routes.json');assert routes['parameters']['total_parameters']==count
report['natural_record_scope']=routes['scope'];report['natural_parameters']=routes['parameters'];report['natural_audio_input_route']={'asr_transcript':routes['audio']['transcript'],'speech_user':routes['speech_response']['user'],'same_text':routes['speech_response']['user']==routes['audio']['transcript'],'output_as_text':isinstance(routes['speech_response']['prediction'],str)}
assert report['natural_audio_input_route']['same_text'] and report['natural_audio_input_route']['output_as_text']
finite=build_records();assert finite and all(len(row['candidates'])==4 for row in finite)
report['finite_policy_candidate_count']={'records':len(finite),'candidates_per_record':4,'first_candidates':finite[0]['candidates']}
post=read('docs/course-experiments/results/posttraining.json');report['finite_policy_scope']={k:post['results'][k] for k in ['scope','candidate_source']}
report['licenses']={'root_license_first_line':Path('LICENSE').read_text().splitlines()[0],'natural_model_card_licenses':{n:re.search(r'^license: (.+)$',Path('docs/technical-reviews/artifacts/natural-r4-sources',n).read_text(),re.M).group(1) for n in ['qwen-card.md','whisper-card.md']},'dataset_document_has_distinct_licenses':all(k in Path('assets/training/README.md').read_text() for k in ['CC-BY-NC-4.0','CDLA-Sharing-1.0','CC-BY-SA-4.0','OFL','CC BY 4.0'])}
deployment=read('docs/course-experiments/results/capstone_deployment.json')['results']
report['teacher_failures']={'teacher_stage':'dpo','count':deployment['stages']['dpo']['count'],'end_to_end_correct':deployment['stages']['dpo']['end_to_end_correct'],'by_task':deployment['stages']['dpo']['by_task']}
assert report['teacher_failures']['end_to_end_correct']<report['teacher_failures']['count']
report['result']='passed'
(A/'natural-r4-audit-records.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Independent audit passed:',json.dumps({k:report[k] for k in ['environment','qwen_header','manifests']},ensure_ascii=False))
