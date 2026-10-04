import collections, datetime, hashlib, importlib.util, importlib.metadata, json, math, pathlib, platform, re, struct, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[3]
A=ROOT/'docs/technical-reviews/artifacts'
E=ROOT/'docs/natural-assistant/evidence'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
inputs={}
def read(p):
 p=ROOT/p; inputs[str(p.relative_to(ROOT))]=sha(p); return json.loads(p.read_text())
def distance(a,b):
 table=[[0]*(len(b)+1) for _ in range(len(a)+1)]
 for i in range(len(a)+1):table[i][0]=i
 for j in range(len(b)+1):table[0][j]=j
 for i,x in enumerate(a,1):
  for j,y in enumerate(b,1):table[i][j]=min(table[i-1][j]+1,table[i][j-1]+1,table[i-1][j-1]+(x!=y))
 return table[-1][-1]
def header(p,label):
 with p.open('rb') as f:
  size=struct.unpack('<Q',f.read(8))[0];raw=f.read(size);h=json.loads(raw)
 (A/('natural-final-fact-20.8-'+label+'-header.json')).write_text(json.dumps(h,indent=2)+'\n')
 vals=[v for k,v in h.items() if k!='__metadata__']; n=sum(math.prod(v['shape']) for v in vals)
 return {'parameters':n,'tensor_count':len(vals),'dtypes':dict(collections.Counter(v['dtype'] for v in vals)),'header_bytes':size,'file_bytes':p.stat().st_size,'header_sha256':hashlib.sha256(raw).hexdigest()}
start=time.monotonic()
m=read('docs/natural-assistant/public-release.json')
train=read('docs/natural-assistant/evidence/train/result.json')
assert train['status']=='completed' and train['completed_steps']==180 and train['trainable_parameters']==1605632
assert train['optimizer_only_lora'] and len(train['optimizer_parameter_names'])==112 and all('lora_' in n for n in train['optimizer_parameter_names'])
assert train['changed_adapter_tensor_count']==112 and train['frozen_parameter_samples_unchanged']
final=read('docs/natural-assistant/evidence/final/result.json')
assert final['status']=='completed' and final['device']=='cuda' and final['dtype']=='bfloat16'
assert final['model_revision']==m['base_model']['revision'] and final['asr_revision']==m['asr_model']['revision']
gen=read('docs/natural-assistant/evidence/final/generations-adapter.json')
sem=read('docs/natural-assistant/evidence/semantic-final/final-semantic-review.json')
records=[z for z in sem['all_records'] if z['variant']=='adapter']
key=lambda z:(z['id'],z['task'])
raw={key(z):z for z in gen};assert len(raw)==len(gen)==90
counts=collections.defaultdict(lambda:{'pass':0,'total':0})
row_checks=[]
for z in records:
 g=raw[key(z)];assert g==z['raw_generation_record_preserved'] and g['prediction']==z['raw_generated_suffix']
 assert len(g['generated_token_ids'])==g['generated_tokens']
 assert g['ended_with_eos']==bool(g['generated_token_ids'] and g['generated_token_ids'][-1] in g['eos_token_ids'])
 decision=z['semantic_pass'] and z['instruction_pass'] and z['required_content_pass'] and not z['unsupported_claims'] and not g['truncated'] and not g['completion_unknown']
 assert bool(decision)==z['completed_task_success']
 domain=z['domain'];counts[domain]['total']+=1;counts[domain]['pass']+=int(decision)
 row_checks.append({'id':g['id'],'task':g['task'],'domain':domain,'completed_success_recalculated':bool(decision),'whole_answer_reason':z['reason'],'raw_prediction_sha256':hashlib.sha256(g['prediction'].encode()).hexdigest(),'raw_record_matches_semantic_record':True})
for domain,expected in {'text_chat':(4,4),'text_dialogue':(1,2),'scene':(17,36),'ocr':(18,18),'text_presence':(6,6),'speech_typed_control':(4,12),'speech_chat':(1,12)}.items():
 assert (counts[domain]['pass'],counts[domain]['total'])==expected
scenes=[z for z in records if z['domain']=='scene'];descriptions=[z for z in scenes if z['id'].endswith('/scene')]
assert len({z['raw_generation_record_preserved']['image'] for z in scenes})==12
assert len(descriptions)==12 and sum(z['completed_task_success'] for z in descriptions)==0
for z in records:
 if z['domain'] in ('ocr','text_presence'):
  assert z['raw_generated_suffix']==z['gold_reference_unchanged']
trans=read('docs/natural-assistant/evidence/final/transcripts.json')
manifest=read('docs/natural-assistant/manifest.json');rows={z['id']:z for z in manifest['audio_rows']}
tsv=ROOT/'outputs/natural-extension/data/speech/official-sources/test.tsv';inputs[str(tsv.relative_to(ROOT))]=sha(tsv)
tsvrows=[line.split('\t') for line in tsv.read_text().splitlines()]
cer=[]
for z in trans:
 assert z['reference_transcript']==rows[z['id']]['raw_transcription']
 member=rows[z['id']]['source_member'].split('/')[-1]
 orig=next(r for r in tsvrows if r[1]==member)
 assert orig[2]==z['reference_transcript']
 errors=distance(z['reference_transcript'],z['transcript']);den=len(z['reference_transcript'])
 assert errors==z['raw_errors'] and den==z['raw_reference_characters']
 cer.append({'id':z['id'],'errors':errors,'reference_characters':den,'hypothesis_preserved':True})
assert len(cer)==12 and sum(z['errors'] for z in cer)==150 and sum(z['reference_characters'] for z in cer)==481
ext=read('docs/natural-assistant/evidence/external-ocr/generations-adapter.json')
extsem=read('docs/natural-assistant/evidence/semantic-external/external-semantic-review.json')
ex=read('docs/natural-assistant/external-ocr.json');exrows={z['id']:z for z in ex['rows']}
external=[]
for g in ext:
 assert g['reference_answer']==exrows[g['id']]['answer']
 z=next(z for z in extsem['all_20_records'] if z['variant']=='adapter' and z['id']==g['id']);assert z['raw_generation_record_preserved']==g
 exact=g['prediction']==g['reference_answer'];success=exact and g['ended_with_eos'] and not g['truncated']
 assert success==z['completed_task_success']
 external.append({'id':g['id'],'raw_errors':distance(g['reference_answer'],g['prediction']),'raw_reference_characters':len(g['reference_answer']),'success':success,'reference_unchanged':True,'image_grounded_review':z['image_grounded_review']})
assert len(ext)==10 and sum(z['success'] for z in external)==1
events_path=ROOT/'docs/natural-assistant/evidence/student-trial/events.jsonl';inputs[str(events_path.relative_to(ROOT))]=sha(events_path)
ev=[json.loads(x) for x in events_path.read_text().splitlines()];by={z['index']:z for z in ev}
b=read('docs/natural-assistant/evidence/student-trial/browser-report.json')
s=read('docs/natural-assistant/evidence/student-trial/student-trial-summary.json')
chats=[c for c in b['cases'] if c['endpoint']=='/api/chat'];asrs=[c for c in b['cases'] if c['endpoint']=='/api/transcribe']
assert len(chats)==8 and len(asrs)==2
history=[];trace=[]
for c in b['cases']:
 assert c['http_status']==200
 if c['endpoint']=='/api/reset':history=[]
 if c['endpoint']=='/api/chat':
  z=by[c['core_event_index']];g=z['result'];row=z['row'];assert row['history']==history and row['user']==c['request']['prompt'].strip()
  assert g['prediction']==c['response']['prediction'] and row.get('answer') is None and z['original_function_called_once'] and z['returned_result_unchanged']
  content=[]
  if row['image']:content.append({'type':'image','image':row['image']})
  content.append({'type':'text','text':row['user']})
  history.extend([{'role':'user','content':content},{'role':'assistant','content':[{'type':'text','text':g['prediction']}]}])
  trace.append({'case':c['name'],'core_event_index':z['index'],'prior_history_turns':len(row['history']),'image':row['image'],'generation_seconds':g['generation_seconds'],'prediction':g['prediction'],'ended_with_eos':g['ended_with_eos'],'history_exact':True})
for c in asrs:
 z=by[c['asr_event_index']];assert c['response']['asr']==z['result'] and not z['gold_transcript_supplied']
assert all(z['ended_with_eos'] for z in trace)
timings=[z['generation_seconds'] for z in trace]
asrt=[z['result'] for z in ev if z['event']=='actual_asr_transcription']
load=next(z for z in ev if z['event']=='actual_core_loaded');asrload=next(z for z in ev if z['event']=='actual_asr_loaded');closed=ev[-1]
assert s['wall_seconds']==b['elapsed_seconds']==289.19217826100066
assert closed['peak_rss_kib']==12959448 and closed['temporary_uploads_deleted']
assert len([z for z in ev if z['event']=='actual_image_control_generation'])==2
cache=ROOT/'outputs/natural-extension/student-base-cache/hf';snapshots=ev[2]['snapshots'];cached=[];headers={}
for k,v in snapshots.items():
 d=cache/('models--'+v['repo'].replace('/','--'))/'snapshots'/v['revision']
 actual={p.relative_to(d).as_posix() for p in d.rglob('*') if p.is_file()};assert actual=={f['path'] for f in v['files']}
 for f in v['files']:
  p=d/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];cached.append({'model':k,**f,'independently_hashed':True})
 headers[k]=header(d/'model.safetensors',k)
 for name in ['README.md','config.json']:
  (A/('natural-final-fact-20.8-'+k+'-'+name)).write_bytes((d/name).read_bytes())
assert len(cached)==23 and sum(f['bytes'] for f in cached)==5238035532
headers['adapter']=header(ROOT/'outputs/natural-extension/fact-20.8-public/adapter_model.safetensors','adapter')
assert headers['base_model']['parameters']==2127532032 and headers['asr_model']['parameters']==241734912 and headers['adapter']['parameters']==1605632
assert headers['adapter']['tensor_count']==112
for p,digest in m['code_files'].items():assert sha(ROOT/p)==digest;inputs[p]=sha(ROOT/p)
assert sha(ROOT/'requirements-natural.txt')==m['requirements_sha256'];inputs['requirements-natural.txt']=sha(ROOT/'requirements-natural.txt')
spec=importlib.util.spec_from_file_location('release',ROOT/'scripts/fetch_natural_release.py');release=importlib.util.module_from_spec(spec);spec.loader.exec_module(release)
release.validate_manifest(m);release.verify_release(m,ROOT/'outputs/natural-extension/fact-20.8-public',receipt=False);release.check_runtime(m)
packages={p:importlib.metadata.version(p) for p in m['dependency_versions']}
commands=[]
def run(argv,name):
 x=subprocess.run(argv,cwd=ROOT,text=True,capture_output=True)
 (A/('natural-final-fact-20.8-'+name+'.stdout.txt')).write_text(x.stdout);(A/('natural-final-fact-20.8-'+name+'.stderr.txt')).write_text(x.stderr)
 assert x.returncode==0
 commands.append({'argv':argv,'exit_code':x.returncode,'stdout':x.stdout,'stderr':x.stderr})
run([sys.executable,'scripts/fetch_natural_release.py','--list'],'list')
run([sys.executable,'scripts/fetch_natural_release.py','--output','outputs/natural-extension/student-release-smoke/student-public-8dab26-20261004/adapter','--verify'],'verify')
source=(A/'natural-final-fact-20.8-read2-current.md').read_text();code=re.search(r'```python\n(.*?)```',source,re.S).group(1)
for name,txt in [('original',code),('exercise',code.replace('received = b"base revision A, adapter revision C"','received = original'))]:
 p=A/('natural-final-fact-20.8-'+name+'.py');p.write_text(txt);run([sys.executable,str(p.relative_to(ROOT))],name)
probe=read('docs/natural-assistant/evidence/student-trial/official-cli-smoke.json')
runtime_record=read('docs/natural-assistant/evidence/student-trial/official-manifest-verification.json')
assert runtime_record['python'].startswith('3.12.14 ') and runtime_record['runtime_check']=='passed'
assert runtime_record['manifest_sha256']==inputs['docs/natural-assistant/public-release.json']
q=ROOT/'outputs/natural-extension/student-release-smoke/official-cli-8dab26-20261004'
assert (q/'official-page.html').stat().st_size==probe['get']['body_bytes'] and sha(q/'official-page.html')==probe['get']['body_sha256']
assert json.loads((q/'official-raw-chat-response.json').read_text())==probe['chat']['raw_response']
assert probe['chat']['http_status']==200 and probe['chat']['request']['prompt']=='你好' and probe['shutdown']['exit_code']==0 and probe['shutdown']['owned_temporary_upload_directories_deleted']
scriptdir=ROOT/'outputs/natural-extension/student-release-smoke'
scripts={}
for name,digest in b['scripts'].items():
 assert sha(scriptdir/name)==digest;scripts[name]=digest
scripts['official_cli_probe.py']=sha(scriptdir/'official_cli_probe.py')
pngs={p.name:{'sha256':sha(p),'bytes':p.stat().st_size,'actually_viewed_by_reviewer':True} for p in (E/'student-trial').glob('*.png')};assert len(pngs)==6
result={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_scope':'Fresh CPU stdlib calculations, cached SHA/header inspection, package metadata validation, --list and --verify only; no model forward/training/GPU/installation/Git mutation.', 'environment':{'python':platform.python_version(),'platform':platform.platform(),'package_versions':packages},'input_sha256':inputs,'matrix_recount_from_per_item_records':dict(counts),'scene_denominators':{'questions':36,'unique_images':12,'descriptions':12,'description_success':0},'all90_adapter_item_checks':row_checks,'raw_ASR_independent_full_matrix_CER':{'cases':cer,'errors':150,'reference_characters':481,'cer':150/481},'external_original_GT_independent_checks':external,'student_HTTP_to_original_core_trace':trace,'student_counts':{'UI_chats':8,'ASR_calls':2,'separate_image_controls':2},'CPU_timing':{'browser_total_wall':b['elapsed_seconds'],'core_load':load['elapsed_seconds'],'generation_min':min(timings),'generation_max':max(timings),'ASR_generate_decode':[z['seconds'] for z in asrt],'ASR_audio_seconds':[z['audio_seconds'] for z in asrt],'Whisper_cached_load':asrload['elapsed_seconds'],'RSS_KiB':closed['peak_rss_kib'],'RSS_bytes':closed['peak_rss_kib']*1024,'RSS_GiB':closed['peak_rss_kib']/1024**2,'wall_scope':'browser subprocess workflow includes public adapter download/hash, cached core load, 2 image controls, Chromium操作/8 chats/2 ASR/reset/shutdown; not fresh base/ASR download','core_generation_scope':'time.monotonic starts after encode/device transfer and stops after model.generate synchronization; decoder text conversion excluded','ASR_scope':'starts after read/downmix/resample/processor; includes generate, stopping metadata and batch_decode; excludes file/preprocess/load/hash'},'23_cached_public_files':cached,'cached_total_bytes':sum(f['bytes'] for f in cached),'serialized_header_counts':headers,'weights_only_float32_bytes':headers['base_model']['parameters']*4,'fresh_anonymous_adapter_total_bytes':sum(f['bytes'] for f in m['files']),'upload_limit':{'bytes':8*1024*1024,'MiB':8,'decimal_MB':8*1024*1024/10**6},'commands_executed':commands,'official_CLI_supplement_checked':{'record_sha256':inputs['docs/natural-assistant/evidence/student-trial/official-cli-smoke.json'],'HTTP_page_hash_raw_response_match':True,'cached_CPU_only':True,'ASR_not_invoked':True,'actual_hello_response':probe['chat']['raw_response']['prediction'],'cleanup_checked':True,'original_8_UI_workflow_rerun':False},'trial_scripts_sha256':scripts,'screenshots_actually_viewed':pngs,'elapsed_seconds':time.monotonic()-start}
result['original_training_record_check']={'steps':180,'optimizer_only_lora':True,'trainable_parameters':1605632,'changed_LoRA_tensors':112,'frozen_samples_checked':len(train['frozen_parameter_samples_initial']),'full_frozen_base_comparison_claim':False,'ASR_training':False}
result['official_manifest_runtime_record_check']={'python':runtime_record['python'],'recorded_versions_match_present':runtime_record['installed_versions']==packages,'same_public_manifest':True}
(A/'natural-final-fact-20.8-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['matrix_recount_from_per_item_records','scene_denominators','cached_total_bytes','serialized_header_counts','CPU_timing','elapsed_seconds']},ensure_ascii=False))
