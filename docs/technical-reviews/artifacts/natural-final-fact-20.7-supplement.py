from pathlib import Path
from urllib.request import Request,urlopen
import hashlib,json,subprocess,platform,importlib.metadata,sys
from tiny_perceptron import natural_assistant as a
P='docs/technical-reviews/artifacts/natural-final-fact-20.7-'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
url='https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/whisper/modeling_whisper.py'
with urlopen(Request(url,headers={'User-Agent':'technical-source-review/1'}),timeout=25) as r: raw=r.read();status=r.status
Path(P+'transformers-whisper.py').write_bytes(raw)
manifest=json.loads(Path('docs/natural-assistant/manifest.json').read_text());transcripts=json.loads(Path('docs/natural-assistant/evidence/final/transcripts.json').read_text()); rows={r['id']:r for r in manifest['audio_rows']}
contexts=[]
for t in transcripts:
 row=rows[t['id']]; typed=a.messages_for(dict(row,task='typed_chat'),Path('data/natural')); speech=a.messages_for(dict(row,user=t['transcript'],task='speech_chat'),Path('data/natural'))
 assert typed[:-1]==speech[:-1]==[{'role':'system','content':[{'type':'text','text':'請用繁體中文簡短回應使用者所說的內容，不要只重複原話。'}]}]
 assert typed[-1]['content']==[{'type':'text','text':t['reference_transcript']}];assert speech[-1]['content']==[{'type':'text','text':t['transcript']}]
 contexts.append({'id':row['id'],'typed_messages_current_function':typed,'speech_messages_current_function':speech,'same_prior_messages':True,'images':0,'prior_user_turns':0,'scope':'Current messages_for executed against frozen source-hash-identical manifest; this is not an original inference-time serialized prompt log.'})
source_receipts=[]
for revision,file in [('638dc13c1598fb1cbfc097598081b0c27c35de55','tiny_perceptron/natural_assistant.py'),('638dc13c1598fb1cbfc097598081b0c27c35de55','scripts/natural_assistant.py'),('638dc13c1598fb1cbfc097598081b0c27c35de55','docs/natural-assistant/manifest.json'),('96af0243c5cd461cea867d9907e72f1b0d1587fb','docs/natural-assistant/rubric.json'),('96af0243c5cd461cea867d9907e72f1b0d1587fb','tiny_perceptron/natural_assistant.py'),('96af0243c5cd461cea867d9907e72f1b0d1587fb','scripts/natural_assistant.py'),('213496fe63f338a16026c2eb131e3e39faa24721','tiny_perceptron/natural_assistant.py'),('213496fe63f338a16026c2eb131e3e39faa24721','scripts/natural_assistant.py')]:
 cmd=['git','show',revision+':'+file];r=subprocess.run(cmd,capture_output=True);assert r.returncode==0
 name=P+'version-'+revision[:8]+'-'+file.replace('/','__');Path(name).write_bytes(r.stdout)
 source_receipts.append({'command':' '.join(cmd),'exit_code':r.returncode,'path':name,'sha256':sha(name),'current_path':file,'current_sha256':sha(file),'matches_current':sha(name)==sha(file)})
cmd=['git','log','-4','--format=%H %cI %s','--','docs/natural-assistant/rubric.json'];r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0
chronology={'command':' '.join(cmd),'exit_code':r.returncode,'stdout':r.stdout,'rubric_commit':'96af0243c5cd461cea867d9907e72f1b0d1587fb','rubric_commit_timestamp':'2026-10-04T08:36:09Z','rubric_sha256':sha('docs/natural-assistant/rubric.json'),'pre_output_review_timestamp':json.loads(Path('docs/natural-assistant/evidence/semantic-validation/pre-output-review.json').read_text())['created_utc'],'final_started_timestamp':json.loads(Path('docs/natural-assistant/evidence/final/result.json').read_text())['observed_at'],'manifest_system_context_preexisted':True,'scope':'Repository documentary chronology; no claim of independent wall-clock witnessing original model execution.'}
# Parameter sum from original primary safetensors header, rather than rounded model-card label.
hdr=json.loads(Path(P+'whisper-header.json').read_text());total=0
for k,v in hdr.items():
 if k=='__metadata__':continue
 n=1
 for d in v['shape']:n*=d
 total+=n
assert total==241734912
config=json.loads(Path(P+'whisper-generation-config.json').read_text());assert config['lang_to_id']['<|zh|>']==50260 and config['task_to_id']['transcribe']==50359 and config['eos_token_id']==50257
valraw=json.loads(Path('docs/natural-assistant/evidence/validation/generations.json').read_text()); valtrs=json.loads(Path('docs/natural-assistant/evidence/validation/transcripts.json').read_text());valrecords=[]
# Independent manual outcomes: only adapter ASR Amazon limitation response passes; every other row is replay or invented/contradicted content.
for x in valraw:
 if x['task'] not in ['typed_chat','speech_chat']:continue
 assert x['generated_token_ids'][-1] in x['eos_token_ids'] and x['ended_with_eos'] and not x['truncated'] and not x['completion_unknown']
 tr=next(t for t in valtrs if t['id']==x['id']);ref=rows[x['id']]['user'];assert x['user']==(ref if x['task']=='typed_chat' else tr['transcript'])
 success=x['variant']=='adapter' and x['task']=='speech_chat' and x['id'].endswith('10291819997967779677')
 valrecords.append({'id':x['id'],'variant':x['variant'],'task':x['task'],'reference':ref,'actual_user':x['user'],'prediction':x['prediction'],'ended_with_eos':True,'independent_task_success':success})
vals={v:{t:{'success':sum(x['independent_task_success'] for x in valrecords if x['variant']==v and x['task']==t),'denominator':sum(1 for x in valrecords if x['variant']==v and x['task']==t)} for t in ['typed_chat','speech_chat']} for v in ['base','adapter']}
assert vals=={'base':{'typed_chat':{'success':0,'denominator':6},'speech_chat':{'success':0,'denominator':6}},'adapter':{'typed_chat':{'success':0,'denominator':6},'speech_chat':{'success':1,'denominator':6}}}
exe=json.loads(Path('docs/natural-assistant/evidence/validation/execution.json').read_text());assert exe['runner_arguments'][exe['runner_arguments'].index('--max-new-tokens')+1]=='384'
# Before comparison file preserves own review; after reading all 24 relevant semantic dimensions compare exact pass choices, keeping caveats.
ind=json.loads(Path(P+'independent-semantic-before-comparison.json').read_text());sem=json.loads(Path('docs/natural-assistant/evidence/semantic-final/final-semantic-review.json').read_text());comp=[]
for x in ind['rows']:
 for task,own in [('typed_chat',x['typed_task_success']),('speech_chat',x['speech_task_success'])]:
  record=next(r for r in sem['all_records'] if r['id']==x['id'] and r['variant']=='adapter' and r['task']==task)
  assert record['completed_task_success']==own
  comp.append({'id':x['id'],'task':task,'independent_success':own,'reported_success':record['completed_task_success'],'agreement':True,'reported_reason':record['reason'],'reported_unsupported':record['unsupported_claims'],'reported_official_faithfulness':record['faithfulness_to_official_source'],'reported_received_faithfulness':record['faithfulness_to_actual_received_input'],'reported_beyond_replay':record['responds_beyond_replay'],'reported_traditional_script':record['Traditional_Chinese_instruction_pass'],'reported_instruction':record['instruction_pass']})
res={'command':'.venv/bin/python docs/technical-reviews/artifacts/natural-final-fact-20.7-supplement.py','environment':{'python':platform.python_version(),'torch':importlib.metadata.version('torch'),'soundfile':importlib.metadata.version('soundfile'),'device':'CPU API/metadata/text checks; no models loaded'},'official_source_receipt':{'url':url,'status':status,'path':P+'transformers-whisper.py','sha256':sha(P+'transformers-whisper.py')},'parameter_total':total,'generation_config_verified':{'zh':50260,'transcribe':50359,'EOS':50257},'contexts':contexts,'source_versions':source_receipts,'pre_output_rubric_chronology':chronology,'validation_records':valrecords,'validation_totals':vals,'final_success_comparison':comp,'notes':['Exact full-success choices agree 24/24; source faithfulness 10/12 typed and 4/12 ASR is distinct from task completion. Pure faithful summaries fail the frozen response requirement.','Borderline blog/organization summaries remain unaccepted under strict replay rule; metal/telescope acknowledgement and fierce-animal characterization accepted; no task success claim for merely shortened paraphrase.','ASR is operationally frozen: load_asr eval plus inference_mode, and training uses only LoRA core tensors; load_asr does not explicitly set ASR requires_grad=False. No ASR optimization/checkpoint update path exists in the course runner.','Normal EOS is not audio correctness, semantic success, or evidence a person listened.','Card small=244M is a coarse family label, not the exact count of this fixed checkpoint header.','The rubric phrase neither changes punctuation is understood as no punctuation deletion: NFKC does fold compatibility comma/semicolon/brackets, as chapter20.7 correctly discloses.'], 'audio_played':False,'models_loaded':False}
Path(P+'supplement.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'parameter_total':total,'contexts':len(contexts),'version_receipts':len(source_receipts),'rubric_chronology':chronology,'validation_totals':vals,'final_success_agreement':len(comp),'notes':res['notes']},ensure_ascii=False,indent=2))
