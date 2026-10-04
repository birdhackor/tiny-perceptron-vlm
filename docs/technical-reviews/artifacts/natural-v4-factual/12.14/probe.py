"""Independent bounded probes: manual example, HTTP wiring, fixed-record audit.
No pretrained inference or benchmark replication is performed.
"""
from pathlib import Path
import sys,json,hashlib,re,copy,base64,io,tempfile,threading,http.client
from types import SimpleNamespace
from unittest.mock import patch
import torch,numpy as np,soundfile as sf
from tiny_perceptron import natural_concepts as nc, natural_assistant as na, natural_ui as ui
ROOT=Path(__file__).resolve().parents[5]
ART=Path(__file__).resolve().parent

def read(path): return json.loads((ROOT/path).read_text())
def sha(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def emit(label,obj): print(label,json.dumps(obj,ensure_ascii=False,sort_keys=True))

env={'python':sys.version.split()[0],'torch':str(torch.__version__),'numpy':str(np.__version__),'soundfile':str(sf.__version__),'device':'cpu','cuda_available':str(torch.cuda.is_available())}
emit('environment',env)
section=(ART/'section-12.14.md').read_text()
code=re.findall(r'```python\n(.*?)\n```',section,re.S)[0]
print('literal_section_example_begin')
exec(compile(code,'course/chapters/12.md#12.14','exec'))
print('literal_section_example_end')
report=nc.speech_stages('我不吃辣','我不吃拉','選清淡的湯麵。','選清淡的湯麵。')
assert report['transcription']['edits']==1 and report['transcription']['reference_characters']==4
assert report['transcription']['cer']==0.25 and report['answers_identical'] is True and report['answers_correct'] is None
# Independently derive lower bound: unequal strings need >=1 edit; replacing 拉 by 辣 achieves 1.
assert '我不吃拉'.replace('拉','辣')=='我不吃辣'
emit('manual_report',report)
for reference,prediction,want in [('我不吃辣','我不吃',1),('我不吃辣','我真的不吃辣',2),('我不吃辣','我不吃辣',0),('請推薦不辣的晚餐。','請推薦辣的晚餐。',1)]:
 observed=nc.text_error_report(reference,prediction)
 assert observed['edits']==want
 emit('metric_edge',observed)
# Equal wrong answers remain unjudged. Distinct wording can encode the same declared restriction.
wrong=nc.speech_stages('我不吃辣','我不吃辣','吃辣椒。','吃辣椒。')
paraphrases=nc.speech_stages('我不吃辣','我不吃辣','選不辣的餐點。','挑清淡、沒有辣椒的餐點。')
assert wrong['answers_identical'] and wrong['answers_correct'] is None
assert not paraphrases['answers_identical'] and paraphrases['answers_correct'] is None
emit('answer_agreement_counterexamples',{'identical_wrong':wrong,'distinct_paraphrases':paraphrases})
emit('prerequisite_audio_order',nc.audio_order_report())
# Real HTTP/session/upload/chat paths, deliberately injected ASR/generator; not model ability.
calls=[]
def runner(model,processor,row,data_root,options):
 calls.append(copy.deepcopy(row)); return {'prediction':'已收到：'+row['user']}
def recognize(model,processor,path):
 assert path.is_file();return {'transcript':'給我晚餐建議','audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def req(server,route,data):
 conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
 try:
  conn.request('POST',route,json.dumps(data),{'Content-Type':'application/json'})
  response=conn.getresponse();body=json.loads(response.read());assert response.status==200,(response.status,body);return body
 finally:conn.close()
with tempfile.TemporaryDirectory(dir=ROOT/'outputs/natural-v4/factual-research/12.14') as td:
 with patch.object(na,'generate',runner),patch.object(na,'transcribe',recognize):
  server=ui.create_server(object(),object(),SimpleNamespace(adapter=None),td,port=0,asr=(object(),object()))
  thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':0.01},daemon=True);thread.start()
  try:
   session=req(server,'/api/session',{})['session']
   first=req(server,'/api/chat',{'session':session,'prompt':'我不吃辣'})
   audio=io.BytesIO();sf.write(audio,np.zeros(1600,dtype='float32'),16000,format='WAV')
   upload=req(server,'/api/upload',{'session':session,'filename':'example.wav','kind':'audio','base64':base64.b64encode(audio.getvalue()).decode()})
   transcript=req(server,'/api/transcribe',{'session':session,'audio':upload['asset']})
   assert len(calls)==1 # ASR does not generate a chat answer.
   second=req(server,'/api/chat',{'session':session,'prompt':transcript['asr']['transcript'],'speech':upload['asset']})
   assert calls[1]['user']=='給我晚餐建議' and len(calls[1]['history'])==2
   assert calls[1]['history'][0]['content']==[{'type':'text','text':'我不吃辣'}]
   assert calls[1]['history'][1]['content']==[{'type':'text','text':first['prediction']}]
   assert 'audio' not in calls[1] and not second['asr']['corrected']
   corrected=req(server,'/api/chat',{'session':session,'prompt':'給我不辣的晚餐建議','speech':upload['asset']})
   assert corrected['asr']['corrected'] and corrected['asr']['transcript']=='給我晚餐建議'
   assert corrected['asr']['submitted_text']=='給我不辣的晚餐建議'
   emit('two_round_exercise_and_explicit_correction',{'received_rows':calls,'original_transcript':transcript['asr']['transcript'],'uncorrected_submission':second['user'],'correction_record':corrected['asr']})
  finally:server.shutdown();server.server_close();thread.join(2)
# Fixed-record audit, only section-relevant selection and speech pair data.
prefix='docs/natural-assistant/evidence/v4-runtime/'
vp=prefix+'validation-37219466611/blind-review/scored/scores.json'
sp='docs/natural-assistant/v4/selection.json'; protocol=read('docs/natural-assistant/v4/validation-protocol-lower-lr.json')
selection=read(sp); validation=read(vp)
assert selection==read(prefix+'validation-37219466611/blind-review/scored/selection.json')
weights=validation['integer_weights']; numerators={}
for variant,record in validation['variants'].items():
 n=sum(record['correct_counts'][name]*weight for name,weight in weights.items())
 assert n==record['primary_numerator'] and record['primary_denominator']==75600
 numerators[variant]=n
assert numerators=={'base':53277,'adapter-step-001039':52285,'adapter-step-002077':52705}
assert validation['selected_variant']==selection['selected_variant']=='base'
assert all(not record['eligible'] for name,record in validation['variants'].items() if name!='base')
emit('independent_selection_arithmetic',{'weighted_numerators':numerators,'denominator':75600,'voice_counts':{k:{n:v for n,v in d['correct_counts'].items() if 'voice' in n} for k,d in validation['variants'].items()},'validation_denominators':validation['denominators'],'selected_variant':'base','selection_sha256':sha(sp),'scope':'Recomputed fixed validation counts/decision, not independent semantic regrading or own GPU training.'})
test=prefix+'evaluate-37221188153/blind-review/'
result=read(test+'result.json'); gens=read(test+'generations-base.json'); transcripts=read(test+'transcripts.json'); scores=read(test+'scored/scores.json'); manifest=read('docs/natural-assistant/v4/manifest.json')
assert result['selected_only'] and list(result['variants'])==['base'] and result['trainable_parameters']==0
assert result['execution']['adapters']==[] and result['execution']['adapter_run_id'] is None
assert result['model_revision']==selection['base_model']['revision']
assert sha(test+'generations-base.json')==scores['artifact_binding']['generation_file']['sha256']
assert sha(test+'transcripts.json')==scores['artifact_binding']['transcripts_sha256']
rows={row['id']:row for row in manifest['audio_rows']}
pairs=[]
for transcript in transcripts:
 if transcript['task']!='speech_chat':continue
 identifier=transcript['id'];row=rows[identifier]
 speech=next(g for g in gens if g['id']==identifier and g['task']=='speech_chat')
 typed=next(g for g in gens if g['id']==identifier and g['task']=='typed_chat')
 assert speech['user']==transcript['transcript'] and typed['user']==row['user']==transcript['reference_transcript']
 assert speech['image']==typed['image']==row.get('image')
 assert na.messages_for(dict(row,user=transcript['transcript']),ROOT)==na.messages_for(row,ROOT)
 assert speech['prediction']==typed['prediction'] and speech['generated_token_ids']==typed['generated_token_ids']
 assert nc.edit_distance(row['user'],transcript['transcript'])==0
 decisions=[d for d in scores['case_decisions'] if d['case_id'] in ['speech_chat:'+identifier,'typed_chat:'+identifier]]
 assert len(decisions)==2
 pairs.append({'id':identifier,'reference':row['user'],'recognized':transcript['transcript'],'history':row['history'],'system':row['system'],'image':row.get('image'),'typed_prediction':typed['prediction'],'spoken_prediction':speech['prediction'],'raw_token_ids_equal':True,'case_decisions':decisions})
assert len(pairs)==4 and sum(len(p['reference']) for p in pairs)==42
assert sum(d['passed'] for p in pairs for d in p['case_decisions'])==4
emit('fixed_paired_speech_records',{'pairs':pairs,'denominators':{'questions':4,'reference_characters':42,'typed_answers':4,'spoken_answers':4,'seed':result['seed'],'max_new_tokens':protocol['generation']['max_new_tokens']},'record_environment':result['versions'],'record_device':result['device'],'record_dtype':result['dtype'],'typed_passed':2,'spoken_passed':2,'raw_cer':0.0,'scope':'Actual source records inspected and independently recomputed on CPU; no own audio recognition, LLM inference, or GPU replication. System/history derived from bound manifest; raw records do not include rendered input tokens.'})
emit('all_checks','passed')
