"""Full high-level result verifier controls using explicitly artificial non-neural records."""
import copy,hashlib,importlib.util,io,json,struct,sys,wave,zipfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'current-source';sys.path.insert(0,str(SOURCE))
from scripts.omni_revision import research_execution as rx
spec=importlib.util.spec_from_file_location('mature_current_result_fixture',SOURCE/'tests/test_omni_mature_research.py');fx=importlib.util.module_from_spec(spec);spec.loader.exec_module(fx)
root=BASE/'probes/current-full-results';root.mkdir(parents=True,exist_ok=False)
raw=lambda v:json.dumps(v,ensure_ascii=False,allow_nan=False).encode();jsonl=lambda rows:('\n'.join(json.dumps(v) for v in rows)+'\n').encode()
sha=lambda b:hashlib.sha256(b).hexdigest()
base_core={'core_selection_sha256':'fixture-selection','core_final_opening_sha256':'fixture-opening','files':{}}
data=(SOURCE/'docs/omni-products/v1/data-definition.json').read_bytes();data_binding={'bytes':len(data),'sha256':sha(data)}
kev_result,kev_files=fx.kev_complete_fixture.__wrapped__();qwen_cases,qwen_result,qwen_files=fx.qwen_failed_fixture.__wrapped__()
results=[]
for mode in ('Kev_complete','Kev_cache_failure_still_complete','Kev_empty_views','Kev_missing_raw','Qwen_all104_failed_complete','Qwen_one_legal_PCM','Qwen_invalid_PCM','Qwen_empty_rows','Qwen_missing_raw','Qwen_wrong_runtime','Qwen_wrong_opening'):
 job_name='kev' if mode.startswith('Kev') else 'qwen25';folder=root/mode;folder.mkdir();job={**rx.job_config(job_name,base_core),'config_sha256':'a'*64,'input_files':{'data-definition.json':data_binding}}
 if job_name=='kev':result=copy.deepcopy(kev_result);files=copy.deepcopy(kev_files)
 else:
  result=copy.deepcopy(qwen_result);files=copy.deepcopy(qwen_files);cases_raw=raw(qwen_cases);job['input_files']['cases/cases.json']={'bytes':len(cases_raw),'sha256':sha(cases_raw)}
  with zipfile.ZipFile(folder/'input.zip','w') as archive:archive.writestr('cases/cases.json',cases_raw)
 if mode in ('Qwen_one_legal_PCM','Qwen_invalid_PCM'):
  predicted={'text':'no','new_tokens':1,'generated_ids':[100],'seed':20263029,'whole_completion_seconds':0.1,'generation':{'return_audio':True,'use_audio_in_video':False,'thinker_max_new_tokens':16,'talker_max_new_tokens':192,'thinker_do_sample':False,'talker_do_sample':False,'speaker':'Chelsie'},'input':{'input_ids':{'shape':[1,2],'dtype':'torch.int64','sha256':sha(struct.pack('<qq',1,2)),'values':[[1,2]]}}}
  buffer=io.BytesIO()
  with wave.open(buffer,'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000 if mode=='Qwen_invalid_PCM' else 24000);w.writeframes(b'\0\0'*240)
  wav=buffer.getvalue();files['speech/0.wav']=wav
  spoken={'id':qwen_cases['speech_output_prompts'][0]['id'],'gold_word':'no','generated':True,'prediction':predicted,'audio':{'sample_rate':24000,'samples':240,'legal':True,'path':'result/speech/0.wav','sha256':sha(wav),'float32_sha256':sha(b'\0'*960)}}
  original=copy.deepcopy(spoken);spoken.update(ASR_proxy_attempted=True,ASR_proxy_correct=True,ASR_proxy_text='no');result['speech_outputs'][0]=spoken
  originals=[json.loads(x) for x in files['qwen-speech-raw.jsonl'].splitlines()];originals[0]=original;files['qwen-speech-raw.jsonl']=jsonl(originals);files['qwen-asr-raw.jsonl']=jsonl(result['speech_outputs'])
  result['speech_counts'].update(generated=1,legal=1,ASR_proxy_attempted=1,ASR_proxy_correct_out_of_all8=1)
 if mode=='Kev_cache_failure_still_complete':result['cache']['rows'][0]['prefix_after']='2'*64;result['cache']['all_observed_within_tolerance']=False
 if mode=='Kev_empty_views':files['kev-final-english.json']=b'{}';files['kev-final-chinese.json']=b'{}'
 if mode=='Kev_missing_raw':files['kev-raw-predictions.jsonl']=b'\n'.join(files['kev-raw-predictions.jsonl'].splitlines()[:-1])
 if mode=='Qwen_empty_rows':files['qwen-raw-predictions.jsonl']=(b'{}\n'*96)
 if mode=='Qwen_missing_raw':files['qwen-raw-predictions.jsonl']=b'\n'.join(files['qwen-raw-predictions.jsonl'].splitlines()[:-1])
 result.update(schema='explicit-artificial-review-fixture',completed=True,final_opened=True)
 runtime={'python':'3.12.10','torch':'2.8.0+cu128','cuda':'12.8','gpu':'NVIDIA '+job['gpu'],'gpu_bytes':48*1024**3,'base_torch_location':'explicit-fixture-only'}
 lock=dict(line.split('==') for line in (SOURCE/f'scripts/omni_revision/research_requirements/{job_name}.txt').read_text().splitlines());isolated={'torch':runtime['torch'],'cuda':runtime['cuda'],'packages':{k:lock[k] for k in ('transformers','huggingface-hub','tokenizers','numpy','accelerate','safetensors')}}
 if mode=='Qwen_wrong_runtime':isolated['packages']['transformers']='0.0.0'
 opening={'job':job_name,'core_selection_sha256':job['core_selection_sha256'],'core_final_opening_sha256':job['core_final_opening_sha256'],'job_config_sha256':job['config_sha256'],'data_definition_sha256':data_binding['sha256'],'input_files':job['input_files'],'core_models_immutable':True}
 if mode=='Qwen_wrong_opening':opening['core_final_opening_sha256']='wrong'
 result_raw=raw(result);summary={'schema':'omni-mature-worker-summary-v1','job':job_name,'core_selection_sha256':job['core_selection_sha256'],'own_training_permitted':False,'success':True,'final_opened':True,'research_result_sha256':sha(result_raw),'elapsed_seconds':1.0,'runtime':runtime,'isolated_runtime':isolated}
 files.update({'research-result.json':result_raw,'worker-summary.json':raw(summary),'runtime-before.json':raw(runtime),'isolated-runtime.json':raw(isolated),'research-opening.json':raw(opening)})
 manifest={n:{'bytes':len(b),'sha256':sha(b)} for n,b in files.items()};path=folder/'result.zip'
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
  for n,b in files.items():archive.writestr(n,b)
  archive.writestr('file-manifest.json',raw(manifest))
 try:verified=rx.verify_results(path,job);entry={'case':mode,'accepted':True,'verified':verified}
 except Exception as e:entry={'case':mode,'accepted':False,'error_type':type(e).__name__,'reason':str(e)}
 expected=mode in ('Kev_complete','Kev_cache_failure_still_complete','Qwen_all104_failed_complete','Qwen_one_legal_PCM');entry['expected_acceptance']=expected;assert entry['accepted'] is expected,entry;results.append(entry)
record={'scope':'Actual high-level verify_results including runtime/opening/manifest and measurement helpers; entire model observations/runtime are explicit artificial fixtures. No real neural, teacher, core, final or provider run.','cases':results,'all_expected_outcomes':True,'real_neural_or_paid_or_final_calls':0}
p=BASE/'current-full-results-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
