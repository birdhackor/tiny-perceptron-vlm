"""Semantic completion negative fixtures; no model/pretrained/final inference."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'current-source';sys.path.insert(0,str(SOURCE))
from scripts.omni_revision import research_execution as rx
root=BASE/'probes/current-result-contract';root.mkdir(parents=True,exist_ok=False)
def put(path,files):
 manifest={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()}
 with zipfile.ZipFile(path,'w') as z:
  for n,b in files.items():z.writestr(n,b)
  z.writestr('file-manifest.json',json.dumps(manifest))
def raw(v):return json.dumps(v).encode()
core={'core_selection_sha256':'explicit-fixture-selection','core_final_opening_sha256':'explicit-fixture-opening','files':{}}
cases=[]
for job in ('kev','qwen25'):
 summary={'job':job,'core_selection_sha256':core['core_selection_sha256'],'success':True,'final_opened':True,'test_fixture':True}
 result={'completed':True,'final_opened':True,'test_fixture':True}
 files={'worker-summary.json':raw(summary)}
 if job=='kev':
  files.update({'kev-final-english.json':raw({}),'kev-final-chinese.json':raw({}),'kev-calibration.json':raw([{}]*80),'kev-policy-predictions.json':raw([{}]*60)})
 else:result.update(audio_text_pairs=[{}]*24,AV_shift_pairs=[{}]*24,speech_outputs=[{}]*8)
 files['research-result.json']=raw(result);path=root/(job+'.zip');put(path,files)
 try:verified=rx.verify_results(path,rx.job_config(job,core));cases.append({'job':job,'accepted':True,'verified':verified,'missing':'All actual predictions/IDs/logits/counts and calibration/policy evidence; Kev final views are empty dictionaries.'})
 except Exception as e:cases.append({'job':job,'accepted':False,'error_type':type(e).__name__,'reason':str(e)})
record={'scope':'Manifest/SHA-correct but deliberately semantically empty completion fixtures. No real worker/model/NN/final calls.','cases':cases,'paid_or_neural_or_final_calls':0}
p=BASE/'current-result-contract-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
