"""Hash-only supplemental checks of original core science and external Kev pins. No NN/import of models."""
from pathlib import Path
import hashlib,json,sys,zipfile

BASE=Path('/workspace/work/omni-kev-actual-independent-verification-v1');SOURCE=BASE/'source-010a639f'
sys.path.insert(0,str(SOURCE))
from scripts.omni_revision import research_upstream
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
with zipfile.ZipFile(BASE/'download/input.zip') as z:
 job=json.loads(z.read('job.json'));selection=json.loads(z.read('model-selection.json'));opening=json.loads(z.read('final-opening.json'))
 for n,v in job['files'].items():assert sha(SOURCE/'docs/omni-products/v1'/n)==v
 own_eval=opening['evaluation_sources_sha256'];assert len(own_eval)==57
 for n,v in own_eval.items():assert sha(SOURCE/n)==v
 recipes=read(SOURCE/'docs/omni-products/v1/formal-recipes.json')
 for n,v in recipes['execution_sources_sha256'].items():assert sha(SOURCE/n)==v
 for n,v in recipes['implementation_sha256'].items():assert sha(SOURCE/'tiny_perceptron/omni_products'/n)==v
 note=read(BASE/'verified/external-source-verification.json');assert set(note)==set(research_upstream.FILES)
 external=[]
 for n,(size,digest) in research_upstream.FILES.items():
  raw=z.read('thirdparty/kev/'+n);assert len(raw)==size and hashlib.sha256(raw).hexdigest()==digest
  assert note[n]=={'source_url':research_upstream.source_url(n),'bytes':size,'sha256':digest}
  external.append({'path':n,**note[n]})
assert selection['temperature']==0.6484197773255048 and selection['threshold']==0.6
result={'reviewer_task':'/root/omni_cpu_final_execution_review','status':'pass','core_metadata_bindings_checked':len(job['files']),'own_evaluation_sources':len(own_eval),'formal_execution_sources':len(recipes['execution_sources_sha256']),'own_model_implementation_sources':len(recipes['implementation_sha256']),'own_C_temperature_unchanged':selection['temperature'],'own_C_threshold_unchanged':selection['threshold'],'core_selection_sha256':job['core_selection_sha256'],'core_final_opening_sha256':job['core_final_opening_sha256'],'external_source_revision':research_upstream.REVISION,'all5_external_files_in_original_input_and_worker_receipt_match_fixed_sha':external,'adapter_id':'jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52','base_id':'Qwen/Qwen3.5-4B-Base','base_revision':'1001bb4d826a52d1f399e183466143f4da7b741b','head_pt_sha256_enforced_by_executed_worker':'dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c','adapter_config_sha256_enforced_by_executed_worker':'8a05dfd6c5e7f61a62e6db5d8de8093a8dbbd105aa7ee7b5e7b73f6249f45617','reviewer_weight_download_or_model_load':False,'NN_forward_calls':0}
p=BASE/'binding-verification-results-v1.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print({k:result[k] for k in ['status','core_metadata_bindings_checked','own_evaluation_sources','formal_execution_sources','own_model_implementation_sources','own_C_temperature_unchanged','own_C_threshold_unchanged']})
