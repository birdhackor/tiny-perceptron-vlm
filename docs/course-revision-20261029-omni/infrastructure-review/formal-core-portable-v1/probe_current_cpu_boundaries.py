"""Offline CPU launch interruption/finish predicates; no remote writes or formal training."""
import copy,json,os,subprocess,sys
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'current-source'))
from scripts.omni_revision import cpu_execution as cpu
from tiny_perceptron.omni_products.training_runtime import json_file
probe=BASE/'probes/current-cpu-boundaries';probe.mkdir(parents=True,exist_ok=False)
real_popen=subprocess.Popen;children=[]
state={'all_training_processes_exited':True}
def start_then_interrupt(*args,**kwargs):
 # Represent cancellation after OS creation but before bounded_child records the start response.
 child=real_popen([sys.executable,'-c','import time; time.sleep(30)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 children.append(child)
 raise KeyboardInterrupt('review cancellation at child-start return boundary')
try:
 with patch.object(cpu.subprocess,'Popen',side_effect=start_then_interrupt):
  try:cpu.bounded_child(['-c','review fixture only'],probe/'launch.log',5,state)
  except BaseException as error:launch_error=type(error).__name__
 alive=children[0].poll() is None
 launch={'injected_exception':launch_error,'real_fixture_child_alive_after_return':alive,'all_training_processes_exited_flag':state['all_training_processes_exited'],'false_stopped_flag':alive and state['all_training_processes_exited'] is True}
finally:
 for child in children:
  if child.poll() is None:child.kill()
  child.wait(timeout=10)
launch['review_fixture_child_cleanup_observed']=all(c.poll() is not None for c in children)
reserved={'attempt_id':'cpu-core-1-1','status':'reserved','revision':'1'*40,'purpose':'private review fixture','reserved_usd':'.10'}
base_receipt={'success':True,'all_training_processes_exited':True,'no_modal_compute_created':True,'release_id':9,'release_tag':'private-fixture-tag','assets':[{'id':10,'name':'model.zip','bytes':4,'sha256':'a'*64,'roundtrip_verified':True}]}
base_owned={'draft':True,'tag_name':'private-fixture-tag','body':'private-fixture-body','assets':[{'id':10,'name':'model.zip','size':4}]}
base_artifacts={'total_count':1,'artifacts':[{'id':20,'name':'omni-core-1-1','size_in_bytes':100,'created_at':'2026-10-30T00:00:00Z','expires_at':'2026-11-06T00:00:00Z'}]}
cases=[]
for mode in ('success','known_failed_stopped','unknown_child','unrecorded_release_asset','foreign_owner','retention_eight_days','upload_failed','no_modal_unknown','lost_start_false_flag'):
 output=probe/mode;output.mkdir();receipt=copy.deepcopy(base_receipt);owned=copy.deepcopy(base_owned);artifacts=copy.deepcopy(base_artifacts);publication='success'
 if mode=='known_failed_stopped':receipt['success']=False
 if mode=='unknown_child':receipt['all_training_processes_exited']=False
 if mode=='unrecorded_release_asset':owned['assets'].append({'id':11,'name':'unknown.zip','size':5})
 if mode=='foreign_owner':owned['body']='foreign'
 if mode=='retention_eight_days':artifacts['artifacts'][0]['expires_at']='2026-11-07T00:00:00Z'
 if mode=='upload_failed':publication='failure'
 if mode=='no_modal_unknown':receipt['no_modal_compute_created']=False
 if mode=='lost_start_false_flag':receipt.update(success=False,all_training_processes_exited=state['all_training_processes_exited'])
 json_file(output/'reservation.json',{'attempt':reserved});json_file(output/'cpu-receipt.json',receipt);json_file(output/'release-ownership.json',{'body':'private-fixture-body'})
 saved=[]
 class Store:
  def read(self):return {'attempts':[copy.deepcopy(reserved)]},'fixture-blob'
  def write(self,result,blob):saved.append(result);return {'commit':'fixture','blob':'fixture'}
 def api(path,*args):return artifacts if path.startswith('actions/') else owned
 with patch.object(cpu,'identity',return_value=None),patch.object(cpu,'api',side_effect=api),patch.object(cpu,'GitHubLedger',Store),patch.dict(os.environ,{'GITHUB_RUN_ID':'1','GITHUB_RUN_ATTEMPT':'1','OMNI_ARTIFACT_OUTCOME':publication}):
  cpu.finish_cpu(reserved['attempt_id'],output)
 attempt=saved[-1]['attempts'][0]
 cases.append({'case':mode,'status':attempt['status'],'compute_stopped':attempt['receipt']['compute_stopped'],'resources_verified':attempt['receipt']['resources_verified']})
result={'scope':'Real private sleeping child only, killed and waited by reviewer; launch/finish fault fixtures are not actual GitHub or formal receipts. All APIs/ledger writes mocked.','launch_interruption':launch,'finish_cases':cases,'paid_or_formal_or_final_calls':0}
json_file(BASE/'current-cpu-boundaries-result.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
