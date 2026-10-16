"""固定HF32檔匿名GET與publication/default pointer/借用CPU收據鏈；不重跑推論。"""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import UTC,datetime
from pathlib import Path
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[4]
TREE=Path('/workspace/work/lesson-products-transport')
ART=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from scripts.lesson_products.publication import verify_release


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 started=datetime.now(UTC).isoformat()
 files=['docs/lesson-products/results/publication-v1.json','docs/lesson-products/results/publication-spec-v1.json','docs/lesson-products/results/public-cpu-acceptance-v1.json','tiny_perceptron/lesson_products/release.json','tiny_perceptron/lesson_products/delivery.py','tiny_perceptron/lesson_products/__main__.py','tiny_perceptron/lesson_products/inference.py','.github/workflows/lesson-products-publish.yml','scripts/lesson_products/publish.py','scripts/lesson_products/publication.py']
 before={name:sha(ROOT/name) for name in files}
 receipt=json.loads((ROOT/files[0]).read_bytes());execution=receipt['execution'];spec=json.loads((ROOT/files[1]).read_bytes());cpu=json.loads((ROOT/files[2]).read_bytes());pointer=json.loads((ROOT/files[3]).read_bytes())
 public=execution['public'];repo='birdhackor/tiny-perceptron-course-models';revision='2c6a1062f3e66a602cf83eda6c703456ec18bca0';prefix='lesson-products/v1-20261016';source='b137418e7e4fda757ab295fafa8515bc98a53f56'
 assert receipt['status']==execution['status']=='completed' and execution['revision']==source and receipt['run_id']=='lesson-release-38871109465-1'
 assert receipt['all_bounds_retained'] is True and execution['gpu_used'] is False
 assert public=={'repo':repo,'revision':revision,'prefix':prefix}
 assert pointer['repo']==repo and pointer['revision']==revision and set(pointer['products'])=={'b','c'}
 actual_downloads={item['path']:{key:item[key] for key in ('sha256','bytes')} for item in execution['anonymous_downloads']}
 assert len(execution['anonymous_downloads'])==len(actual_downloads)==32 and actual_downloads==spec['files']
 assert cpu['release_sha256']==before[files[3]] and cpu['release']==pointer
 # Original descriptor stays in isolated jobs; main results is a byte-exact evidence copy.
 original_spec=subprocess.check_output(['git','show',source+':docs/lesson-products/jobs/release-v1.json'],cwd=TREE)
 assert original_spec==(ROOT/files[1]).read_bytes()
 source_checks=[]
 for name,digest in execution['source_hashes'].items():
  raw=subprocess.check_output(['git','show',source+':'+name],cwd=TREE)
  assert hashlib.sha256(raw).hexdigest()==digest
  source_checks.append({'path':name,'sha256':digest,'matches_actual_publication_commit':True})
 responses=[]
 with tempfile.TemporaryDirectory(dir=ROOT/'work',prefix='completed-publication-audit-') as temporary:
  downloaded=Path(temporary)
  def fetch(item):
   name,expected=item;url=f'https://huggingface.co/{repo}/resolve/{revision}/{prefix}/{quote(name,safe="/")}'
   attempts=[]
   for attempt in (1,2):
    try:
     request=urllib.request.Request(url,headers={'User-Agent':'tiny-perceptron-independent-engineering-audit/1'})
     assert request.get_header('Authorization') is None and request.get_header('Cookie') is None
     with urllib.request.urlopen(request,timeout=45) as response:
      raw=response.read(expected['bytes']+1);status=response.status;commit=response.headers.get('X-Repo-Commit')
     assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'],name
     target=downloaded/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
     attempts.append({'attempt':attempt,'status':status,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
     return {'path':name,'url':url,'expected':expected,'attempts':attempts,'anonymous_request':True,'no_auth_header_or_cookie':True,'reported_repo_commit':commit,'passed':True}
    except Exception as error:
     attempts.append({'attempt':attempt,'error_type':type(error).__name__,'http_status':getattr(error,'code',None)})
     if attempt==2:return {'path':name,'url':url,'expected':expected,'attempts':attempts,'anonymous_request':True,'passed':False}
   raise AssertionError('unreachable')
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:responses=list(pool.map(fetch,sorted(spec['files'].items())))
  assert all(row['passed'] for row in responses),[row for row in responses if not row['passed']]
  verified=verify_release(downloaded,spec)
  assert verified==execution['verified_payload'] and verified['files']==32
  manifests={}
  for product in ('b','c'):
   location=downloaded/product/'artifact-manifest.json';manifest=json.loads(location.read_bytes());entry=pointer['products'][product]
   assert entry=={'prefix':prefix+'/'+product,'manifest_sha256':sha(location)}
   assert set(manifest['files'])=={'model.pt','config.json','tokenizer.json','training-receipt.json','auto-policy.json' if product=='b' else 'calibration.json'}
   manifests[product]={'manifest_sha256':sha(location),'settings_freeze_sha256':manifest['settings_freeze_sha256'],'data_manifest_sha256':manifest['data_manifest_sha256'],'files':manifest['files']}
 # Check root CPU receipt and raw current outputs, explicitly not rerunning acceptance.
 assert cpu['status']=='passed' and len(cpu['cli_runs'])==7 and len(cpu['http_requests'])==3
 commands=cpu['cli_runs'];assert all(row['exit_code']==0 and not row['stderr'] for row in commands)
 outputs=[json.loads(row['stdout']) for row in commands]
 for index,product in enumerate(('b','c')):
  assert outputs[index]['manifest_sha256']==pointer['products'][product]['manifest_sha256'] and outputs[index]['revision']==revision and outputs[index]['repo']==repo and outputs[index]['files_verified']==5 and outputs[index]['anonymous'] is True
 assert outputs[2]['answer']==2 and outputs[2]['loops']==3
 assert outputs[3]['selected']==outputs[6]['selected']=='no'
 assert outputs[4]['status']=='answered' and outputs[4]['b_loops']==3 and outputs[4]['solution']['answer']==2
 assert outputs[5]['status']=='budget_limited' and outputs[5]['b_loops']==outputs[5]['max_loops']==2
 assert [row['endpoint'] for row in cpu['http_requests']]==['/solve','/judge','/auto']
 assert cpu['http_requests'][0]['output']['answer']==2 and cpu['http_requests'][1]['output']['selected']=='no' and cpu['http_requests'][2]['output']['solution']['answer']==2
 raw_root=Path('/workspace/work/lesson-products-clean-cpu-acceptance');raw_files=[]
 for product,manifest in manifests.items():
  for name,expected in {'artifact-manifest.json':spec['files'][product+'/artifact-manifest.json'],**manifest['files']}.items():
   path=raw_root/'outputs/lesson-products'/product/name;assert sha(path)==expected['sha256'] and path.stat().st_size==expected['bytes']
   raw_files.append({'path':str(path),**expected})
 http_log=raw_root/'http-server.log';log=http_log.read_text()
 for route in ('GET /health','POST /solve','POST /judge','POST /auto'):assert route+' HTTP/1.1" 200' in log
 runner=Path('/workspace/work/curriculum-20261016/clean_cpu_acceptance.py');script=runner.read_text();assert 'W.mkdir(exist_ok=False)' in script and 'HF_' in script and 'HUGGING_FACE_HUB_TOKEN' in script and 'server.terminate();server.wait' in script
 cpu_python=Path(commands[0]['command'][0]);cfg=cpu_python.parents[1]/'pyvenv.cfg';assert 'include-system-site-packages = false' in cfg.read_text()
 env={key:value for key,value in os.environ.items() if not(key.startswith(('HF_','HUGGINGFACE_','TRANSFORMERS_')) or key=='HUGGING_FACE_HUB_TOKEN')}
 code='import importlib.util,sys,json;print(json.dumps({"executable":sys.executable,"prefix":sys.prefix,"torch_origin":importlib.util.find_spec("torch").origin,"project_origin":importlib.util.find_spec("tiny_perceptron.lesson_products").origin}))'
 observed=subprocess.run([str(cpu_python),'-c',code],cwd=raw_root,env=env,capture_output=True,text=True);assert observed.returncode==0,observed.stderr
 origins=json.loads(observed.stdout);assert origins['torch_origin'].startswith(str(cpu_python.parents[1]))
 assert before=={name:sha(ROOT/name) for name in files}
 result={'reviewer_task':'/root/tech_20261016_review_tools','context':'informed completed-publication callback','started_at':started,'completed_at':datetime.now(UTC).isoformat(),'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform()},'checked_files':[{'path':str(ROOT/name),'sha256':digest} for name,digest in before.items()],'publication_commit':source,'actual_source_blob_checks':source_checks,'personally_downloaded_remote':{'repo':repo,'revision':revision,'prefix':prefix,'files':responses,'count':len(responses),'total_bytes':sum(row['expected']['bytes'] for row in responses),'verified_payload':verified,'all_expected':True,'transport':'urllib.request explicit User-Agent only, no auth/cookie header or HF client/cache used; transient attempt outcomes retained, bytes in own temp then removed'},'default_release_manifest_chain':manifests,'borrowed_cpu_receipt_checked':{'author_task':cpu['author_task'],'receipt_sha256':before[files[2]],'cli_runs':len(commands),'http_requests':len(cpu['http_requests']),'release_sha256':cpu['release_sha256'],'all_exit_codes_zero':True,'raw_current_files_checked':raw_files,'http_log':{'path':str(http_log),'sha256':sha(http_log)},'acceptance_script':{'path':str(runner),'sha256':sha(runner)},'venv_cfg':{'path':str(cfg),'sha256':sha(cfg)},'current_import_origin_probe':origins,'did_reexecute_cli_or_http':False},'all_expected':True,'limits':['This reviewer independently fetched all32 anonymous immutable-commit remote files and checked source blobs/pointers/raw outputs. Actual publication Modal run and seven CLI/three HTTP runs are borrowed root receipts, not personally rerun.','No weights/Torch model loaded, training/formaltest/GPU/paid/write operation. Current CPU environment probe inspects module origins only; not historical proof of all original environment values.','Current downloader/CLI source inspected for default pointer wiring and auth behavior; not a full new code or scientific technical audit.']}
 path=ART/'engineering-publication-completed-results-callback-01.json'
 with path.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps({'remote_files':len(responses),'all_expected':True,'total_bytes':result['personally_downloaded_remote']['total_bytes'],'cpu_receipt_cli_http_counts':[7,3],'receipt_sha256':sha(path)}))


if __name__=='__main__':main()
