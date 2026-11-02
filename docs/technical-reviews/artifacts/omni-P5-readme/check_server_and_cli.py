"""Small local UI/server and command-parser inspection; no neural inference."""
import ast,datetime,hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path('/workspace/work/omni-book-integrated-v1');OUT=ROOT/'docs/technical-reviews/artifacts/omni-P5-readme';RAW=Path('/workspace/work/omni-P5-readme-independent-raw');PYTHON='/workspace/tiny-perceptron-vlm/.venv/bin/python'
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),JUPYTER_RUNTIME_DIR=str(RAW/'jupyter-runtime'),JUPYTER_CONFIG_DIR=str(RAW/'jupyter-config'))
p={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_readme','scope':'Local JupyterLab startup and content listing; parser/source inspection of maintenance and download commands. No workflow dispatch or final model inference.'}
cmd=[PYTHON,'-m','jupyterlab','notebooks','--no-browser','--ServerApp.port=18739','--ServerApp.ip=127.0.0.1','--ServerApp.allow_root=True','--IdentityProvider.token=','--ServerApp.open_browser=False'];process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
try:
 responses=[]
 for attempt in range(80):
  try:
   with urllib.request.urlopen('http://127.0.0.1:18739/api/status',timeout=1) as r:
    d=json.load(r);responses.append({'endpoint':'/api/status','status':r.status,'kernel_count':d['kernels']});break
  except Exception:time.sleep(.1)
 with urllib.request.urlopen('http://127.0.0.1:18739/api/contents/01/1.1.ipynb?content=0',timeout=5) as r:
  d=json.load(r);responses.append({'endpoint':'/api/contents/01/1.1.ipynb?content=0','status':r.status,'path':d['path'],'type':d['type']})
 p['jupyterlab']={'command':cmd,'responses':responses,'notebook_source_sha256':hashlib.sha256((ROOT/'notebooks/01/1.1.ipynb').read_bytes()).hexdigest(),'scope':'Notebook directory startup and first file API listing only; no notebook cell run through this server, no browser authentication or Windows execution.'}
finally:
 process.terminate()
 try:stdout,stderr=process.communicate(timeout=10)
 except subprocess.TimeoutExpired:process.kill();stdout,stderr=process.communicate()
 p['jupyterlab_shutdown_returncode']=process.returncode
# CLI contract inspection executes AST parsing, not commands which rewrite frozen textbook.
files=['scripts/build_visuals.py','scripts/format_course_code.py','scripts/build_course.py','scripts/check_course_reviews.py','scripts/check_technical_reviews.py','docs/review-tools/check_review_round.py','scripts/check_notebooks.py','scripts/fetch_training_assets.py','scripts/fetch_course_models.py','scripts/check_course_models.py']
p['source_cli_inspection']=[]
for f in files:
 q=ROOT/f;txt=q.read_text();tree=ast.parse(txt,filename=f);flags=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument':
   flags.append([a.value for a in n.args if isinstance(a,ast.Constant) and isinstance(a.value,str)])
 p['source_cli_inspection'].append({'path':f,'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'ast_parsed':True,'arguments':flags,'has_main_guard':"if __name__ == \"__main__\"" in txt})
# Record installed kernel and exact --sys-prefix option behavior via package's actual help.
for key,args in [('ipykernel_install_help',[PYTHON,'-m','ipykernel','install','--help']),('kernelspecs',[PYTHON,'-m','jupyter','kernelspec','list','--json'])]:
 r=subprocess.run(args,cwd=ROOT,env=env,text=True,capture_output=True,timeout=15)
 p[key]={'command':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
with (OUT/'server-cli-receipts-v1.json').open('x') as f:json.dump(p,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in p.items() if k not in ['ipykernel_install_help','kernelspecs','source_cli_inspection']},ensure_ascii=False,indent=2))
