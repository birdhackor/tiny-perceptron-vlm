"""實跑當版Notebook CLI的有限CPU樣本、隔離及故障回報；另重核原五節點。"""
import datetime,hashlib,json,os,platform,shutil,subprocess,sys,time
from pathlib import Path
import torch,nbclient,ipykernel,pytest

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
CACHE=Path('/workspace/work/reference/omni-technical-validation-callback-v2')
CACHE.mkdir(parents=True,exist_ok=True)
SANDBOX=CACHE/'notebook-cli-sandbox'
SANDBOX.mkdir()
(SANDBOX/'scripts').mkdir();(SANDBOX/'notebooks').mkdir()
current=ROOT/'scripts/check_notebooks.py';copy=SANDBOX/'scripts/check_notebooks.py';shutil.copyfile(current,copy)
assert current.read_bytes()==copy.read_bytes()
ENV={**os.environ,'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def notebook(source,path):
    import nbformat
    n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    n.metadata['kernelspec']=dict(name='tiny-perceptron',display_name='Tiny Perceptron',language='python')
    nbformat.write(n,path)
def run(name,argv,cwd,limit=50):
    path=CACHE/(name+'.log');start=time.monotonic()
    p=subprocess.run(argv,cwd=cwd,env=ENV,capture_output=True,text=True,timeout=limit)
    path.write_text(p.stdout+p.stderr)
    return dict(command=argv,cwd=str(cwd),returncode=p.returncode,seconds=round(time.monotonic()-start,3),stdout=p.stdout,stderr=p.stderr,log_path=str(path),log_sha256=sha(path))

if __name__=='__main__':
    first="import os,json,torch\nsentinel = 'only first kernel'\nprint(json.dumps({'pid':os.getpid(),'device':str(torch.tensor([1]).device),'first':True}))"
    second="import os,json,torch\nprint(json.dumps({'pid':os.getpid(),'device':str(torch.tensor([1]).device),'sentinel_absent':'sentinel' not in globals()}))\nassert 'sentinel' not in globals()"
    notebook(first,SANDBOX/'notebooks/01-a.ipynb');notebook(second,SANDBOX/'notebooks/02-b.ipynb')
    success=run('kernel-complete-two',['python','scripts/check_notebooks.py','--mode','kernel','--workers','1'],SANDBOX)
    summary=json.loads((SANDBOX/'outputs/notebooks/validation-kernel.json').read_text());parsed=[]
    for name in ['01-a.ipynb','02-b.ipynb']:
        executed=json.loads((SANDBOX/'outputs/notebooks'/name).read_text());text=''.join(executed['cells'][0]['outputs'][0]['text']);parsed.append(json.loads(text))
    notebook("raise ValueError('intentional validation callback probe')",SANDBOX/'notebooks/03-failure.ipynb')
    failed=run('kernel-intentional-failure',['python','scripts/check_notebooks.py','--mode','kernel','--workers','1'],SANDBOX)
    failure_summary=json.loads((SANDBOX/'outputs/notebooks/validation-kernel.json').read_text())
    failure_file=SANDBOX/'notebooks/03-failure.ipynb';failure_file.unlink()
    old=ROOT/'docs/technical-reviews/artifacts/omni-technical-validation-v1/execution-failure-classification.json'
    ids=[x['node_id'] for x in json.loads(old.read_text())['current_worktree_failures']]
    tests=run('current-five-and-historical-metadata-tests',['pytest','-ra',*ids,'tests/test_historical_rechecks.py','tests/test_historical_rechecks_v2.py'],ROOT,90)
    gates=[]
    for name,script in [('reader-gate','scripts/check_course_reviews.py'),('technical-gate','scripts/check_technical_reviews.py'),('all-gate','docs/review-tools/check_review_round.py')]:
        x=run(name,['python',script],ROOT,70)
        # CLI JSON is small; retain exact output only when it is bounded, otherwise fingerprint outside Git.
        if len(x['stdout'])>12000:x['stdout']=x['stdout'][-1500:];x['stdout_scope']='trailing excerpt; full output retained outside Git with SHA'
        gates.append(x)
    checks=dict(success_exit=success['returncode']==0,complete_summary=summary['total']==summary['passed']==2 and summary['failed']==0 and summary['failures']==[],different_pids=parsed[0]['pid']!=parsed[1]['pid'],cpu_only=all(x['device']=='cpu' for x in parsed),independent_namespace=parsed[1]['sentinel_absent'] is True,failure_exit=failed['returncode']==1,failure_summary=failure_summary['total']==3 and failure_summary['passed']==2 and failure_summary['failed']==1 and len(failure_summary['failures'])==1)
    assert all(checks.values())
    value=dict(created_at=datetime.datetime.now(datetime.UTC).isoformat(),command='uv run --extra cpu python '+str(Path(__file__).relative_to(ROOT)),environment=dict(python=sys.version,torch=torch.__version__,nbclient=nbclient.__version__,ipykernel=ipykernel.__version__,pytest=pytest.__version__,device='CPU',platform=platform.platform()),actual_cli_source=dict(path='scripts/check_notebooks.py',sha256=sha(current),sandbox_path=str(copy),sandbox_sha256=sha(copy)),success=success,success_summary=summary,success_outputs=parsed,failure=failed,failure_summary=failure_summary,checks=checks,current_regression_nodes_and_metadata_tests=tests,current_global_gates=gates,scope='Finite two-notebook CPU contract check plus one deliberately failing notebook. This is not the full331-notebook run. Existing original partial/global-failure receipts remain unchanged; current results are dated separately. No training, GPU or paid action.')
    out=HERE/'bounded-probes-v2.json'
    with out.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(output=str(out.relative_to(ROOT)),sha256=sha(out),checks=checks,tests_returncode=tests['returncode'],gate_returncodes=[x['returncode'] for x in gates]),ensure_ascii=False))
