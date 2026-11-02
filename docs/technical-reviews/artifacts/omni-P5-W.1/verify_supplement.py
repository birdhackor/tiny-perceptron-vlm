"""Small remaining W.1 routing/configuration probes; all clone/install effects mocked or local."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import types
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
DEST=Path(__file__).resolve().parent
RUNTIME=Path('/workspace/work/omni-P5-W.1-runtime-v2/supplement')

def run(argv,cwd=ROOT,env=None):
    p=subprocess.run(argv,cwd=cwd,env=env,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,(argv,p.stdout,p.stderr)
    return {'argv':argv,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

if __name__=='__main__':
    RUNTIME.mkdir(exist_ok=False)
    record={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_w_1','command':'/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/omni-P5-W.1/verify_supplement.py','environment':{'python':platform.python_version(),'os':platform.platform(),'device':'CPU; branch routing only, no neural model'},'checks':{}}
    checks=record['checks']
    checks['notebook_generation']=run([sys.executable,'scripts/build_course.py','--check'])
    env={**os.environ,'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':str(RUNTIME/'global-gitconfig')}
    checks['longpaths_config']=run(['git','config','--global','core.longpaths','true'],cwd=RUNTIME,env=env)
    checks['longpaths_value']=run(['git','config','--global','--get','core.longpaths'],cwd=RUNTIME,env=env)
    run(['git','init','--quiet'],cwd=RUNTIME,env=env)
    checks['lfs_version']=run(['git','lfs','version'],cwd=RUNTIME,env=env)
    checks['lfs_local_install']=run(['git','lfs','install','--local'],cwd=RUNTIME,env=env)
    checks['lfs_local_filter']=run(['git','config','--local','--get','filter.lfs.process'],cwd=RUNTIME,env=env)
    assert checks['lfs_version']['stdout'].startswith('git-lfs/')
    assert checks['longpaths_value']['stdout']=='true\n'
    assert checks['lfs_local_filter']['stdout']=='git-lfs filter-process\n'
    nb=json.loads((ROOT/'notebooks/01/1.1.ipynb').read_text())
    setup=next(''.join(c['source']) for c in nb['cells'] if c.get('metadata',{}).get('course_setup'))
    google=types.ModuleType('google');google.__path__=[]
    colab=types.ModuleType('google.colab');google.colab=colab
    prior_exists=Path.exists
    def exists(p):
        return False if str(p)=='/content/tiny-perceptron-vlm' else prior_exists(p)
    scope={}
    with patch.dict(sys.modules,{'google':google,'google.colab':colab}),patch('subprocess.run') as calls,patch('os.chdir') as change,patch.object(Path,'exists',exists):
        exec(compile(setup,'notebooks/01/1.1.ipynb:course_setup','exec'),scope)
    routing=[]
    for c in calls.call_args_list:
        kwargs=c.kwargs
        routing.append({'argv':c.args[0],'check':kwargs.get('check'),'GIT_LFS_SKIP_SMUDGE':kwargs.get('env',{}).get('GIT_LFS_SKIP_SMUDGE')})
    assert len(routing)==2 and routing[0]['argv'][0:3]==['git','clone','--depth']
    assert routing[0]['GIT_LFS_SKIP_SMUDGE']=='1' and routing[1]['argv'][1:4]==['-m','pip','install']
    assert change.call_args.args==(Path('/content/tiny-perceptron-vlm'),)
    checks['colab_bootstrap_routing']={'notebook_sha256':sha256((ROOT/'notebooks/01/1.1.ipynb').read_bytes()).hexdigest(),'in_colab':scope['in_colab'],'subprocess_calls':routing,'requested_chdir':str(change.call_args.args[0]),'scope':'Simulated import of google.colab and missing /content checkout; subprocess.run and os.chdir were spies. No remote clone, pip install, account login, or Colab runtime was used.'}
    record['completed_at']=datetime.now(timezone.utc).isoformat()
    (DEST/'supplement-execution-v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))
