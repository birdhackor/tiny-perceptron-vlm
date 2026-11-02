"""Independent bounded W.1 checks; no final/teacher model inference or training."""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
RUNTIME = Path('/workspace/work/omni-P5-W.1-runtime-v2')
PYTHON = Path('/workspace/tiny-perceptron-vlm/.venv/bin/python')

def digest(p):
    return sha256(Path(p).read_bytes()).hexdigest()

def command(argv, cwd=ROOT, env=None, check=True):
    p = subprocess.run(argv,cwd=cwd,env=env,text=True,capture_output=True,timeout=90)
    result = {'argv':[str(x) for x in argv],'cwd':str(cwd),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    if check and p.returncode:
        raise RuntimeError(json.dumps(result,ensure_ascii=False))
    return result

def server_probe(argv, fixture, env):
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        port=s.getsockname()[1]
    config = Path(env['JUPYTER_CONFIG_DIR'])
    config.mkdir(parents=True,exist_ok=True)
    (config/'jupyter_server_config.json').write_text(json.dumps({'ServerApp':{'allow_root':True,'open_browser':False,'ip':'127.0.0.1','port':port,'port_retries':0},'IdentityProvider':{'token':'w1-local-artificial-probe'}}))
    logfile=RUNTIME / ('server-'+str(port)+'.log')
    with logfile.open('w') as output:
        process=subprocess.Popen(argv,cwd=fixture,env=env,stdout=output,stderr=subprocess.STDOUT,text=True)
        try:
            base=f'http://127.0.0.1:{port}'
            headers={'Authorization':'token w1-local-artificial-probe'}
            def request(path,method='GET'):
                with urllib.request.urlopen(urllib.request.Request(base+path,headers=headers,method=method),timeout=2) as response:
                    raw=response.read()
                    return json.loads(raw) if raw else None
            for _ in range(150):
                try:
                    status=request('/api/status')
                    break
                except Exception:
                    if process.poll() is not None:
                        raise RuntimeError('Jupyter server exited: '+logfile.read_text().replace('w1-local-artificial-probe','REDACTED'))
                    time.sleep(.2)
            else:
                raise RuntimeError('Jupyter server did not become ready')
            notebook=request('/api/contents/01/1.1.ipynb?content=0')
            kernels=request('/api/kernelspecs')['kernelspecs']
            result={'argv':argv,'status':'ready','notebook_path':notebook['path'],'notebook_type':notebook['type'],'kernel_display_name':kernels['tiny-perceptron']['spec']['display_name'],'root_listing_verified':True,'browser_ui_used':False}
            assert result['notebook_type']=='notebook' and result['kernel_display_name']=='Tiny Perceptron'
            request('/api/shutdown','POST')
            process.wait(timeout=15)
            result['returncode']=process.returncode
            return result
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)

def main():
    RUNTIME.mkdir(parents=True,exist_ok=False)
    record={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_w_1','command':'/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/omni-P5-W.1/verify_w1.py','environment':{'python':platform.python_version(),'os':platform.platform(),'torch':importlib.metadata.version('torch'),'jupyterlab':importlib.metadata.version('jupyterlab'),'ipykernel':importlib.metadata.version('ipykernel'),'nbclient':importlib.metadata.version('nbclient'),'device':'CPU; all executed examples are artificial or the character-ID exercise'},'checks':{}}
    checks=record['checks']
    freeze_path=ROOT/'docs/course-revision-20261029-omni/reviews/P5-actual/technical-numbered-source-freeze-v5.json'
    freeze=json.loads(freeze_path.read_text())
    page=next(p for p in freeze['pages'] if p['page_id']=='W.1')
    raw=(ROOT/page['snapshot']).read_bytes()
    text=(ROOT/page['source']).read_text()
    start=text.index('## W.1 '); end=text.index('\n## ',start)+1
    assert text[start:end].encode()==raw
    assert digest(ROOT/page['snapshot'])==page['source_sha256']
    checks['frozen_source']={'snapshot':page['snapshot'],'bytes':len(raw),'sha256':page['source_sha256'],'current_equal':True,'figures':page['figures_sha256'],'numbered_freeze_sha256':digest(freeze_path),'canonical_freeze_sha256':digest(ROOT/freeze['canonical_source_freeze']['path'])}

    index=json.loads((ROOT/'course/lesson-index.json').read_text())
    if isinstance(index,dict):
        index=index['lessons']
    receipt_path=ROOT/'outputs/omni-P5-current-kernel-bundle-v9-receipt.json'
    receipt=json.loads(receipt_path.read_text())
    rows={r['id']:r for r in receipt['rows']}
    from scripts.export_course import checked_notebook, output_html
    total_code=0; teaching_code=0; output_count=0; with_teaching=0; downloaded=0
    signature=[]
    for item in index:
        source=ROOT/item['notebook']
        executed=ROOT/'outputs/omni-P5-current-kernel-bundle-v9'/Path(item['notebook']).relative_to('notebooks')
        nb=json.loads(source.read_text())
        checked=checked_notebook(nb,executed)
        row=rows[item['id']]
        assert digest(source)==row['source']['sha256'] and digest(executed)==row['joined']['sha256']
        download=ROOT/'outputs/omni-P5-preview-v10'/item['notebook']
        assert download.read_bytes()==source.read_bytes()
        downloaded+=1
        codes=[c for c in checked['cells'] if c['cell_type']=='code']
        lessons=[c for c in codes if not c.get('metadata',{}).get('course_setup') and 'course_figure' not in c.get('metadata',{})]
        total_code+=len(codes); teaching_code+=len(lessons); with_teaching+=bool(lessons)
        output_count+=sum(bool(output_html(c.get('outputs',[]))) for c in lessons)
        signature.append([item['id'],digest(source),digest(executed)])
    checks['notebook_and_output_audit']={'lesson_count':len(index),'receipt_sha256':digest(receipt_path),'exact_source_cell_type_and_code_execution_verified':True,'code_cells':total_code,'teaching_code_cells':teaching_code,'lessons_with_teaching_code':with_teaching,'teaching_cells_with_renderable_outputs':output_count,'download_notebook_equal_count':downloaded,'inventory_signature_sha256':sha256(json.dumps(signature,ensure_ascii=False).encode()).hexdigest(),'new_model_forward_calls':0,'scope':'Inspected preserved execution outputs and the current exporter. This audit does not rerun the source bundle or independently prove every scientific claim.'}
    checks['uv_dry_run']=command(['uv','sync','--frozen','--extra','cpu','--group','notebook','--dry-run','--offline'],env={**os.environ,'UV_PROJECT_ENVIRONMENT':str(PYTHON.parent.parent)})
    checks['uv_dry_run']['scope']='Dependency/lock/group resolution only; this review did not download packages or create a fresh full installation.'

    # A clean Git configuration and PATH without git-lfs reproduce the fresh Git-only case.
    source=RUNTIME/'git-source'; source.mkdir()
    clean={**os.environ,'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null','GIT_LFS_SKIP_SMUDGE':'1'}
    command(['git','init','--quiet'],cwd=source,env=clean)
    pointer='version https://git-lfs.github.com/spec/v1\noid sha256:'+'0'*64+'\nsize 123\n'
    (source/'.gitattributes').write_text('*.bin filter=lfs diff=lfs merge=lfs -text\n')
    (source/'data.bin').write_text(pointer)
    command(['git','add','.'],cwd=source,env=clean)
    command(['git','-c','user.name=W1 Reviewer','-c','user.email=w1@example.invalid','commit','--quiet','-m','artificial LFS pointer'],cwd=source,env=clean)
    bin_dir=RUNTIME/'git-only-bin'; bin_dir.mkdir()
    import shutil
    for name in ('git','sh'):
        (bin_dir/name).symlink_to(shutil.which(name))
    clone_env={**clean,'PATH':str(bin_dir)}
    absent=command(['git','lfs','version'],cwd=RUNTIME,env=clone_env,check=False)
    assert absent['returncode']!=0
    clone=command(['git','clone',str(source),'tiny-perceptron-vlm'],cwd=RUNTIME,env=clone_env)
    assert (RUNTIME/'tiny-perceptron-vlm/data.bin').read_text()==pointer
    checks['git_only_clone']={'git_version':command(['git','--version'])['stdout'].strip(),'git_lfs_absent_returncode':absent['returncode'],'clone':clone,'pointer_retained':True,'scope':'Local artificial repository, fresh Git config, no git-lfs on PATH. No public repository clone or network LFS download was performed.'}

    fixture=RUNTIME/'fixture'; fixture.mkdir()
    command([str(PYTHON),'-m','venv','--without-pip','.venv'],cwd=fixture)
    prefix=fixture/'.venv'
    site=next((prefix/'lib').glob('python*/site-packages'))
    source_site=next((PYTHON.parent.parent/'lib').glob('python*/site-packages'))
    (site/'review-existing-packages.pth').write_text(str(source_site)+'\n')
    # Recreate the two installed console-entry-point launchers in the private prefix.
    # The first attempt inherited packages through .pth but had no console launchers.
    entries={entry.name:entry for entry in importlib.metadata.entry_points(group='console_scripts')}
    for name in ('jupyter','jupyter-lab'):
        module, attr=entries[name].value.split(':')
        script=prefix/'bin'/name
        script.write_text(f'#!{prefix}/bin/python\nfrom {module} import {attr}\nraise SystemExit({attr}())\n')
        script.chmod(0o755)
    for name in ('scripts','notebooks','course','tiny_perceptron','pyproject.toml'):
        (fixture/name).symlink_to(ROOT/name,target_is_directory=(ROOT/name).is_dir())
    env={**os.environ,'JUPYTER_CONFIG_DIR':str(RUNTIME/'jupyter-config'),'JUPYTER_RUNTIME_DIR':str(RUNTIME/'jupyter-runtime'),'JUPYTER_DATA_DIR':str(prefix/'share/jupyter'),'PYTHONPATH':str(ROOT)}
    checks['activated_python']=command(['/bin/bash','-c','source .venv/bin/activate\npython -c "import sys; print(sys.prefix)"'],cwd=fixture,env=env)
    assert str(prefix) in checks['activated_python']['stdout']
    checks['check_env']=command([str(prefix/'bin/python'),'scripts/check_env.py'],cwd=fixture,env=env)
    assert 'compute         OK（cpu）' in checks['check_env']['stdout'] and '未安裝' not in checks['check_env']['stdout']
    checks['kernel_install']=command([str(prefix/'bin/python'),'-m','ipykernel','install','--sys-prefix','--name','tiny-perceptron','--display-name','Tiny Perceptron'],cwd=fixture,env=env)
    spec=json.loads((prefix/'share/jupyter/kernels/tiny-perceptron/kernel.json').read_text())
    assert spec['argv'][0]==str(prefix/'bin/python') and spec['display_name']=='Tiny Perceptron'
    checks['kernel_install']['kernel_json']={'argv':spec['argv'],'display_name':spec['display_name']}
    checks['activated_jupyterlab']=server_probe(['/bin/bash','-c','source .venv/bin/activate\njupyter lab notebooks'],fixture,env)
    checks['direct_jupyterlab']=server_probe([str(prefix/'bin/python'),'-m','jupyterlab','notebooks'],fixture,env)
    exercise_code='''
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
def output(nb):
    return ''.join(''.join(o.get('text',[])) for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='stream')
p=Path('notebooks/01/1.1.ipynb')
original=nbformat.read(p,as_version=4)
def run(nb):
    return NotebookClient(nb,timeout=45,kernel_name='tiny-perceptron',resources={'metadata':{'path':str(p.parent.resolve())}}).execute()
old=output(run(original))
modified=nbformat.read(p,as_version=4)
matches=0
for c in modified.cells:
    if c.cell_type=='code' and 'text = "貓看狗，狗看貓。"' in c.source:
        c.source=c.source.replace('text = "貓看狗，狗看貓。"','text = "鳥看狗，狗看鳥。"')
        matches+=1
new=output(run(modified))
assert matches==1 and old.splitlines()[-1]=='貓看狗，狗看貓。' and new.splitlines()[-1]=='鳥看狗，狗看鳥。'
setup=next(c for c in original.cells if c.get('metadata',{}).get('course_setup'))
warmup=nbformat.v4.new_notebook(cells=[setup,nbformat.v4.new_code_cell('values = [2, 4]\\ntotal = 0\\nfor value in values:\\n    total += value\\nprint(total)')])
warmup_text=output(run(warmup))
assert warmup_text=='6\\n'
state=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in ['x = 1','y = x + 1','x = 10','print(y)','y = x + 1\\nprint(y)']])
state_text=output(run(state))
assert state_text=='2\\n11\\n'
print(json.dumps({'original_stdout':old,'modified_stdout':new,'matching_input_cells':matches,'warmup_pasted_cell_stdout':warmup_text,'persistent_state_stdout':state_text,'isolated_fresh_kernels':4,'notebook_saved_or_changed':False},ensure_ascii=False))
'''
    result=command([str(prefix/'bin/python'),'-c',exercise_code],cwd=fixture,env=env)
    checks['notebook_exercise']={'argv':[str(prefix/'bin/python'),'-c','(inline saved in verify_w1.py: exercise_code)'],'returncode':result['returncode'],'result':json.loads(result['stdout']),'stderr':result['stderr'],'scope':'Four artificial/character-ID kernels only; no model, training, retuning, or Colab execution.'}
    record['completed_at']=datetime.now(timezone.utc).isoformat()
    (DEST/'bounded-execution-v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
