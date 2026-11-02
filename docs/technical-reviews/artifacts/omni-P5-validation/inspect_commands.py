"""Read-only current notebook-output audit and two tiny namespace/kernel probes."""
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).with_name('command-and-notebook-inspection-v1.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(path):
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}


def run(args):
    p=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
    return {'argv':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}


def main():
    import nbformat
    import nbclient
    import ipykernel
    import torch
    recpath=ROOT/'outputs/omni-P5-current-kernel-bundle-v9-receipt.json'
    rec=json.loads(recpath.read_text());errors=[];total_cells=0;executed=0
    for r in rec['rows']:
        s=ROOT/r['source']['path'];j=ROOT/r['joined']['path'];old=ROOT/r['executed_source']['path']
        for path,want in [(s,r['source']['sha256']),(j,r['joined']['sha256']),(old,r['executed_source']['sha256'])]:
            if sha(path)!=want:errors.append([r['id'],'hash',str(path)])
        src=json.loads(s.read_text());joined=json.loads(j.read_text())
        if [(c['cell_type'],c['source']) for c in src['cells']] != [(c['cell_type'],c['source']) for c in joined['cells']]:errors.append([r['id'],'cell/type/source mismatch'])
        for c in joined['cells']:
            if c['cell_type']=='code' and ''.join(c['source']).strip():
                total_cells+=1
                if isinstance(c.get('execution_count'),int):executed+=1
                else:errors.append([r['id'],'unexecuted nonempty code'])
                if any(o['output_type']=='error' for o in c.get('outputs',[])):errors.append([r['id'],'saved error output'])
    assert not errors
    spec=importlib.util.spec_from_file_location('reviewer_notebook_module',ROOT/'scripts/check_notebooks.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    scratch=Path('/workspace/work/reference/omni-P5-validation/synthetic-kernels')
    (scratch/'notebooks').mkdir(parents=True,exist_ok=True)
    mod.ROOT=scratch
    source=scratch/'notebooks/probe.ipynb'
    n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("assert 'review_probe' not in globals()\nreview_probe=1\nprint('fresh namespace', review_probe)")])
    nbformat.write(n,source)
    probes=[]
    for label in ['first','second']:
        target=Path(mod.run_kernel(source,scratch/label,timeout=30))
        done=nbformat.read(target,as_version=4)
        probes.append({'source':receipt(source),'executed':receipt(target),'execution_count':done.cells[0].execution_count,'output':done.cells[0].outputs[0]['text']})
    assert all(r['output']=='fresh namespace 1\n' for r in probes)
    commands=[run([sys.executable,'scripts/build_course.py','--check']),
              run([sys.executable,'-m','ipykernel','--version']),run([sys.executable,'-m','pytest','--version']),run(['git','--version']),
              run(['bash','-c','false; echo $?; true; echo $?'])]
    help_result=run([sys.executable,'-m','pytest','--help'])
    # Only keep the option definition used in the page, not all third-party help text.
    help_lines=help_result['stdout'].splitlines()
    i=next(i for i,x in enumerate(help_lines) if x.strip().startswith('-r, '))
    pytest_excerpt='\n'.join(help_lines[i:i+9])
    result={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_validation',
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'nbclient':nbclient.__version__,'ipykernel':ipykernel.__version__,'device':'CPU'},
            'current_preserved_execution_bundle':receipt(recpath),'notebooks':len(rec['rows']),'nonempty_code_cells':total_cells,'executed_nonempty_code_cells':executed,'errors':errors,
            'scope':'Audited existing execution bytes/source equality for all331. Executed only two synthetic assert/print notebooks in fresh kernels, no neural forwards.',
            'kernel_probes':probes,'commands':commands,'pytest_r_option_excerpt':pytest_excerpt,
            'not_executed':['Full331 notebooks were not rerun','Full pytest was not run because this review forbids final-family neural inference','No Windows PowerShell or Colab runtime execution','No public publishing or full current review-gate assertion']}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)),'sha256':sha(OUT),'bytes':OUT.stat().st_size,'notebooks':len(rec['rows']),'code_cells':executed,'kernel_probes':len(probes),'commands':[{k:v for k,v in x.items() if k!='stderr'} for x in commands]},ensure_ascii=False))


if __name__=='__main__':main()
