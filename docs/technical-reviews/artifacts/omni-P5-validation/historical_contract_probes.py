"""Prompted historical c03/reader-output checks using tiny artificial CPU programs only."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
AD=Path(__file__).parent
SCRATCH=Path('/workspace/work/reference/omni-P5-validation/historical-contract-v1')

def rec(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    import nbformat
    import torch
    (SCRATCH/'notebooks').mkdir(parents=True,exist_ok=True)
    code={
        '00-good':"import torch\nassert torch.ones(1).device.type == 'cpu'\nassert not torch.cuda.is_available()\nreview_kernel_value=42\nprint('actual CPU', torch.__version__, torch.ones(1).device.type, torch.cuda.is_available())",
        '01-intentional-failure':"assert 'review_kernel_value' not in globals()\nraise ValueError('intentional reviewer failure; no neural model')",
    }
    for name,source in code.items():
        nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]),SCRATCH/'notebooks'/f'{name}.ipynb')
    out=AD/'historical-kernel-contract-v1'
    worker=SCRATCH/'run_original_checker.py'
    worker.write_text("import importlib.util,sys\nfrom pathlib import Path\n"+
        f"s=importlib.util.spec_from_file_location('probe_checker',{str(ROOT/'scripts/check_notebooks.py')!r})\n"+
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"+
        f"m.ROOT=Path({str(SCRATCH)!r})\nsys.argv=['check_notebooks.py','--mode','kernel','--workers','1','--output',{str(out)!r}]\nm.main()\n")
    cmd=[sys.executable,str(worker)];kernel=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    result=json.loads((out/'validation-kernel.json').read_text())
    assert kernel.returncode==1 and result['total']==2 and result['passed']==1 and result['failed']==1
    assert len(result['failures'])==1 and 'intentional reviewer failure' in result['failures'][0]['error']
    done=nbformat.read(out/'00-good.ipynb',as_version=4)
    output=''.join(o.get('text','') for o in done.cells[0].outputs)
    assert 'actual CPU' in output and 'cpu False' in output
    tests=SCRATCH/'test_artificial_status.py'
    tests.write_text("import pytest\ndef test_pass():\n    assert 2+2==4\ndef test_skip():\n    pytest.skip('intentional reviewer skip')\ndef test_fail():\n    assert False, 'intentional reviewer failure'\n")
    env=os.environ.copy();env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
    pcmd=[sys.executable,'-m','pytest','-ra','-q','--noconftest','-c','/dev/null',str(tests)]
    pytest=subprocess.run(pcmd,cwd=SCRATCH,env=env,text=True,capture_output=True)
    assert pytest.returncode==1 and all(x in pytest.stdout for x in ['1 failed','1 passed','1 skipped','intentional reviewer skip'])
    result={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_validation',
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'CPU; no neural model'},
            'scope':'Post-original prompted historical reader/c03 verification. Executes only assert/print and deliberate failures/skips in tiny artificial fixtures. No course final/teacher neural forward or training.',
            'checker_source':rec(ROOT/'scripts/check_notebooks.py'),'fixture_sources':code,
            'kernel_command':cmd,'kernel_returncode':kernel.returncode,'kernel_stdout':kernel.stdout,'kernel_stderr':kernel.stderr,
            'kernel_result':result,'kernel_result_artifact':rec(out/'validation-kernel.json'),'actual_good_output':output,'actual_good_executed_copy':rec(out/'00-good.ipynb'),
            'pytest_command':pcmd,'pytest_returncode':pytest.returncode,'pytest_stdout':pytest.stdout,'pytest_stderr':pytest.stderr,
            'interpretation':'Intentional failures are the expected probe outcome: the checker reports exactly one failed Notebook and writes the same result to validation-kernel.json; pytest-ra shows pass/fail/skip totals and skip reason. The failed Notebook first confirms no variable from the successful kernel, then raises its deliberate error.',
            'limits':['Not a new331Notebook/fullpytest run','No Windows/Colab/student-learning-effect measurement','One success and one synthetic failure verify output/namespace/CPU mechanics only']}
    p=AD/'historical-contract-results-v1.json'
    with p.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'output':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':rec(p)['sha256'],'kernel_total':2,'kernel_passed':1,'kernel_failed':1,'pytest_expected_nonzero':pytest.returncode},ensure_ascii=False))

if __name__=='__main__':main()
