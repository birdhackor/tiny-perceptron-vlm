from pathlib import Path
from unittest.mock import patch
import contextlib,io,json,sys,types,copy,hashlib,importlib.metadata,tomllib,subprocess
root=Path.cwd();sys.path.insert(0,str(root));out=root/'docs/technical-reviews/artifacts/natural-v4-supplemental/site-validation-publishing'
from scripts import build_course,export_course,check_notebooks,reading_time
from tiny_perceptron.model import ModelConfig,TinyLM
import torch,nbformat
result={'environment':{'python':sys.version,'torch':str(torch.__version__),'device':'cpu','cuda_available':torch.cuda.is_available(),'nbclient':importlib.metadata.version('nbclient'),'ipykernel':importlib.metadata.version('ipykernel'),'zensical':importlib.metadata.version('zensical')}}
result['smoke_parameter_count']=sum(p.numel() for p in TinyLM(ModelConfig(max_length=64)).parameters());assert result['smoke_parameter_count']==31584
record=json.loads((root/'docs/gpu-smoke-result.json').read_text());remote=json.loads((root/'outputs/modal-smoke/remote-result.json').read_text());assert record==remote
result['smoke_record_audit']={'returned_original_record_equal':True,'steps_before_resume':record['resume']['resumed_from'],'final_step':record['resume']['resumed_to'],'additional_updates':record['resume']['resumed_to']-record['resume']['resumed_from'],'recorded_weight_difference':record['resume']['max_weight_difference'],'comparison_scope':'Original CUDA record and implementation audit only; no original GPU checkpoint tensors or GPU replication performed.'}
# Confirm local bootstrap and first-code-cell construction without installing anything.
body='Mechanism prose.\n\n```python\nprint(2+3)\n```\n'; nb=build_course.notebook('probe','Probe',body,root/'course/chapters/01.md'); assert nb['cells'][2]['metadata']['course_setup'];assert ''.join(nb['cells'][3]['source'])=='print(2+3)'
ns={};exec(build_course.BOOTSTRAP,ns);result['local_bootstrap']={'root':str(ns['root']),'in_colab':ns['in_colab'],'torch_imported':str(ns['torch'].__version__)}
commands=[]; google=types.ModuleType('google'); colab=types.ModuleType('google.colab');google.colab=colab
exists=Path.exists
with patch.dict(sys.modules,{'google':google,'google.colab':colab}),patch('pathlib.Path.exists',lambda p:False if str(p)=='/content/tiny-perceptron-vlm' else exists(p)),patch('subprocess.run',lambda args,**kw: commands.append({'args':args,'check':kw.get('check')})),patch('os.chdir',lambda path:None):exec(build_course.BOOTSTRAP,{})
result['colab_controlled_branch']=commands; assert commands[0]['args'][:4]==['git','clone','--depth','1'];assert commands[1]['args'][1:4]==['-m','pip','install'];assert not any(x in commands[0]['args'] for x in ['--branch','--revision']);
# Use genuine notebook execution in two fresh runner kernels, with fixture ROOT confined here.
fixture=out/'kernel-fixture';(fixture/'notebooks').mkdir(parents=True,exist_ok=True);old=check_notebooks.ROOT;check_notebooks.ROOT=fixture
kernels=[]
for ident in ['a','b']:
 path=fixture/'notebooks'/f'{ident}.ipynb'; notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("assert 'review_probe_state' not in globals()\nreview_probe_state=17\nprint('new_namespace',review_probe_state)")]);nbformat.write(notebook,path)
 target=check_notebooks.run_kernel(path,fixture/'executed',30); executed=nbformat.read(target,as_version=4); kernels.append({'notebook':ident,'execution_count':executed.cells[0].execution_count,'outputs':executed.cells[0].outputs})
check_notebooks.ROOT=old;result['two_independent_kernels']=kernels
# Reject stale/unexecuted/error data: do not run whole exporter/build.
original={'cells':[{'cell_type':'code','source':['print(5)'],'metadata':{},'outputs':[],'execution_count':None}]}
valid=copy.deepcopy(original);valid['cells'][0].update(execution_count=1,outputs=[{'output_type':'stream','name':'stdout','text':'5\n'}]);cases={'valid':valid,'stale':copy.deepcopy(valid),'unexecuted':copy.deepcopy(valid),'error':copy.deepcopy(valid)};cases['stale']['cells'][0]['source']=['print(6)'];cases['unexecuted']['cells'][0]['execution_count']=None;cases['error']['cells'][0]['outputs']=[{'output_type':'error','ename':'E','evalue':'x','traceback':[]}];outcomes={}
for label,case in cases.items():
 p=out/'probes'/f'export-{label}.json';p.write_text(json.dumps(case));
 try:export_course.checked_notebook(original,p);outcomes[label]='accepted'
 except ValueError as e:outcomes[label]=str(e)
assert outcomes['valid']=='accepted' and all(outcomes[k]!='accepted' for k in ['stale','unexecuted','error']);result['export_guards']=outcomes
# Check reading-time staleness and completeness on one bounded synthetic page.
p=out/'probes/time-fixture.json';inventory={'pages':[{'page_id':'fixture','source_sha256':'a'*64,'figures_sha256':{}}]};record={'pages':[{'page_id':'fixture','source_sha256':'a'*64,'figures_sha256':{},'minutes_min':1,'minutes_max':2,'reviewer_task':'actual-fixture','reason':'fixture only'}]};p.write_text(json.dumps(record));assert reading_time.load_estimates(p,inventory,require_complete=True);record['pages'][0]['source_sha256']='b'*64;p.write_text(json.dumps(record))
try:reading_time.load_estimates(p,inventory,require_complete=True);raise AssertionError('stale accepted')
except ValueError as e:result['time_stale_rejection']=str(e)
# Help only, never install/build/whole-check.
help_commands=[[sys.executable,'scripts/export_course.py','--help'],[sys.executable,'scripts/check_notebooks.py','--help'],[sys.executable,'-m','zensical','build','--help'],['uv','sync','--help']]
result['helps']=[]
for cmd in help_commands:
 r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0; name='help-'+str(len(result['helps']))+'.txt';(out/'probes'/name).write_text(r.stdout+r.stderr);result['helps'].append({'command':cmd,'returncode':r.returncode,'path':name})
result['project_groups']=tomllib.loads((root/'pyproject.toml').read_text())['dependency-groups']
(out/'software-receipt.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
