from pathlib import Path
import ast,json,sys,tempfile
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import torch
from scripts.course_experiments.run import experiment_spec,execute
from scripts.course_experiments.common import Context
from scripts.course_experiments.architecture import _sft_dataset
out=Path(__file__).parent
result={}
for name in ['modern','moe','efficiency','precision','flash_probe']:
 d=experiment_spec(name);result[name]={k:d[k] for k in ['id','module','function','assets']}
with tempfile.TemporaryDirectory(prefix='t8-entry-') as d:
 p=Path(d);ctx=Context('cpu',p,p,p)
 try:_sft_dataset(ctx)
 except FileNotFoundError as e:result['required_sft_dataset']={'expected_error':str(e)}
 else:raise AssertionError('missing required sft dataset was accepted')
 try:ctx.dependency('sft')
 except FileNotFoundError as e:result['required_sft_model']={'expected_error':str(e)}
 else:raise AssertionError('missing required sft model was accepted')
 try:execute('flash_probe','cuda',p,p,p)
 except RuntimeError as e:result['no_silent_cpu_fallback']={'expected_error':str(e)}
 else:raise AssertionError('CPU wheel accepted CUDA run')
# Execute the original non-CUDA FP16 guard without the surrounding 200-step training workflow.
src=(ROOT/'scripts/course_experiments/architecture.py').read_text();tree=ast.parse(src);fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_precision');loop=next(n for n in fun.body if isinstance(n,ast.For));branch=next(n for n in loop.body if n.lineno==1009)
wrapper=ast.For(target=ast.Name(id='_once',ctx=ast.Store()),iter=ast.List(elts=[ast.Constant(value=1)],ctx=ast.Load()),body=[branch],orelse=[]);module=ast.Module(body=[wrapper],type_ignores=[]);ast.fix_missing_locations(module)
ns={'torch':torch,'dtype':torch.float16,'device_type':'cpu','results':{},'name':'fp16'};exec(compile(module,'original-run_precision-fp16-guard','exec'),ns);assert ns['results']['fp16']['status']=='not_run';result['actual_fp16_cpu_guard']=ns['results']['fp16']
(out/'entry_contract_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
