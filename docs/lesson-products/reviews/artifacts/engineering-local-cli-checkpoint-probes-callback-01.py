"""真torch.save/load weights_only本機fixture；只mock train，零訓練。"""
import contextlib
import copy
import hashlib
import importlib
import io
import json
import platform
import sys
import tempfile
from datetime import UTC,datetime
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
ART=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from scripts.lesson_products import train_local as cli
import torch
trainer=importlib.import_module('tiny_perceptron.lesson_products.train')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 start=datetime.now(UTC).isoformat();before=sha(ROOT/'scripts/lesson_products/train_local.py');results=[]
 with tempfile.TemporaryDirectory(dir=ROOT/'work',prefix='local-cli-checkpoint-callback-') as temporary:
  work=Path(temporary)
  def run(kind,product='b'):
   case=work/(kind+'-'+product);case.mkdir();data=case/'data';data.mkdir();(data/'manifest.json').write_bytes((ROOT/cli.DATA_MANIFEST).read_bytes())
   job=json.loads((ROOT/f'docs/lesson-products/jobs/train-{product}-v1.json').read_bytes());jobpath=case/'job.json';jobpath.write_text(json.dumps(job))
   resume=case/'resume';resume.mkdir();out=case/'out'
   stored={'metadata':{'seed':job['seed'],'training_recipe':{'batch_size':job['batch_size'],'learning_rate':job['learning_rate']}},'optimizer':{'param_groups':[{'lr':job['learning_rate']}]}}
   def invoke():
    caught=None;calls=[];error=io.StringIO();console=io.StringIO()
    def train(*args,**kwargs):
     calls.append({'args':args,'kwargs':kwargs});return {'product':product,'last_step':1}
    argv=['local','--job',str(jobpath),'--data',str(data),'--output',str(out),'--resume-from',str(resume)]
    with patch.object(sys,'argv',argv),patch.object(trainer,'train',side_effect=train),contextlib.redirect_stdout(console),contextlib.redirect_stderr(error):
     try:cli.main()
     except SystemExit as raised:caught=raised.code
    return caught,calls,error.getvalue()
   torch.save(stored,resume/'last.pt');torch.save(stored,resume/'best.pt')
   code,calls,error=invoke();assert code is None and len(calls)==1
   assert calls[0]['kwargs']['resume']==resume/'last.pt' and calls[0]['kwargs']['resume_best']==resume/'best.pt'
   if kind=='same_recipe':
    results.append({'id':kind+'-'+product,'passed':True,'actual_torch_save_load_weights_only':True,'train_calls':1,'best_and_last_passed':True});return
   broken=copy.deepcopy(stored)
   if kind=='seed':broken['metadata']['seed']+=1
   elif kind=='metadata_lr':broken['metadata']['training_recipe']['learning_rate']=.002
   elif kind=='optimizer_lr':broken['optimizer']['param_groups'][0]['lr']=.002
   elif kind=='coherent_old_lr':broken['metadata']['training_recipe']['learning_rate']=.002;broken['optimizer']['param_groups'][0]['lr']=.002
   elif kind=='metadata_batch':broken['metadata']['training_recipe']['batch_size']+=1
   elif kind=='second_optimizer_group':broken['optimizer']['param_groups'].append({'lr':.002})
   elif kind=='missing_recipe':broken['metadata'].pop('training_recipe')
   elif kind=='nonfinite_optimizer_lr':broken['optimizer']['param_groups'][0]['lr']=float('nan')
   torch.save(broken,resume/'last.pt')
   inventory={p.relative_to(case).as_posix():p.read_bytes() for p in case.rglob('*') if p.is_file()}
   code,calls,error=invoke()
   assert code==2 and not calls and not out.exists()
   assert 'checkpoint' in error
   assert inventory=={p.relative_to(case).as_posix():p.read_bytes() for p in case.rglob('*') if p.is_file()}
   results.append({'id':kind+'-'+product,'base_passed':True,'passed':True,'actual_torch_save_load_weights_only':True,'train_calls':0,'exit_code':code,'files_unchanged':True,'error':error.strip()})
  run('same_recipe','b');run('same_recipe','c')
  for kind in ('seed','metadata_lr','optimizer_lr','coherent_old_lr','metadata_batch','second_optimizer_group','missing_recipe','nonfinite_optimizer_lr'):run(kind)
 assert before==sha(ROOT/'scripts/lesson_products/train_local.py')
 report={'reviewer_task':'/root/tech_20261016_review_tools','context':'informed callback-01','started_at':start,'completed_at':datetime.now(UTC).isoformat(),'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform(),'torch':str(torch.__version__)},'checked_files':[{'path':'scripts/lesson_products/train_local.py','sha256':before}],'results':results,'all_expected':True,'limits':['Real CPU torch serialization of plain fixture checkpoint metadata/optimizer groups; no model/tensor/optimizer step or full checkpoint restore. Train alone is mocked.','Local metadata guard and last/best routing proven; original train remains responsible for full paired weights/config/run_identity validation.','No cloud/paid/GPU/HF/external writes or access to root smoke outputs.']}
 path=ART/'engineering-local-cli-checkpoint-results-callback-01.json'
 with path.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps({'cases':len(results),'all_expected':True,'sha256':sha(path)}))


if __name__=='__main__':main()
