"""本機CLI原始離線審核；mock train mapping，真resume驗證prefix零訓練步。"""
import ast
import contextlib
import copy
import hashlib
import importlib
import io
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
ART=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from scripts.lesson_products import train_local as cli
train_module=importlib.import_module('tiny_perceptron.lesson_products.train')
data_module=importlib.import_module('tiny_perceptron.lesson_products.data')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started=datetime.now(UTC).isoformat()
    names=['scripts/lesson_products/train_local.py','scripts/lesson_products/modal_runner.py','tiny_perceptron/lesson_products/train.py','tiny_perceptron/lesson_products/training.py',cli.DATA_MANIFEST]
    before={name:sha(ROOT/name) for name in names}
    results=[]
    with tempfile.TemporaryDirectory(dir=ROOT/'work',prefix='local-cli-audit-') as folder:
        work=Path(folder); count=0
        def run(kind,job_path='docs/lesson-products/jobs/train-b-v1.json',device='cpu',resume=None,train_override=None):
            nonlocal count
            count+=1;case=work/str(count);case.mkdir();data=case/'data';out=case/'out'
            job=json.loads((ROOT/job_path).read_bytes())
            if kind=='remote_resume':job['resume']={'run_id':'old','source':'volume','hf_revision':None,'bundle_sha256':'a'*64}
            jobfile=case/'job.json';jobfile.write_text(json.dumps(job))
            if kind not in ('generate','nonempty_no_manifest'):
                data.mkdir();(data/'manifest.json').write_bytes((ROOT/cli.DATA_MANIFEST).read_bytes())
            if kind=='manifest_mismatch':(data/'manifest.json').write_text('{}')
            if kind=='nonempty_no_manifest':data.mkdir();(data/'retain.txt').write_text('must retain')
            if kind=='existing_output':out.mkdir();(out/'retain.txt').write_text('must retain')
            initial={p.relative_to(case).as_posix():p.read_bytes() for p in case.rglob('*') if p.is_file()}
            calls=[];writes=[]
            def write_dataset(path,**kwargs):
                writes.append(kwargs);path.mkdir(parents=True,exist_ok=True)
                (path/'manifest.json').write_bytes((ROOT/cli.DATA_MANIFEST).read_bytes())
            def train(*args,**kwargs):
                calls.append({'args':args,'kwargs':kwargs})
                if train_override:return train_override(*args,**kwargs)
                return {'product':job['product'],'last_step':20}
            argv=['train_local','--job',str(jobfile),'--data',str(data),'--output',str(out),'--device',device]
            if resume:argv+=['--resume-from',str(resume)]
            caught=None;stdout=io.StringIO();stderr=io.StringIO()
            with patch.object(sys,'argv',argv),patch.object(data_module,'write_dataset',side_effect=write_dataset),patch.object(train_module,'train',side_effect=train),contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                try:cli.main()
                except SystemExit as error:caught=error.code
            rejected=kind in ('manifest_mismatch','nonempty_no_manifest','existing_output','remote_resume')
            if rejected:
                assert caught==2 and not calls
                assert {p.relative_to(case).as_posix():p.read_bytes() for p in case.rglob('*') if p.is_file()}==initial
            else:
                assert caught is None and len(calls)==1
                args,kwargs=calls[0]['args'],calls[0]['kwargs']
                assert args==(data,out,job['product'],job['mode'])
                expected={key:job[source] for key,source in [('max_steps','max_steps'),('max_seconds','train_seconds'),('batch_size','batch_size'),('learning_rate','learning_rate'),('seed','seed'),('depth_weights','depth_weights')]}
                expected.update(device=device,resume=resume/'last.pt' if resume else None,resume_best=resume/'best.pt' if resume else None)
                assert kwargs==expected
                if kind=='generate':
                    config=json.loads((ROOT/cli.DATA_MANIFEST).read_bytes())['config']
                    assert writes==[{key:config[key] for key in ('per_length','stress_per_length','services')}]
                else:assert not writes
            return {'id':kind+'_'+Path(job_path).stem+'_'+device,'passed':True,'train_calls':len(calls),'generated':bool(writes),'exit_code':caught,'stdout':stdout.getvalue().strip(),'stderr':stderr.getvalue().strip()}
        for job in ('train-b-v1','train-c-v1','mechanism-fixed3-v1','mechanism-odd135-v1'):
            results.append(run('exact_argmapping',f'docs/lesson-products/jobs/{job}.json'))
        for device in ('mps','cuda'):results.append(run('exact_argmapping',device=device))
        for kind in ('generate','manifest_mismatch','nonempty_no_manifest','existing_output','remote_resume'):results.append(run(kind))
        resumed=work/'explicit-resume';resumed.mkdir();(resumed/'best.pt').write_text('synthetic');(resumed/'last.pt').write_text('synthetic')
        results.append(run('resume_pair_argmapping',resume=resumed))

        # Execute exact existing train validation prefix, stopping before model/data work.
        source=(ROOT/'tiny_perceptron/lesson_products/train.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='train')
        cut=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and ast.unparse(n)=='config = model.config')
        node=copy.deepcopy(node);node.body=node.body[:cut]+ast.parse("return {'product': product, 'last_step': start_step, 'actual_seed': seed, 'actual_learning_rate': learning_rate}").body
        module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
        namespace=train_module.__dict__.copy();exec(compile(module,'actual_train_resume_prefix','exec'),namespace);prefix=namespace['train']
        checkpoint_seed=99;checkpoint_lr=.002;metadata={};best={}
        def restore(path,product,device,learning_rate):
            output=Path(path).parent.parent/'unused'
            # Tokenizer is saved by the actual prefix in the output chosen below.
            tokenizer_sha=sha(current_output/'tokenizer.json')
            config=train_module.Config(len(train_module.Tokenizer()),width=128,ff=256)
            metadata.clear();metadata.update(product='b',mode='multi',data_manifest_sha256=sha(current_data/'manifest.json'),tokenizer_sha256=tokenizer_sha,depth_weights=[1]*5,initial_weights_sha256='a'*64,step=100,seed=checkpoint_seed,run_identity='consistent-synthetic-run',best_step=100,best_key=[.5,-1.0],best_weights_sha256='c'*64,reference_depth=5,training_recipe={'batch_size':128,'learning_rate':checkpoint_lr})
            best.clear();best.update(metadata=copy.deepcopy(metadata),config=asdict(config),model={'synthetic':'no tensors'})
            return SimpleNamespace(config=config),SimpleNamespace(param_groups=[{'lr':checkpoint_lr}]),None,metadata.copy()
        def prefix_train(data,out,*args,**kwargs):
            nonlocal current_data,current_output
            current_data,current_output=data,out
            with patch.object(train_module.torch,'set_num_threads'),patch.object(train_module.torch,'manual_seed'),patch.object(train_module.torch,'use_deterministic_algorithms'),patch.dict(namespace,{'restore':restore,'state_digest':lambda _: 'c'*64}),patch.object(train_module.torch,'load',side_effect=lambda *a,**k:best):
                observed=prefix(data,out,*args,**kwargs)
            actual.append(observed)
            return observed
        current_data=current_output=None;actual=[]
        # First pass same-checkpoint recipe as explicit legal baseline.
        baseline_job=json.loads((ROOT/'docs/lesson-products/jobs/train-b-v1.json').read_bytes())
        direct_case=work/'resume-valid-base';direct_case.mkdir();current_data=direct_case/'data';current_data.mkdir();(current_data/'manifest.json').write_bytes((ROOT/cli.DATA_MANIFEST).read_bytes());current_output=direct_case/'out'
        with patch.object(train_module.torch,'set_num_threads'),patch.object(train_module.torch,'manual_seed'),patch.object(train_module.torch,'use_deterministic_algorithms'),patch.dict(namespace,{'restore':restore,'state_digest':lambda _:'c'*64}),patch.object(train_module.torch,'load',side_effect=lambda *a,**k:best):
            legal=prefix(current_data,current_output,'b','multi',max_steps=6000,max_seconds=1300,batch_size=128,learning_rate=checkpoint_lr,seed=checkpoint_seed,resume=resumed/'last.pt',resume_best=resumed/'best.pt')
        assert legal['actual_seed']==checkpoint_seed and legal['actual_learning_rate']==checkpoint_lr
        result=run('different_resume_recipe',resume=resumed,train_override=prefix_train)
        assert actual[-1]['actual_seed']!=baseline_job['seed'] and actual[-1]['actual_learning_rate']!=baseline_job['learning_rate']
        result.update(passed=False,expected='reject mismatched job/checkpoint seed or learning_rate',actual='accepted; requested recipe silently replaced by checkpoint seed/optimizer lr',base_passed=True,requested={'seed':baseline_job['seed'],'learning_rate':baseline_job['learning_rate']},observed=actual[-1],scope='Actual CLI mapping plus actual train resume validation prefix; restore/load return internally consistent synthetic metadata/config, no tensor/model/optimizer step.')
        results.append(result)
        help_run=subprocess.run([sys.executable,'-m','scripts.lesson_products.train_local','--help'],cwd=ROOT,capture_output=True,text=True)
        assert help_run.returncode==0 and '--resume-from' in help_run.stdout
        results.append({'id':'module_help','passed':True,'exit_code':0,'stdout':help_run.stdout})
    assert before=={name:sha(ROOT/name) for name in names}
    failures=[r['id'] for r in results if not r['passed']]
    assert failures==['different_resume_recipe_train-b-v1_cpu']
    report={'reviewer_task':'/root/tech_20261016_review_tools','context':'informed engineering original','started_at':started,'completed_at':datetime.now(UTC).isoformat(),'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform()},'checked_files':[{'path':p,'sha256':digest} for p,digest in before.items()],'results':results,'necessary_counterexamples':failures,'verdict':'revise','limits':['No model training, optimizer step, real GPU/MPS, cloud, HF, paid service or external writes. All fixtures in own isolated work directory, no root smoke outputs touched.','Valid resume metadata/config are explicit stub values; actual train validation prefix executes only before config/data/model work. This establishes ignored seed/lr checks without claiming a full trained checkpoint end-to-end result.']}
    path=ART/'engineering-local-cli-probes-results-original.json'
    with path.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'cases':len(results),'necessary_counterexamples':failures,'sha256':sha(path)}))


if __name__=='__main__':main()
