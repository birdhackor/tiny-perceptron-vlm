"""知情重跑原14個案例；唯一fixture改動是提供目前CLI會預讀的checkpoint資料。"""
import hashlib
from pathlib import Path
original=Path(__file__).with_name('engineering-local-cli-probes-original.py')
# Bind to original immutable report, rather than an assumed source digest.
import json
report=json.loads(Path(__file__).parents[1].joinpath('engineering-local-cli-original.json').read_bytes())
reference=next(x for x in report['artifacts'] if x['id']=='probe_code')
assert hashlib.sha256(original.read_bytes()).hexdigest()==reference['sha256']
source=original.read_text()
def replace(old,new):
 global source
 assert source.count(old)==1,old
 source=source.replace(old,new)
replace("caught=None;stdout=io.StringIO();stderr=io.StringIO()", "stored={'metadata':{'seed':99 if kind=='different_resume_recipe' else job['seed'], 'training_recipe':{'batch_size':job['batch_size'],'learning_rate':.002 if kind=='different_resume_recipe' else job['learning_rate']}},'optimizer':{'param_groups':[{'lr':.002 if kind=='different_resume_recipe' else job['learning_rate']}]}}\n            caught=None;stdout=io.StringIO();stderr=io.StringIO()")
replace("patch.object(data_module,'write_dataset',side_effect=write_dataset)", "patch.object(data_module,'write_dataset',side_effect=write_dataset),patch.object(train_module.torch,'load',return_value=stored)")
replace("rejected=kind in ('manifest_mismatch','nonempty_no_manifest','existing_output','remote_resume')", "rejected=kind in ('manifest_mismatch','nonempty_no_manifest','existing_output','remote_resume','different_resume_recipe')")
replace("assert actual[-1]['actual_seed']!=baseline_job['seed'] and actual[-1]['actual_learning_rate']!=baseline_job['learning_rate']", "assert not actual  # current CLI rejects before calling train, unlike the original")
old="result.update(passed=False,expected='reject mismatched job/checkpoint seed or learning_rate',actual='accepted; requested recipe silently replaced by checkpoint seed/optimizer lr',base_passed=True,requested={'seed':baseline_job['seed'],'learning_rate':baseline_job['learning_rate']},observed=actual[-1],scope='Actual CLI mapping plus actual train resume validation prefix; restore/load return internally consistent synthetic metadata/config, no tensor/model/optimizer step.')"
new="result.update(passed=True,expected='reject mismatched job/checkpoint seed or learning_rate',actual='rejected by current CLI before train',base_passed=True,requested={'seed':baseline_job['seed'],'learning_rate':baseline_job['learning_rate']},checkpoint={'seed':99,'learning_rate':.002},scope='Original internally consistent checkpoint mismatch; current CLI load provider explicit fixture. Legal actual train resume-prefix base still passes, no model/optimizer step.')"
replace(old,new)
replace("assert failures==['different_resume_recipe_train-b-v1_cpu']", "assert failures==[]")
replace("'verdict':'revise'", "'verdict':'pass'")
replace("'context':'informed engineering original'", "'context':'informed engineering callback-01'")
replace('engineering-local-cli-probes-results-original.json','engineering-local-cli-probes-results-callback-01-confirmed.json')
exec(compile(source,str(__file__),'exec'),globals())
