import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
OUT = Path(__file__).parent
REPO_REVISION = subprocess.run(['git','rev-parse','HEAD'],capture_output=True,check=True,text=True).stdout.strip()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
records=[]
for stage in ['moe-pretrain','moe-sft','moe-native','dense-weighted']:
    source=Path('docs/selftrained/results/training-raw')/stage
    training=json.loads((source/'raw/train-receipt.json').read_text())
    receipt=json.loads((source/'receipt.json').read_text())
    public_stage={'moe-native':'moe-joint','dense-weighted':'dense-joint'}.get(stage,stage)
    manifest_path=OUT/'public-metadata'/public_stage/'inference-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    files={item['path']:item for item in receipt['files']}
    assert receipt['returncode']==0 and receipt['status']=='completed'
    assert training['completed_requested_steps'] and not training['interrupted']
    assert files['best.pt']['sha256']==manifest['selected_checkpoint_sha256']
    for name in ['model.safetensors','model-config.json','tokenizer.json']:
        assert files[name]['sha256']==manifest['files'][name]
    assert files['inference-manifest.json']['sha256']==sha(manifest_path)
    selected={key:training[key] for key in ['stage','architecture','steps','completed_requested_steps','interrupted','test_used_for_selection','selection','stage_history','origin']}
    records.append({'stage':stage,'raw_source':str(source),'raw_sha256':{'receipt.json':sha(source/'receipt.json'),'raw/train-receipt.json':sha(source/'raw/train-receipt.json')},
        'inspected_pointers':{'raw/train-receipt.json':['/'+key for key in selected],
            'receipt.json':['/status','/returncode','/revision','/manifest_sha256','/files','/job/architecture','/job/stage','/job/steps']+(['/job/init_checkpoint'] if 'init_checkpoint' in receipt['job'] else [])},
        'training_values':selected,'execution_values':{key:receipt[key] for key in ['status','returncode','revision','manifest_sha256']},
        'job_selected':{key:receipt['job'][key] for key in ['architecture','stage','steps','init_checkpoint'] if key in receipt['job']},
        'completed_steps':training['steps'],'public_selected_step':manifest['selected_step'],'best_checkpoint_and_all_exports_match':True})
(OUT/'training-inspected-pointers.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
raw_dir=Path('docs/selftrained/results/public-cpu-raw')
scenario_rows=[]
for path in sorted(raw_dir.glob('*-argv.json')):
    value=json.loads(path.read_text());argv=value['argv']
    target=OUT/'inputs'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    assert sha(path)==sha(target)
    scenario_rows.append({'file':str(path),'snapshot':str(target),'sha256':sha(path),'pointers':['/argv','/cwd'],
        'task':argv[argv.index('--task')+1] if '--task' in argv else 'history_append',
        'chat':any(x.endswith('scripts/selftrained/chat.py') for x in argv),
        'device':argv[argv.index('--device')+1] if '--device' in argv else None})
scenario={'records':scenario_rows,'chat_invocations':sum(row['chat'] for row in scenario_rows),
          'history_append_invocations':sum(not row['chat'] for row in scenario_rows),
          'distinct_chat_tasks':sorted(set(row['task'] for row in scenario_rows if row['chat']))}
assert scenario['chat_invocations']==8 and scenario['history_append_invocations']==1 and len(scenario['distinct_chat_tasks'])==7
(OUT/'scenario-argv-check.json').write_text(json.dumps(scenario,indent=2)+'\n')
text=json.loads((raw_dir/'text-1-stdout.txt').read_text())
argv=json.loads((raw_dir/'text-1-argv.json').read_text())
expected=json.loads(Path('docs/selftrained/examples/v2/text.messages.json').read_text())
assert argv['public_messages_before_execution']==expected
assert text['model']['manifest_sha256']=='f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e'
assert text['public_source']['revision']=='979cdfacc588ad0536f1c64fff96f264571cf054'
current=json.loads((OUT/'current-cli-stdout.json').read_text())
assert current['answer']==text['answer'] and current['generations'][0]['generated_ids']==text['generations'][0]['generated_ids']
result={'repository_revision':REPO_REVISION,'training_exports_verified':4,
        'completion_and_selection':[{key:row[key] for key in ['stage','completed_steps','public_selected_step','best_checkpoint_and_all_exports_match']} for row in records],
        'scenario_chat_invocations':8,'scenario_history_append':1,'scenario_groups':7,
        'historical_text_raw_pointers':['/answer','/generations','/model','/public_source'],
        'historical_text_argv_pointers':['/argv','/cwd','/public_messages_before_execution'],
        'historical_stdout_sha256':sha(raw_dir/'text-1-stdout.txt'),
        'historical_messages_match_current_example':True,'current_answer_and_ids_match_historical_raw':True,
        'author_notes_review_and_scope_correction_read':False}
(OUT/'saved-raw-check-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
