"""Read-only owner callback for freeze16 19.13; retain original reports/evidence unchanged."""
import difflib
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
from scripts.reading_time import lesson_slices, source_sha256

BASE=Path('docs/technical-reviews/artifacts/revision-20261016/backfill')
def read(path):
    return json.loads(Path(path).read_bytes())
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
original_path=Path('docs/course-revision-20261016/reviews/technical-19.13-backfill-original-01.json')
original=read(original_path)
old_execution=read(BASE/'19.13-execution.json')
old_seal=read(BASE/'19.13-self-check.json')
assert all(sha(p)==h for p,h in old_seal['sealed_sha256'].items())
entry=next(p for p in read('docs/course-revision-20261016/freeze-16.json')['pages'] if p['page_id']=='19.13')
new=Path(entry['snapshot']).read_text()
old=Path('docs/course-revision-20261016/freeze-10/sources/19.13.md').read_text()
current=lesson_slices(Path('course/chapters/19.md').read_text())['19.13']
assert new==current and source_sha256(new)==entry['source_sha256']
expected=old.replace('yes／no機率；建議澄清與其依據','yes／no機率、選中類別與集中度').replace('共同介面會讓人選成品；','三個成品各有自己的推論入口，使用者依任務選擇；')
assert new==expected, 'Unexpected changes outside the two reviewed replacements'
source_checks={s['path']:{'original_sha256':s['sha256'],'current_sha256':sha(s['path']),'same_bytes':sha(s['path'])==s['sha256']} for s in original['sources'] if s.get('path') and s['id']!='s_frozen'}
assert all(v['same_bytes'] for v in source_checks.values())
commands=[
 ['.venv/bin/python','scripts/selftrained/chat.py','--help'],
 ['.venv/bin/python','-m','tiny_perceptron.lesson_products','solve','--b','outputs/lesson-products/products/b','--state','兩盒追蹤;A=2;B=6;事件=A設5,A設1;查詢=A','--loops','3'],
 ['.venv/bin/python','-m','tiny_perceptron.lesson_products','judge','--c','outputs/lesson-products/products/c','--state','卡片進度;申請編號=?','--question','info'],
]
outputs=[]
for command in commands:
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    outputs.append({'argv':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
b=json.loads(outputs[1]['stdout']); c=json.loads(outputs[2]['stdout'])
assert b['answer']==1 and b['loops']==3
assert c['selected']=='no' and set(c['probabilities'])=={'yes','no'}
assert abs(sum(c['probabilities'].values())-1)<1e-6 and 0<=c['concentration']<=1
assert '--messages' in outputs[0]['stdout'] and '--effort' not in outputs[0]['stdout']
settings=read('docs/lesson-products/results/settings-freeze-v1.json')
product_files={f'{p}/{name}':{'sha256':sha(f'outputs/lesson-products/products/{p}/{name}'),'same_bytes':sha(f'outputs/lesson-products/products/{p}/{name}')==expected} for p,files in settings['products'].items() for name,expected in files.items()}
assert all(v['same_bytes'] for v in product_files.values())
primary=read('docs/selftrained/results/public-cpu-raw/ocr-1-result.json')
assert primary['returncode']==0 and primary['chat'] is True and primary['actual_answer']=='大小'
raw_predictions='outputs/lesson-products/final-report-v1.predictions.jsonl'
assert sha(raw_predictions)==old_execution['existing_predictions']['sha256']
new_sources=['scripts/selftrained/chat.py','tiny_perceptron/selftrained/inference.py','docs/selftrained/results/public-cpu-raw/ocr-1-result.json','docs/selftrained/results/public-cpu-raw/text-1-argv.json']
record={
 'reviewer_task':'/root/backfill_tech_19_13','reviewer_context_at_initial_assignment':'fresh','current_context':'informed',
 'command':'uv run --frozen --extra cpu python docs/technical-reviews/artifacts/revision-20261016/backfill/19.13-callback-probe-01.py',
 'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','platform':platform.platform()},
 'freeze':{'path':'docs/course-revision-20261016/freeze-16.json','snapshot':entry['snapshot'],'source_sha256':source_sha256(new),'snapshot_sha256':sha(entry['snapshot']),'current_equals_snapshot':True,'only_two_expected_replacements':True},
 'complete_section_reread':'Reviewer reread all freeze16 19.13 text, not only the diff. No additional substantive changes outside the two fixes were found.',
 'diff_from_original':''.join(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),fromfile='freeze15/19.13',tofile='freeze16/19.13')),
 'original_report_path':str(original_path),'original_report_sha256':sha(original_path),'all_original_sealed_files_unchanged':True,
 'prior_sources_same_bytes':source_checks,'product_files_same_bytes':product_files,
 'existing_raw_predictions_same_sha256':sha(raw_predictions),
 'newly_executed_commands':outputs,
 'c_table_checks':{'actual_keys':sorted(c),'selected':c['selected'],'probabilities':c['probabilities'],'concentration':c['concentration'],'sum_probabilities':sum(c['probabilities'].values()),'concentration_is_not_correctness_probability':c['concentration_scope']},
 'independent_entry_checks':{'a':'scripts/selftrained/chat.py --help runs; code directly constructs InferenceAssistant and calls reply, no B/C entry dependency; actual generation evidence remains existing primary OCR receipt.','b':'solve command above loads only --b and returns answer=1, loops=3.','c':'judge command above loads only --c and returns probabilities, selected class and concentration.','http':'Reread delivery.serve POST /solve and /judge dispatch directly to B and C; this is not a new HTTP rerun.'},
 'a_existing_primary_receipt':{'path':'docs/selftrained/results/public-cpu-raw/ocr-1-result.json','sha256':sha('docs/selftrained/results/public-cpu-raw/ocr-1-result.json'),'returncode':primary['returncode'],'actual_answer':primary['actual_answer'],'model_sha256':primary['model']['files']['model.safetensors'],'public_source':primary['public_source'],'scope':'Read existing raw execution result, not a new A generation.'},
 'newly_inspected_source_sha256':{p:sha(p) for p in new_sources},
 'not_rerun':'Initial random tensor reconstruction, hook depth checks, all formal-test counts, training receipts/selected checkpoint digests, anonymous network download and A model generation are retained from original 19.13 evidence after same-byte checks. No training, parameter update, formal-test rerun, paid service or remote write.',
 'script_sha256':sha(__file__),
}
out=BASE/'19.13-callback-execution-01.json';out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'saved':str(out),'source_sha256':source_sha256(new),'c_output':c,'a_help_exit':outputs[0]['returncode'],'b_answer':b,'original_unchanged':True,'same_byte_sources':len(source_checks)},ensure_ascii=False,indent=2))
