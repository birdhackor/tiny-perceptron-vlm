"""Actual callback: reread current section, match all quoted outputs to raw IDs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=BASE.parents[3]
sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer

sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
spec=importlib.util.spec_from_file_location('section_facts_callback',ROOT/'docs/review-tools/section_facts.py')
facts=importlib.util.module_from_spec(spec);spec.loader.exec_module(facts)
body,whole,first=facts.original_section(ROOT/'course/chapters/08.md','8.4')
assert sha(body)=='8bb856a9eb9becb816c3dba16cb770702c5d3a2724cbc65165a3a2ff5dd0b909'
(OUT/'section.md').write_bytes(body)
oldbody=(BASE/'section.md').read_bytes()
before=b'| `style=json` | `{"answer":3}` |'
after=b'| `style=json` | `{"answer": 3}` |'
assert oldbody.count(before)==1
assert oldbody.replace(before,after)==body
newfence=facts.fences(body,first)[0]['raw']
assert newfence==(BASE/'fence-1.py').read_bytes()
(OUT/'fence-1.py').write_bytes(newfence)
dump('change-proof.json',{'initial_section_sha256':sha(oldbody),'current_section_sha256':sha(body),
    'old_line':before.decode(),'new_line':after.decode(),'only_change':'One ASCII space after colon in main actual-output table.',
    'fence_unchanged':True,'fence_sha256':sha(newfence),'figures':[],'first_line':first,
    'newline_policy':'Read source bytes, no newline normalization.'})

# Check every old artifact hash before reusing initial actual CPU execution.
initial_archive=ROOT/'docs/technical-reviews/history/phase4-8_4-own-initial-revise-ec78152c3ec559aa128b90005249fbaf27b13028c69d5887ed1274f43ea6d74c.json'
assert sha(initial_archive.read_bytes())=='ec78152c3ec559aa128b90005249fbaf27b13028c69d5887ed1274f43ea6d74c'
shutil.copyfile(initial_archive,OUT/'own-initial-revise.json')
initial=json.loads(initial_archive.read_bytes())
reused=[]
for artifact in initial['artifacts']:
    p=ROOT/artifact['path']
    assert sha(p.read_bytes())==artifact['sha256'],artifact['id']
    reused.append({'id':artifact['id'],'path':artifact['path'],'sha256':artifact['sha256'],'unchanged':True})
rawpath=ROOT/'docs/course-experiments/results/style.json'
raw=rawpath.read_bytes()
assert raw==(BASE/'inputs/docs/course-experiments/results/style.json').read_bytes()
d=json.loads(raw);tok=ByteTokenizer()
text=body.decode('utf-8')
rows=re.findall(r'^\| `style=(concise|vivid|json)` \| `([^`]+)` \|',text,re.M)
assert len(rows)==6
matches=[]
for style,visible in rows:
    sample=d['results']['prompt_only_comparison_same_weights'][style]['samples'][0]
    ids=sample['generated_ids'];assert ids[-1]==tok.eos_id
    decoded=tok.decode(ids[:-1])
    assert visible==sample['generated']==decoded
    matches.append({'style':style,'displayed':visible,'raw_generated':sample['generated'],
        'raw_generated_ids':ids,'decoded_before_eos':decoded,'exact_utf8_bytes_equal':True})
dump('raw-output-proof.json',{'original_result_path':rawpath.relative_to(ROOT).as_posix(),
    'original_result_sha256':sha(raw),'original_run_revision':d['revision'],
    'locator':'results.prompt_only_comparison_same_weights.{concise,vivid,json}.samples[0].{generated,generated_ids}',
    'same_arithmetic_question':'2+2=?','tables_checked':2,'quoted_outputs_checked':6,
    'json_colon_space_ids':[66,40,59],'all_quoted_outputs_exact':True,'samples':matches})
for source in initial['sources']:
    if source['kind']=='repository_code':assert sha((ROOT/source['path']).read_bytes())==source['sha256']
codechecks=[]
for p in ['scripts/course_experiments/common.py','scripts/course_experiments/text.py','tiny_perceptron/data.py','tiny_perceptron/model.py','tiny_perceptron/tokenization.py']:
    expected=d['code_sha256'][p];assert sha((ROOT/p).read_bytes())==expected
    codechecks.append({'path':p,'sha256':expected,'matches_initial_original_run_source':True})
current_behavior=sha((ROOT/'scripts/course_experiments/behavior.py').read_bytes())
assert current_behavior=='2ccdd39e666bee64eeeddfa9df629af48f8db90fbff549818932cfb27f4db50a'
codechecks.append({'path':'scripts/course_experiments/behavior.py','sha256':current_behavior,
    'scope':'Matches current helper SHA personally obtained in initial review; historical helper snapshot independently remains original-run manifest exact.'})
dump('reused-evidence-hashes.json',{'artifacts':reused,'current_helper_checks':codechecks,
    'execution_reuse':'Initial original-fence and bounded CPU checks are reused only after current fence/helper and all 37 formal artifact SHA verification; not rerun in this callback.',
    'callback_execution':'Only source byte comparison, raw ID decoding, six quoted-output equality checks and evidence SHA validation ran now.'})
dump('environment.json',{'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,
    'device':'cpu','cwd':str(Path.cwd()),'login':'false','training_runs_this_callback':0,
    'model_weights_downloaded_loaded_saved_this_callback':0})
dump('personal-reading.json',{'reviewer_task':'/root/phase4_factual_coordinator/factual_8_4','context':'same independent original technical reviewer callback',
    'date':'2026-10-05','current_section_read':'Personally reread full current8.4 including both output tables, code, exercise and details; tool output delivered complete section.',
    'original_source_read':'Personally read live original JSON metadata and full first probe samples for concise/vivid/json, including generated_ids, expected, eos and actual strings; no third-party judgment used.',
    'finding':'Main table now preserves literal JSON space exactly. Both tables\' six quoted answers match raw generation strings and decoded IDs. Entire section differs solely by this one ASCII space; no new technical claim arose.',
    'limits':'Initial run-code/CPU/source evidence reused by exact hashes. No new neural training or generation. No figure referenced by8.4.'})
print(json.dumps({'current_source_sha256':sha(body),'all_six_table_outputs_exact':True,'only_one_ascii_space_changed':True,
    'reused_formal_artifacts_sha_verified':len(reused),'initial_cpu_examples_rerun':False,'callback_training_runs':0},ensure_ascii=False))
