"""The original 19.5 reviewer checks the concrete wording repair and evidence continuity."""
from pathlib import Path
import difflib
import hashlib
import json
import platform
import sys

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def section(raw, identifier, next_id):
    start=raw.index(('## '+identifier+' ').encode())
    end=raw.index(('## '+next_id+' ').encode(),start)
    return raw[start:end]

own_path=ROOT/'docs/technical-reviews/19.5.json'
own=json.loads(own_path.read_text())
assert own['reviewer_task']=='/root/p6_fact_19_5' and own['verdict']=='revise'
initial_path=BASE/'initial-report-revise.json'
if initial_path.exists():
    assert initial_path.read_bytes()==own_path.read_bytes()
else:
    initial_path.write_bytes(own_path.read_bytes())
initial_section=(BASE/'section.md').read_bytes()
current_chapter=(ROOT/'course/chapters/19.md').read_bytes()
current_section=section(current_chapter,'19.5','19.6')
old='這個回答同時用了兩種線索：當前問題決定內容，先前要求決定兩點格式。'.encode()
new='這個回答的內容對應當前 App 問題，也符合先前要求的兩點格式。'.encode()
assert initial_section.count(old)==1
assert current_section==initial_section.replace(old,new)
(BASE/'current-section.md').write_bytes(current_section)

artifact_checks=[]
for a in own['artifacts']:
    actual=sha(ROOT/a['path'])
    assert actual==a['sha256'],a['id']
    artifact_checks.append({'id':a['id'],'path':a['path'],'sha256':actual,'unchanged':True})
code_checks=[]
for s in own['sources']:
    if s['kind']=='repository_code':
        actual=sha(ROOT/s['path'])
        assert actual==s['sha256'],s['id']
        code_checks.append({'id':s['id'],'path':s['path'],'sha256':actual,'unchanged':True})
raw_checks=[]
for p in (BASE/'raw').rglob('*'):
    if p.is_file():
        relative=p.relative_to(BASE/'raw')
        assert sha(p)==sha(ROOT/relative),str(relative)
        raw_checks.append({'original':str(relative),'sha256':sha(p),'unchanged':True})
old_chapter=(BASE/'chapter-frozen-input.md').read_bytes()
dependencies={identifier:section(old_chapter,identifier,next_id)==section(current_chapter,identifier,next_id) for identifier,next_id in [('19.3','19.4'),('19.4','19.5')]}
old_19_3=section(old_chapter,'19.3','19.4')
new_19_3=section(current_chapter,'19.3','19.4')
old_ocr='同一畫布的區域問題及訓練增強版本不跨份'.encode()
new_ocr='同一來源家族的區域問題及訓練增強版本留在同一份'.encode()
assert new_19_3==old_19_3.replace(old_ocr,new_ocr)
(BASE/'reinspection-dependency-19.3.md').write_bytes(new_19_3)
manifest=json.loads((ROOT/'docs/selftrained/v2-manifest.json').read_text())
data_checks=[]
for r in manifest['records']:
    p=ROOT/'outputs/selftrained-v2/data'/r['path']
    actual=sha(p)
    assert actual==r['sha256'] and p.stat().st_size==r['bytes']
    data_checks.append({'path':str(p.relative_to(ROOT)),'sha256':actual,'bytes':p.stat().st_size,'unchanged':True})
result=json.loads((ROOT/'docs/selftrained/results/public-cpu-raw/text-1-result.json').read_text())
messages=json.loads((ROOT/'docs/selftrained/examples/v2/text.messages.json').read_text())
answer='1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。'
assert result['actual_answer']==answer
assert result['generation_count']==1 and result['returncode']==0
assert result['actual_generations'][0]['prompt_messages']==messages
assert messages[-1]['role']=='user' and messages[-1]['content']=='App錯誤訊息一直重複出現。'
assert messages[1]['content']=='接下來請用兩點回答。'
assert result['actual_generations'][0]['raw_output']==answer
assert result['actual_generations'][0]['generated_ids'][-1]==2 and result['actual_generations'][0]['eos']
payload={
    'reviewer_task':'/root/p6_fact_19_5',
    'python':sys.version,'platform':platform.platform(),'device':'cpu',
    'initial_report_sha256':sha(initial_path),
    'initial_section_sha256':sha(BASE/'section.md'),
    'current_section_sha256':sha(BASE/'current-section.md'),
    'section_change':'Exactly the originally requested one-sentence replacement; all other 19.5 bytes identical.',
    'current_full_section_read':True,
    'dependency_sections_unchanged':dependencies,
    'changed_dependency_read':'Read current 19.3 completely; only the OCR family wording changed from canvas to source family. The text/dialogue and calculator family rules relied on by 19.5 are byte-identical, and 19.4 is entirely unchanged.',
    'artifact_hash_checks':artifact_checks,
    'repository_source_hash_checks':code_checks,
    'original_raw_hash_checks':raw_checks,
    'fixed_data_hash_checks':data_checks,
    'rechecked_original_result_pointers':['/actual_answer','/generation_count','/returncode','/actual_generations/0/prompt_messages','/actual_generations/0/raw_output','/actual_generations/0/generated_ids','/actual_generations/0/eos'],
    'observed_answer':answer,
    'support':'Content corresponds to the App question and the two-line numbered format satisfies the previous request. The revised sentence states correspondence, not isolated causal usage.',
    'new_model_generations':0,'training_steps_run':0,'heldout_rows_rescored':0,
    'result':'Original issue i1 resolved by the exact requested wording change; previous independent evidence reused only after full SHA checks.',
}
(BASE/'reinspection-output.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in payload.items() if not k.endswith('_checks')},ensure_ascii=False,indent=2))
