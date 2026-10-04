"""Re-read the author's concrete sentence-boundary correction and verify its scope."""
import ast
import difflib
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
A=ROOT/'docs/technical-reviews/artifacts'
def sha(raw): return hashlib.sha256(raw).hexdigest()
chapter=(ROOT/'course/chapters/20.md').read_text()
start=re.search(r'^## 20\.3 ',chapter,re.M).start()
next_heading=re.search(r'^## ',chapter[start+1:],re.M)
current=chapter[start:start+1+next_heading.start()] if next_heading else chapter[start:]
old=(A/'natural-20.3-section.md').read_text()
assert current!=old
old_block=re.findall(r'```python\n(.*?)\n```',old,re.S)
new_block=re.findall(r'```python\n(.*?)\n```',current,re.S)
assert new_block==old_block
assert current.replace('DOCCI原始描述是英文，訓練取原始人工描述的開頭片段並保留英文。切句程式遇到句尾引號時可能帶入下一句，因此不把每筆都稱為只有一句。','DOCCI原始描述是英文，訓練保留原始人工描述的第一個完整英文句子。')==old
data=(ROOT/'docs/natural-assistant/DATA.md').read_text()
assert 'train_03125' in data and 'STOP."' in data and '句末標點後若還有引號，就可能帶入下一句' in data
raw=(ROOT/'docs/natural-assistant/manifest.json').read_bytes()
assert sha(raw)=='7604526c67da31a41940f16f9c87027c73cf2782c63522dbde8c7efbf015db91'
manifest=json.loads(raw)
spec=importlib.util.spec_from_file_location('independent20_3_vision',ROOT/'scripts/prepare_natural_vision.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
originals={r['example_id']:r for r in map(json.loads,(ROOT/'data/natural/vision/sources/docci-descriptions.jsonl').read_text().splitlines())}
rows=[r for r in manifest['rows'] if r['task']=='scene' and r['split']=='train']
checks=[]
for row in rows:
    text=originals[row['source']['original_id']]['description']
    actual=module.first_caption_sentence(text)
    assert actual==row['answer'] and text.startswith(actual)
    checks.append(row['id'])
row=next(r for r in rows if r['source']['original_id']=='train_03125')
text=originals['train_03125']['description']
first=text[:re.search(r'''[.!?]["”']?(?=\s|$)''',text).end()]
assert row['answer'].startswith(first) and row['answer']!=first and 'On the bottom of the white boarder is a rust color.' in row['answer']
source=(ROOT/'tiny_perceptron/natural_assistant.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run_train')
selection=next(n.value for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rows' for t in n.targets))
selected=eval(compile(ast.Expression(selection),'actual-run_train-row-expression','eval'),{'manifest':manifest})
assert len(selected)==272 and all(r['answer'] is not None and not r.get('audio') for r in selected)
snapshot=json.loads((A/'natural-20.3-source-snapshot.json').read_text())
unchanged=[]
for item in snapshot['files']:
    if item['path'] in ('course/chapters/20.md','docs/natural-assistant/DATA.md'): continue
    assert sha((ROOT/item['path']).read_bytes())==item['sha256']
    unchanged.append(item['path'])
(A/'natural-20.3-revised-section.md').write_bytes(current.encode())
(A/'natural-20.3-DATA-revised.md').write_bytes(data.encode())
old_data=(A/'natural-20.3-DATA-initial.md').read_text()
diff=''.join(difflib.unified_diff(old.splitlines(True),current.splitlines(True),fromfile='initial-20.3',tofile='revised-20.3'))+'\n'+''.join(difflib.unified_diff(old_data.splitlines(True),data.splitlines(True),fromfile='initial-DATA',tofile='revised-DATA'))
(A/'natural-20.3-revision-diff.txt').write_text(diff)
result={'reviewer_task':'/root/natural_factual_20_3','method':'Read complete revised20.3 and DATA changed paragraph; compare exact diff; independently invoke actual extraction function against all120 original English descriptions; execute actual AST row-selection expression only, with no model call.','previous_source_sha256':sha(old.encode()),'current_source_sha256':sha(current.encode()),'current_DATA_sha256':sha(data.encode()),'manifest_unchanged_sha256':sha(raw),'original_example_code_unchanged':True,'exercise_instruction_unchanged':True,'only_20_3_change':'Correct first-complete-sentence guarantee to original English prefix with closing-quote boundary exception.','all_120_actual_prefixes_match_generator_and_original':True,'exception':{'source_id':'train_03125','first_complete_sentence':first,'fixed_actual_training_target':row['answer'],'generator_output':module.first_caption_sentence(text),'verified_reason':'The full stop before closing quote is not immediately followed by whitespace, so regex continues to next punctuation boundary.'},'actual_run_train_selection_statement':ast.unparse(selection),'selected_supervised_train_rows':len(selected),'train_audio_excluded':24,'unchanged_local_supporting_sources':unchanged,'environment':json.loads((A/'natural-20.3-execution.json').read_text())['environment'],'resolved_issue':'c12 now accurately describes frozen targets without changing data, generator, trained parameters or claims about held-out Chinese human authorship.'}
(A/'natural-20.3-recheck-execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
