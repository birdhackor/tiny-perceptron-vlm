import ast,json,hashlib
from pathlib import Path
A=Path(__file__).resolve().parent; S=A/'sources'
spans={'generation-config.py':[(107,113),(125,127),(212,214),(266,272),(545,546)],'generation-utils.py':[(1648,1707),(2821,2845)],'stopping-criteria.py':[(47,88),(452,473)],'logits-process.py':[(1615,1671)],'chat-templating.md':[(23,28),(74,84),(129,173)],'cache-explanation.md':[(27,27),(44,58),(76,95)],'pytorch-utils.py':[(324,347)],'vllm-config.py':[(238,239),(254,257)]}
selected={'stopping-criteria.py':['StoppingCriteria','MaxLengthCriteria','EosTokenCriteria'],'logits-process.py':['ForcedEOSTokenLogitsProcessor'],'pytorch-utils.py':['isin_mps_friendly'],'generation-utils.py':['_prepare_generated_length']}
fragments=[]
for name,names in selected.items():
 p=S/name; raw=p.read_bytes();txt=raw.decode();tree=ast.parse(txt)
 for target in names:
  nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==target];assert len(nodes)==1
  n=nodes[0];code=''.join(txt.splitlines(keepends=True)[n.lineno-1:n.end_lineno]); fragments.append({'source_file':name,'source_sha256':hashlib.sha256(raw).hexdigest(),'name':target,'original_start_line':n.lineno,'original_end_line':n.end_lineno,'raw_code':code,'sha256':hashlib.sha256(code.encode()).hexdigest()})
(S/'executed-original-fragments.json').write_text(json.dumps(fragments,ensure_ascii=False,indent=2)+'\n')
for name,ranges in spans.items():
 p=S/name;raw=p.read_bytes();lines=raw.decode().splitlines(keepends=True)
 excerpts=[{'original_start_line':start,'original_end_line':end,'raw_text':''.join(lines[start-1:end])} for start,end in ranges]
 (S/(name+'.excerpts.json')).write_text(json.dumps({'original_file':name,'original_sha256':hashlib.sha256(raw).hexdigest(),'excerpts':excerpts},ensure_ascii=False,indent=2)+'\n')
 p.unlink()
print('Saved exact original source fragments and bounded line excerpts; removed unrelated full snapshots.')
