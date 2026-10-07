import ast,json,hashlib,torch
from pathlib import Path
p=Path('docs/course-experiments/results/encoders.json');raw=json.loads(p.read_text());a=raw['results']['audio'];c=a['calibration'];source=Path('scripts/course_experiments/modalities.py');assert hashlib.sha256(source.read_bytes()).hexdigest()==raw['code_sha256'][str(source)]
tree=ast.parse(source.read_text());ns={};exec(compile(ast.Module(body=[f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=='_audio_records'],type_ignores=[]),str(source),'exec'),ns)
rows=ns['_audio_records']()['test'];i=next(i for i,row in enumerate(rows) if row['frequency']==320 and row['amplitude']==.5 and row['seconds']==.12)
logits=torch.tensor(c['test']['logits']);labels=torch.tensor(c['test']['labels']);grid=c['temperature_grid'];vl=torch.tensor(c['validation']['logits']);vy=torch.tensor(c['validation']['labels']);selection=[(t,torch.nn.functional.cross_entropy(vl/t,vy).item()) for t in grid];chosen=min(selection,key=lambda x:x[1])[0]
assert chosen==c['chosen_temperature']==.5
original=logits.softmax(-1);calibrated=(logits/chosen).softmax(-1)
assert torch.equal(original.argmax(-1),calibrated.argmax(-1)) and original[i].argmax().item()==0 and labels[i].item()==1
assert round(original[i].max().item(),4)==.7490 and round(calibrated[i].max().item(),4)==.8990
print(json.dumps({'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'row_index':i,'input_record':rows[i],'classes':a['classes'],'label':labels[i].item(),'prediction':original[i].argmax().item(),'original_confidence':original[i].max().item(),'scaled_confidence':calibrated[i].max().item(),'validation_only_selection':selection,'chosen_temperature':chosen,'argmax_invariant_all_test':True},ensure_ascii=False,indent=2))
