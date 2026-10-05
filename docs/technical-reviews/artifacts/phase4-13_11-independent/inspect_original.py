import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/workspace/tiny-perceptron-vlm')
BASE=Path(__file__).resolve().parent
records=[]
for file,names in [
 ('tiny_perceptron/posttraining.py',{'bandit_advantage','ppo_clipped_objective','FiniteResponsePolicy','FiniteRewardModel','FiniteValueModel'}),
 ('scripts/course_experiments/posttraining.py',{'_update','_normalized_scores'}),
]:
 p=ROOT/file;raw=p.read_bytes();lines=raw.decode().splitlines();tree=ast.parse(raw)
 entry={'path':file,'sha256':hashlib.sha256(raw).hexdigest(),'inspected_nodes':[]}
 for node in tree.body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
   entry['inspected_nodes'].append({'name':node.name,'lines':[node.lineno,node.end_lineno]})
   print(file,node.name,node.lineno,node.end_lineno)
   print('\n'.join(f'{j+1}: {lines[j]}' for j in range(node.lineno-1,node.end_lineno)))
 records.append(entry)
file='scripts/course_experiments/posttraining.py';p=ROOT/file;raw=p.read_bytes();lines=raw.decode().splitlines()
print(file,'exact calculation branch',315,389)
print('\n'.join(f'{j+1}: {lines[j]}' for j in range(314,389)))
records[-1]['additional_inspected_ranges']=[[315,389]]
(BASE/'original-method-inspection.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
