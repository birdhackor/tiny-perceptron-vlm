import json,hashlib,ast
from pathlib import Path
r=json.load(open('docs/course-experiments/results/moe.json'))
for p in ['tiny_perceptron/model.py','tiny_perceptron/modern.py']:
 h=hashlib.sha256(Path(p).read_bytes()).hexdigest();print(p,h,h==r['code_sha256'][p]);assert h==r['code_sha256'][p]
hist=Path('docs/technical-reviews/artifacts/p7_technical_d/modern-historical-scripts_course_experiments_architecture.py')
h=hashlib.sha256(hist.read_bytes()).hexdigest();print('historical_architecture_sha',h,h==r['code_sha256']['scripts/course_experiments/architecture.py']);assert h==r['code_sha256']['scripts/course_experiments/architecture.py']
def funcs(p): return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(Path(p).read_text()).body if isinstance(x,ast.FunctionDef)}
a=funcs(hist);b=funcs('scripts/course_experiments/architecture.py')
for n in ['run_moe','_nll','_train','_heldout','_text_dataset','_clone_config']:
 print(n,'AST_same',a[n]==b[n]);assert a[n]==b[n]
