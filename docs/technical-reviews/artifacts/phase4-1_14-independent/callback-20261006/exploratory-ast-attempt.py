# Exact Python heredoc from the exploratory source inspection that exited 1.
from pathlib import Path
import ast,json
p=Path('scripts/course_experiments/text.py'); raw=p.read_bytes(); tree=ast.parse(raw)
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name=='run_simple_models':
  for node in ast.walk(n):
   if isinstance(node,ast.For) and isinstance(node.target,ast.Tuple):
    print(json.dumps({'locator':'run_simple_models line '+str(node.lineno),'configurations':ast.literal_eval(node.iter)},ensure_ascii=False))
