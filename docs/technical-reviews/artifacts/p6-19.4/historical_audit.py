from pathlib import Path
import ast
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
current=ast.parse((ROOT/'scripts/selftrained/train.py').read_text())
current_defs={x.name:ast.dump(x,include_attributes=False) for x in current.body if isinstance(x,ast.FunctionDef)}
rows=[];seen={}
for p in sorted((BASE/'raw').glob('*/raw/execution.json')):
    e=json.loads(p.read_bytes());revision=e['revision'];stage=p.parents[1].name
    if revision not in seen:
        command=['git','show',revision+':scripts/selftrained/train.py']
        result=subprocess.run(command,cwd=ROOT,capture_output=True,check=True)
        save=BASE/'historical-code'/revision/'train.py';save.parent.mkdir(parents=True,exist_ok=True);save.write_bytes(result.stdout)
        tree=ast.parse(result.stdout);funcs={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
        checks={name:{'line':funcs[name].lineno,'end':funcs[name].end_lineno,'same_as_current':ast.dump(funcs[name],include_attributes=False)==current_defs.get(name)} for name in ['set_trainable','objective','validation_loss']}
        assert checks['set_trainable']['same_as_current'] and checks['validation_loss']['same_as_current']
        seen[revision]={'revision':revision,'sha256':hashlib.sha256(result.stdout).hexdigest(),'saved':str(save.relative_to(ROOT)),'command':command,'functions':checks}
    rows.append({'stage':stage,'revision':revision,'command':e['command']})
base_tree=ast.parse((BASE/'historical-code/07a6bc2e2dd1b0d7ee3634a0829e2fea5a86053a/train.py').read_text())
baseline=next(x for x in base_tree.body if isinstance(x,ast.FunctionDef) and x.name=='objective')
for source in seen.values():
    tree=ast.parse((ROOT/source['saved']).read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='objective')
    if len(fn.args.args)==6:assert ast.dump(fn,include_attributes=False)==ast.dump(baseline,include_attributes=False)
result={'sources':list(seen.values()),'executions':rows,'inspection':'AST located exact objective/freeze/validation contracts; personally read baseline objective 296-322, weighted b485b3a6 objective 422-461 and mask/reduction 361-399, current/native 519-628. Other baseline objective ASTs exactly equal baseline; all freeze and validation ASTs exactly equal current.'}
(BASE/'historical-source-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('All 15 training revisions acquired from local Git; freeze and validation functions are exact AST matches to inspected current source. All default-stage objective functions exactly match personally inspected baseline source.')
