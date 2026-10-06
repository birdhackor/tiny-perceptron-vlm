"""Persist official installed-source and archived training batch comparisons."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def batch_node(raw):
    tree=ast.parse(raw)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RecordEncoder')
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='batch')

base=Path(torch.__file__).parent
torch_matches=[]
for filename,relative in [('functional-installed-commit.py','nn/functional.py'),
    ('module-installed-commit.py','nn/modules/module.py'),('grad_mode-installed-commit.py','autograd/grad_mode.py')]:
    snapshot=OUT/'external'/filename;installed=base/relative
    equal=snapshot.read_bytes()==installed.read_bytes()
    torch_matches.append({'official_snapshot':str(snapshot.relative_to(ROOT)),
        'installed_path':str(installed),'sha256':sha(snapshot),'bytes_identical':equal})
    assert equal

current=batch_node((ROOT/'tiny_perceptron/selftrained/dataset.py').read_bytes())
pipeline=[]
for stage in ['moe-pretrain','moe-sft','moe-vision','moe-ocr','moe-audio','moe-joint','moe-weighted','moe-native']:
    p=ROOT/'docs/selftrained/results/training-raw'/stage/'raw/execution.json'
    j=json.loads(p.read_text());top={k:type(v).__name__ for k,v in j.items()}
    pointers={k:j[k[1:]] for k in ['/revision','/stage','/returncode']}
    commit=pointers['/revision']
    raw=subprocess.run(['git','show',commit+':tiny_perceptron/selftrained/dataset.py'],
        capture_output=True,check=True,cwd=ROOT).stdout
    target=OUT/'pipeline-revisions'/commit/'dataset.py';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    node=batch_node(raw);same=ast.dump(node,include_attributes=False)==ast.dump(current,include_attributes=False)
    pipeline.append({'receipt':str(p.relative_to(ROOT)),'receipt_sha256':sha(p),'top_keys_and_types':top,
        'checked_pointers':pointers,'snapshot':str(target.relative_to(ROOT)),'snapshot_sha256':sha(target),
        'batch_lines':[node.lineno,node.end_lineno],'batch_AST_same_as_current':same})
    assert same
result={'environment':{'python':sys.version,'torch':torch.__version__,'torch_git':torch.version.git_version,'device':'cpu'},
    'official_torch_source_matches':torch_matches,'training_pipeline_batches':pipeline,'all_assertions_passed':True}
(OUT/'source-audit-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'official_torch_files_matched':len(torch_matches),'archived_MoE_stage_batches_matched':len(pipeline),
    'all_assertions_passed':True},indent=2))
