"""Navigation reinspection and bounded original CLI parser checks only."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import shlex
import sys

OUT=Path(__file__).resolve().parent
OWN=OUT.parent
ROOT=OUT.parents[4]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def section(path,key):
    raw=path.read_bytes(); heads=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
    i=next(i for i,h in enumerate(heads) if h[0].startswith(('## '+key+' ').encode()))
    return raw[heads[i].start():heads[i+1].start() if i+1<len(heads) else len(raw)]

initial=json.loads((OWN/'initial-R.4.json').read_text())
old=(OWN/'source-R.4.md').read_bytes()
new=section(ROOT/'course/README.md','R.4')
assert new==(OUT/'current-R.4.md').read_bytes()
old_line='MoE、PPO、DPO、量化、蒸餾各自回答不同問題，也有互為替代的安排。第一次完成小助理，不必把每個方法都套一次。先看任務需要什麼，再選架構、訓練訊號與交付方式；成品的能力與限制集中在[19.12](chapters/19.md#19.12)，取得模型與重做的步驟放在[操作頁](training.md)。'
new_line='MoE、PPO、DPO、量化、蒸餾各自回答不同問題，也有互為替代的安排。第一次完成小助理，不必把每個方法都套一次。先看任務需要什麼，再選架構、訓練訊號與交付方式；成品的能力與限制集中在[19.12](chapters/19.md#19.12)；已公開合成示範的下載與重做步驟見[19.11](chapters/19.md#19.11)，各局部實驗的操作見[操作頁](training.md)。'
assert old.count(old_line.encode())==1
assert old.replace(old_line.encode(),new_line.encode())==new

unchanged_sources=[]
for source in initial['sources']:
    if source['kind']=='repository_code':
        path=ROOT/source['path']; assert sha(path)==source['sha256']
        unchanged_sources.append({'source_id':source['id'],'path':source['path'],'sha256':source['sha256']})
for key,path in [('19.11','course/chapters/19.md'),('T.3','course/training.md'),('T.11','course/training.md')]:
    current=section(ROOT/path,key)
    assert current==(OUT/('current-'+key+'.md')).read_bytes()
    assert current==(OWN/('navigation-source-'+key+'.md')).read_bytes()

routes=[]
for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',new.decode()):
    relative,sep,anchor=target.partition('#'); path=(ROOT/'course'/relative).resolve()
    assert path.is_file()
    if anchor: assert re.search(r'^## '+re.escape(anchor)+r' ',path.read_text(),re.M)
    routes.append({'label':label,'path':str(path.relative_to(ROOT)),'anchor':anchor,'exists':True})
assert any(r['label']=='19.11' and r['anchor']=='19.11' for r in routes)
assert '各局部實驗的操作見[操作頁](training.md)' in new.decode()
target=(OUT/'current-19.11.md').read_text()
assert '新成品的操作與權重待實作完成後另回填' in target
assert 'python scripts/fetch_capstone.py --stage joint' in target
assert 'python scripts/capstone.py train --stage pretrain' in target
assert 'scripts/fetch_course_models.py --model text_foundation' in (OUT/'current-T.3.md').read_text()

parser_locations={}
def original_parser(relative):
    source=(ROOT/relative).read_text();tree=ast.parse(source)
    main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    i=next(i for i,node in enumerate(main.body) if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='parse_args')
    namespace={'argparse':argparse,'Path':Path,'ROOT':ROOT,'MANIFEST':Path('__review_no_manifest_loaded__'),'__doc__':'Bounded parser-contract check; no manifest/model/trainer invoked.'}
    # Execute only the exact original parser construction statements, stopping
    # before parse_args, file checks, imports/loaders and training/download dispatch.
    parser_module=ast.fix_missing_locations(ast.Module(body=main.body[:i],type_ignores=[]))
    exec(compile(parser_module,relative+':original-parser-only','exec'),namespace)
    parser_locations[relative]={'main_line':main.lineno,'last_parser_statement_line':main.body[i-1].end_lineno,'parse_args_line':main.body[i].lineno,'source_sha256':sha(ROOT/relative),'bound':'Option acceptance only; isolated unused MANIFEST placeholder. No default manifest content, download, weights, training dispatch or scores validated.'}
    return namespace['parser']

fetch=original_parser('scripts/fetch_capstone.py')
capstone=original_parser('scripts/capstone.py')
accepted=[]
for fence in re.findall(r'```bash\n(.*?)```',target,re.S):
    for line in fence.splitlines():
        if not line.startswith(('python scripts/fetch_capstone.py ','python scripts/capstone.py train ')): continue
        tokens=shlex.split(line);parser=fetch if tokens[1]=='scripts/fetch_capstone.py' else capstone
        args=parser.parse_args(tokens[2:]); parsed={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}; accepted.append({'original_recipe_line':line,'parsed':parsed,'verified':'Original CLI parser accepted; action deliberately not invoked.'})
stages={row['parsed']['stage'] for row in accepted if row['parsed'].get('command')=='train'}
assert stages=={'pretrain','sft','joint','dpo'}
assert any(row['parsed'].get('list') for row in accepted)
assert any(row['parsed'].get('stage')=='joint' and 'command' not in row['parsed'] for row in accepted)

environment={'python':sys.version,'python_executable':sys.executable,'device':'cpu (text/AST/argparse only)','cwd':str(ROOT),'scope':'Navigation/source-byte and original parser checks; no network, data/model downloads, GPU, training, new weights or capability scoring.'}
result={'reviewer_task':initial['reviewer_task'],'current_source_sha256':hashlib.sha256(new).hexdigest(),'initial_source_sha256':initial['source_sha256'],'only_R4_change':'Operation destination sentence now separates published synthetic integration19.11 and local experiment operation page. Remaining R.4 bytes are unchanged.','new_navigation_routes':routes,'personally_reread':['Full currentR.4','Full current19.11','Current operation pageT.3,T.11'],'initial_repository_sources_still_exact':unchanged_sources,'original_cli_parser_locations':parser_locations,'accepted_long_recipe_contracts':accepted,'resolved_issue':'R4-operation-destination','finding':'Current19.11 actually contains the old synthetic download and four-stage recipe and says new-product operations/weights are pending. Training page correctly covers local experiments. No unresolved navigation/scope issue found.'}
(OUT/'recheck-environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
(OUT/'recheck-observations.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
print('All navigation reinspection and bounded original-parser assertions passed.')
