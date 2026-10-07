import ast,json,hashlib,random
from pathlib import Path
from scripts.course_experiments.common import split_records,records_sha256,text_examples
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.data import IGNORE,ByteTokenizer
p=Path('docs/course-experiments/results/safety.json');raw=json.loads(p.read_text());r=raw['results'];source=Path('docs/technical-reviews/artifacts/p7_technical_b/sources/behavior-a864a60.py');ns={};tree=ast.parse(source.read_text());exec(compile(ast.Module(body=[f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name in ['_conversation','_safety_records']],type_ignores=[]),str(source),'exec'),ns)
parts=split_records(ns['_safety_records'](),seed=42);arith=split_records(arithmetic_records(),seed=42);tok=ByteTokenizer();out={'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'seed':raw['seed'],'runs':{}}
for name,rows in [('safety-only',parts['train']),('model',parts['train']+arith['train'])]:
    tr=r['runs'][name]['training'];assert tr['records']==len(rows) and tr['steps']==900 and tr['records_sha256']==records_sha256(rows)
    examples=text_examples(rows,'sft',128);rng=random.Random(42);targets=sum(sum(int(z!=IGNORE) for z in y) for _ in range(900) for _,y in rng.choices(examples,k=16));assert targets==tr['effective_tokens']
    metrics={}
    for category in ['safety','arithmetic']:
        metrics[category]={}
        for split,rr in r['runs'][name][category].items():
            ss=rr['samples']
            for s in ss:
                ids=s['generated_ids'];content=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
                assert s['exact']==(content==tok.encode(s['expected'])) and s['eos']==(tok.eos_id in ids)
            assert rr['matches']==sum(s['exact'] for s in ss) and rr['records']==len(ss)
            metrics[category][split]={'matches':rr['matches'],'records':len(ss),'eos':sum(s['eos'] for s in ss)}
    out['runs'][name]={'records':len(rows),'records_sha256':records_sha256(rows),'steps':900,'targets':targets,'metrics':metrics}
print(json.dumps(out,ensure_ascii=False,indent=2))
