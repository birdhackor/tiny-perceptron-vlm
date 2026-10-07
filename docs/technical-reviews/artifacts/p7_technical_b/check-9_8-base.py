import json,hashlib
from pathlib import Path
from tiny_perceptron.data import ByteTokenizer
p=Path('docs/course-experiments/results/style.json');x=json.loads(p.read_text());r=x['results']['content_evaluation']['test'];tok=ByteTokenizer()
for s in r['samples']:
    ids=s['generated_ids'];content=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
    assert s['exact']==(content==tok.encode(s['expected']))
assert r['records']==7 and r['matches']==sum(s['exact'] for s in r['samples'])==0
print(json.dumps({'raw_style_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'matches':0,'records':7,'samples':r['samples']},ensure_ascii=False,indent=2))
