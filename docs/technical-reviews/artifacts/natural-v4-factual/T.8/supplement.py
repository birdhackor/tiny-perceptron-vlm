from pathlib import Path
from dataclasses import asdict
import json,hashlib,torch
from scripts.train import parser
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig,TinyLM
root=Path.cwd();tok=ByteTokenizer()
for name,target in [('modern','baseline.pt'),('moe','top2_aux0.01.pt'),('efficiency','ordinary.pt'),('precision','fp32.pt')]:
    d=json.loads((root/f'docs/course-experiments/results/{name}.json').read_text());artifacts={v['path']:v['sha256'] for v in d['artifacts']}
    assert artifacts['model.pt']==artifacts[target]
    print('FIXED CHECKPOINT',name,target,artifacts[target])
sft=json.loads((root/'docs/course-experiments/results/sft.json').read_text())['results']['after']
for split,h in sft.items():
    matches=0
    for s in h['samples']:
        ids=s['generated_ids'];raw=ids[:ids.index(2)] if 2 in ids else ids
        exact=raw==tok.encode(s['expected']);assert exact==s['exact'];matches+=exact
    assert matches==h['matches']
    print('ORIGINAL SFT ANSWER ID RECOUNT',split,matches,h['records'],h['effective_tokens'])
assert sft['test']['matches']==5 and sft['test']['records']==10
args=parser().parse_args(['--rotary','--norm','rms','--activation','swiglu','--tied','--heads','4','--kv-heads','1','--backend','sdpa'])
config=ModelConfig(**{k:getattr(args,k) for k in ['width','layers','heads','max_length','experts','top_k','rotary','norm','activation','backend','tied','kv_heads']})
m=TinyLM(config);y=m(torch.tensor([[1,2,3]]))['logits'];assert list(y.shape)==[1,3,264] and m.output.weight is m.embedding.weight
print('OWN CLI FLAGS',asdict(config),'shape',list(y.shape),'weight_identity',m.output.weight is m.embedding.weight)
tmp=root/'outputs/natural-v4/factual-research/T.8/toy-data/toy-text';counts=[]
for split in ['train','validation','test']:
    p=tmp/f'{split}.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()];counts.append(len(rows))
    print('OWN TOY DATA',split,len(rows),len({r['family'] for r in rows}),hashlib.sha256(p.read_bytes()).hexdigest())
assert counts==[9,1,2]
print('ARITHMETIC MiB',2**20,'bytes; training parameter FP32 bytes',141568*4)
