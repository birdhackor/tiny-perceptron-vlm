import pathlib,sys,json,hashlib
r=pathlib.Path('/workspace/work/tutorial-audit-20261008');sys.path.insert(0,str(r/'freeze/implementation'))
import torch
from tiny_perceptron.simple import BigramLM, ContextMLP
from scripts.course_experiments.text import _simple_examples,_simple_sample
from tiny_perceptron.model import TinyLM,ModelConfig
p=pathlib.Path('/workspace/tiny-perceptron-vlm/outputs/course-experiments/course-v1/simple_models')
j=json.loads((r/'technical-foundations-evidence/simple_models-raw.json').read_text());torch.set_num_threads(1)
vocab=json.loads((p/'vocabulary.json').read_text());splits={s:[json.loads(l)['text'] for l in (p/'data'/f'{s}.jsonl').read_text().splitlines()] for s in ['train','validation','test']}
report={'scope':'Evaluate hash-matched historical simple-v1 checkpoints only. No updates, no training. Frozen evaluator implementation, CPU torch '+torch.__version__,'denominators':{s:sum(len(t)+1 for t in texts) for s,texts in splits.items()},'runs':{},'initial_attempt':'First strict hash guard failed before loading any checkpoint; this revised audit records each mismatch and skips evaluation rather than treating local files as historical artifacts.'}
for name in ['bigram','mlp1','mlp3','mlp5']:
 f=p/f'{name}.pt';expected=next(a['sha256'] for a in j['artifacts'] if a['path']==f'{name}.pt');actual=hashlib.sha256(f.read_bytes()).hexdigest()
 if expected!=actual:
  report['runs'][name]={'checkpoint_sha256':actual,'expected_historical_sha256':expected,'status':'unverified: local file differs from historical artifact; no load/evaluation'}
  continue
 state=torch.load(f,weights_only=True,map_location='cpu');context=state['context'];m=BigramLM(len(vocab)+2) if name=='bigram' else ContextMLP(len(vocab)+2,context,16);m.load_state_dict(state['model']);m.eval()
 nll={}
 with torch.no_grad():
  for s,texts in splits.items():
   x,y=_simple_examples(texts,vocab,context);nll[s]=torch.nn.functional.cross_entropy(m(x),y).item()
 report['runs'][name]={'checkpoint_sha256':actual,'post_update_nll':nll,'absolute_difference_from_raw':{s:abs(nll[s]-j['results']['runs'][name]['after_nll_same_post_update_time'][s]) for s in nll},'samples':[_simple_sample(m,vocab,context,'顏色=','cpu')]}
report['text_foundation_config_count'] = TinyLM(ModelConfig(width=64,layers=2)).description()['parameters']
(r/'technical-foundations-evidence/simple-checkpoint-evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
