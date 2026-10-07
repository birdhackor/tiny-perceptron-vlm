import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
frequencies = [200, 220, 440, 660]
truth = ["high" if f > 300 else "low" for f in frequencies]
no_audio = ["high"] * len(frequencies)
accuracy = sum(a == b for a, b in zip(truth, no_audio)) / len(truth)
print("頻率", frequencies)
print("標準答案", truth)
print("不聽聲音的固定猜測", accuracy)


# Owner supplied proportional check
import json
from pathlib import Path
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_c/originals/audio-raw.json').read_text())['results'];d=r['data']['splits']['test']['records']
print('majority-counts',sum(x['answer']=='low' for x in d),sum(x['answer']=='high' for x in d),len(d))
for key in ['test','blank','shuffle']:
 a=r[key];rows=a['samples'];old=sum(x['target']==x['generated'] for x in rows);assert len(rows)==a['examples'] and old==a['correct'];print('old-recount',key,old,len(rows),'outputs',sorted({x['generated'] for x in rows}))
 if key=='shuffle':
  new=sum(x['generated']==d[x['donor_row']]['answer'] for x in rows);same=sum(x['target']==d[x['donor_row']]['answer'] for x in rows)
  print('new-recount',new,len(rows),'same-label',same,'wrong-rows',[x['row'] for x in rows if x['generated']!=d[x['donor_row']]['answer']]);assert new==11 and same==2
print('nine-low-baseline',9/10)

