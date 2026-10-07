import json
from pathlib import Path
t=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/distillation.json').read_text())['results']['tasks']['attributes'];a=t['runs']['w32_ce_kl']['test']['generated_samples'];b=t['runs']['w32_ce_kl_packed4']['test']['generated_samples'];assert len(a)==len(b)==10
for x,y in zip(a,b):print('floating',json.dumps(x,ensure_ascii=False));print('packed4',json.dumps(y,ensure_ascii=False))
print('same_arch',t['runs']['w32_ce']['storage']['parameter_count']==t['runs']['w32_ce_kl']['storage']['parameter_count'])
for k in ['w32_ce','w32_ce_kl','w32_ce_kl_packed4']:print(k,'bytes',t['runs'][k]['storage']['tensor_bytes'],'test',t['runs'][k]['test']['correct'],t['runs'][k]['test']['examples'])
