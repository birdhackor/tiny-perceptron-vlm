import json,subprocess,sys
from pathlib import Path
import torch
root=Path('outputs/natural-v4/factual-research/T.6');out=[]
for modality in ['vision','audio']:
 path=root/(modality+'-one-update.pt');command=[sys.executable,'scripts/pretrain_encoders.py','--modality',modality,'--train','--steps','1','--device','cpu','--output',str(path)]
 p=subprocess.run(command,capture_output=True,text=True);assert p.returncode==0,p.stderr;r=json.loads(p.stdout);assert r['mode']=='train' and r['holdout_examples']==2
 out.append({'command':command,'returncode':p.returncode,'stdout':r,'stderr':p.stderr,'checkpoint_fields':list(torch.load(path,weights_only=True))})
print(json.dumps(out,indent=2))
