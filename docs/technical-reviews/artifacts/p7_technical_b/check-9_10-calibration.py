import json,hashlib,torch
from pathlib import Path
p=Path('docs/course-experiments/results/encoders.json');raw=json.loads(p.read_text());out={'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'metrics':{}}
for modality in ['vision','audio']:
    c=raw['results'][modality]['calibration'];vl=torch.tensor(c['validation']['logits']);vy=torch.tensor(c['validation']['labels']);grid=c['temperature_grid'];selected=min(grid,key=lambda t:torch.nn.functional.cross_entropy(vl/t,vy).item());assert selected==c['chosen_temperature']==.5
    metrics={'chosen_temperature':selected}
    for split in ['validation','test']:
        logits=torch.tensor(c[split]['logits']);labels=torch.tensor(c[split]['labels']);metrics[split]={}
        for name,temp in [('original',1.),('calibrated',selected)]:
            prob=(logits/temp).softmax(-1);conf,pred=prob.max(-1);correct=pred==labels;N=len(labels);ece=0.;bins=[]
            for i in range(5):
                mask=(conf>=i/5)&(conf<=1 if i==4 else conf<(i+1)/5)
                if mask.any():
                    mean=conf[mask].mean().item();accuracy=correct[mask].float().mean().item();count=mask.sum().item();ece+=count/N*abs(accuracy-mean);bins.append({'count':count,'mean_confidence':mean,'accuracy':accuracy})
            measured={'count':N,'correct':correct.sum().item(),'accuracy':correct.float().mean().item(),'mean_confidence':conf.mean().item(),'nll':torch.nn.functional.cross_entropy(logits/temp,labels).item(),'brier':((prob-torch.nn.functional.one_hot(labels,prob.shape[-1])).square().sum(-1).mean()).item(),'ece':ece}
            for k,v in measured.items():assert abs(c[split][name][k]-v)<2e-7,(modality,split,name,k,v,c[split][name][k])
            assert torch.equal(logits.argmax(-1),(logits/temp).argmax(-1))
            metrics[split][name]={**measured,'bins':bins}
        assert set(c['validation']['families']).isdisjoint(c['test']['families'])
    out['metrics'][modality]=metrics
print(json.dumps(out,ensure_ascii=False,indent=2))
