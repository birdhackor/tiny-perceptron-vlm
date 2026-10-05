"""Read only original optimizer update counters, without interpreting quality."""
import hashlib
import json
from pathlib import Path
OUT=Path('/workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-15_4-independent')
raw=OUT/'inputs/moe-original-result-opaque.json'
data=json.loads(raw.read_text())
records=[]
for name in ('top1_aux0','top1_aux0.01','top2_aux0','top2_aux0.01'):
    record={'variant':name,'pointers':[],'counters':{}}
    for key in ('requested_steps','steps','optimizer_updates','skipped_updates','all_requested_attempts_completed','effective_tokens'):
        record['pointers'].append('/results/variants/'+name+'/training/'+key)
        record['counters'][key]=data['results']['variants'][name]['training'][key]
    assert record['counters']['requested_steps']==record['counters']['steps']==record['counters']['optimizer_updates']==180
    assert record['counters']['skipped_updates']==0 and record['counters']['all_requested_attempts_completed'] is True
    records.append(record)
result={'original_result_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'records':records,'scope':'Existing original training counters only. No training or quality re-evaluation.'}
(OUT/'training-provenance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
