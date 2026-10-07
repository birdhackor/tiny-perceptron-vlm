import json
from pathlib import Path
x=json.load(open('docs/technical-reviews/artifacts/p7_technical_b/check-9_10-calibration-output.json'))['metrics'];v=x['vision']['test'];a=x['audio']['test']
out={'brier_toy':.8**2+(.2-1)**2,'vision_original':[round(v['original']['ece'],5),round(v['original']['brier'],6)],'vision_scaled':[round(v['calibrated']['ece'],6),round(v['calibrated']['brier'],9)],'audio_original':[round(a['original']['ece'],5),round(a['original']['brier'],5)],'audio_scaled':[round(a['calibrated']['ece'],5),round(a['calibrated']['brier'],5)],'audio_summary':{k:[round(a['original'][k],4),round(a['calibrated'][k],4)] for k in ['mean_confidence','accuracy','nll']}}
assert abs(out['brier_toy']-1.28)<1e-12
assert out['vision_original']==[.02858,.001485] and out['vision_scaled']==[.000572,.000000859]
assert out['audio_original']==[.13186,.20207] and out['audio_scaled']==[.14154,.25337]
print(json.dumps(out,indent=2))
