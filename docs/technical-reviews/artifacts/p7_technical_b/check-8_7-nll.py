import json
from pathlib import Path
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_b/sources/style-result.json').read_text())['results']['default_style_runs']
out={}
for name,expected in [('concise',8.41357),('vivid',0.26805)]:
 d=r[name]['after']['test']; count=d['effective_tokens']; value=d['nll_sum']/count
 assert count=={'concise':14,'vivid':308}[name] and round(value,5)==expected and d['matches']==0 and d['records']==7
 out[name]={'nll_sum':d['nll_sum'],'effective_targets':count,'mean':value,'rounded5':round(value,5),'matches':d['matches'],'records':d['records']}
print(json.dumps(out,ensure_ascii=False,indent=2))
