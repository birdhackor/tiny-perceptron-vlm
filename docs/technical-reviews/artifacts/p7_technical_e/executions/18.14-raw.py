import json
from pathlib import Path
t=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/distillation.json').read_text())['results']['tasks']['attributes'];print('keys',list(t));print('teacher_storage',t['teacher_storage']);print('studentwidthkeys',list(t['student_widths']))
w=t['student_widths']['32'];print('runkeys',list(w['runs']));print('keys',list(w))
for k,r in w['runs'].items():
 print(k,'storage',r.get('storage'),'test', {x:r['test'][x] for x in ['examples','correct','effective_tokens']})
print('packed',json.dumps(w.get('packed',w.get('quantized',{})),ensure_ascii=False))
