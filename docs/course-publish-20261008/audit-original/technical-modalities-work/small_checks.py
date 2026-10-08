import pathlib,re,sys,json,io,contextlib,traceback
sys.path.insert(0,'/workspace/work/tutorial-audit-20261008/freeze/implementation')
import torch
torch.set_num_threads(1)
B=pathlib.Path('/workspace/work/tutorial-audit-20261008')
results={}
for pid in ['10.5','10.7','10.8','11.2','11.3','11.5','12.4','12.5','12.9','12.12','13.6','13.13','13.15']:
 source=(B/'freeze/sources'/f'{pid}.md').read_text()
 code=re.findall(r'```python\n(.*?)```',source,re.S)[0]
 output=io.StringIO()
 try:
  with contextlib.redirect_stdout(output):exec(compile(code,f'{pid}.md:first-python-block','exec'),{})
  results[pid]={'status':'completed','output':output.getvalue(),'scope':'正文第一個Python程式塊，使用凍結實作；CPU，不做課程實驗訓練'}
 except Exception:
  results[pid]={'status':'error','output':output.getvalue(),'error':traceback.format_exc()}
print(json.dumps(results,ensure_ascii=False,indent=2))
(B/'technical-modalities-work/small-check-results.json').write_text(json.dumps({'torch':torch.__version__,'device':'cpu','threads':1,'results':results},ensure_ascii=False,indent=2))
