import contextlib,io,json,pathlib,re,sys,os,traceback,hashlib,platform
root=pathlib.Path('/workspace/work/tutorial-audit-20261008')
impl=root/'freeze/implementation'
sys.path.insert(0,str(impl));os.chdir(impl)
import torch
torch.set_num_threads(1)
m=json.loads((root/'manifest.json').read_text())
records=[]
for pid in m['groups']['foundations']['pages']:
 s=(root/'freeze/sources'/f'{pid}.md').read_text();cells=re.findall(r'```python\n(.*?)\n```',s,re.S)
 ns={'__name__':'__audit_snippet__'};out=io.StringIO();error=None
 try:
  with contextlib.redirect_stdout(out):
   for n,c in enumerate(cells,1):exec(compile(c,f'{pid}.md#python-cell-{n}','exec'),ns)
 except Exception:
  error=traceback.format_exc()
 records.append({'page_id':pid,'cells':len(cells),'output':out.getvalue(),'error':error,'scope':'exact authored Python code blocks, sequential within page; frozen imports; CPU; no notebook/bootstrap/shell install'})
result={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','source_root':str(impl),'pages':records}
(root/'technical-foundations-evidence/snippet-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for p in records:
 if p['cells']:print(p['page_id'],p['cells'],'PASS' if p['error'] is None else 'FAIL',repr(p['output'])[:300],p['error'] or '')
print('total cells',sum(p['cells'] for p in records),'fail pages',sum(p['error'] is not None for p in records))
