import hashlib,json,urllib.request
from pathlib import Path
OUT=Path('/workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-16_8-independent-fresh')
sources=OUT/'sources';records=[]
for name,path in [('autograd-5c488690.py','torch/autograd/__init__.py'),('tensor-attributes-5c488690.rst','docs/source/tensor_attributes.rst')]:
 url='https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/'+path
 try:
  with urllib.request.urlopen(url,timeout=30) as r:raw=r.read()
 except Exception as error:
  records.append(dict(url=url,error=type(error).__name__,message=str(error),accessed_on='2026-10-05'))
  continue
 (sources/name).write_bytes(raw)
 records.append(dict(url=url,path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),accessed_on='2026-10-05',version='5c4886908584029761b579af026dcfb627c84070'))
flash=json.loads((OUT/'inputs/flash_probe.json').read_bytes())
records.append({'gpu_runtime_selected_provenance':{k:flash['results']['runtime'][k] for k in ['torch','torch_git_version','cuda_runtime_version','gpu','installed_source_sha256']}})
for name,spans in [('functional-5c488690.py',[(6367,6536)]),('torch-docs-5c488690.py',[(833,863),(7907,7975)]),('attention-1706.03762v7.txt',[(150,202),(233,243)]),('flashattention-2205.14135v2.txt',[(1,25),(110,137)])]:
 lines=(sources/name).read_text().splitlines(keepends=True)
 excerpt=''.join('SOURCE '+name+' lines '+str(first)+'-'+str(last)+'\n'+''.join(lines[first-1:last])+'\n' for first,last in spans)
 (sources/(name+'.inspected-excerpt.txt')).write_text(excerpt)
(sources/'supplemental_fetch_log.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
