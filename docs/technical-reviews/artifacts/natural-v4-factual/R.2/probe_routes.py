"""Bounded R.2 entry examples, each with fresh globals; no training or weights downloads."""
import contextlib, hashlib, io, json, pathlib, re, sys, traceback
import torch
ROOT = pathlib.Path(__file__).resolve().parents[5]
OUT = pathlib.Path(__file__).resolve().parent
torch.set_num_threads(1)
source = (OUT / 'source-section.md').read_text()
links = re.findall(r'\]\(([^)]+\.md)#([A-Z\d]+\.\d+)\)', source)
results=[]
for rel, ident in dict.fromkeys(links):
    path = ROOT / 'course' / rel
    raw = path.read_text()
    body = re.search(r'^## '+re.escape(ident)+r'\s.*?(?=^## |\Z)', raw, re.M|re.S).group(0)
    blocks = re.findall(r'```python\n(.*?)```', body, re.S)
    record={'lesson':ident,'path':str(path.relative_to(ROOT)),'section_sha256':hashlib.sha256(body.encode()).hexdigest(),'python_blocks':len(blocks),'stdout':'','stderr':'','result':'no Python block; instructions only'}
    if blocks:
        stdout,stderr=io.StringIO(),io.StringIO()
        torch.manual_seed(42)
        try:
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                exec(compile('\n\n'.join(blocks),str(path)+'#'+ident,'exec'), {'__name__':'__main__'})
            record['result']='executed successfully with fresh globals'
        except Exception:
            record['result']='failed'
            traceback.print_exc(file=stderr)
        record.update(stdout=stdout.getvalue(),stderr=stderr.getvalue())
    results.append(record)
# Independently recompute the only quantitative example explicitly described by R.2.
size=[{'dtype':str(dtype),'elements':(w:=torch.zeros(64,32).to(dtype)).numel(),'bytes_per_element':w.element_size(),'payload_bytes':w.numel()*w.element_size()} for dtype in (torch.float32,torch.float16,torch.int8)]
assert [r['payload_bytes'] for r in size] == [8192,4096,2048]
report={'environment':{'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu','threads':torch.get_num_threads()},'scope':'Only R.2 explicitly linked entry/prerequisite short Python blocks, in fresh globals; Bash install/download/training examples not executed; no GPU or full-course kernel audit.','linked_sections':len(results),'sections_with_python':sum(bool(r['python_blocks']) for r in results),'executed_python_blocks':sum(r['python_blocks'] for r in results),'failures':sum(r['result']=='failed' for r in results),'quantization_storage_probe':size,'results':results}
(OUT/'cpu-route-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False,indent=2))
for r in results: print(r['lesson'],r['python_blocks'],r['result'])
if report['failures']:sys.exit(1)
