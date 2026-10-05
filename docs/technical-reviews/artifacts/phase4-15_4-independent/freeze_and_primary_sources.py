"""Save this review's inputs and primary-source extracts; no model execution."""
import ast
import hashlib
import json
import re
import subprocess
import urllib.request
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/phase4-15_4-independent'
TMP = Path('/tmp/phase4-15_4-authorities')
TMP.mkdir(exist_ok=True)
for name in ('inputs', 'sources', 'code'):
    (OUT / name).mkdir(exist_ok=True)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def write(name, raw):
    (OUT / name).write_bytes(raw)
    return {'path': str((OUT / name).relative_to(ROOT)), 'sha256': sha(raw), 'bytes': len(raw)}

def dump(name, obj):
    return write(name, (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode())

facts = json.loads(Path('/tmp/phase4-15_4-independent-facts/extraction.json').read_text())
chapter = (ROOT / 'course/chapters/15.md').read_bytes()
assert sha(chapter) == facts['source_file_sha256'], 'Need preserve actual first extraction input'
manifest = {
    'reviewer_task': '/root/phase4_factual_coordinator/factual_15_4',
    'frozen_at_utc': datetime.now(UTC).isoformat(),
    'frozen_chapter_meaning': 'Actual full UTF-8 input read by this review; not a claim about a later whole-chapter version.',
    'files': [], 'code_read_scope': [], 'primary_sources': [],
}
manifest['files'].append(write('inputs/chapter-15-frozen-input.md', chapter))
headers = list(re.finditer(rb'(?m)^## [^\r\n]+', chapter))
intro = chapter[:headers[0].start()]
manifest['files'].append(write('inputs/chapter-15-introduction-unreviewed.md', intro))
manifest['introduction_scope'] = 'Frozen without content inspection; this reviewer is not reviewing 15.1 or the introduction.'
for lesson in ('15.3', '15.4', '15.13'):
    i = next(i for i,h in enumerate(headers) if h[0].startswith(('## '+lesson+' ').encode()))
    raw = chapter[headers[i].start():headers[i+1].start() if i+1 < len(headers) else len(chapter)]
    manifest['files'].append(write('inputs/section-'+lesson+'.md', raw))
manifest['files'].append(write('code/original-fence-1.py', Path('/tmp/phase4-15_4-independent-facts/fence-1.py').read_bytes()))
manifest['files'].append(write('inputs/extraction.json', Path('/tmp/phase4-15_4-independent-facts/extraction.json').read_bytes()))
manifest['files'].append(write('inputs/moe-original-result-opaque.json', (ROOT/'docs/course-experiments/results/moe.json').read_bytes()))
manifest['files'].append(write('inputs/tinystories-original-512.jsonl', (ROOT/'data/training/text-initial/tinystories-train-512.jsonl').read_bytes()))

def nodes(raw, requested):
    source=raw.decode(); tree=ast.parse(source); lines=source.splitlines(keepends=True)
    selected=[]
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in requested:
            selected.append((n.name,n.lineno,n.end_lineno,''.join(lines[n.lineno-1:n.end_lineno])))
    assert {r[0] for r in selected} == set(requested)
    return selected

for path,names in (
    ('tiny_perceptron/modern.py', ['DenseFFN','MoEFFN']),
    ('tiny_perceptron/data.py', ['pad_batch']),
    ('tiny_perceptron/model.py', ['ModelConfig','Block']),
    ('scripts/course_experiments/common.py', ['new_lm','split_records','records_sha256','text_examples']),
    ('scripts/course_experiments/architecture.py', ['_text_dataset','_routing']),
):
    raw=(ROOT/path).read_bytes()
    inspected=nodes(raw,names)
    entry={'repository_path':path,'whole_file_sha256':sha(raw),'inspected_nodes':[]}
    for name,start,end,segment in inspected:
        saved=write('code/'+path.replace('/','--')+'--'+name+'.py',segment.encode())
        entry['inspected_nodes'].append({'name':name,'lines':[start,end],**saved})
    manifest['code_read_scope'].append(entry)

historical_revision='48a4f3e912b483d70aee57c42c2aac226534a9a6'
hist=subprocess.run(['git','show',historical_revision+':scripts/course_experiments/architecture.py'],cwd=ROOT,capture_output=True,check=True).stdout
assert sha(hist)=='1ad0b0789318236d5e1a8fc1117cb65758bbd6e3ad62dc2d646a111ae1ea37a3'
hist_nodes=nodes(hist,['_routing','_text_dataset'])
now_nodes={n[0]:n[3] for n in nodes((ROOT/'scripts/course_experiments/architecture.py').read_bytes(),['_routing','_text_dataset'])}
entry={'repository_path':'scripts/course_experiments/architecture.py','revision':historical_revision,'whole_file_sha256':sha(hist),'inspected_nodes':[]}
for name,start,end,segment in hist_nodes:
    assert segment == now_nodes[name], 'Recorded calculation differs from current helper'
    entry['inspected_nodes'].append({'name':name,'lines':[start,end],'identical_to_current':True,**write('code/historical-architecture--'+name+'.py',segment.encode())})
manifest['historical_result_code_scope']=entry

# The original PDF was located via an index; the reviewer reads the actual PDF-derived pages.
pdf=ROOT/'docs/technical-reviews/artifacts/fact_finish_r_4/originals/moe.pdf'
assert sha(pdf.read_bytes())=='2567713d198a376b65b92f0c60b9f58c8c5f66755530b92203533a17c4ec4b8e'
extract=TMP/'shazeer-pages-1-4.txt'
subprocess.run(['pdftotext','-f','1','-l','4','-layout',str(pdf),str(extract)],check=True)
manifest['primary_sources'].append({
    'id':'shazeer2017','url':'https://arxiv.org/pdf/1701.06538v1',
    'version':'arXiv:1701.06538v1, 23 Jan 2017; identifier checked on page 1',
    'full_original_pdf_sha256':sha(pdf.read_bytes()),
    'snapshot':write('sources/shazeer-1701.06538v1-pages-1-4.txt',extract.read_bytes()),
    'read_locators':['p1 title/version','p2 section1.2','p3 section2 equation1 and computation paragraph','p4 section2.1 equations2-5 and Training the Gating Network'],
})

commit='5c4886908584029761b579af026dcfb627c84070'
requests=[
    ('torch-docs',f'https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/_torch_docs.py'),
    ('tensor-docs',f'https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/_tensor_docs.py'),
    ('derivatives',f'https://raw.githubusercontent.com/pytorch/pytorch/{commit}/tools/autograd/derivatives.yaml'),
    ('floatingpoint','https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/tutorial/floatingpoint.rst'),
]
for name,url in requests:
    with urllib.request.urlopen(url,timeout=30) as response:
        raw=response.read(); status=response.status
    (TMP/(name+'.raw')).write_bytes(raw)
    source=raw.decode(); lines=source.splitlines(keepends=True)
    segments=[]
    if name in ('torch-docs','tensor-docs'):
        tree=ast.parse(source)
        wanted={'torch.tensor','torch.topk','torch.sum'} if name=='torch-docs' else {'"topk"','"tolist"','"sum"'}
        for n in tree.body:
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and n.value.args:
                arg=ast.unparse(n.value.args[0])
                if isinstance(n.value.args[0],ast.Constant): arg=json.dumps(n.value.args[0].value)
                if arg in wanted:
                    segments.append((n.lineno,n.end_lineno,''.join(lines[n.lineno-1:n.end_lineno])))
        assert len(segments)==3,(name,len(segments))
    elif name=='derivatives':
        start=next(i for i,line in enumerate(lines) if line.startswith('- name: topk('))
        end=next((i for i in range(start+1,len(lines)) if lines[i].startswith('- name: ')),len(lines))
        segments=[(start+1,end,''.join(lines[start:end]))]
    else:
        end=next(i for i,line in enumerate(lines) if 'Representation Error' in line)
        segments=[(1,end,''.join(lines[:end]))]
    excerpt=''.join(f'PRIMARY SOURCE {url}\nORIGINAL LINES {start}-{end}\n'+segment+'\n' for start,end,segment in segments)
    manifest['primary_sources'].append({'id':name,'url':url,'http_status':status,'accessed_on':'2026-10-05','full_response_sha256':sha(raw),'version':commit if name!='floatingpoint' else 'CPython v3.13.5','read_line_ranges':[[x[0],x[1]] for x in segments],'snapshot':write('sources/'+name+'-primary-excerpt.txt',excerpt.encode())})

dump('input-and-primary-manifest.json',manifest)
print(json.dumps({'frozen_chapter_sha256':sha(chapter),'source_sha256':facts['source_sha256'],'primary_sources':[{'id':s['id'],'version':s['version']} for s in manifest['primary_sources']],'historical_routing_equal_to_current':True},ensure_ascii=False,indent=2))
