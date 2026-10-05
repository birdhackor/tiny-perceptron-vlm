"""Same technical owner: check only actual C.7 delta, target, and evidence hashes."""
import ast
import hashlib
import json
import platform
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

BASE=Path(__file__).resolve().parents[1]
ROOT=Path(__file__).resolve().parents[5]
PRIOR=ROOT/'docs/technical-reviews/artifacts/phase4-20261005-c_7-independent'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def section(raw,id):
    headers=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
    i=next(i for i,h in enumerate(headers) if h[0].startswith(b'## '+id.encode()+b' '))
    return raw[headers[i].start():headers[i+1].start() if i+1<len(headers) else len(raw)]

prior=json.loads((BASE/'prior-opaque/C.7.json').read_bytes())
assert prior['reviewer_task']=='/root/phase4_factual_coordinator/factual_c_7'
for a in prior['artifacts']:assert sha(ROOT/a['path'])==a['sha256']
whole=(ROOT/'course/chapters/0C.md').read_bytes()
current=section(whole,'C.7');old=(PRIOR/'inputs/C.7.raw.md').read_bytes()
before='與上節保持權重的作答計算不同。'.encode()
after='與[C.5保持權重的作答計算](0C.md#C.5)不同。'.encode()
assert old.count(before)==1 and current==old.replace(before,after)
assert hashlib.sha256(current).hexdigest()=='f5bfba461df392317a557141a257a56452d4c0eb37b4bd1f36203ef9f6e8e2ff'
assert current==(BASE/'current/C.7.raw.md').read_bytes()
assert section(whole,'C.6')==(BASE/'current/C.6.context.raw.md').read_bytes()
assert section(whole,'C.5')==(BASE/'current/C.5.target.raw.md').read_bytes()
assert section(whole,'C.6')==section((PRIOR/'inputs/0C.frozen-input.md').read_bytes(),'C.6')
fences=re.findall(rb'```(?:python|bash)\n(.*?)```',current,re.S)
assert len(fences)==2
for raw,name in zip(fences,['fence-1.py','fence-2.sh'],strict=True):assert raw==(PRIOR/'inputs'/name).read_bytes()
for name,id in [('scripts/course_experiments/applications.py','application-code'),('scripts/course_experiments/common.py','common-code'),('scripts/course_experiments/run.py','runner-code'),('docs/course-experiments/results/reasoning.json','raw-results'),('course/figures/rewrite-C-policy-update.svg','raw-figure')]:
    assert sha(ROOT/name)==next(a['sha256'] for a in prior['artifacts'] if a['id']==id)
target=section(whole,'C.5').decode()
assert '可保持模型權重固定，只改作答流程' in target

code=ROOT/'scripts/course_experiments/applications.py';rawcode=code.read_bytes();tree=ast.parse(rawcode)
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_sample','_reasoning_samples']]
assert len(nodes)==2
sample=next(n for n in nodes if n.name=='_sample')
assert any(ast.unparse(d)=='torch.no_grad()' for d in sample.decorator_list)
calls=[{'line':n.lineno,'call':ast.unparse(n.func)} for node in nodes for n in ast.walk(node) if isinstance(n,ast.Call)]
assert not any(x['call'].endswith(('.backward','.step','.load_state_dict')) for x in calls)
lines=rawcode.splitlines(keepends=True)
excerpts=b''.join(b''.join(lines[min([n.lineno]+[d.lineno for d in n.decorator_list])-1:n.end_lineno]) for n in nodes)
(BASE/'current/original-inference-method.excerpts.py').write_bytes(excerpts)

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.open=None
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.open={'attrs':dict(attrs),'text':''}
    def handle_data(self,data):
        if self.open is not None:self.open['text']+=data
    def handle_endtag(self,tag):
        if tag=='a' and self.open is not None:self.links.append(self.open);self.open=None

url='http://127.0.0.1:8765/C.7.html'
with urllib.request.urlopen(url,timeout=15) as response:html=response.read();assert response.status==200
parser=Links();parser.feed(html.decode())
links=[link for link in parser.links if link['text']=='C.5保持權重的作答計算']
assert len(links)==1
href=links[0]['attrs']['href'];targeturl=urljoin(url,href)
assert targeturl=='http://127.0.0.1:8765/C.5.html'
with urllib.request.urlopen(targeturl,timeout=15) as response:targethtml=response.read();assert response.status==200
assert '可保持模型權重固定，只改作答流程' in targethtml.decode()
navigation={'source_url':url,'source_html_sha256':hashlib.sha256(html).hexdigest(),
            'actual_link':links[0],'target_url':targeturl,'target_html_sha256':hashlib.sha256(targethtml).hexdigest(),
            'target_text_inspected':'可保持模型權重固定，只改作答流程','http_statuses':[200,200],
            'scope':'Parsed only the changed navigation and its target sentence, not a new full-page visual review.'}
(BASE/'checks/navigation.json').write_text(json.dumps(navigation,ensure_ascii=False,indent=2)+'\n')
receipt={
 'schema_version':1,'kind':'same_owner_technical_recheck_receipt','artifact_id':'c7-current-recheck-receipt-r1',
 'reviewer_task':'/root/phase4_factual_coordinator/factual_c_7','lesson_id':'C.7','verdict':'pass',
 'current_source':{'path':'course/chapters/0C.md#C.7','snapshot':str((BASE/'current/C.7.raw.md').relative_to(ROOT)),
                   'sha256':hashlib.sha256(current).hexdigest()},
 'current_frozen_whole_input':{'path':str((BASE/'current/0C.frozen-input.md').relative_to(ROOT)),
                             'sha256':sha(BASE/'current/0C.frozen-input.md'),
                             'meaning':'Actual current frozen input; only intro,C.5,C.6,C.7 content was read.'},
 'current_intro':{'path':str((BASE/'current/0C.intro.raw.md').relative_to(ROOT)),
                  'sha256':sha(BASE/'current/0C.intro.raw.md'),'read':True,'scope':'Current chapter introduction; does not change C.7 technical claim support.'},
 'current_context':[
     {'section':'C.6','path':str((BASE/'current/C.6.context.raw.md').relative_to(ROOT)),
      'sha256':sha(BASE/'current/C.6.context.raw.md'),'read_scope':'Complete necessary reward-criteria context; bytes identical to original own frozen input.'},
     {'section':'C.5','path':str((BASE/'current/C.5.target.raw.md').relative_to(ROOT)),
      'sha256':sha(BASE/'current/C.5.target.raw.md'),'read_scope':'Complete current changed-link target; assessed only fixed-weight test-time-compute compatibility, not its unchanged metrics.'}],
 'current_figures':{'course/figures/rewrite-C-policy-update.svg':sha(ROOT/'course/figures/rewrite-C-policy-update.svg')},
 'prior_opaque':{'path':str((BASE/'prior-opaque/C.7.json').relative_to(ROOT)),
                 'sha256':sha(BASE/'prior-opaque/C.7.json'),'original_issues':'Preserved unchanged in prior report; no unresolved original issues.',
                 'proof_manifest':str((BASE/'prior-opaque/proof-manifest.json').relative_to(ROOT))},
 'actual_change':{'before':before.decode(),'after':after.decode(),
                 'judgment':'Explicit C.5 link correctly points to the fixed-weight answering-compute section; inference and policy-training distinction remains accurate.'},
 'unchanged_support_reused':{
     'artifact_hashes_checked':len(prior['artifacts']),
     'original_artifact_records':json.loads((BASE/'checks/reused-evidence-sha256.json').read_bytes())['verified_original_artifacts'],
     'original_source_locators':[{'source_id':s['id'],'url':s.get('url'),
                                  'version':s.get('version'),'inspection_note':s.get('inspection_note')}
                                 for s in prior['sources'] if s['kind'] in ['official_source','official_docs','paper','repository_code']],
     'reuse_reason':'Only navigation text changed. Both fences, SVG, original code, raw measurements and all 33 original evidence artifacts have exact matching SHA. Own original CPU calculations, raw recomputation, render/view and primary-source inspections are reused, not rerun.'},
 'additional_original_method_inspection':{'path':'scripts/course_experiments/applications.py',
                                       'sha256':sha(code),'locators':'_sample lines41–103 including no_grad decorator; _reasoning_samples lines750–811',
                                       'support':'Candidate generation/selection calls no training update/backward/optimizer method; reinforces C.5 fixed-weight workflow distinction.',
                                       'bounded_check':'AST only; no LM creation or generated model scores.'},
 'execution':{'command':'.venv/bin/python '+str((BASE/'checks/recheck.py').relative_to(ROOT)),
              'code_path':str((BASE/'checks/recheck.py').relative_to(ROOT)),'code_sha256':sha(BASE/'checks/recheck.py'),
              'environment':{'python':platform.python_version(),'device':'cpu','network':'Local read-only HTTP GET only'},
              'result':'All exact-delta, current-input, unchanged-evidence, AST contract and navigation assertions succeeded. No numerical claims changed; no new numeric run needed.'},
 'navigation':navigation,'unresolved_dependencies':[],
 'restrictions_observed':'No author revision notes, third-reader opinions or other reviewer judgments read; no paper refetch, CPU suite rerun, GPU, train, full recipe, download, retained weights or unrelated screenshot.'}
(BASE/'checks/receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'receipt_id':receipt['artifact_id'],'receipt_path':str((BASE/'checks/receipt.json').relative_to(ROOT)),
                  'receipt_sha256':sha(BASE/'checks/receipt.json'),'source_sha256':receipt['current_source']['sha256'],
                  'intro_sha256':receipt['current_intro']['sha256'],'target_sha256':receipt['current_context'][1]['sha256'],
                  'figure_sha256':receipt['current_figures'],'verdict':receipt['verdict'],'navigation':navigation,
                  'unchanged_artifacts_checked':len(prior['artifacts']),'unresolved_dependencies':[]},ensure_ascii=False,indent=2))
