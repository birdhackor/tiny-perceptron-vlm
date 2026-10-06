"""Original-owner current callback for 11.4; source reading and hash checks only."""
import ast
from datetime import datetime, UTC
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK = '/root/phase4_factual_coordinator/factual_11_4'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def section(path, lesson):
    raw = path.read_bytes()
    headings = list(re.finditer(rb'(?m)^## [^\r\n]+', raw))
    index = next(i for i,h in enumerate(headings) if re.match(rb'^## '+lesson.encode()+rb' ',h[0]))
    start = headings[index].start()
    end = headings[index+1].start() if index+1 < len(headings) else len(raw)
    return raw[start:end], raw[:start].count(b'\n')+1
def emit(key, value): print(json.dumps({key:value}, ensure_ascii=False, sort_keys=True))

prior_path = HERE/'prior-opaque/docs/technical-reviews/11.4.json'
prior_raw = prior_path.read_bytes()
prior = json.loads(prior_raw)  # Same original owner, after opaque preservation.
primary, first_line = section(ROOT/'course/chapters/11.md', '11.4')
assert sha(primary) == prior['source_sha256'] == '12c45a42e56a01208a0441c4050faff168af6496407e286b22c3b306b85a2795'
(HERE/'current-11.4.md').write_bytes(primary)
emit('current_primary', {'source':'course/chapters/11.md#11.4','sha256':sha(primary),'first_line':first_line,'original_utf8':primary.decode()})

context, context_first_line = section(ROOT/'course/chapters/07.md','7.1')
(HERE/'current-7.1.md').write_bytes(context)
prefixes = [b'## 7.1 ', '一位使用者問'.encode(), '一筆對話是'.encode(),
            '`render_chat` 將'.encode(), 'Y前八項'.encode(), '用正確助理示範'.encode()]
blocks = context.split(b'\n\n')
selected = [block for block in blocks if any(block.startswith(prefix) for prefix in prefixes)]
assert len(selected) == 6
slice_raw = b'\n\n'.join(selected) + b'\n'
(HERE/'necessary-7.1-context-slice.md').write_bytes(slice_raw)
emit('current_necessary_context', {'source':'course/chapters/07.md#7.1','section_sha256':sha(context),
    'section_first_line':context_first_line,'slice_sha256':sha(slice_raw),'selected_paragraphs':6,
    'slice_original_paragraph_bytes':slice_raw.decode(),
    'scope':'11.4 linked role/content, question context, assistant answer and EOS supervision only; no independent verdict on 7.1 exercise, numeric lesson or the whole chapter'})

data_path = ROOT/'tiny_perceptron/data.py'
data_raw = data_path.read_bytes()
tree = ast.parse(data_raw)
lines = data_raw.splitlines(keepends=True)
extracts=[]
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ['ByteTokenizer','render_chat']:
        extracts.append({'name':node.name,'first_line':node.lineno,'last_line':node.end_lineno,
            'original_utf8':b''.join(lines[node.lineno-1:node.end_lineno]).decode()})
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['SPECIALS','IGNORE'] for t in node.targets):
        extracts.append({'name':','.join(t.id for t in node.targets if isinstance(t,ast.Name)),
            'first_line':node.lineno,'last_line':node.end_lineno,
            'original_utf8':b''.join(lines[node.lineno-1:node.end_lineno]).decode()})
(HERE/'current-data.py').write_bytes(data_raw)
old_result = ROOT/'docs/technical-reviews/artifacts/phase4-11_4-independent/vqa-original.json'
old_result_raw = old_result.read_bytes()
old_original = json.loads(old_result_raw)
data_pointer = '/code_sha256/tiny_perceptron~1data.py'
original_data_sha = old_original['code_sha256']['tiny_perceptron/data.py']
assert sha(data_raw) == original_data_sha
emit('necessary_original_contract', {'path':'tiny_perceptron/data.py','sha256':sha(data_raw),
    'matching_original_result_pointer':data_pointer,'original_result_full_sha256':sha(old_result_raw),
    'ast_selected_original_code':extracts})

reused=[]
for artifact in prior['artifacts']:
    path=ROOT/artifact['path']
    observed=sha(path.read_bytes())
    assert observed == artifact['sha256'], (artifact['id'], 'prior proof changed')
    reused.append({'artifact_id':artifact['id'],'path':artifact['path'],'sha256':observed,'kind':artifact['kind']})
for source in prior['sources']:
    if source['kind']=='repository_code':
        assert sha((ROOT/source['path']).read_bytes()) == source['sha256'], source['id']
svg_path='course/figures/rewrite-11-same-image-questions.svg'
svg_sha=sha((ROOT/svg_path).read_bytes())
assert svg_sha == prior['figure_sha256'][svg_path] == '4195cd648e5c2707e5acad33acdb0a93bdfa543d2ea1b8e39357d8ab48ad36be'
environment={'python':platform.python_version(),'python_executable':sys.executable,
    'torch_distribution':metadata.version('torch'),'device_scope':'source/hash inspection on CPU host; no torch import/model execution',
    'shell':'bash login:false','cwd':str(ROOT)}
emit('environment',environment)
emit('reused_proof',{'all_artifacts_match':True,'count':len(reused),'all_registered_repository_source_hashes_match':True,
    'current_svg':{'path':svg_path,'sha256':svg_sha},
    'reuse_scope':'Prior original CPU fence/audit, 256pixel check, historical raw measurement audit, downloaded authoritative source inspections and original render/view; these are not new executions or current-page render claims.'})
receipt={'schema_version':1,'reviewer_task':TASK,'same_original_owner':True,'inspected_at':datetime.now(UTC).isoformat(),
    'prior_canonical_opaque':{'path':str(prior_path.relative_to(ROOT)),'sha256':sha(prior_raw)},
    'primary':{'source':'course/chapters/11.md#11.4','sha256':sha(primary),'snapshot':str((HERE/'current-11.4.md').relative_to(ROOT)),'fully_read':True},
    'intro':None,'figure':{'path':svg_path,'sha256':svg_sha,'reuse_scope':'same SVG and claim content; original render/view retained'},
    'necessary_context':{'source':'course/chapters/07.md#7.1','current_section_sha256':sha(context),
        'section_snapshot':str((HERE/'current-7.1.md').relative_to(ROOT)),
        'necessary_slice_sha256':sha(slice_raw),'necessary_slice_snapshot':str((HERE/'necessary-7.1-context-slice.md').relative_to(ROOT)),
        'actual_read_scope':'Current 7.1 whole short section was read to identify dependency; substantive callback uses only six selected heading/paragraph blocks, role/content and assistant/EOS supervision. No independent review of 7.1.',
        'original_contract':{'path':'tiny_perceptron/data.py','sha256':sha(data_raw),'snapshot':str((HERE/'current-data.py').relative_to(ROOT)),
            'ast_inspected':[{k:v for k,v in x.items() if k!='original_utf8'} for x in extracts],
            'original_result_pointer':data_pointer,'original_result_sha256':sha(old_result_raw)},
        'finding':'Current necessary explanations are compatible with 11.4 image/question/answer pairing and answer-only targets. Role boundaries come from role, question bytes remain context, assistant answer+EOS are supervised. They do not turn the 11.4 records-name example into image ingestion, training or a learned-answer demonstration.',
        'support_boundary':'7.1 current prose is checked against original contracts and prior LLaVA Eq.(3)/Table2 authority, not treated as its own proof. Exercise or chapter-wide claims are outside this callback.'},
    'other_prior_context_scope':'Original frozen 11.1–11.3 and 11.5 opening read scope remains historical; no claim to have re-read or refreshed the whole chapter. 11.3 prior necessary description/answer target context is explicitly frozen, not a current whole-section assertion.',
    'instructions_read':[{'path':p,'sha256':sha((ROOT/p).read_bytes())} for p in ['docs/review-tools/factual-reviewer-instructions.md','.agents/skills/clear-tutorial/references/review-protocol.md']],
    'reused_artifacts':reused,'environment':environment,
    'new_actions':'Current source/necessary-context AST reading and exact byte/hash verification only; no re-fetch, model/data download, CPU model re-run, GPU, training, .pt save or re-render.',
    'events':[],'judgment':'Current callback finds no unresolved substantive issue within 11.4 and its necessary 7.1 dependency.'}
(HERE/'current-inspection-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
(HERE/'current-environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
emit('receipt',{'path':str((HERE/'current-inspection-receipt.json').relative_to(ROOT)),
    'sha256':sha((HERE/'current-inspection-receipt.json').read_bytes()),'result':'all exact hash/source checks passed; current dependency compatible'})
