"""Narrow source/AST/fingerprint verification; no model execution or training."""
import ast
import hashlib
import importlib.metadata
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).resolve().parent
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
prior = json.loads((A/'prior-opaque/accepted-13.14-report.json').read_bytes())
assert digest(A/'prior-opaque/accepted-13.14-report.json') == '14a2ddf6e20cfe47334edf1b1264b0c6d006e292d9077c33265d4bd9a7c065e7'
fingerprints=[]
for entry in prior['artifacts']:
    p=ROOT/entry['path'];assert p.is_file() and not p.is_symlink()
    observed=digest(p);assert observed==entry['sha256']
    fingerprints.append({'artifact_id':entry['id'],'path':entry['path'],'sha256':observed,'unchanged':True})
raw=(ROOT/'course/chapters/13.md').read_bytes();heads=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
i=next(i for i,h in enumerate(heads) if h[0].startswith(b'## 13.14 '))
section=raw[heads[i].start():heads[i+1].start()]
assert hashlib.sha256(section).hexdigest()=='37faa95f02e041ded9cfbf85895630693204fe32d37c5c6a90896fe559f7a531'
assert section==(A/'current-13.14.md').read_bytes()
quote='reference保留整段後訓練起點。'
assert quote in section.decode()
figure=ROOT/'course/figures/rewrite-13-model-roles.svg'
assert digest(figure)=='b18bbd67c517b81f3f215755d70c948156283de29aac2aa4f1a068381ec53459'
tree=ET.fromstring(figure.read_bytes());ns={'s':'http://www.w3.org/2000/svg'}
label=next(n for n in tree.findall('s:text',ns) if n.attrib.get('x')=='50' and n.attrib.get('y')=='928')
assert label.text=='整段後訓練都不更新'
desc=tree.find('s:desc',ns).text
source=ROOT/'scripts/course_experiments/posttraining.py'
assert digest(source)=='862dc12680fee8374679d52a1cc55a2dd325a7aebbf634668731fa7def08aa04'
ast_source=ast.parse(source.read_bytes())
run=next(n for n in ast_source.body if isinstance(n,ast.FunctionDef) and n.name=='run_posttraining')
ref=next(n for n in run.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reference' for t in n.targets))
sft_loop=next(n for n in run.body if isinstance(n,ast.For) and any(isinstance(v,ast.Constant) and v.value=='sft_steps' for v in ast.walk(n.iter)))
assert sft_loop.end_lineno<ref.lineno
sft_update=next(n for n in ast.walk(sft_loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_update')
assert isinstance(sft_update.args[0],ast.Name) and sft_update.args[0].id=='sft_optimizer'
out={
 'environment':{'python':sys.version,'torch':importlib.metadata.version('torch'),'device':'source/AST/hash analysis only','model_computation':'none'},
 'source_sha256':hashlib.sha256(section).hexdigest(),'figure_sha256':digest(figure),'intro_sha256':None,
 'exact_body_quote':quote,'body_locator':'course/chapters/13.md#13.14, figure-explanation paragraph, source line463',
 'exact_figure_quote':label.text,'figure_locator':'SVG reference card text x=50,y=928',
 'figure_accessibility_description':desc,
 'implementation':{'path':source.relative_to(ROOT).as_posix(),'sha256':digest(source),
    'sft_loop_range':[sft_loop.lineno,sft_loop.end_lineno],'sft_optimizer_update_line':sft_update.lineno,
    'reference_assignment_line':ref.lineno,'reference_assignment':ast.unparse(ref),
    'observed_order':'SFT updates policy first; completed policy is then copied and frozen as reference.'},
 'prior_artifact_fingerprints_verified':fingerprints,
 'primary_locator_scope':{
    'dpo':'arXiv2305.18290v3 §3 PDFpp3–4: three RLHF phases include SFT; following Eq3 names reference as initial SFT model.',
    'instructgpt':'arXiv2203.02155v1 §3.1 three steps and §3.5 PDFp9 Eq2: supervise first, then PPO with SFT reference.'},
 'reader_context':'Current7.17 explicitly says SFT is a kind of post-training; current glossary says demonstrations/preferences/feedback. Current13.4 says reference usually preserves completed demonstration-finetuning start; current13.12 repeats whole-post-training-start wording.',
 'supported_freeze_scope':'Fixed completed-SFT snapshot during subsequent PPO/preference optimization in the inspected recipe, not the beginning of all post-training including SFT.'
}
(A/'inspection-facts.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
