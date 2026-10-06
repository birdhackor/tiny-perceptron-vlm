"""Actual V4 source/AST/fingerprint checks; no model computation or training."""
import ast
import hashlib
import importlib.metadata
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
A=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
prior=json.loads((A/'prior-opaque/revise-13.14-report.json').read_bytes())
assert sha(A/'prior-opaque/revise-13.14-report.json')=='77ff086690cd25ba76569f3549dfea9ae019e2606b30103970ea22ed05a0b9d2'
fingerprints=[]
for e in prior['artifacts']:
 p=ROOT/e['path'];assert p.is_file() and not p.is_symlink();observed=sha(p);assert observed==e['sha256'];fingerprints.append({'artifact_id':e['id'],'path':e['path'],'sha256':observed,'unchanged':True})
current_files=[]
for path,expected in [('scripts/course_experiments/posttraining.py','862dc12680fee8374679d52a1cc55a2dd325a7aebbf634668731fa7def08aa04'),('tiny_perceptron/posttraining.py','3d0e2ae3b29f95abebde9b10d0cd106639a131fab1b2c7a509e732ba8daf295b'),('docs/course-experiments/results/posttraining.json','95c8ee6c89b009a9646903c1bccaa6a0ace53786a608307d50ba7d9fb0a09172')]:
 assert sha(ROOT/path)==expected;current_files.append({'path':path,'sha256':expected,'unchanged':True})
body=(A/'current-13.14.md').read_bytes();assert sha(A/'current-13.14.md')=='ba2361526193f3321b9d8974a6889117fe4602a5e2d6d132413d0d22ab9d6ab4'
phrase='reference保留PPO開始時、已完成示範微調的策略，在這段PPO期間固定。';assert phrase in body.decode()
old=(ROOT/'docs/technical-reviews/artifacts/phase4-13_14-reference-scope/current-13.14.md').read_bytes()
old_fences=re.findall(rb'(?ms)^```python\n(.*?)^```',old);new_fences=re.findall(rb'(?ms)^```python\n(.*?)^```',body)
assert old_fences==new_fences and len(new_fences)==2
fences=[]
for n,code in enumerate(new_fences,1):
 p=A/f'current-fence-{n}.py';p.write_bytes(code);fences.append({'fence':n,'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'byte_identical_to_own_prior_executable':True})
svg=ROOT/'course/figures/rewrite-13-model-roles.svg';assert sha(svg)=='d7589ab9e44118540111e8f331b555d4d6c4db3f548e7d373bf7d40750eba406'
t=ET.fromstring(svg.read_bytes());ns={'s':'http://www.w3.org/2000/svg'};labels=[n.text for n in t.findall('s:text',ns)][-3:];desc=t.find('s:desc',ns).text
assert labels==['reference：PPO起點','看情境 → 起點各卡機率','整段PPO期間不更新']
assert '已完成示範微調' in desc and '整段PPO期間固定' in desc
tree=ast.parse((ROOT/'scripts/course_experiments/posttraining.py').read_bytes());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_posttraining')
assignments={t.id:n for n in run.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
ref=assignments['reference'];ppo=assignments['ppo'];assert ref.lineno==289 and ppo.lineno==326
sft_loop=next(n for n in run.body if isinstance(n,ast.For) and any(isinstance(x,ast.Constant) and x.value=='sft_steps' for x in ast.walk(n.iter)))
assert sft_loop.end_lineno<ref.lineno<ppo.lineno
out={'environment':{'python':sys.version,'torch':importlib.metadata.version('torch'),'device':'source/AST/hash analysis only','model_computation':'none','shell':'bash login:false'},
 'source_sha256':sha(A/'current-13.14.md'),'figure_sha256':sha(svg),'intro_sha256':None,
 'current_body_quote':phrase,'current_visible_figure_labels':labels,'current_accessibility_description':desc,
 'prior_full_revise_report':{'path':(A/'prior-opaque/revise-13.14-report.json').relative_to(ROOT).as_posix(),'sha256':sha(A/'prior-opaque/revise-13.14-report.json')},
 'prior_fingerprints_verified':fingerprints,'current_unchanged_files':current_files,'current_fences':fences,
 'implementation_contract':{'sft_loop_range':[sft_loop.lineno,sft_loop.end_lineno],'reference_line':ref.lineno,'reference_assignment':ast.unparse(ref),'ppo_initialization_line':ppo.lineno,'ppo_initialization':ast.unparse(ppo)},
 'primary_support_locators':['DPOarXiv2305.18290v3§3PDFpp3–4Eq3 and following initialSFT reference definition','InstructGPTarXiv2203.02155v1§3.1three steps and§3.5PDFp9Eq2','Original recipe278–289 and326–354'],
 'actual_visual_action':'Current SVG independently rendered with Inkscape at640px, actual PNG loaded via view_image and reference card personally inspected.',
 'actual_served_page':json.loads((A/'page-fetch.json').read_bytes()),
 'support_scope':'Completed-SFT policy is the initial PPO policy and stays fixed as reference during this PPO stage; no all-post-training interval remains in current13.14 text or figure.'}
(A/'inspection-facts.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
