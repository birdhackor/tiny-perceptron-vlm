import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
current_report=ROOT/'docs/technical-reviews/6.8.json'
prior_raw=current_report.read_bytes()
prior_sha=sha(prior_raw)
assert prior_sha=='612601e21bbefaa5e5de9de72023be28daaaa7cdbb8726d6d82de5d5daceb287'
history=ROOT/'docs/technical-reviews/history'/f'phase4-6_8-own-before-context-reinspection-{prior_sha}.json'
if history.exists():assert history.read_bytes()==prior_raw
else:history.write_bytes(prior_raw)
assert history.read_bytes()==prior_raw

def section(raw,key):
 headings=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
 index=next(i for i,h in enumerate(headings) if h.group().split()[1].decode()==key)
 return raw[headings[index].start():headings[index+1].start() if index+1<len(headings) else len(raw)]
old_chapter=(OUT.parent/'inputs/current/course/chapters/06.md').read_bytes()
current_chapter=(ROOT/'course/chapters/06.md').read_bytes()
metadata={'prior_history_path':history.relative_to(ROOT).as_posix(),'prior_history_sha256':prior_sha,'sections':{},'figure_sources':[],'snapshots':[],'scope':'Only own6.8 and necessary6.5/6.6 context, not whole-chapter review. Full-chapter old hash denotes own frozen historical input only.'}
(OUT/'inputs').mkdir(exist_ok=True)
for key in ('6.8','6.5','6.6'):
 new=section(current_chapter,key)
 old=section(old_chapter,key)
 (OUT/'inputs'/f'current-{key}.md').write_bytes(new)
 (OUT/'inputs'/f'own-frozen-{key}.md').write_bytes(old)
 delta=''.join(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile=f'own-frozen-{key}',tofile=f'current-{key}'))
 (OUT/'inputs'/f'{key}.diff').write_text(delta)
 metadata['sections'][key]={'current_sha256':sha(new),'own_frozen_sha256':sha(old),'unchanged_vs_own_original_frozen':new==old,'current_path':(OUT/'inputs'/f'current-{key}.md').relative_to(ROOT).as_posix()}
 if key=='6.8':assert sha(new)=='270ca0d9b3408029ec1da29fc3c38d02273fe85d411323835043866cf4b1709e'
 if key=='6.5':
  for reference in re.findall(r'!\[[^\]]*\]\(([^)]+\.svg)\)',new.decode()):
   p=(ROOT/'course/chapters'/reference).resolve()
   target=OUT/'figures'/p.name
   target.parent.mkdir(exist_ok=True)
   target.write_bytes(p.read_bytes())
   metadata['figure_sources'].append({'path':p.relative_to(ROOT).as_posix(),'snapshot':target.relative_to(ROOT).as_posix(),'sha256':sha(p.read_bytes()),'equal_to_snapshot':target.read_bytes()==p.read_bytes()})
for name in ('docs/review-tools/factual-reviewer-instructions.md','scripts/check_technical_reviews.py','docs/review-tools/section_facts.py'):
 target=OUT/'inputs'/Path(name).name
 target.write_bytes((ROOT/name).read_bytes())
 metadata['snapshots'].append({'origin':name,'snapshot':target.relative_to(ROOT).as_posix(),'sha256':sha(target.read_bytes())})
report=json.loads(prior_raw)
metadata['prior_artifact_hash_checks']=[]
for a in report['artifacts']:
 digest=sha((ROOT/a['path']).read_bytes())
 assert digest==a['sha256'],a['path']
 metadata['prior_artifact_hash_checks'].append({'path':a['path'],'sha256':digest,'matches':True})
metadata['reused_current_code_checks']=[]
for name in ('tiny_perceptron/data.py','tiny_perceptron/model.py','scripts/course_experiments/common.py','scripts/course_experiments/text.py'):
 snapshot=OUT.parent/'inputs/current'/name
 assert (ROOT/name).read_bytes()==snapshot.read_bytes(),name
 metadata['reused_current_code_checks'].append({'path':name,'sha256':sha(snapshot.read_bytes()),'matches_own_original':True})
for lesson in ('6.8','6.5'):
 url=f'http://127.0.0.1:8765/{lesson}.html'
 with urllib.request.urlopen(url,timeout=10) as response:
  raw=response.read()
  (OUT/f'preview-{lesson}.html').write_bytes(raw)
  metadata.setdefault('preview',[]).append({'url':url,'http_status':response.status,'sha256':sha(raw),'path':(OUT/f'preview-{lesson}.html').relative_to(ROOT).as_posix()})
commands=[]
for figure in metadata['figure_sources']:
 source=ROOT/figure['snapshot']
 target=source.with_suffix('.png')
 argv=['inkscape',str(source),'--export-type=png','--export-width=1400',f'--export-filename={target}']
 result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=30,check=False)
 name=source.stem
 (OUT/f'render-{name}.stdout.txt').write_text(result.stdout)
 (OUT/f'render-{name}.stderr.txt').write_text(result.stderr)
 assert result.returncode==0 and target.is_file(),result.stderr
 commands.append({'command_argv':argv,'exit_code':result.returncode,'output':target.relative_to(ROOT).as_posix(),'output_sha256':sha(target.read_bytes())})
argv=['chromium','--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--hide-scrollbars','--window-size=1280,1500',f'--screenshot={OUT/"preview-6.8.png"}','http://127.0.0.1:8765/6.8.html']
try:
 result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=20,check=False)
 (OUT/'chromium.stdout.txt').write_text(result.stdout)
 (OUT/'chromium.stderr.txt').write_text(result.stderr)
 commands.append({'command_argv':argv,'exit_code':result.returncode,'scope':'Own6.8 live preview screenshot, bounded20s.'})
except subprocess.TimeoutExpired as error:
 (OUT/'chromium.stdout.txt').write_bytes(error.stdout or b'')
 (OUT/'chromium.stderr.txt').write_bytes(error.stderr or b'')
 commands.append({'command_argv':argv,'exit_code':124,'timed_out':True,'scope':'No browser visibility claim on timeout; current necessarySVGs separately rendered by Inkscape.'})
metadata['render_commands']=commands
metadata['tool_versions']={name:subprocess.run([name,'--version'],capture_output=True,text=True,check=True).stdout.strip() for name in ('inkscape','chromium')}
sys.path.insert(0,str(ROOT))
import torch
import tokenizers
assert torch.version.cuda is None and not torch.cuda.is_available()
metadata['environment']={'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),'tokenizers':tokenizers.__version__,'device':'cpu','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'cwd':str(ROOT),**metadata['tool_versions']}
(OUT/'context-input-and-render-receipt.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'prior_history_path':metadata['prior_history_path'],'prior_history_sha256':prior_sha,'sections':metadata['sections'],'figure_sources':metadata['figure_sources'],'prior_artifact_hash_count':len(metadata['prior_artifact_hash_checks']),'renders':commands,'environment':metadata['environment']},ensure_ascii=False,indent=2))
