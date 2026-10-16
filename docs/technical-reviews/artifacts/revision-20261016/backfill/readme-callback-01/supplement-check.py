"""Local-only product loads, recorded training jobs, and current changed dependency inspection."""
from datetime import datetime, UTC
import difflib
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from unittest.mock import patch
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).parent
SCRATCH=ROOT/'work/readme-backfill-callback-01-attempt02'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
import torch
from tiny_perceptron.lesson_products.inference import Product,auto_solve
from tiny_perceptron.lesson_products.training import state_digest
torch.set_num_threads(2)
b,c=Product(SCRATCH/'products/b','b'),Product(SCRATCH/'products/c','c')
r={'at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/tech_transport_readme_20261016','current_context':'informed','environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'}}
with patch('urllib.request.urlopen',side_effect=AssertionError('Inference must not contact the network')):
 r['offline_library_calls']={
  'solve':b.solve('兩盒追蹤;A=2;B=6;事件=交換;查詢=B',3),
  'judge':c.judge('卡片進度;申請編號=?','info'),
  'auto':auto_solve(b,c,'兩盒追蹤;A=2;B=6;事件=交換;查詢=B','auto'),
 }
r['actual_loaded_weight_bindings']={}
for product,model in (('b',b),('c',c)):
 receipt=json.loads((SCRATCH/'products'/product/'training-receipt.json').read_bytes())
 digest=state_digest(model.model.state_dict())
 state=torch.load(SCRATCH/'products'/product/'model.pt',weights_only=True,map_location='cpu')
 r['actual_loaded_weight_bindings'][product]={'state_digest':digest,'training_selected_digest':receipt['selected_weights_sha256'],'matches':digest==receipt['selected_weights_sha256'],'inference_file_is_tensor_state_only':all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()),'class':type(model.model).__name__}
 assert digest==receipt['selected_weights_sha256']
r['training_run_api']=[]
for run_id in (38864076350,38864352441):
 url=f'https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/actions/runs/{run_id}'
 with urlopen(Request(url,headers={'User-Agent':'readme-owner-informed-callback'}),timeout=30) as response:raw=response.read()
 value=json.loads(raw)
 r['training_run_api'].append({'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'id':value['id'],'head_sha':value['head_sha'],'status':value['status'],'conclusion':value['conclusion']})
f14=json.loads((ROOT/'docs/course-revision-20261016/freeze-14.json').read_bytes())
f15=json.loads((ROOT/'docs/course-revision-20261016/freeze-15.json').read_bytes())
page=next(p for p in f14['pages'] if p['page_id']=='19.19')
before=(ROOT/page['snapshot']).read_text()
raw=(ROOT/'course/chapters/19.md').read_text()
current=re.search(r'^## 19\.19 .*?(?=^## |\Z)',raw,re.S|re.M)[0]
r['dependency_callback_15']={'chapter19_path':'course/chapters/19.md','current_sha256':sha(ROOT/'course/chapters/19.md'),'readme14':next(p for p in f14['pages'] if p['page_id']=='readme'),'readme15':next(p for p in f15['pages'] if p['page_id']=='readme'),'diff_19_19_from_14_to_current':''.join(difflib.unified_diff(before.splitlines(True),current.splitlines(True),fromfile=page['snapshot'],tofile='current19.19')),'judgment':'Actually read both source changes: standalone B takes --loops; effort and --max-loops belong to Auto. These match the personally inspected CLI/parser and actual commands. README bytes and its figure did not change from freeze14 to15; 19.13/19.20 claim support is unchanged. This records source inspection only and does not adopt another section owner review.'}
assert r['dependency_callback_15']['readme14']['source_sha256']==r['dependency_callback_15']['readme15']['source_sha256']==sha(ROOT/'README.md')
r['inspection_notes']={
 'readme_change':'Personally read the entire current README. Only the final 教材提供什麼 paragraph changed relative to the old reviewed body: pending B/C acceptance became actual results and product-guide links. Its A v2 and old experiment claims remain the same.',
 'route_change':'Read the current course/README diff: A results remain19.12/19.11; new B/C results now19.20 and operations19.19. Required effort/loop/probability routes remain the same.',
 'publication':'Read current publication.py verify_release and publish.py publication flow plus workflow_dispatch and contents permission. Matched every receipt source hash at actual execution commit b137418e..., not assumed current-main path. Historical release-v1 descriptor path differs from its retained results/publication-spec-v1.json copy as RELEASE explicitly explains.',
 'bounded_scope':'No training, optimizer step, formal evaluate_frozen/select_settings call, paid job, HF/GitHub write, or HTTP service launch. Independently re-counted saved raw scores and ran only necessary local inference/download CLI checks. Other reviewer conclusions were not read.',
}
(OUT/'supplement-results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'offline':{k:{a:v[a] for a in ('answer','selected','status','b_loops') if a in v} for k,v in r['offline_library_calls'].items()},'weight_bindings':r['actual_loaded_weight_bindings'],'training_runs':r['training_run_api'],'chapter19_sha256':r['dependency_callback_15']['current_sha256']},ensure_ascii=False))
