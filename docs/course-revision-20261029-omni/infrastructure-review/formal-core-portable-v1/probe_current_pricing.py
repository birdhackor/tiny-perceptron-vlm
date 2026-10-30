"""Read-only current official cost evidence, local raw source outside Git."""
import concurrent.futures,hashlib,json,sys,urllib.request
from datetime import datetime,UTC
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'current-source'))
from scripts.omni_revision import cpu_execution as cpu
folder=BASE/'official-pricing-current';folder.mkdir(exist_ok=False)
urls={'actions':cpu.STORAGE_URL,'repository':'https://api.github.com/repos/birdhackor/tiny-perceptron-vlm','releases':'https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases'}
def get(item):
 name,url=item
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'independent-omni-core-review','Accept':'application/json' if name=='repository' else 'text/html'}),timeout=30) as response:
  raw=response.read(2*1024**2+1);final=response.url
 if len(raw)>2*1024**2:raise ValueError(name+' source too large')
 (folder/(name+'.raw')).write_bytes(raw)
 return name,{'source_url':url,'final_url':final,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'observed_at_utc':datetime.now(UTC).isoformat()},raw
results={}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for n,m,raw in pool.map(get,urls.items()):results[n]=(m,raw)
class Reply:
 url=cpu.STORAGE_URL
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def read(self,n):return results['actions'][1][:n]
with patch.object(cpu,'command',return_value=results['repository'][1]),patch.object(cpu.urllib.request,'urlopen',return_value=Reply()):quote=cpu.pricing()
parser=cpu.PricingText();parser.feed(results['releases'][1].decode());text=' '.join(parser.parts);needle='Neither the total size of a release nor the bandwidth used to deliver binary files is limited.'
assert needle in text, 'release scope excerpt not found'
from decimal import Decimal,ROUND_CEILING
storage=(Decimal(cpu.MAX_ARTIFACT)/1024**3*7*Decimal(quote['storage']['conservative_daily_usd_per_gib'])).quantize(Decimal('.01'),rounding=ROUND_CEILING)
record={'scope':'Actual readonly official URLs fetched once; production pricing function parsed those exact current bytes with only transport fixtures, no GH/Modal writes.','sources':{n:m for n,(m,_) in results.items()},'actual_production_pricing':quote,'release_scope_excerpt':needle,'release_scope_limit':'This excerpt establishes release delivery size/bandwidth policy; it is not a successful model retention receipt.','artifact_storage_ceiling_usd':str(storage),'cleanup_margin_usd':str(Decimal('.10')-storage),'artifact_bound_within_retained_point10':storage<=Decimal('.10'),'formal_or_paid_calls':0}
p=BASE/'current-pricing-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
