"""Parse the exact official bytes fetched by probe_current_pricing.py; no network."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,UTC
from decimal import Decimal,ROUND_CEILING
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'current-source'))
from scripts.omni_revision import cpu_execution as cpu
folder=BASE/'official-pricing-current';urls={'actions':cpu.STORAGE_URL,'repository':'https://api.github.com/repos/birdhackor/tiny-perceptron-vlm','releases':'https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases'}
raws={n:(folder/(n+'.raw')).read_bytes() for n in urls}
class Reply:
 url=cpu.STORAGE_URL
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def read(self,n):return raws['actions'][:n]
with patch.object(cpu,'command',return_value=raws['repository']),patch.object(cpu.urllib.request,'urlopen',return_value=Reply()):quote=cpu.pricing()
p=cpu.PricingText();p.feed(raws['releases'].decode());text=' '.join(p.parts);excerpt='Each file included in a release must be under 2 GiB. There is no limit on the total size of a release, nor bandwidth usage.';assert excerpt in text
storage=(Decimal(cpu.MAX_ARTIFACT)/1024**3*7*Decimal(quote['storage']['conservative_daily_usd_per_gib'])).quantize(Decimal('.01'),rounding=ROUND_CEILING)
record={'scope':'Read-only official URLs fetched once by probe_current_pricing.py. Production pricing parsed those exact actual bytes with readonly transport fixtures. Original fetch probe stopped only on stale wording for an auxiliary Release quotation, not pricing validation.','sources':{n:{'source_url':u,'raw_path':str(folder/(n+'.raw')),'raw_sha256':hashlib.sha256(raws[n]).hexdigest(),'bytes':len(raws[n]),'saved_at_utc':datetime.fromtimestamp((folder/(n+'.raw')).stat().st_mtime,UTC).isoformat()} for n,u in urls.items()},'production_pricing':quote,'release_scope_excerpt':excerpt,'artifact_storage_ceiling_usd':str(storage),'cleanup_margin_usd':str(Decimal('.10')-storage),'artifact_bound_within_retained_point10':storage<=Decimal('.10'),'formal_or_paid_calls':0}
p=BASE/'current-pricing-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
