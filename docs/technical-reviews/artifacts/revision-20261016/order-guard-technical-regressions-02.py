"""以未變原17案例對當前Windows-path修正重跑；原收據保持不動。"""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[4]
report=root/'docs/course-revision-20261016/reviews/order-guard-technical-callback-01.json'
assert hashlib.sha256(report.read_bytes()).hexdigest()=='cf8881287231ffdab61a0d2acc93d2e9c02c95399318d2495bfad4bb1e350deb'
reference=next(item for item in json.loads(report.read_bytes())['artifacts'] if item['id']=='probe_code')
path=root/reference['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['sha256']
source=path.read_text();assert source.count('order-guard-technical-probes-results-01.json')==1
source=source.replace('order-guard-technical-probes-results-01.json','order-guard-technical-probes-results-02.json')
exec(compile(source,str(__file__),'exec'),globals())
