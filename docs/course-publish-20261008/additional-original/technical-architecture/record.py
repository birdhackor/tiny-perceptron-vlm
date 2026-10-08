import json, hashlib, pathlib, datetime
BASE=pathlib.Path('/workspace/work/tutorial-audit-20261008')
OUT=pathlib.Path('/workspace/work/technical-architecture')
DB=OUT/'page-work.json'
rows=json.loads(DB.read_text()) if DB.exists() else []
def add(pid, quote, mechanism, purpose, checks, status='adequate', unverified=None):
 s=(BASE/'freeze/sources'/f'{pid}.md').read_text()
 assert quote in s,(pid,quote)
 rows.append({'page_id':pid,'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'actual_read_scope':'完整凍結頁（正文與details）；技術輪，並非逐段無提示首讀','source_judgment':{'status':status,'mechanism_property':mechanism,'need_purpose':purpose},'quoted_basis':[quote],'checks':checks,'unverified':unverified or ['尚未驗實際網站桌面/手機版面；未重演長訓或跨裝置品質。'],'discovery_source':'source-only initial technical judgment; no other reviewer or historical findings read','recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
def save(): DB.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
