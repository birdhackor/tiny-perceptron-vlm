import json,pathlib,hashlib,datetime
ROOT=pathlib.Path('/workspace/work/tutorial-audit-20261008')
DRAFT=ROOT/'reviewer-work/technical-learning/page-records.json'
M=json.load(open(ROOT/'manifest.json'))
INVENTORY={p['page_id']:p for p in M['inventory']['pages']}
pages=json.load(open(DRAFT)) if DRAFT.exists() else []
def add(pid,quote,judgment,mechanism,use,checks,unverified=None):
 s=(ROOT/'freeze/sources'/f'{pid}.md').read_text(); assert quote in s,(pid,quote)
 obj={'page_id':pid,'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'source_judgment':judgment,'quoted_basis':[{'quote':quote,'line':s[:s.index(quote)].count('\n')+1}], 'checks':{'mechanism_property':mechanism,'need_use':use,'technical_evidence':checks}, 'unverified':unverified or ['未逐一執行頁內短程式；判斷基於正文、相稱手算與已記實作回查。','未檢查網站桌面／手機實際頁面、折疊與跳轉路徑。'],'actual_reading':'完整凍結來源，包含details；技術輪，非逐段無提示首讀。','discovery_stage':'source_independent_technical_initial','other_review_access':False}
 assert obj['source_sha256']==INVENTORY[pid]['source_sha256']
 assert not any(p['page_id']==pid for p in pages),pid
 pages.append(obj)
def save(): DRAFT.write_text(json.dumps(pages,ensure_ascii=False,indent=2)+'\n')
