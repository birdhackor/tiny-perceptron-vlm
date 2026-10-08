import json, pathlib
B=pathlib.Path('/workspace/work/tutorial-audit-20261008')
D=B/'work/continuity-modalities'
def add(rows):
 p=D/'actual-page-notes.json'
 old=json.loads(p.read_text()) if p.exists() else []
 assert not (set(r['page_id'] for r in rows)&set(r['page_id'] for r in old))
 for r in rows:
  r['reading_scope']='全文銜接主文審；尚未讀本组折疊，不是逐段盲讀測試'
 p.write_text(json.dumps(old+rows,ensure_ascii=False,indent=2)+'\n')
def n(pid,quote,identity,mechanism,need,operation,example,dependencies,transition,visual='無圖；目前關係可由文字與程式的對應辨認',unknown=None,issues=None):
 return dict(page_id=pid,quoted_basis=[quote],source_judgment=('需補說明' if issues else '已足夠於原承諾範圍'),checks=dict(identity=identity,mechanism={'answer':mechanism,'judgment':'已足夠'},need={'answer':need,'judgment':'已足夠'},operation=operation,example_and_limit=example,dependencies=dependencies,transitions=transition),visual=visual,unverified=unknown or ['未執行程式；輸出僅依正文核算','未驗證實際網頁版面與互動'],issue_ids=issues or [])
