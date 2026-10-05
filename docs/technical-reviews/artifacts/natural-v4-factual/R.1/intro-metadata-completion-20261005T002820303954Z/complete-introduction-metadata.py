"""Add only the original owner's reread introduction metadata to R.1."""
from pathlib import Path
import hashlib
import json
import re

folder=Path(__file__).resolve().parent
report=Path('docs/technical-reviews/R.1.json')
before=(folder/'report-before.json').read_bytes()
assert report.read_bytes()==before
original=json.loads(before)
raw=Path('course/README.md').read_bytes().decode('utf-8')
headings=list(re.finditer(r'^## .+$',raw,re.M))
assert headings[0][0].startswith('## R.1 ')
intro=raw[:headings[0].start()].encode('utf-8')
body=raw[headings[0].start():headings[1].start()].encode('utf-8')
assert intro==(folder/'raw-introduction.md').read_bytes()
assert hashlib.sha256(intro).hexdigest()==original['introduction_sha256']
assert hashlib.sha256(body).hexdigest()==original['source_sha256']
updated=dict(original)
updated['intro_sha256']=hashlib.sha256(intro).hexdigest()
updated['intro_summary']='導言從手機輸入「今天天氣」後出現接續候選字的經驗切入，說明教材先以少量文字觀察記錄例子、猜測、錯誤代價與修改猜法，再把這些步驟組成處理較長文字、圖片與聲音的模型。這是課程路線的介紹，並未宣稱第一個編碼例子已完成模型學習。'
assert {k:v for k,v in updated.items() if k not in {'intro_sha256','intro_summary'}}==original
after=(json.dumps(updated,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
report.write_bytes(after)
(folder/'report-after.json').write_bytes(after)
receipt={'reviewer_task':'/root/v4_review_coordinator/factual_v4_r_1',
    'action':'Original owner fully reread current introduction and added exactly intro_sha256 and intro_summary.',
    'report_before_sha256':hashlib.sha256(before).hexdigest(),
    'report_after_sha256':hashlib.sha256(after).hexdigest(),
    'source_sha256':updated['source_sha256'],
    'intro_sha256':updated['intro_sha256'],
    'intro_summary':updated['intro_summary'],
    'original_report_fields_preserved':True,
    'reused_prior_substantive_evidence':True,
    'reuse_scope':'All prior claims, judgments, sources, code, figures and CPU/browser evidence remain unchanged; no new authority/CPU/render/full-round execution is claimed.',
    'command':'.venv/bin/python '+str(folder.relative_to(Path.cwd()))+'/complete-introduction-metadata.py'}
(folder/'metadata-completion-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
