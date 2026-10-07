"""Formatting only: expand this owner's explicit shorthand into the review schema."""
import json,subprocess,sys
r=json.load(sys.stdin)
for c in r['claims']:
 c['evidence']=[dict(zip(('source_id','locator','supports'),e)) if isinstance(e,list) else e for e in c['evidence']]
 if 'verification' in c and isinstance(c['verification'],list):
  c['verification']=dict(zip(('method','expected','observed','details','tolerance'),c['verification']))
r['checks']={k:dict(zip(('status','details','claim_ids'),v)) if isinstance(v,list) else v for k,v in r['checks'].items()}
subprocess.run(['.venv/bin/python','docs/technical-reviews/artifacts/p7_technical_c/save-page.py'],input=json.dumps(r,ensure_ascii=False),text=True,check=True)
