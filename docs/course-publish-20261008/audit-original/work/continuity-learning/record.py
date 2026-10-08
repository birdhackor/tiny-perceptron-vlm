import json,pathlib,datetime
B=pathlib.Path('/workspace/work/tutorial-audit-20261008'); M=json.loads((B/'manifest.json').read_text()); P={p['page_id']:p for p in M['inventory']['pages']}
def save(rows):
 with (B/'work/continuity-learning/actual-judgments.jsonl').open('a') as f:
  for pid,quote,deps,focus,mechanism,need,example,judgment in rows:
   f.write(json.dumps({'page_id':pid,'source_sha256':P[pid]['source_sha256'],'quoted_basis':quote,'actual_dependencies':deps,'current_learning_check':focus,'mechanism_property':{'answer':mechanism,'judgment':'已足夠' if judgment=='pass' else judgment},'necessity_use':{'answer':need,'judgment':'已足夠' if judgment=='pass' else judgment},'example_scope_and_next_use':example,'source_judgment':judgment,'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'provenance':'獨立凍結全文銜接審；未讀reports/synthesis/notes/traces/checks人類判斷'},ensure_ascii=False)+'\n')
