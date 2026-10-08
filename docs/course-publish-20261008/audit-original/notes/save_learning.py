import sys,json,subprocess
from pathlib import Path
B=Path('/workspace/work/tutorial-audit-20261008')
n=json.load(sys.stdin)
n['reviewer']='/root/read_learning'
s=json.loads((B/'sessions/learning.json').read_text())
assert n['page_id']==json.loads((B/'manifest.json').read_text())['groups']['learning']['pages'][s['page_index']]
assert n['unit_index']==s['unit_index']
p=B/'notes'/f"learning-{s['phase']}-{n['page_id']}-{n['unit_index']}.json"
p.write_text(json.dumps(n,ensure_ascii=False,indent=2))
subprocess.run(['python',str(B/'reader.py'),'next','learning','--note',str(p)],check=True)
