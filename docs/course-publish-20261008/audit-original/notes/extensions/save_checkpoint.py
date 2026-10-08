from pathlib import Path
import json,sys,subprocess
b=Path(__file__).resolve().parent
n=json.load(sys.stdin)
n['reviewer']='/root/read_extensions'
p=b/(n['page_id']+'-u'+str(n['unit_index']).zfill(2)+'.json')
assert not p.exists(), 'Do not overwrite original checkpoints'
p.write_text(json.dumps(n,ensure_ascii=False,indent=2))
subprocess.run(['python',str(b.parent.parent/'reader.py'),'next','extensions','--note',str(p)],check=True)
