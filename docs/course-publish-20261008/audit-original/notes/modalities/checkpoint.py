import json,sys,subprocess
from pathlib import Path
root=Path('/workspace/work/tutorial-audit-20261008')
n=json.load(sys.stdin)
p=root/'notes/modalities'/f"{n['page_id']}-{n['unit_index']}.json"
p.write_text(json.dumps(n,ensure_ascii=False,indent=2))
subprocess.run([sys.executable,str(root/'reader.py'),'next','modalities','--note',str(p)],check=True,stdout=subprocess.DEVNULL)
subprocess.run([sys.executable,str(root/'reader.py'),'show','modalities'],check=True)
