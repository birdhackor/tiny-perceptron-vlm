import json, sys, subprocess
from pathlib import Path
b=Path(__file__).resolve().parents[1]
n=json.load(sys.stdin)
p=b/'notes'/'architecture-checkpoint.json'
p.write_text(json.dumps(n,ensure_ascii=False,indent=2))
subprocess.run([sys.executable,str(b/'reader.py'),'next','architecture','--note',str(p)],check=True)
