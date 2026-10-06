from pathlib import Path
import hashlib
import json
import re
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.selftrained import train

commands=re.findall(r'```bash\n(.*?)```',(BASE/'inputs/19.4.md').read_text(),re.S)
rows=[]
for raw in commands:
    args=shlex.split(raw.replace('\\\n',' '))
    assert args[:9]==['uv','run','--frozen','--extra','cpu','--extra','selftrained','python','scripts/selftrained/train_local_stage.py']
    sep=args.index('--');wrapper=args[9:sep]
    manifest=wrapper[wrapper.index('--manifest')+1];pin=wrapper[wrapper.index('--manifest-sha256')+1]
    assert hashlib.sha256((ROOT/manifest).read_bytes()).hexdigest()==pin
    forwarded=args[sep+1:]+['--records','synthetic.jsonl','--asset-dir','synthetic-assets']
    parser=train.parser();parser.allow_abbrev=False
    parsed=parser.parse_args(forwarded)
    rows.append({'command':raw.strip(),'parsed':vars(parsed),'manifest_sha256_verified':True,'fully_executed':False,'reason':'Original recipe is full training; exact parser and separate real bounded synthetic wrapper runs validate the software contract without original full data.'})
version=subprocess.run(['uv','--version'],capture_output=True,text=True,check=True).stdout.strip()
out={'uv_version':version,'commands':rows}
(BASE/'command-contract.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('Two documented shell recipes parsed with exact arguments; manifest pins matched;',version)
