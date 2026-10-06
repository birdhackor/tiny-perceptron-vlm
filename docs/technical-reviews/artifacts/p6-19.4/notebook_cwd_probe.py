from pathlib import Path
import json
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
fence=re.findall(r'```python\n(.*?)```',(BASE/'inputs/19.4.md').read_text(),re.S)[0]
expected='moe-pretrain 完成 1000 選定 250\nmoe-sft 完成 8000 選定 8000\nmoe-vision 完成 1200 選定 1200\nmoe-ocr 完成 3000 選定 3000\nmoe-audio 完成 1000 選定 200\nmoe-joint 完成 6000 選定 2000\nmoe-native 完成 4000 選定 1000\n'
result=subprocess.run([sys.executable,'-c',fence],cwd=ROOT/'notebooks/19',capture_output=True,text=True,timeout=10)
assert result.returncode==0 and result.stdout==expected
(BASE/'notebook-cwd-probe.json').write_text(json.dumps({'python':sys.version.split()[0],'cwd':str(ROOT/'notebooks/19'),'command':[sys.executable,'-c',fence],'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'imports':'json, pathlib only; no network or checkpoint load'},ensure_ascii=False,indent=2)+'\n')
print(result.stdout,end='')
