"""Record commands, stdout/stderr, exit status, versions, and limits for this review."""
from pathlib import Path
import hashlib, json, subprocess, sys, time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
receipts = []
commands = [
    ("verify", [str(ROOT/".venv/bin/python"), str(BASE/"verify.py")], 30),
    ("render", ["inkscape", "course/figures/rewrite-04-shift-causal.svg", "--export-type=png", "--export-filename="+str(BASE/"figure.png")], 30),
    ("inkscape-version", ["inkscape", "--version"], 10),
]
for name, argv, timeout in commands:
    started = time.perf_counter()
    result = subprocess.run(argv,cwd=ROOT,capture_output=True,timeout=timeout)
    for stream in ["stdout", "stderr"]:
        (BASE/f"{name}.{stream}.txt").write_bytes(getattr(result,stream))
    receipt = {"name":name,"command_argv":argv,"cwd":str(ROOT),"timeout_seconds":timeout,"exit_code":result.returncode,"elapsed_seconds":time.perf_counter()-started,"stdout_path":f"{name}.stdout.txt","stderr_path":f"{name}.stderr.txt","python":sys.version,"device":"CPU"}
    if name == "verify" and result.returncode == 0:
        receipt["environment"] = json.loads((BASE/"environment.json").read_text())
    if name == "render" and result.returncode == 0:
        receipt["output_sha256"] = hashlib.sha256((BASE/"figure.png").read_bytes()).hexdigest()
    receipts.append(receipt)
    print(json.dumps(receipt))
    (BASE/"checks-receipt.json").write_text(json.dumps(receipts,indent=2)+"\n")
    result.check_returncode()
