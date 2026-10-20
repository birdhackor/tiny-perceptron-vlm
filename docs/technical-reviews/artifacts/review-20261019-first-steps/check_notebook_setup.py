"""在證據目錄內暫建 Notebook 環境，核對 kernel 與伺服器後移除暫建環境。"""

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path.cwd()
ART = Path(__file__).resolve().parent
RUNTIME = ART / ".runtime"

if "--child" in sys.argv:
    import importlib.metadata
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(ROOT / "notebooks/01/1.1.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace('text = "貓看狗，狗看貓。"', 'text = "鳥看狗，狗看鳥。"')
    client = NotebookClient(notebook, kernel_name="tiny-perceptron", timeout=40,
                            resources={"metadata":{"path":str(ROOT / "notebooks/01")}})
    client.execute()
    outputs = [o.get("text","") for c in notebook.cells if c.cell_type == "code" for o in c.get("outputs",[])
               if o.output_type == "stream"]
    print(json.dumps({"versions":{p:importlib.metadata.version(p) for p in ["ipykernel","jupyterlab","nbclient"]},
        "kernel_stdout": outputs, "error_outputs": sum(o.output_type=="error" for c in notebook.cells
                    if c.cell_type=="code" for o in c.get("outputs",[]))},ensure_ascii=False))
    raise SystemExit()

RUNTIME.mkdir()
env = dict(os.environ)
for key, relative in {"UV_CACHE_DIR":"cache", "TMPDIR":"tmp", "XDG_CACHE_HOME":"cache",
                      "IPYTHONDIR":"ipython", "JUPYTER_RUNTIME_DIR":"jupyter-runtime",
                      "JUPYTER_CONFIG_DIR":"jupyter-config", "JUPYTER_DATA_DIR":"jupyter-data",
                      "MPLCONFIGDIR":"matplotlib"}.items():
    directory = RUNTIME / relative
    directory.mkdir(exist_ok=True)
    env[key] = str(directory)
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["JUPYTER_PLATFORM_DIRS"] = "1"
env["PYTHONPATH"] = str(ROOT)
records = []


def run(command, timeout=120):
    p = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    records.append({"command":command,"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr})
    if p.returncode:
        raise RuntimeError("命令執行失敗")


server = None
try:
    run(["uv","venv","--python",sys.executable,str(RUNTIME / "env")])
    python = str(RUNTIME / "env/bin/python")
    run(["uv","pip","install","--python",python,"-r",str(ART / "notebook-requirements.txt")])
    # 重用既有 PyTorch，只在暫建環境加入讀取路徑。
    site = RUNTIME / "env/lib" / ("python" + platform.python_version().rsplit(".",1)[0]) / "site-packages"
    existing = ROOT / ".venv/lib" / ("python" + platform.python_version().rsplit(".",1)[0]) / "site-packages"
    (site / "project-torch.pth").write_text(str(existing) + "\n")
    run([python,"-m","ipykernel","install","--sys-prefix","--name","tiny-perceptron",
         "--display-name","Tiny Perceptron"])
    spec = RUNTIME / "env/share/jupyter/kernels/tiny-perceptron/kernel.json"
    records.append({"kernel_spec":json.loads(spec.read_text())})
    run([python,str(Path(__file__).resolve()),"--child"],timeout=90)
    command = [python,"-m","jupyterlab","notebooks","--no-browser","--ServerApp.port=0",
               "--ServerApp.port_retries=0","--IdentityProvider.token=review-local"]
    server_log = RUNTIME / "server.log"
    with server_log.open("w") as stream:
        server = subprocess.Popen(command,env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,text=True)
    info = None
    for _ in range(100):
        for file in (RUNTIME / "jupyter-runtime").glob("jpserver-*.json"):
            info = json.loads(file.read_text())
        if info:
            break
        if server.poll() is not None:
            break
        time.sleep(0.1)
    if not info:
        raise RuntimeError("伺服器沒有提供啟動資訊")
    url = info["url"].replace("localhost","127.0.0.1") + "api/contents/01/1.1.ipynb?content=0"
    request = urllib.request.Request(url,headers={"Authorization":"token review-local"})
    with urllib.request.urlopen(request,timeout=10) as response:
        listing = json.load(response)
    records.append({"command":command,"server_root":info.get("root_dir"),
                    "notebook_api":listing,"server_log":server_log.read_text()})
except Exception as error:
    records.append({"error":type(error).__name__,"details":str(error)})
finally:
    if server:
        server.terminate()
        server.wait(timeout=15)
    shutil.rmtree(RUNTIME)
print(json.dumps({"command":"PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "environment":{"python":platform.python_version(),"device":"cpu","os":platform.platform()},
    "records":records,"temporary_environment_removed":not RUNTIME.exists()},ensure_ascii=False,indent=2))
