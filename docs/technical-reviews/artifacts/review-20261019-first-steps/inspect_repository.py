"""保存導覽、安裝計畫、現行程式與凍結圖的核對證據。"""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path.cwd()
ART = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


records = {"command":"PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "environment":{"python":platform.python_version(),"device":"cpu","os":platform.platform()}}
records["repository_sources"] = []
for name, needles in {
    "pyproject.toml":["[dependency-groups]", "[tool.uv.sources]", "requires-python"],
    "scripts/build_course.py":["BOOTSTRAP =", "def build"],
    "scripts/export_course.py":["def output_html", "def checked_notebook", "def section_footer"],
    "scripts/check_notebooks.py":["def run_kernel"],
    ".github/workflows/pages.yml":["- run: uv sync", "Register notebook kernel"],
    "scripts/check_env.py":["def main", "# 在選到的裝置"],
    "scripts/train.py":["if args.train:", '"mode":'],
    ".gitignore":["data/", "checkpoints/", "outputs/", ".venv"],
    ".gitattributes":["filter=lfs"],
}.items():
    path = ROOT / name
    text = path.read_text()
    excerpts = []
    for needle in needles:
        position = text.find(needle)
        excerpts.append({"locator":needle,"found":position >= 0,
                         "text":text[position:position+1800] if position>=0 else ""})
    records["repository_sources"].append({"path":name,"sha256":sha(path),"excerpts":excerpts})

raw = (ROOT / "course/first-steps.md").read_text()
records["implementation_table"] = []
for row in raw.splitlines():
    if not row.startswith("| `tiny_perceptron/") or "](" not in row:
        continue
    package = row.split("`")[1]
    cell = row.split(" | ")[1]
    targets = re.findall(r"\]\(chapters/([^#]+)#([^)]+)\)(?:–([A-Z\d]+\.\d+))?",cell)
    entries = []
    for chapter, first, last in targets:
        ids = [first]
        if last:
            prefix = first.rsplit(".",1)[0]
            ids = [prefix + "." + str(i) for i in range(int(first.rsplit(".",1)[1]),int(last.rsplit(".",1)[1])+1)]
        path = ROOT / "course/chapters" / chapter
        text = path.read_text()
        for identifier in ids:
            match = re.search(r"^## " + re.escape(identifier) + r"\s.*?(?=^## |\Z)",text,re.M|re.S)
            part = match.group(0) if match else ""
            lines = [line for line in part.splitlines() if package in line or "節錄自" in line or "精簡自" in line]
            entries.append({"source":str(path.relative_to(ROOT)),"anchor":identifier,"exists":bool(match),
                            "source_sha256":sha(path),"mentions":lines})
    records["implementation_table"].append({"package":package,"exists":(ROOT/package).exists(),"entries":entries})

records["curriculum_order"] = []
for chapter in ["01","02","03"]:
    path = ROOT / "course/chapters" / (chapter + ".md")
    text = path.read_text()
    matches = []
    for m in re.finditer(r"^## (.+)$",text,re.M):
        heading = m.group(1)
        part = text[m.end():]
        part = re.split(r"^## ",part,maxsplit=1,flags=re.M)[0]
        hits = [line for line in part.splitlines() if "from torch import nn" in line or "nn.Linear" in line
                or ".copy_(" in line or " @ " in line or "class CausalAttention" in line]
        if hits:
            matches.append({"heading":heading,"code_mentions":hits})
    records["curriculum_order"].append({"path":str(path.relative_to(ROOT)),"sha256":sha(path),"matches":matches})

records["gradient_svg_coordinates"] = []
tree = ET.parse(ROOT / "course/figures/rewrite-W-6-gradient-step.svg")
ns = {"s":"http://www.w3.org/2000/svg"}
for circle in tree.findall("s:circle",ns):
    x,y = float(circle.attrib["cx"]),float(circle.attrib["cy"])
    w,L = (x-75)/123,(430-y)/34
    records["gradient_svg_coordinates"].append({"pixel":[x,y],"w":w,"L":L,"formula_L":(w-3)**2,
        "absolute_error":abs(L-(w-3)**2)})
curve = next(p for p in tree.findall("s:path",ns) if p.attrib.get("stroke-width")=="3" and p.attrib.get("stroke")=="#64748b")
points = [(float(x),float(y)) for x,y in re.findall(r"[ML]([\d.]+) ([\d.]+)",curve.attrib["d"])]
records["gradient_svg_curve"] = {"points":len(points),"coordinate_mapping":"x=75+123w; y=430-34L",
    "max_pixel_error":max(abs(y-(430-34*((x-75)/123-3)**2)) for x,y in points),"pixel_tolerance":0.02}
assert records["gradient_svg_curve"]["max_pixel_error"] < 0.02
records["axes_svg_labels"] = [t.text for t in ET.parse(ROOT / "course/figures/rewrite-W-3-axes.svg").findall("s:text",ns)]

cache = ART / ".uv-cache"
env = {**os.environ,"UV_CACHE_DIR":str(cache),"PYTHONDONTWRITEBYTECODE":"1"}
command = ["uv","sync","--dry-run","--frozen","--extra","cpu","--group","notebook"]
p = subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True,timeout=45)
records["sync_plan"] = {"command":command,"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
if cache.exists():
    shutil.rmtree(cache)
command = ["zsh","-c","source .venv/bin/activate\ncommand -v python\npython --version"]
p = subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True,timeout=10)
records["activation"] = {"command":command,"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
print(json.dumps(records,ensure_ascii=False,indent=2))
