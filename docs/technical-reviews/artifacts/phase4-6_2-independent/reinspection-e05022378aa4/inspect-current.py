"""Freeze and render current 6.2, preserve own prior report and verify reused evidence."""
from datetime import datetime, UTC
from hashlib import sha256
from importlib.util import spec_from_file_location,module_from_spec
from pathlib import Path
import difflib
import json
import platform
import subprocess
import sys
import urllib.request

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=BASE.parents[3]
EXPECTED="e05022378aa42be8de2cef56ccfc2440b62d7f28f52fd07987678f9f1e20d1a7"
digest=lambda raw:sha256(raw).hexdigest()
helper=ROOT/"docs/review-tools/section_facts.py"
spec=spec_from_file_location("section_facts",helper);facts=module_from_spec(spec);spec.loader.exec_module(facts)
raw,whole,line=facts.original_section(ROOT/"course/chapters/06.md","6.2")
assert digest(raw)==EXPECTED
(OUT/"current-section.md").write_bytes(raw)
prior_raw=(BASE/"original/section.md").read_bytes()
diff="".join(difflib.unified_diff(prior_raw.decode().splitlines(keepends=True),raw.decode().splitlines(keepends=True),fromfile="own frozen original",tofile="current 6.2"))
(OUT/"source-diff.txt").write_text(diff,encoding="utf-8")
report_path=ROOT/"docs/technical-reviews/6.2.json"
prior_report_bytes=report_path.read_bytes()
prior_report_sha=digest(prior_report_bytes)
assert prior_report_sha=="8c37971d58bea16944e4c3cc674ad253de6e41f36f824e4da6426eb4ae6abb62"
history=ROOT/"docs/technical-reviews/history"/f"phase4-6_2-own-before-reinspection-{prior_report_sha}.json"
if history.exists(): assert history.read_bytes()==prior_report_bytes
else: history.write_bytes(prior_report_bytes)
prior_report=json.loads(prior_report_bytes)
assert prior_report["reviewer_task"]=="/root/phase4_factual_coordinator/factual_6_2"
verified=[]
for item in prior_report["artifacts"]:
 path=ROOT/item["path"]
 actual=digest(path.read_bytes())
 assert actual==item["sha256"],item["id"]
 verified.append({"id":item["id"],"path":item["path"],"sha256":actual})
fences=facts.fences(raw,line)
assert len(fences)==1
assert fences[0]["raw"]==(BASE/"original/fence-1.py").read_bytes()
(OUT/"current-fence.py").write_bytes(fences[0]["raw"])
figure=ROOT/"course/figures/rewrite-06-02-bpe-merge.svg"
figure_raw=figure.read_bytes()
assert digest(figure_raw)==prior_report["figure_sha256"][figure.relative_to(ROOT).as_posix()]
(OUT/"current-figure.svg").write_bytes(figure_raw)
command=["inkscape",str(OUT/"current-figure.svg"),"--export-type=png",f"--export-filename={OUT}/current-figure-390.png","--export-width=390"]
render=subprocess.run(command,cwd=ROOT,capture_output=True,timeout=30)
(OUT/"render-stdout.txt").write_bytes(render.stdout)
(OUT/"render-stderr.txt").write_bytes(render.stderr)
assert render.returncode==0
with urllib.request.urlopen("http://127.0.0.1:8765/6.2.html",timeout=10) as response:
 page=response.read(4000000); status=response.status
(OUT/"current-preview.html").write_bytes(page)
receipt={"observed_on":datetime.now(UTC).isoformat(),"reviewer_task":prior_report["reviewer_task"],
 "source":"course/chapters/06.md#6.2","source_sha256":digest(raw),"prior_source_sha256":digest(prior_raw),
 "prior_history_path":history.relative_to(ROOT).as_posix(),"prior_history_sha256":prior_report_sha,
 "current_fence_sha256":digest(fences[0]["raw"]),"current_figure_sha256":digest(figure_raw),
 "source_change":"Only exercise spelling changed: 而不是隻看→而不是只看 and 跟着變→跟著變. No substantive claim, number, input, fence or SVG changed.",
 "reused_evidence_hash_checks":verified,"reused_evidence_count":len(verified),
 "render":{"command_argv":command,"timeout_seconds":30,"exit_code":render.returncode,"png_sha256":digest((OUT/"current-figure-390.png").read_bytes())},
 "preview":{"url":"http://127.0.0.1:8765/6.2.html","http_status":status,"html_sha256":digest(page),"timeout_seconds":10},
 "environment":{"python":sys.version,"python_executable":sys.executable,"platform":platform.platform(),"device":"CPU"},
 "not_run":"Original fence/exhaustive probes/model training were not rerun. Existing original short-CPU evidence is explicitly reused after hash/support checks; no external source or model downloads."}
(OUT/"input-and-render-receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"source_sha256":digest(raw),"history":str(history.relative_to(ROOT)),"prior_history_sha256":prior_report_sha,"unchanged_evidence_files":len(verified),"render_exit":render.returncode,"preview_http_status":status},ensure_ascii=False))
