"""Informed original-owner README callback: primary bytes, raw score recount, read-only download and CPU CLI."""
from collections import Counter
from datetime import datetime, UTC
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[6]
OUT = Path(__file__).parent
SCRATCH = ROOT / "work/readme-backfill-callback-01-attempt02"
SCRATCH.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_bytes())
def run(args, timeout=120):
    p = subprocess.run([str(x) for x in args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return {"command": [str(x) for x in args], "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
old_path = ROOT / "docs/course-revision-20261016/reviews/technical-readme-transport-original-02.json"
old = load(old_path)
r = {"reviewer_task": "/root/tech_transport_readme_20261016", "current_context": "informed", "at": datetime.now(UTC).isoformat(), "source_sha256": sha(ROOT/"README.md"), "baseline_report": {"path":str(old_path.relative_to(ROOT)),"sha256":sha(old_path)}, "environment": {"python": platform.python_version(), "torch": importlib.metadata.version("torch"), "platform": platform.platform(), "device": "cpu"}}
r["baseline_sources"] = []
for source in old["sources"]:
    if source["kind"] == "repository_code":
        current = sha(ROOT/source["path"])
        r["baseline_sources"].append({"id":source["id"],"path":source["path"],"before_sha256":source["sha256"],"current_sha256":current,"changed":current!=source["sha256"]})
f = load(ROOT/"docs/course-revision-20261016/freeze-14.json")
r["freeze14_readme"] = next(p for p in f["pages"] if p["page_id"] == "readme")
assert r["freeze14_readme"]["source_sha256"] == r["source_sha256"]
release_path = ROOT/"tiny_perceptron/lesson_products/release.json"
release = load(release_path)
r["default_release"] = {"path":str(release_path.relative_to(ROOT)),"sha256":sha(release_path),"content":release}
r["cli"] = []
for product in ("b", "c"):
    dest = SCRATCH/"products"/product
    assert not dest.exists(), "Use a newly empty callback destination"
    result = run([sys.executable,"-m","tiny_perceptron.lesson_products","download","--product",product,"--destination",dest])
    r["cli"].append(result)
    if result["returncode"]:
        (OUT/"callback-results.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
        raise SystemExit(f"download {product} failed; preserved actual result")
state = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
bdir,cdir = SCRATCH/"products/b", SCRATCH/"products/c"
commands = [
 ["solve","--b",bdir,"--state",state,"--loops","3"],
 ["judge","--c",cdir,"--state","卡片進度;申請編號=?","--question","info"],
 ["auto","--b",bdir,"--c",cdir,"--state",state,"--effort","auto"],
 ["judge","--c",cdir,"--state","卡片進度;申請編號=?","--question","info","--candidates","no,yes"],
 ["auto","--b",bdir,"--c",cdir,"--state",state,"--effort","high","--max-loops","2"],
]
for args in commands:
    r["cli"].append(run([sys.executable,"-m","tiny_perceptron.lesson_products",*args]))
r["downloaded"] = {}
for product in ("b","c"):
    directory=SCRATCH/"products"/product
    manifest=load(directory/"artifact-manifest.json")
    receipt=load(directory/"training-receipt.json")
    r["downloaded"][product]={
      "directory":str(directory.relative_to(ROOT)),
      "files":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in directory.iterdir() if p.is_file()},
      "manifest":manifest,
      "training_receipt":{k:receipt[k] for k in ("run_identity","product","mode","random_initialization","resume_source","steps_this_attempt","last_step","selected_step","parameters","data_manifest_sha256","initial_weights_sha256","selected_weights_sha256","weights_changed","resume_verification","environment","scope")},
    }
    assert sha(directory/"artifact-manifest.json") == release["products"][product]["manifest_sha256"]
    for name,item in manifest["files"].items():
        assert sha(directory/name)==item["sha256"] and (directory/name).stat().st_size==item["bytes"]

# No model test is restarted: stream the original saved raw score file and derive the README counts.
primary_report=load(ROOT/"docs/lesson-products/results/final-report-v1.json")
raw_path=ROOT/"outputs/lesson-products/final-report-v1.predictions.jsonl"
assert sha(raw_path)==primary_report["predictions"]["sha256"]
assert raw_path.stat().st_size==primary_report["predictions"]["bytes"]
counts=Counter();correct=Counter();families={};selected_fields_consistent=True;error_example=None
def choose_meaning(scores,candidates):
    return min(zip(candidates,scores,strict=True),key=lambda p:(-p[1],p[0]))[0]
with raw_path.open() as stream:
    for line in stream:
        row=json.loads(line)
        if row["product"]=="b" and row["split"]=="test":
            key="b_loop5_"+row["task"]
            scores=row["scores_by_depth"]["5"]
            answer=max(range(len(scores)),key=scores.__getitem__)
            selected_fields_consistent &= answer==row["answers_by_depth"]["5"]
            counts[key]+=1;correct[key]+=answer==row["gold"]
            families.setdefault(key,set()).add(row["family_id"])
        elif row["product"]=="c" and row["split"] in ("test","wording_shift"):
            key="c_"+row["split"]+"_"+row["question"]
            selected=choose_meaning(row["raw_scores"],row["candidates"])
            counts[key]+=1;correct[key]+=selected==row["gold"]
            families.setdefault(key,set()).add(row["family_id"])
            if row["split"]=="test":selected_fields_consistent &= selected==row["selected"]
            elif selected!=row["gold"] and error_example is None:
                error_example={k:row[k] for k in ("id","family_id","question","candidates","gold","raw_scores")}
                error_example["independent_selected"]=selected
r["independent_raw_recount"]={"path":str(raw_path.relative_to(ROOT)),"sha256":sha(raw_path),"bytes":raw_path.stat().st_size,"method":"B: argmax of raw scores at depth5; C: highest raw score, exact ties by semantic ID. No model forward, no formal test restart.","counts":{k:{"count":v,"correct":correct[k],"families":len(families[k]),"accuracy":correct[k]/v} for k,v in counts.items()},"selected_fields_consistent":selected_fields_consistent,"first_c_wording_error":error_example}
assert counts["b_loop5_boxes"]==2956 and correct["b_loop5_boxes"]==2928
assert counts["b_loop5_sum"]==42 and correct["b_loop5_sum"]==2
assert counts["c_test_choice"]==11797 and correct["c_test_choice"]==11797
assert correct["c_wording_shift_choice"]<counts["c_wording_shift_choice"]

# Verify original pre-test sources and freeze receipt bytes against the actual pre-test commit.
freeze_path=ROOT/"docs/lesson-products/results/settings-freeze-v1.json"
freeze=load(freeze_path)
original=subprocess.run(["git","show","821a2c717f05ad7b57e08e88cd965e92ff0c1200:docs/lesson-products/results/settings-freeze-v1.json"],capture_output=True,check=True).stdout
assert hashlib.sha256(original).hexdigest()==sha(freeze_path)==primary_report["settings_freeze_sha256"]
r["pretest_binding"]={"commit":"821a2c717f05ad7b57e08e88cd965e92ff0c1200","commit_metadata":run(["git","show","-s","--format=%H %cI %s","821a2c717f05ad7b57e08e88cd965e92ff0c1200"]),"settings_freeze_sha256":sha(freeze_path),"test_outcomes_observed":freeze["test_outcomes_observed"],"current_frozen_code":{name:{"expected":digest,"current":sha(ROOT/"tiny_perceptron/lesson_products"/name),"match":sha(ROOT/"tiny_perceptron/lesson_products"/name)==digest} for name,digest in freeze["source_sha256s"].items()},"start_marker":load(ROOT/"docs/lesson-products/results/final-report-v1.started.json"),"recorded_report_at":primary_report["recorded_at"]}

# Read actual publication sources at their recorded execution commit, not at current-main assumptions.
pub=load(ROOT/"docs/lesson-products/results/publication-v1.json")
revision=pub["execution"]["revision"]
r["publication_execution_sources"]=[]
for name,digest in pub["execution"]["source_hashes"].items():
    p=subprocess.run(["git","show",revision+":"+name],capture_output=True)
    actual=hashlib.sha256(p.stdout).hexdigest() if p.returncode==0 else None
    r["publication_execution_sources"].append({"path":name,"revision":revision,"expected":digest,"actual":actual,"matches":actual==digest,"current_sha256":sha(ROOT/name) if (ROOT/name).is_file() else None})

spec=load(ROOT/"docs/lesson-products/results/publication-spec-v1.json")
urls=[
 ("hf_tree",f"https://huggingface.co/api/models/{release['repo']}/tree/{release['revision']}/lesson-products/v1-20261016?recursive=true"),
 ("hf_report",f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/lesson-products/v1-20261016/report.json"),
 ("hf_raw_manifest",f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/lesson-products/v1-20261016/raw/predictions-manifest.json"),
 ("hf_provenance",f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/lesson-products/v1-20261016/provenance.json"),
 ("hf_card",f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/lesson-products/v1-20261016/README.md"),
 ("hf_license",f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/lesson-products/v1-20261016/LICENSE"),
 ("publish_run","https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/actions/runs/38871109465"),
]
def fetch(item):
    name,url=item;entry={"id":name,"url":url,"authentication":"No Authorization header supplied"}
    try:
        with urlopen(Request(url,headers={"User-Agent":"independent-readme-callback"}),timeout=30) as response:
            raw=response.read(2*1024*1024)
            entry.update(status=response.status,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        (SCRATCH/(name+".txt")).write_bytes(raw)
        if name=="hf_tree":
            value=json.loads(raw);files=[f for f in value if f["type"]=="file"]
            entry["file_count"]=len(files)
            entry["matches_package"]={f["path"]:f["size"]==spec["files"][f["path"].removeprefix("lesson-products/v1-20261016/")]["bytes"] for f in files}
            entry["raw_compressed_tree_record"]=next(x for x in files if x["path"].endswith(".jsonl.gz"))
        elif name=="hf_report":
            entry["matches_local_report"]=entry["sha256"]==sha(ROOT/"docs/lesson-products/results/final-report-v1.json")
        elif name=="hf_card":entry["excerpts"]=[line for line in raw.decode().splitlines() if any(term in line for term in ("2928","2956","11797","MIT","finite","有限","2/42"))]
        elif name=="hf_license":entry["excerpt"]=raw.decode().splitlines()[:4]
        elif name=="publish_run":
            value=json.loads(raw);entry["fields"]={k:value[k] for k in ("id","head_sha","status","conclusion")}
        else:entry["content"]=json.loads(raw)
    except Exception as error:entry.update(error_type=type(error).__name__,error=str(error))
    return entry
with ThreadPoolExecutor(max_workers=4) as pool:r["network"]=list(pool.map(fetch,urls))

# Existing old evidence remains evidence only because its exact implementation and artifact bytes are unchanged.
r["old_artifact_hashes"]={a["path"]:{"expected":a["sha256"],"current":sha(ROOT/a["path"]),"match":sha(ROOT/a["path"])==a["sha256"]} for a in old["artifacts"]}
r["current_counts"]={"notebooks":len(list((ROOT/"notebooks").glob("**/*.ipynb"))),"numbered_sections":sum(len(re.findall(r"^## [A-Z\d]+\.\d+ ",p.read_text(),re.M)) for p in list((ROOT/"course/chapters").glob("*.md"))+[ROOT/"course"/name for name in ("README.md","first-steps.md","training.md","glossary.md")])}
r["build_check"]=run([sys.executable,"scripts/build_course.py","--check"])
paths=["README.md","course/README.md","course/chapters/19.md","docs/lesson-products/README.md","docs/lesson-products/RELEASE.md","tiny_perceptron/lesson_products/release.json","tiny_perceptron/lesson_products/__main__.py","tiny_perceptron/lesson_products/delivery.py","tiny_perceptron/lesson_products/inference.py","tiny_perceptron/lesson_products/model.py","tiny_perceptron/lesson_products/evaluation.py","scripts/lesson_products/evaluate.py","scripts/lesson_products/train_local.py",".github/workflows/lesson-products-publish.yml","docs/lesson-products/results/final-report-v1.json","docs/lesson-products/results/settings-freeze-v1.json","docs/lesson-products/results/publication-v1.json","docs/lesson-products/results/publication-spec-v1.json","docs/lesson-products/results/publication-package-v1.json","docs/lesson-products/results/public-cpu-acceptance-v1.json","course/figures/rewrite-01-character-ids.svg"]
r["inspected_source_hashes"]={p:sha(ROOT/p) for p in paths}
(OUT/"callback-results.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"source_sha256":r["source_sha256"],"cli_returncodes":[x["returncode"] for x in r["cli"]],"raw_recount":r["independent_raw_recount"]["counts"],"network":[{k:e[k] for k in ("id","status","error","file_count","matches_local_report") if k in e} for e in r["network"]],"counts":r["current_counts"],"build_returncode":r["build_check"]["returncode"]},ensure_ascii=False))
