"""Collector callback checks; preserve all original reports and outputs."""

import hashlib
import json
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix08.json"
BUNDLE = "docs/technical-reviews/artifacts/omni-technical-coordinator-v1/numbered-section-reviews-v1.json"
MANIFEST = "docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-manifest-v1.json"
ORIGINAL = "docs/course-revision-20261029-omni/reviews/technical/technical_glossary/technical-original-v1.json"


def digest(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT/path).read_bytes())


def ref(reference):
    assert digest(reference["path"]) == reference["sha256"], reference
    return read(reference["path"])


def slices(canonical, page_ids=None):
    result = {}
    for page in canonical["pages"]:
        if page_ids is not None and page["page_id"] not in page_ids:
            continue
        raw = (ROOT/page["snapshot"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
        text = raw.decode("utf-8")
        headers = list(re.finditer(r"^## ([^\n]+)\n", text, re.M))
        for n, header in enumerate(headers):
            sid = header[1].split()[0]
            if re.fullmatch(r"[RWTG]\.\d+", sid):
                section = text[header.start():headers[n+1].start() if n+1<len(headers) else len(text)]
                result[sid] = (page["page_id"], section.encode("utf-8"))
    return result


bundle = read(BUNDLE)
assert digest(BUNDLE) == "93d14739cb72c37db4e5f58b499461b252ec700d3f176d334ce251df06a7320f"
canonical = ref(bundle["canonical_source_freeze"])
assert bundle["canonical_source_freeze"]["path"] == FREEZE
numbered = ref(bundle["section_freeze"])
assert numbered["canonical_source_freeze"] == bundle["canonical_source_freeze"]
current_slices = slices(canonical, numbered["canonical_page_ids"])
page_map = {p["page_id"]:p for p in numbered["pages"]}
expected = {f"{prefix}.{n}" for prefix, count in [("R",4),("W",7),("T",11),("G",4)] for n in range(1,count+1)}
assert set(current_slices) == expected == set(page_map)
assert {e["lesson_id"] for e in bundle["sections"]} == expected
assert len(bundle["sections"]) == 26
manifest = read(MANIFEST)
assignment_path = manifest["assignments"]["path"]
assignment_observed_sha = digest(assignment_path)
assignments = read(assignment_path)
author_tasks = set(manifest.get("author_tasks", []))
reader_tasks = {t["task"] for t in assignments["tasks"] if t.get("role")=="reader"}
records = []
owners = []
refs = {}
all_issues = []
for entry in bundle["sections"]:
    sid = entry["lesson_id"]
    current = ref(entry["report"])
    original = ref(entry["original_report"])
    initial = ref(entry["original_section_freeze"])
    initial_page = next(p for p in initial["pages"] if p["page_id"]==sid)
    initial_canonical = ref(initial["canonical_source_freeze"])
    initial_slices = slices(initial_canonical, initial["canonical_page_ids"])
    page = page_map[sid]
    canonical_id, exact = current_slices[sid]
    assert (ROOT/page["snapshot"]).read_bytes() == exact
    assert hashlib.sha256(exact).hexdigest() == page["source_sha256"] == current["source_sha256"]
    assert current["figure_sha256"] == page["figures_sha256"]
    initial_raw = (ROOT/initial_page["snapshot"]).read_bytes()
    assert initial_raw == initial_slices[sid][1]
    assert hashlib.sha256(initial_raw).hexdigest() == initial_page["source_sha256"] == original["source_sha256"]
    assert original["figure_sha256"] == initial_page["figures_sha256"]
    owner = current["reviewer_task"]
    assert owner == original["reviewer_task"]
    assert owner not in reader_tasks | author_tasks | {"/root/omni_technical_coordinator/technical_glossary"}
    assert current["lesson_id"] == original["lesson_id"] == sid
    assert current["reviewer_context"] == original["reviewer_context"] == "fresh"
    task = [t for t in assignments["tasks"] if t["task"]==owner]
    assert len(task)==1 and task[0]["lesson_id"]==sid and task[0]["role"]=="technical" and task[0]["context"]=="fresh"
    at = datetime.fromisoformat
    assert at(task[0]["dispatch_window"]["not_before"]) <= at(original["created_at"]) <= at(entry["original_collection"]["received_at"])
    assert at(original["created_at"]) <= at(current["created_at"]) <= at(entry["current_collection"]["received_at"])
    assert at(entry["original_collection"]["received_at"]) <= at(entry["current_collection"]["received_at"])
    for role in ["original_collection","current_collection"]:
        assert entry[role]["collection_role"] == "technical"
    if "current_dispatch_window" in entry:
        assert at(entry["current_dispatch_window"]["not_before"]) <= at(current["created_at"])
    assert current["verdict"]=="pass" and all(c["status"]=="verified" for c in current["claims"])
    assert all(i.get("status")=="resolved" and i.get("resolution") for i in current.get("issues",[]))
    for key in ["original_report","report"]:
        report_ref = entry[key]
        refs[report_ref["path"]] = report_ref
        r = original if key=="original_report" else current
        for n in range(len(r.get("issues",[]))):
            item = {"report":report_ref,"json_pointer":f"/issues/{n}","lesson_id":sid,"canonical_page_id":canonical_id}
            if item not in all_issues:
                all_issues.append(item)
    owners.append(owner)
    records.append({"lesson_id":sid,"canonical_page_id":canonical_id,"reviewer_task":owner,"report":entry["report"],"original_report":entry["original_report"],"original_section_freeze":entry["original_section_freeze"],"source_sha256":current["source_sha256"],"figures_match":True,"current_claims":len(current["claims"]),"original_verdict":original["verdict"],"current_verdict":current["verdict"],"original_received_at":entry["original_collection"]["received_at"],"current_received_at":entry["current_collection"]["received_at"],"identity_assignment_and_time_checks":True})
assert len(set(owners)) == 26
index = ref(bundle["issue_inventory"])
assert {r["path"]:r for r in index["reports"]} == refs
key = lambda x: (x["report"]["path"], x["json_pointer"])
assert sorted(index["issues"],key=key) == sorted(all_issues,key=key)
assert len(refs)==33 and len(all_issues)==4

original = read(ORIGINAL)
assert digest(ORIGINAL)=="f41987fdfb492664e559e32d5f1b774e57e7464057d989ccb61b17480e981819"
artifact_integrity = []
for a in original["artifacts"]:
    assert digest(a["path"])==a["sha256"]
    artifact_integrity.append({"path":a["path"],"sha256":a["sha256"],"unchanged":True})
source_integrity = []
for s in original["sources"]:
    if s["kind"]=="repository_code":
        assert digest(s["path"])==s["sha256"]
        source_integrity.append({"path":s["path"],"sha256":s["sha256"],"unchanged":True})
glossary = next(p for p in canonical["pages"] if p["page_id"]=="glossary")
raw = (ROOT/glossary["snapshot"]).read_bytes()
assert hashlib.sha256(raw).hexdigest()==glossary["source_sha256"]==digest("course/glossary.md")
text = raw.decode("utf-8")
old_text = (ROOT/original["source_snapshot"]["path"]).read_bytes().decode("utf-8")
old_lines, new_lines = old_text.split("\n"), text.split("\n")
changed = [n for n,(a,b) in enumerate(zip(old_lines,new_lines),1) if a!=b]
assert changed == [28,42,107,108,198,223] and len(old_lines)==len(new_lines)
assert glossary["figures_sha256"]=={} and not re.findall(r"!\[.*?\]\(.*?\)",text)
missing = []
for label,destination in re.findall(r"\[([^\]]+)\]\(([^)]+)\)",text):
    path,_,fragment = destination.partition("#")
    target = ROOT/"course"/path
    if not target.is_file() or fragment and not re.search(r"^##\s+"+re.escape(fragment)+r"\s",target.read_text(),re.M):
        missing.append(destination)
assert not missing
labels = []
sort_failures = []
for line in text.split("### 英文與縮寫索引",1)[1].splitlines():
    if not line.startswith("**"):
        continue
    terms = re.findall(r"\[([^\]]+)\]\([^)]+\)",line)
    if line.startswith("**D**"):
        terms = ["decode","deep supervision","Dense","derivative","distillation","DPO"]
    keys = [re.sub("[^a-z0-9]","",t.lower()) for t in terms]
    if keys != sorted(keys):
        sort_failures.append(terms)
    labels.extend(terms)
assert len(labels)==194 and len(set(labels))==194 and not sort_failures

torch.set_num_threads(1)
logits = torch.tensor([[2.,1.,0.],[0.,1.,2.]],dtype=torch.float64)
targets = torch.tensor([2,2])
none = F.cross_entropy(logits,targets,reduction="none")
ce = {"per_example":none.tolist(),"unbatched_first":F.cross_entropy(logits[0],targets[0]).item(),"sum":F.cross_entropy(logits,targets,reduction="sum").item(),"mean":F.cross_entropy(logits,targets,reduction="mean").item(),"negative_log_p_correct":(-logits.log_softmax(-1)[range(2),targets]).tolist()}
assert torch.allclose(none,-logits.log_softmax(-1)[range(2),targets])
l2_inputs = torch.tensor([[3.,4.],[0.,0.],[1e-14,0.]],dtype=torch.float64)
l2_out = F.normalize(l2_inputs,dim=-1)
ln = torch.nn.LayerNorm(2,dtype=torch.float64)
ln_in = torch.ones(1,2,dtype=torch.float64)
current_checks = {"ce":ce,"l2":{"input":l2_inputs.tolist(),"output":l2_out.tolist(),"norms":l2_out.norm(dim=-1).tolist(),"eps":1e-12},"layernorm":{"constant_input":ln_in.tolist(),"output":ln(ln_in).tolist(),"eps":ln.eps,"has_affine_parameters":ln.elementwise_affine},"json":{"text":'{"answer":4}',"parsed_type":type(json.loads('{"answer":4}')).__name__,"array_json_also_valid":json.loads('[1,2]')},"local_links_checked":len(re.findall(r"\[([^\]]+)\]\(([^)]+)\)",text)),"index_entries":len(labels),"index_duplicate_or_sort_failures":[],"changed_lines":changed,"current_quotes":{str(n):new_lines[n-1] for n in changed}}

result = {"created_at":datetime.now(timezone.utc).isoformat(),"reviewer_task":"/root/omni_technical_coordinator/technical_glossary","environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu"},"scope":"Whole glossary facts and six issue closures independently reviewed; all26 sidecar records checked structurally/source/identity/time. R/W/T facts are not co-signed; no completed technical seal is asserted.","bundle":{"path":BUNDLE,"sha256":digest(BUNDLE)},"canonical_source_freeze":bundle["canonical_source_freeze"],"section_freeze":bundle["section_freeze"],"issue_inventory":bundle["issue_inventory"],"section_count":len(records),"unique_owners":len(set(owners)),"unique_report_refs":len(refs),"indexed_issues":len(all_issues),"section_family_counts":dict(Counter(x["lesson_id"].split('.')[0] for x in records)),"sections":records,"assignment_manifest_reference":{"path":MANIFEST,"sha256":digest(MANIFEST),"assignment_path":assignment_path,"declared_sha256":manifest["assignments"]["sha256"],"observed_sha256":assignment_observed_sha,"reference_current":assignment_observed_sha==manifest["assignments"]["sha256"]},"original_report_and_artifacts_unchanged":artifact_integrity,"repository_sources_unchanged":source_integrity,"glossary_current":{"snapshot":glossary["snapshot"],"source_sha256":glossary["source_sha256"],"figures_sha256":{}},"current_checks":current_checks}
path = OUT/"collector-current-audit-v1.json"
with path.open("x") as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2)
    stream.write("\n")
print(json.dumps({"output":str(path.relative_to(ROOT)),"sha256":digest(path.relative_to(ROOT)),"section_count":len(records),"unique_owners":len(set(owners)),"report_refs":len(refs),"issues":len(all_issues),"changed_lines":changed,"manifest_assignment_reference_current":assignment_observed_sha==manifest["assignments"]["sha256"],"current_checks":current_checks},ensure_ascii=False,indent=2))
