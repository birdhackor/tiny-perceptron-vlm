import json, hashlib, zipfile, math, difflib, sys, platform
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, UTC
import torch
torch.set_num_threads(1)
root=Path("/workspace/work/omni-book-integrated-v1")
started=datetime.now(UTC).isoformat()
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:(root/p).read_bytes()
ref=lambda p:{"path":p,"sha256":sha(read(p))}
prior_path="docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18.json"
freeze_path="docs/course-revision-20261101-c-v3/source-freeze-v21.json"
prior=next(p for p in json.loads(read(prior_path))["pages"] if p["page_id"]=="validation")
page=next(p for p in json.loads(read(freeze_path))["pages"] if p["page_id"]=="validation")
old=read(prior["snapshot"]); new=read(page["snapshot"])
assert sha(old)==prior["source_sha256"] and sha(new)==page["source_sha256"]
assert new==read(page["source"])
a=old.decode().splitlines(keepends=True); b=new.decode().splitlines(keepends=True)
delta=[{"op":tag,"source18_lines":[i+1,j],"source21_lines":[k+1,l],"old":"".join(a[i:j]),"new":"".join(b[k:l])} for tag,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes() if tag!="equal"]
assert len(delta)==1 and delta[0]["source18_lines"]==[42,42] and delta[0]["source21_lines"]==[42,44]
archive=Path("/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip")
descriptor=json.loads(read("docs/decision-product/v3/public-release.json"))
assert archive.stat().st_size==descriptor["bytes"] and sha(archive.read_bytes())==descriptor["sha256"]
members={}
prefix="tiny-perceptron-c-v3/"
with zipfile.ZipFile(archive) as z:
 def member(n):
  assert z.namelist().count(prefix+n)==1
  raw=z.read(prefix+n); members[n]={"bytes":len(raw),"sha256":sha(raw)};return raw
 def load(n):return json.loads(member(n))
 metadata=[]
 for row in json.loads(read("docs/decision-product/v3/metadata/acquisition.json"))["files"]:
  n=row["archive_member"].removeprefix(prefix); raw=member(n)
  assert raw==read(row["path"]) and sha(raw)==row["sha256"] and len(raw)==row["bytes"]
  metadata.append({"path":row["path"],"archive_member":n,"sha256":sha(raw)})
 for n,digest in descriptor["bundle"].items():assert sha(member("bundle/"+n))==digest
 opening=load("evidence/final-opening.json")
 impl=[]
 for n,digest in opening["implementation"].items():
  p="tiny_perceptron/decision_product/"+n
  assert sha(read(p))==sha(member(p))==digest
  impl.append({"path":p,"sha256":digest})
 families=load("evidence/final-families.json"); rows=load("evidence/final-observations.json")
 examples=load("evidence/final-examples.json"); formal=load("evidence/final-results.json")
 registry=load("evidence/registry-before-final.json"); selection=load("evidence/selection.json")
 policy=load("bundle/policy.json")
 keys={f["key"] for f in families}
 assert len(keys)==len(families)==1500 and {len(f["facts"]) for f in families}=={4,5,6}
 for f in families:
  canonical=sorted(tuple(fact) for fact in f["facts"])
  key=sha(json.dumps(canonical,separators=(",",":")).encode())
  assert key==f["key"] and int(sha(("c-product-v3-visible-facts-1:"+key).encode()),16)%1000>=960
 exposed={key for e in registry["entries"] for key in e["family_keys"]}
 assert len(exposed)==opening["previous_exposed_family_count"]==608
 assert not keys&exposed
 assert not keys&set(selection["families"]["calibration"]) and not keys&set(selection["families"]["policy"])
 assert sha(member("evidence/registry-before-final.json"))==opening["previous_registry_sha256"]
 assert sha(member("evidence/selection.json"))==opening["selection_sha256"]
 assert selection["temperatures"]==policy["temperatures"] and selection["selected"]["threshold"]==policy["threshold"]==0.99
 for field,n in [("final_opening_sha256","final-opening"),("observations_sha256","final-observations"),("examples_sha256","final-examples"),("families_sha256","final-families")]:
  assert formal[field]==sha(member("evidence/"+n+".json"))
 assert len(rows)==len(examples)==4500
 pairs=Counter((r["family"],r["kind"]) for r in rows)
 assert set(pairs)=={(key,kind) for key in keys for kind in ("choice","info","rubric")} and set(pairs.values())=={1}
 counts=defaultdict(lambda:[0,0]); answerable=accepted=errors=abstained=correct_abstentions=0
 for row,example in zip(rows,examples):
  assert all(row[k]==example[k] for k in ("family","kind","source","panel"))
  assert row["gold_id"]==example["gold"]==row["ids"][row["label"]]
  assert row["ids"]==[c[0] for c in example["candidates"]]+["none"]
  logits=torch.tensor([-math.inf if v is None else v for v in row["logits"]],dtype=torch.float32)
  idx=int(logits.argmax()); correct=idx==row["label"]
  counts[row["kind"]][0]+=int(correct);counts[row["kind"]][1]+=1
  if row["kind"]=="choice":
   probs=(logits/policy["temperatures"]["choice"]).softmax(-1)
   take=row["ids"][idx]!="none" and float(probs[idx])>=policy["threshold"] and row["eligible"][idx]
   answerable+=int(row["gold_id"]!="none");accepted+=int(take);errors+=int(take and not correct)
   abstained+=int(not take);correct_abstentions+=int(not take and correct and row["gold_id"]=="none")
 assert dict(counts)=={kind:[1500,1500] for kind in ("choice","info","rubric")}
 assert (answerable,accepted,errors,abstained,correct_abstentions)==(1025,1025,0,475,475)
 manual=load("evidence/product-checks.json")
 local_states=set(); manual_counts=defaultdict(lambda:[0,0]); wrong_labels=0
 for row in manual["observations"]:
  task,p,r=row["task"],row["permission"],row["ready"];local_states.add((task,p,r))
  if row["kind"] in ("choice","removed"):
   expected="none" if row["kind"]=="removed" or p is None or r is None else "stop" if p is False else "wait" if r is False else task
  elif row["kind"]=="info":
   complete=(p is not None and r is not None) if task=="record" else p is not None if task=="play" else r is not None
   expected="yes" if complete else "no"
  else:expected="none" if p is None or r is None else str(int(p)+int(r)) if task=="record" else str(2 if p else int(r)) if task=="play" else str(2 if r else int(p))
  wrong_labels+=int(row["expected"]!=expected)
  manual_counts[row["kind"]][0]+=int(row["prediction"]["model_choice"]==expected);manual_counts[row["kind"]][1]+=1
 assert len(local_states)==27 and wrong_labels==0
 assert dict(manual_counts)=={"choice":[972,972],"info":[972,972],"rubric":[972,972],"removed":[432,432]}
 assert {r["devices"] for r in manual["observations"]}=={3,4,5,6}
 assert sha(member("evidence/product-checks.json"))==opening["product_checks_sha256"]
 legacy=load("evidence/legacy-regression.json"); legacy_counts=defaultdict(lambda:[0,0])
 assert sha(member("evidence/legacy-regression.json"))==opening["legacy_regression_sha256"]
 for row in legacy["observations"]:
  legacy_counts[row["kind"]][0]+=int(row["prediction"]["model_choice"]==row["gold"]);legacy_counts[row["kind"]][1]+=1
 assert dict(legacy_counts)=={kind:[60,60] for kind in ("choice","info","rubric")}
 completion=load("evidence/completion.json");lineage=load("evidence/lineage.json");parent=load("evidence/lineage-parent.json")
 trace=[json.loads(line) for line in member("evidence/training.jsonl").splitlines()]
 assert sha(member("evidence/training.jsonl"))==completion["artifacts"]["training.jsonl"]
 assert max(row["step"] for row in trace)==completion["completed_updates"]==1500
 assert sha(member("evidence/initial.safetensors"))==lineage["parent_weights_sha256"]==parent["receipts"]["best.json"]["weights_sha256"]
 for n,field in [("completion.json","parent_completion_sha256"),("source-freeze.json","parent_source_freeze_sha256")]:
  assert sha((json.dumps(parent["receipts"][n],ensure_ascii=False,indent=2)+"\n").encode())==lineage[field]
 assert parent["receipts"]["completion.json"]["completed_updates"]==lineage["parent_best_step"]==4000
 assert sha(member("bundle/weights.safetensors"))==completion["artifacts"]["best.safetensors"]==opening["weights_sha256"]
raw_path="docs/technical-reviews/artifacts/c-v3-v21-numbered-r2/execution.json"
assert sha(read(raw_path))=="375deeebc7d2107171fd38ed80ffbe127823b7865640ca3118a8fa479c6b1959"
raw_cpu=json.loads(read(raw_path));cpu=raw_cpu["new_cpu_examples"]
assert [r["returncode"] for r in cpu]==[0,0,0]
assert [r["request"] for r in cpu]==["record-ready.json","record-unknown.json","answer-absent.json"]
assert [r["response"]["results"]["choice"]["status"] for r in cpu]==["accepted","abstained","abstained"]
out={"schema":"canonical-validation-independent-small-audit-v1","reviewer_task":"/root/c_v3_technical_coord/canonical_validation","started_at":started,"completed_at":datetime.now(UTC).isoformat(),
"environment":{"python":sys.version.split()[0],"torch":str(torch.__version__),"device":"cpu","platform":platform.platform()},
"scope":"Source21 delta and retained raw C-v3 evidence only; no review conclusions, training, fresh inference, final opening, download, or weight copies.",
"source_identity":{"source18_freeze":ref(prior_path),"source21_freeze":ref(freeze_path),"source18_page":prior,"source21_page":page},"delta":delta,
"archive":{"path":str(archive),"bytes":archive.stat().st_size,"sha256":descriptor["sha256"],"members_read":members},
"metadata_byte_identity":metadata,"implementation_identity":impl,
"final_record_check":{"correct_and_total":dict(counts),"unique_families":1500,"shared_family_set":True,"one_per_kind_per_family":True,"devices":[4,5,6],"observations":4500,"previous_exposed":len(exposed),"exposed_overlap":0,"calibration_policy_overlap":0,"all_final_hash_partition":True,"answerable_choice":answerable,"accepted":accepted,"accepted_errors":errors,"abstained":abstained,"correct_abstentions":correct_abstentions},
"manual_record_check":{"local_fact_states":len(local_states),"wrong_expected_labels":wrong_labels,"correct_and_total":dict(manual_counts),"devices":[3,4,5,6],"scope":"correlated engineering observations; no new independent final families"},
"legacy_record_check":{"correct_and_total":dict(legacy_counts),"scope":legacy["scope"]},
"training_record_check":{"own_updates":completion["completed_updates"],"parent_updates":parent["receipts"]["completion.json"]["completed_updates"],"own_trace_rows":len(trace),"initial_matches_parent_best":True,"parent_completion_and_source_freeze_hash_match":True,"weights_match_final_opening":True,"scope":"retained training lineage and actual saved own trace; not new training"},
"frozen_policy":policy,
"retained_cpu_execution":{"path":raw_path,"sha256":sha(read(raw_path)),"environment":raw_cpu["environment"],"examples":cpu,"scope":"saved actual CPU commands/output only; no public transport/performance/all-platform evidence"},
"unresolved_before_original_receipt":["Immutable accepted source18 evidence for source-identical content.","Identity and supporting scope of the three completed product reviews; no prior decisions read."]}
print(json.dumps(out,ensure_ascii=False,indent=2))
