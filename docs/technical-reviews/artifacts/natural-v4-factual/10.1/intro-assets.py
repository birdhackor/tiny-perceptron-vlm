from pathlib import Path
import json,hashlib,tarfile,struct
record_path=Path("docs/course-experiments/results/real_modal.json")
source_path=Path("assets/training/sources/fashion-mnist.json")
record=json.loads(record_path.read_text())
source=json.loads(source_path.read_text())
asset=next(a for a in record["assets"] if a["id"]=="fashion-mnist")
archive=Path(asset["archive"])
actual_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
assert actual_sha==asset["archive_sha256"]
images=[f for f in asset["files"] if f["path"].startswith("vision-initial/images/") and f["path"].endswith(".png")]
verified=[]
with tarfile.open(archive,"r:gz") as tar:
 members={m.name.lstrip("./"):m for m in tar.getmembers()}
 for image in images:
  member=members[image["path"]]
  blob=tar.extractfile(member).read()
  assert hashlib.sha256(blob).hexdigest()==image["sha256"]
  assert blob[:8]==b"\x89PNG\r\n\x1a\n"
  width,height=struct.unpack(">II",blob[16:24])
  assert (width,height)==(28,28)
  verified.append({"path":image["path"],"sha256":image["sha256"]})
assert len(verified)==50
print(json.dumps({
 "scope":"Only checks the introduction's acquired-clothing-input statement; no training/release/metric re-audit.",
 "record_path":str(record_path),"record_sha256":hashlib.sha256(record_path.read_bytes()).hexdigest(),
 "source_metadata_path":str(source_path),"source_metadata_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),
 "dataset":source["dataset"],"source_repository":source["source_repository"],
 "source_repository_revision":source["source_repository_revision"],
 "hf_repository":source["hf_repository"],"hf_revision":source["hf_revision"],
 "source_split":source["source_split"],
 "archive_path":str(archive),"archive_sha256":actual_sha,"acquired_png_inputs_checked":len(verified),
 "png_dimensions":[28,28],"first_and_last_verified":[verified[0],verified[-1]],"assertions":"all passed"
},ensure_ascii=False,indent=2))
