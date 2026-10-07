import json,urllib.request,hashlib
from pathlib import Path
revision="979cdfacc588ad0536f1c64fff96f264571cf054"
base="https://huggingface.co/birdhackor/tiny-perceptron-course-models"
out=Path("docs/technical-reviews/artifacts/p7_technical_e/sources")
url=base.replace("https://huggingface.co/","https://huggingface.co/api/models/")+"/tree/"+revision+"/selftrained/v2?recursive=true&expand=false"
with urllib.request.urlopen(url,timeout=45) as r: tree=json.load(r)
(out/"hf-v2-pinned-tree.json").write_text(json.dumps(tree,ensure_ascii=False,indent=2)+"\n")
for stage in ("moe-pretrain","moe-sft","moe-joint","dense-joint"):
 prefix="selftrained/v2/"+stage
 files=[x for x in tree if x["type"]=="file" and x["path"].startswith(prefix+"/")]
 assert {Path(x["path"]).name for x in files}=={"model.safetensors","model-config.json","tokenizer.json","inference-manifest.json"}
 url=base+"/resolve/"+revision+"/"+prefix+"/inference-manifest.json"
 with urllib.request.urlopen(url,timeout=45) as r: data=r.read()
 m=json.loads(data);(out/("hf-"+stage+"-manifest.json")).write_bytes(data)
 assert set(m["files"])=={"model.safetensors","model-config.json","tokenizer.json"}
 assert m["origin"]["kind"]=="all-neural-weights-random"
 print(stage,"files",len(files),"manifestSHA",hashlib.sha256(data).hexdigest(),"selected_step",m["selected_step"],"weight_bytes",next(x["size"] for x in files if x["path"].endswith(".safetensors")))
 print("manifest_keys",sorted(m))
assert not any(x["path"].endswith((".pt",".pkl")) for x in tree)
print("four_inference_exports_exist_no_optimizer_checkpoint_no_training_rerun")
