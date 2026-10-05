import ast,base64,collections,hashlib,json,random,shutil,sys,xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
base=Path(__file__).resolve().parents[1];root=Path.cwd()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(p): return json.loads(Path(p).read_text())
def saved(src,dst):
 src=Path(src);dst=base/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(src)==sha(dst);return {"original_path":str(src),"original_sha256":sha(src),"copy_path":str(dst),"copy_sha256":sha(dst)}
manifest=j(base/"inputs/manifest.json");vision=j(base/"inputs/vision-sources.json");ocr=j(base/"inputs/ocr-sources.json");voice=j(base/"inputs/voice-sources.json");questions=j(base/"inputs/voice-question-sources.json")
raw_lines=Path("docs/natural-assistant/evidence/v4-research/vision/sources/docci-descriptions.jsonl").read_bytes().splitlines(keepends=True)
ci,nextline=next((i,b) for i,b in enumerate(raw_lines,1) if json.loads(b).get("example_id")=="train_01827")
caption=json.loads(nextline);(base/"sources/cat-caption-original.jsonl").write_bytes(nextline)
assert sha("docs/natural-assistant/evidence/v4-research/vision/sources/docci-descriptions.jsonl")==vision["upstream"]["full_captions"]["sha256"]
vr=next(r for r in vision["rows"] if r["example_id"]=="train_01827");assert vr["full_original_caption"]==caption["description"]
assert hashlib.sha256(caption["description"].encode()).hexdigest()==vr["caption_sha256"]
catcopy=saved(vr["local_file"],"sources/cat-original.jpg");assert catcopy["original_sha256"]==vr["sha256"]
svg=ET.fromstring((base/"figures/natural-v4-training-cat.svg").read_bytes());img=svg.find("{http://www.w3.org/2000/svg}image");href=img.attrib.get("href") or img.attrib.get("{http://www.w3.org/1999/xlink}href");embedded=base64.b64decode(href.split(",",1)[1]);assert embedded==Path(vr["local_file"]).read_bytes()
labels=[json.loads(x) for x in (base/"inputs/train-02.jsonl").read_text().splitlines()[:2]]
catrows=[r for r in manifest["rows"] if isinstance(r.get("source"),dict) and r["source"].get("original_id")=="train_01827"]
for r in labels:
 mr=next(x for x in catrows if x["id"]==r["id"]);assert mr==r;assert r["split"]=="train";assert r["source"]["label_kind"].startswith("AI-authored");assert r["references"]["official_description"]==caption["description"];assert r["source"]["image_sha256"]==vr["sha256"]
photo_provenance=labels[0]["source"]["image_review"];assert "train_01827" in photo_provenance["viewed_image_ids"];contact=saved(photo_provenance["artifact"],"sources/author-cat-contact-original.jpg");assert contact["original_sha256"]==photo_provenance["artifact_sha256"]
commons=[r for r in ocr["sources"] if "commons" in r["dataset"].lower()];cp=[];checks=[]
for name in sorted({r["metadata_query"] for r in commons}): cp.append(saved(name,"sources/commons/"+Path(name).name))
for n,r in enumerate(ocr["sources"]):
 if r not in commons:continue
 response=j(r["metadata_query"]);page=response["query"]["pages"][str(r["source_page_id"])];info=page["imageinfo"][0];meta=info["extmetadata"];val=lambda k:meta.get(k,{}).get("value","")
 assert r["title"]==page["title"] and r["original_sha1"]==info["sha1"] and r["original_timestamp"]==info.get("timestamp","")
 assert r["license"]==val("LicenseShortName") and r["license_url"]==val("LicenseUrl") and r["artist_html"]==val("Artist")
 assert r["source_page"]==info["descriptionurl"] and r["artist_html"] and r["transform"]
 checks.append({"source_pointer":"/sources/"+str(n),"source_id":r["source_id"],"pageid":r["source_page_id"],"timestamp":info.get("timestamp",""),"license":r["license"],"metadata_pointer":"/query/pages/"+str(r["source_page_id"])+"/imageinfo/0/{timestamp,sha1,descriptionurl,extmetadata/{Artist,LicenseShortName,LicenseUrl}}"})
(base/"sources/commons-checked-provenance.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2)+"\n")
audio=manifest["audio_rows"];f=[r for r in audio if r["task"]=="speech_transcription"];q=[r for r in audio if r["task"]=="speech_chat"];assert len(audio)==len(f)+len(q);assert len(f)==len(voice["audio_rows"]) and len(q)==len(questions["audio_rows"]);assert all(r["answer"] is None and r["split"] in {"validation","test"} and r["synthetic"] is False for r in audio);assert all(r["user"]==r["raw_transcription"] for r in f);assert all(r["reference_kind"]=="semantic rubric, not source assistant answer" for r in q)
# Execute only the original bounded repository functions; no model imports or model/data fetches.
src=Path("tiny_perceptron/natural_assistant.py");tree=ast.parse(src.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {"asset_path","messages_for","training_row_at"}];namespace={"Path":Path,"random":random};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),"exec"),namespace)
messages=namespace["messages_for"];data_root=Path(vr["local_file"]).parents[2];ms=[messages(r,data_root,assistant=True) for r in labels];assert ms[0][0]["content"][0]==ms[1][0]["content"][0];assert ms[0][0]["content"][1]!=ms[1][0]["content"][1];assert ms[0][-1]["content"]!=ms[1][-1]["content"]
# Minimal history variation checks preservation of the supplied condition.
changed=dict(labels[1],history=[{"role":"user","content":"請直接回答數量。"},{"role":"assistant","content":"好的。"}]);mh=messages(changed,data_root,assistant=True);assert mh[:2]==[{"role":"user","content":[{"type":"text","text":"請直接回答數量。"}]},{"role":"assistant","content":[{"type":"text","text":"好的。"}]}]
original=[labels[0],labels[1]];repeated=[labels[0],labels[0],labels[1]];sampler=namespace["training_row_at"];oc=collections.Counter(sampler(original,i,42)["id"] for i in range(len(original)));rc=collections.Counter(sampler(repeated,i,42)["id"] for i in range(len(repeated)));assert len(oc)==len(rc)==2 and rc[labels[0]["id"]]==2 and oc[labels[0]["id"]]==1
result={"cat_count":2,"cat_manifest_rows":len(catrows),"caption_original_line":ci,"caption_original_file_sha256":sha("docs/natural-assistant/evidence/v4-research/vision/sources/docci-descriptions.jsonl"),"cat_jpeg_embedded_bytes_exact":True,"cat_dimensions":list(Image.open(vr["local_file"]).size),"commons_checked":len(checks),"commons_license_counts":dict(collections.Counter(r["license"] for r in commons)),"fleurs_rows":len(f),"aishell_question_rows":len(q),"audio_training_rows":sum(r["split"]=="train" for r in audio),"source_audio_answer_is_null":True,"messages_same_image_different_request_answer":True,"history_preserved":True,"repeated_row_unique_examples":len(rc),"original_sampling_counts":dict(oc),"repeated_sampling_counts":dict(rc),"retained_original_copies":[catcopy,contact]+cp,"inspection_pointers":["vision-sources /upstream /attribution /rows/40 selected provenance+caption","train-02 JSONL lines 1-2: labels,references and source provenance; no review_status or author judgments","manifest /rows cat labels /audio_rows task,split,user,raw_transcription,answer,synthetic,reference_kind","voice sources /revision /license /source_files and /audio_rows","voice question sources /official_corpus_license /scoring_policy and /audio_rows","ocr sources /nvidia and /sources provenance fields only; no peer_review or review_status"],"function_coverage":[{"path":str(src),"sha256":sha(src),"methods":[{"name":n.name,"lines":[n.lineno,n.end_lineno]} for n in nodes]}]}
(base/"execution/audit-results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
env={"python":sys.version,"pillow":__import__("PIL").__version__,"device":"cpu","model_loaded":"none","training":"not run","network_in_audit":"not used"};(base/"execution/environment.json").write_text(json.dumps(env,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in {"retained_original_copies","inspection_pointers","function_coverage"}},ensure_ascii=False,indent=2))
