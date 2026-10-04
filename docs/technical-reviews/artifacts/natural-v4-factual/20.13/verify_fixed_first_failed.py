"""Bounded independent 20.13 verification. No inference, grading or data edits."""
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron import natural_assistant as core
from tiny_perceptron.natural_concepts import text_error_report
from scripts import fetch_natural_release as release
import torch

ART = Path(__file__).parent
RESEARCH = ROOT / "outputs/natural-v4/factual-research/20.13"
FINAL = ROOT / "docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review"
VALIDATION = ROOT / "docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review"
PUBLIC = ROOT / "docs/natural-assistant/evidence/v4-research/student-selected-public-ui"
INSPECTED = {}

def read(path):
    p = Path(path)
    b = p.read_bytes()
    INSPECTED[str(p.relative_to(ROOT))] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
    return json.loads(b)

def norm(s, remove_space=False):
    s = unicodedata.normalize("NFKC", s).strip()
    return "".join(s.split()) if remove_space else s

def distance(a, b):
    # Full matrix recurrence, independent of the project's two-row implementation.
    d = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): d[i][0] = i
    for j in range(len(b)+1): d[0][j] = j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(a[i-1]!=b[j-1]))
    return d[-1][-1]

def complete(g, cap=384, count_key="generated_tokens"):
    ids=g["generated_token_ids"]
    return (bool(ids) and len(ids)<=cap and all(type(x) is int and x>=0 for x in ids)
            and g[count_key]==len(ids) and ids[-1] in g["eos_token_ids"]
            and g["ended_with_eos"] is True and g["stop_reason"]=="eos"
            and g["truncated"] is False and g["completion_unknown"] is False)

def group(row, task=None):
    task=task or row["task"]
    if task=="scene": return "photo_summary" if row["references"]["qa_type"]=="scene" else "photo_fact"
    return {"chat":"text_chat","text_presence":"text_presence","ocr":"single_ocr",
            "ocr_order":"ordered_ocr","typed_chat":"voice_typed_reference_chat",
            "speech_chat":"voice_actual_asr_chat"}[task]

environment={"python":sys.version, "torch":str(torch.__version__),"device":"cpu",
             "cuda_available":str(torch.cuda.is_available()),"platform":platform.platform(),
             "unicode_database":unicodedata.unidata_version}
section=(RESEARCH/"20.13.raw.md").read_bytes()
assert hashlib.sha256(section).hexdigest()=="06be2d3b1f6facfa0a653b93df9cb9a3d3527928307c77cae2f6ce94b131d781"
example=re.search(rb"```python\n(.*?)\n```", section,re.S).group(1).decode()
print("EXACT CHAPTER EXAMPLE")
exec(example)
print("EXACT EXERCISE (received equals expected)")
exec(example.replace('b"base A, adapter C"','b"base A, adapter B"'))
assert text_error_report("請推薦不辣的晚餐。","請推薦辣的晚餐。")["edits"]==1
assert len("請推薦不辣的晚餐。") == 9
assert text_error_report("請推薦不辣的晚餐。","請推薦不辣的晚餐。")["cer"]==0
probe_pairs=[("１２ \tＡ！","12A!",True),("簡體","简体",False),("A!","a!",False),
             ("甲\n乙","甲乙",False),("甲\n乙","甲乙",True)]
metric_probes=[]
for a,b,remove in probe_pairs:
    own=norm(a,remove)==norm(b,remove)
    assert core.normalized(a,remove)==norm(a,remove)
    metric_probes.append({"a":a,"b":b,"remove_whitespace":remove,"equal":own})
assert [x["equal"] for x in metric_probes]==[True,False,False,False,True]
assert distance("甲乙","甲丙乙")==1
assert distance("甲乙","甲")==1
assert distance("甲乙","甲丙")==1

manifest=read(ROOT/"docs/natural-assistant/v4/manifest.json")
test_rows=[x for x in manifest["rows"] if x["split"]=="test"]
audio=[x for x in manifest["audio_rows"] if x["split"]=="test"]
cases={x["task"]+":"+x["id"]:x for x in test_rows}
for x in audio:
    if x["task"]=="speech_chat":
        for task in ["typed_chat","speech_chat"]:cases[task+":"+x["id"]]=x
generations=read(FINAL/"generations-base.json")
gen={x["task"]+":"+x["id"]:x for x in generations}
assert len(generations)==len(gen)==len(cases)==178 and set(gen)==set(cases)
transcripts=read(FINAL/"transcripts.json")
tr={x["id"]:x for x in transcripts}
assert len(transcripts)==len(tr)==len(audio)==22 and set(tr)=={x["id"] for x in audio}
grades=read(FINAL/"combined/grades.json")
manual={x["case_id"]:x for x in grades["grades"]}
assert len(manual)==len(grades["grades"])==147
assert grades["independent_of_training_and_selection"] and not grades["private_mapping_consulted"]
scores=read(FINAL/"scored/scores.json")
packet_sha=hashlib.sha256((FINAL/"blind-packet.json").read_bytes()).hexdigest()
assert grades["blind_packet_sha256"]==scores["blind_packet_sha256"]==packet_sha
assert scores["manual_grades_sha256"]==INSPECTED[str((FINAL/"combined/grades.json").relative_to(ROOT))]["sha256"]
den=Counter();correct=Counter();incomplete=[];decisions=[]
for key,row in cases.items():
    g=gen[key];kind=group(row,g["task"]);den[kind]+=1
    assert g["variant"]=="base" and g["split"]=="test"
    assert g["family"]==row["family"] and g["image"]==row.get("image")
    assert g["user"]==(tr[row["id"]]["transcript"] if g["task"]=="speech_chat" else row["user"])
    done=complete(g)
    if not done:
        incomplete.append(key)
        assert kind=="text_chat" and len(g["generated_token_ids"])==384 and g["stop_reason"]=="max_new_tokens"
        assert g["truncated"] and not g["ended_with_eos"]
    if kind in {"photo_summary","photo_fact","text_chat","voice_typed_reference_chat","voice_actual_asr_chat"}:
        m=manual[key];assert m["candidate"]=="A" and type(m["passed"]) is bool and m["reason"].strip()
        if kind.startswith("photo_") and done:assert m["source_image_inspected"] is True
        semantic=m["passed"]
    else:
        semantic=norm(row["answer"],kind=="single_ocr")==norm(g["prediction"],kind=="single_ocr")
    passed=done and semantic;correct[kind]+=int(passed)
    decisions.append({"case_id":key,"group":kind,"decoder_complete":done,"rubric_passed":semantic,"passed":passed})
assert correct==scores["correct_counts"]
assert den=={k:scores["denominators"][k] for k in den}
assert decisions==scores["case_decisions"]
assert len(incomplete)==7 and set(incomplete)==set(scores["incomplete_case_ids"])
assert den=={"photo_summary":42,"photo_fact":84,"text_presence":18,"single_ocr":10,"ordered_ocr":3,
             "text_chat":13,"voice_typed_reference_chat":4,"voice_actual_asr_chat":4}
assert correct=={"photo_summary":25,"photo_fact":58,"text_presence":18,"single_ocr":8,"ordered_ocr":1,
                 "text_chat":2,"voice_typed_reference_chat":2,"voice_actual_asr_chat":2}
with (ART/"case-verification.tsv").open("w") as f:
    f.write("case_id\tgroup\ttoken_count\tcomplete\trubric_pass\tfinal_pass\n")
    for d in decisions:f.write("\t".join(map(str,[d["case_id"],d["group"],gen[d["case_id"]]["generated_tokens"],d["decoder_complete"],d["rubric_passed"],d["passed"]]))+"\n")

subsets=defaultdict(lambda:[0,0]);hist=[]
for d in decisions:
    row=cases[d["case_id"]]
    if d["group"]=="photo_fact":
        q=row["references"]["qa_type"];subsets[q][0]+=int(d["passed"]);subsets[q][1]+=1
    if d["group"]=="text_chat" and row["history"]:hist.append({"case_id":d["case_id"],"passed":d["passed"]})
assert subsets["action"]==[3,11] and subsets["relation"]==[21,29]
assert len(hist)==1 and hist[0]["passed"] is False
saved_subsets=read(FINAL/"descriptive-subsets.json")
for k,(c,n) in subsets.items():
    assert (c,n)==(saved_subsets["photo_fact_subsets"][k]["correct"],saved_subsets["photo_fact_subsets"][k]["denominator"])

cer=defaultdict(lambda:[0,0,0,0,0]);cer_rows=[]
for row in audio:
    t=tr[row["id"]];assert t["reference_transcript"]==row["user"]
    assert t["raw_token_ids"]==t["expected_decoder_prompt_ids"]+t["generated_token_ids"]
    assert complete(t,128,"generated_token_count") and t["decoder_prompt_matches"]
    raw=distance(row["user"],t["transcript"]); target=norm(row["user"],True);prediction=norm(t["transcript"],True)
    normalized=distance(target,prediction)
    vals=[raw,len(row["user"]),normalized,len(target),1]
    assert vals[:4]==[t["raw_errors"],t["raw_reference_characters"],t["errors"],t["reference_characters"]]
    assert t["normalized_reference"]==target and t["normalized_prediction"]==prediction
    for k in [t["task"],"all"]:cer[k]=[a+b for a,b in zip(cer[k],vals)]
    cer_rows.append({"id":row["id"],"source":t["source"],"raw_edits":raw,"raw_reference":vals[1],"normalized_edits":normalized,"normalized_reference":vals[3]})
assert cer["all"]==[112,674,96,661,22]
assert cer["speech_transcription"]==[112,632,96,619,18] and cer["speech_chat"]==[0,42,0,42,4]
assert Counter(t["source"] for t in transcripts)=={"fleurs-mandarin-v4":18,"aishell1-human-read-questions-v4":4}

images={k:{x["image"] for x in test_rows if group(x)==k} for k in ["photo_summary","photo_fact","text_presence","single_ocr","ordered_ocr"]}
assert len(images["photo_summary"])==len(images["photo_fact"])==42 and images["photo_summary"]==images["photo_fact"]
assert len(images["text_presence"])==18 and len(images["single_ocr"])==10 and len(images["ordered_ocr"])==3
assert Counter(x["answer"] for x in test_rows if x["task"]=="text_presence")=={"有":10,"沒有":8}
assert all("nvidia" not in json.dumps(x["source"]).lower() for x in test_rows if x.get("image"))
test_families={x["family"] for x in test_rows+audio}
assert not test_families.intersection(x["family"] for x in manifest["rows"]+manifest["audio_rows"] if x["split"] in ["train","validation"])

result=read(FINAL/"result.json");selection=read(ROOT/"docs/natural-assistant/v4/selection.json")
protocol=read(ROOT/"docs/natural-assistant/v4/validation-protocol-lower-lr.json")
vscore=read(VALIDATION/"scored/scores.json")
assert result["total_parameters"]==2_127_532_032 and result["asr"]["parameters"]==808_878_080
assert result["selected_only"] and list(result["variants"])==["base"] and result["trainable_parameters"]==0
assert result["execution"]["selection_sha256"]==hashlib.sha256((ROOT/"docs/natural-assistant/v4/selection.json").read_bytes()).hexdigest()
assert result["execution"]["pretest_selection"]["selected_variant"]==selection["selected_variant"]=="base"
assert selection["validation_scoring_sha256"]==hashlib.sha256((VALIDATION/"scored/scores.json").read_bytes()).hexdigest()
assert selection["validation_result_sha256"]==hashlib.sha256((VALIDATION/"raw/result.json").read_bytes()).hexdigest()
assert selection["manual_grades_sha256"]==hashlib.sha256((VALIDATION/"combined/grades.json").read_bytes()).hexdigest()
assert selection["validation_protocol_sha256"]==hashlib.sha256((ROOT/"docs/natural-assistant/v4/validation-protocol-lower-lr.json").read_bytes()).hexdigest()
weights={"photo_summary":270,"photo_fact":135,"text_presence":840,"single_ocr":1512,"ordered_ocr":5040,"text_chat":560,"voice_typed_reference_chat":1260,"voice_actual_asr_chat":1260}
selection_arithmetic={}
for name,candidate in vscore["variants"].items():
    numerator=sum(candidate["correct_counts"][k]*weights[k] for k in weights)
    assert numerator==candidate["primary_numerator"] and candidate["primary_denominator"]==75600
    selection_arithmetic[name]={"numerator":numerator,"denominator":75600,"incomplete":len(candidate["incomplete_case_ids"])}
    if name!="base":
        gates={k:candidate["correct_counts"][k]>=max(0,vscore["variants"]["base"]["correct_counts"][k]-int(protocol["adapter_nonregression_gates_vs_base"][k+"_correct_count_minimum"]=="base minus 1")) for k in weights}
        assert all(gates[k]==candidate["gates"][k]["passed"] for k in gates)
        assert not candidate["eligible"] and not all(gates.values()) and not candidate["all_generations_complete"]
        selection_arithmetic[name]["failed_gates"]=[k for k,v in gates.items() if not v]
assert all(x["numerator"]<selection_arithmetic["base"]["numerator"] for k,x in selection_arithmetic.items() if k!="base")
rev=result["execution"]["revision"]
for p in ["docs/natural-assistant/v4/selection.json","docs/natural-assistant/v4/manifest.json","docs/natural-assistant/v4/validation-protocol-lower-lr.json","docs/natural-assistant/v4/asr-selection.json"]:
    old=subprocess.run(["git","show",f"{rev}:{p}"],cwd=ROOT,capture_output=True)
    assert old.returncode==0 and old.stdout==(ROOT/p).read_bytes(),p
qwen=read(RESEARCH/"qwen-api.raw");whisper=read(RESEARCH/"whisper-api.raw")
assert qwen["safetensors"]["total"]==2_127_532_032 and whisper["safetensors"]["total"]==808_878_080
# Count unique Whisper parameters from pinned config and inspected 4.57.6 source;
# encoder/decoder attention K has no bias; output projection ties token embedding.
h=1280; ff=5120
attention=4*h*h+3*h
mlp=2*h*ff+ff+h
encoder=(h*128*3+h)+(h*h*3+h)+1500*h+32*(attention+mlp+4*h)+2*h
decoder=51866*h+448*h+4*(2*attention+mlp+6*h)+2*h
assert encoder+decoder==808_878_080

public=read(ROOT/"docs/natural-assistant/v4/public-release.json")
assert public["selected_variant"]=="base" and public["adapter_parameters"]==0
assert public["base_model"]["revision"]==qwen["sha"] and public["asr_model"]["revision"]==whisper["sha"]
fresh=RESEARCH/"anonymous-small-public";fresh.mkdir(exist_ok=True)
download=[]
for x in public["files"]:
    url=f'https://huggingface.co/{public["repo"]}/resolve/{public["revision"]}/{x["path"]}'
    request=urllib.request.Request(url,headers={"User-Agent":"20.13-independent-review"})
    assert not request.has_header("Authorization")
    with urllib.request.urlopen(request,timeout=30) as response:
        b=response.read(100001);assert len(b)<=100000
    assert len(b)==x["bytes"] and hashlib.sha256(b).hexdigest()==x["sha256"]
    (fresh/x["output"]).write_bytes(b)
    download.append({"url":url,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),"request_authentication_header":False})
release.verify_release(public,fresh,receipt=False)
assert sum(x["bytes"] for x in download)==9262

browser=read(PUBLIC/"actual-ui/browser/report.json");requests=read(PUBLIC/"actual-ui/browser/requests.json")
ui=read(PUBLIC/"actual-ui/report.json");observer=[json.loads(s) for s in (PUBLIC/"actual-ui/observer.jsonl").read_text().splitlines()]
assert browser["requests"]==requests and browser["browser"]=="/usr/bin/chromium" and browser["actual_browser_interactions_complete"]
assert len([x for x in requests if x["route"]=="/api/chat"])==4 and all(x["status"]==200 for x in requests)
assert len([x for x in requests if x["route"]=="/api/transcribe"])==1
assert requests[-1]["route"]=="/api/reset" and requests[-1]["response"]["reset"] and browser["reset_rendered_turns"]==0
assert ui["mode"]=="release" and ui["interaction"]=="browser" and ui["offline"] and ui["torch_threads"]==5 and ui["torch_interop_threads"]==1
assert ui["public_release_prelaunch_gate"]["base_model"]==public["base_model"] and ui["public_release_prelaunch_gate"]["asr_model"]==public["asr_model"]
assert ui["model_load_counts"]=={"load_core":1,"load_asr":1} and ui["reset_states"][0]["history_messages"]==0 and ui["reset_states"][0]["assets"]==0
assert ui["server_cpu_resources"]["maximum_resident_set_kib"]==13_496_104
assert any(x.get("kind")=="process_resources" and x.get("maximum_resident_set_kib")==13_496_104 for x in observer)
assert ui["dependencies"]["torch"]=="2.8.0+cpu"

out={"environment":environment,"source_sha256":hashlib.sha256(section).hexdigest(),
     "generation_counts":dict(den),"correct_counts":dict(correct),"completion":{"eos":171,"capped":7,"capped_group":"text_chat","cap":384,"ids":incomplete},
     "subsets":dict(subsets),"context_chat":hist,"cer_aggregate_raw_normalized_count":dict(cer),"cer_records":cer_rows,
     "percentages":{"raw":round(112/674*100,2),"normalized":round(96/661*100,2)},
     "metric_probes":metric_probes,"prerequisite_cer":{"edits":1,"reference":9,"percent":round(1/9*100,1),"corrected":0},
     "image_reuse":{"unique_photo_summary":42,"unique_photo_fact":42,"same_photos":True,"presence_single_shared":len(images["text_presence"]&images["single_ocr"]),"single_order_shared":len(images["single_ocr"]&images["ordered_ocr"])},
     "test_family_disjoint_from_train_validation":True,"no_test_synthetic_nvidia":True,
     "pretest_git_revision":rev,"selected_score_arithmetic":selection_arithmetic,
     "parameter_counts":{"qwen_official_api_and_record":2127532032,"whisper_encoder_derivation":encoder,"whisper_decoder_derivation":decoder,"whisper_total":encoder+decoder},
     "anonymous_public_download":download,"browser_observation":{"linux_cpu":True,"chromium":browser["browser_version"],"torch":ui["dependencies"]["torch"],"max_server_rss_KiB":13496104,"rss_GiB":round(13496104/1024**2,4),"ready_seconds":ui["ready_seconds"],"chat_seconds":browser["chat_seconds"],"asr_seconds":browser["asr_seconds_including_load"],"whole_operation_seconds":ui["wall_seconds"],"no_new_model_download":True,"no_quality_grading":True},
     "inspected_original_json":INSPECTED,"limits":"Independent fixed-record and code verification, not new model inference, training, regrading or GPU replication; no minimum hardware claim."}
(ART/"fixed-results.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in out.items() if k not in ["cer_records","inspected_original_json"]},ensure_ascii=False,indent=2))
print("ALL BOUNDED ASSERTIONS PASSED")
