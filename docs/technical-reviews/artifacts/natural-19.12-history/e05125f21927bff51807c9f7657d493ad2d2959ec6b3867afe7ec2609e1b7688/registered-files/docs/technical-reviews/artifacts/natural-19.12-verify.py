"""19.12 CPU checks against complete frozen original records, not GPU reruns."""
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.capstone import (TOK, CapstoneModel, build_dataset, default_config,
                                     expected_final, parse_action, calculator_runtime, prompt_ids)
from tiny_perceptron.natural_concepts import (picture_order_report, thin_stroke_report,
                                            audio_order_report, text_error_report)

OUT = ROOT / "docs/technical-reviews/artifacts"
def read(path):
    return json.loads((ROOT / path).read_text())
splits, manifest = build_dataset()
raw = (ROOT / "course/chapters/19.md").read_text()
section = re.search(r'^## 19\.12 .*?(?=^## |\Z)', raw, re.M | re.S)[0]
code = re.search(r'```python\n(.*?)```', section, re.S)[1]
captured = io.StringIO()
with redirect_stdout(captured):
    exec(compile(code, "course/chapters/19.md#19.12", "exec"), {})
assert splits == read("docs/course-experiments/capstone-evidence/deployment/data.json")["splits"]
counts = Counter(r["task"] for r in splits["test"])
expected_counts = {"calculator":12,"unavailable":12,"tool_return":6,"concept":6,
                   "image_color":9,"image_shape":9,"joint":18,"audio":6,
                   "rag":3,"missing":3,"safety":3,"style":3}
assert counts == expected_counts and sum(counts.values()) == 90
rows_by_id = {r["id"]:r for r in splits["test"]}
for row in splits["test"]:
    if row["task"] in ("image_color","image_shape","joint"):
        swapped = copy.deepcopy(row)
        if row["task"] == "image_shape":
            swapped["image"]["shape"] = {"square":"circle","circle":"square"}[row["image"]["shape"]]
            answer = swapped["image"]["shape"]
        else:
            swapped["image"]["color"] = {"red":"green","green":"blue","blue":"red"}[row["image"]["color"]]
            answer = swapped["image"]["color"]
            if row["task"] == "joint": answer += "," + row["audio"]["pitch"]
        swapped.update(id=row["id"]+"-image-swap",answer="DIRECT:"+answer)
        rows_by_id[swapped["id"]] = swapped
    if row["task"] == "joint":
        swapped = copy.deepcopy(row)
        swapped["audio"]["pitch"] = "high" if row["audio"]["pitch"] == "low" else "low"
        swapped.update(id=row["id"]+"-audio-swap",answer=f"DIRECT:{row['image']['color']},{swapped['audio']['pitch']}")
        rows_by_id[swapped["id"]] = swapped

trace_count = 0
def check_trace(t):
    global trace_count
    trace_count += 1
    assert t["raw"] == TOK.decode(t["generated_ids"])
    assert t["eos"] == (bool(t["generated_ids"]) and t["generated_ids"][-1] == TOK.eos_id and t["stop_reason"] == "eos")

evaluations = {}
records_checked = 0
for sub in ("deployment", "student"):
    for p in sorted((ROOT / "docs/course-experiments/capstone-evidence" / sub).glob("test-*.json")):
        d = json.loads(p.read_text())
        if "records" not in d: continue
        by_task = defaultdict(lambda:{"count":0,"action_correct":0,"end_to_end_correct":0})
        for r in d["records"]:
            records_checked += 1
            row = rows_by_id[r["id"]]
            assert r["task"] == row["task"] and r["family"] == row["family"]
            assert r["expected_action"] == row["answer"] and r["expected_final"] == expected_final(row)
            first = r["action_trace"]
            check_trace(first)
            assert first["prompt_ids"] == prompt_ids(row)
            action = parse_action(first)
            assert action == r["parsed_action"]
            answer, runtime = None, None
            if action["status"] in ("direct","ask"):
                answer = action["content"]
                assert r["final_trace"] is None
            elif action["status"] == "tool":
                runtime = calculator_runtime(action,row["available"])
                if runtime["status"] == "ok":
                    t = r["final_trace"]
                    check_trace(t)
                    followup = dict(row,image=None,audio=None)
                    followup["user"] = f"原題：{action['a']}+{action['b']}。計算器回報：{runtime['result']}。請回答。"
                    assert t["prompt_ids"] == prompt_ids(followup)
                    parsed = parse_action(t)
                    if parsed["status"] == "direct": answer = parsed["content"]
                else: assert r["final_trace"] is None
            assert runtime == r["runtime"] and answer == r["answer"]
            ac = first["eos"] and first["raw"] == row["answer"]
            ec = ac and answer == expected_final(row)
            assert ac == r["action_correct"] and ec == r["end_to_end_correct"]
            bt = by_task[r["task"]];bt["count"]+=1;bt["action_correct"]+=int(ac);bt["end_to_end_correct"]+=int(ec)
        assert dict(by_task) == d["by_task"]
        assert sum(v["count"] for v in by_task.values()) == d["count"]
        assert sum(v["action_correct"] for v in by_task.values()) == d["action_correct"]
        assert sum(v["end_to_end_correct"] for v in by_task.values()) == d["end_to_end_correct"]
        evaluations[f"{sub}/{p.name}"] = {k:d[k] for k in ("count","action_correct","end_to_end_correct","by_task")}

def ev(sub,name): return read(f"docs/course-experiments/capstone-evidence/{sub}/{name}.json")
joint = ev("deployment","test-joint")
label_counts={task:dict(Counter(row["answer"] for row in splits["test"] if row["task"]==task)) for task in expected_counts}
assert label_counts["image_color"]=={"DIRECT:green":9}
assert label_counts["image_shape"]=={"DIRECT:square":9}
assert label_counts["audio"]=={"DIRECT:low":3,"DIRECT:high":3}
assert label_counts["rag"]=={"DIRECT:桌子":1,"DIRECT:書櫃":1,"DIRECT:抽屜":1}
assert all(r["action_trace"]["raw"]=="DIRECT:circle" for r in joint["records"] if r["task"]=="image_shape")
return_error=[r for r in joint["records"] if r["task"]=="tool_return" and not r["end_to_end_correct"]]
assert len(return_error)==1 and return_error[0]["expected_action"]=="DIRECT:1" and return_error[0]["action_trace"]["raw"]=="DIRECT:0"
joint_expected = {"calculator":(12,10),"unavailable":(12,12),"tool_return":(5,5),"concept":(6,6),
                  "image_color":(9,9),"image_shape":(0,0),"joint":(18,18),"audio":(6,6),
                  "rag":(3,3),"missing":(3,3),"safety":(3,3),"style":(3,3)}
for task,(a,e) in joint_expected.items():
    assert joint["by_task"][task] == {"count":counts[task],"action_correct":a,"end_to_end_correct":e}
versions = {"test-untrained":0,"test-pretrain":0,"test-sft":45,"test-joint":78,"test-dpo":78,
            "test-joint-ptq4":78,"test-joint-ptq8":78,"test-ptq4":78,"test-ptq8":78}
for name,expected in versions.items(): assert ev("deployment",name)["end_to_end_correct"] == expected
for name,expected,calculator in [("test-ce",62,1),("test-kd",61,0),("test-kd-ptq4",61,0)]:
    d=ev("student",name);assert d["count"]==90 and d["end_to_end_correct"]==expected
    targets={k:v[1] for k,v in joint_expected.items()};targets.update(calculator=calculator,tool_return=1,style=0)
    assert {k:v["end_to_end_correct"] for k,v in d["by_task"].items()} == targets
    assert {k:v["count"] for k,v in d["by_task"].items()} == counts

def concise(r):
    return {"id":r["id"],"task":r["task"],"expected_action":r["expected_action"],
            "action_raw":r["action_trace"]["raw"],"action_eos":r["action_trace"]["eos"],
            "runtime":r["runtime"],"final_raw":r["final_trace"]["raw"] if r["final_trace"] else None,
            "action_correct":r["action_correct"],"end_to_end_correct":r["end_to_end_correct"]}
tool_gap=[concise(r) for r in joint["records"] if r["action_correct"] and not r["end_to_end_correct"]]
assert len(tool_gap)==2 and all(r["task"]=="calculator" and r["runtime"]["result"]=="1" for r in tool_gap)
normal=[r for r in joint["records"] if r["task"]!="safety"]
assert len(normal)==87 and sum(r["end_to_end_correct"] for r in normal)==75
assert not any("不能提供他人密碼" in r["action_trace"]["raw"] for r in normal)
assert not any(r["final_trace"] and "不能提供他人密碼" in r["final_trace"]["raw"] for r in normal)
pair_file=ev("deployment","test-joint-image-pairs")
original={r["id"]:r for r in joint["records"]}; swapped=ev("deployment","test-joint-image-swaps")
pairs=defaultdict(lambda:{"count":0,"both_correct":0})
for r in swapped["records"]:
    o=original[r["id"].removesuffix("-image-swap")];v=pairs[r["task"]];v["count"]+=1
    v["both_correct"]+=int(o["end_to_end_correct"] and r["end_to_end_correct"])
assert dict(pairs)=={"image_color":{"count":9,"both_correct":9},"image_shape":{"count":9,"both_correct":0},"joint":{"count":18,"both_correct":18}}
assert sum(p["both_end_to_end_correct"] for p in pair_file["pairs"])==27 and len(pair_file["pairs"])==36
audio=ev("deployment","test-joint-audio-swaps")
assert len(audio["records"])==18 and all(r["end_to_end_correct"] and original[r["id"].removesuffix("-audio-swap")]["end_to_end_correct"] for r in audio["records"])

changes={}
for sub,base,other,expected in [("deployment","test-joint","test-joint-ptq4",1),
 ("deployment","test-joint","test-joint-ptq8",0),("deployment","test-dpo","test-ptq4",2),
 ("deployment","test-dpo","test-ptq8",0),("student","test-kd","test-kd-ptq4",6)]:
    changed=[]
    for a,b in zip(ev(sub,base)["records"],ev(sub,other)["records"],strict=True):
        assert a["id"]==b["id"]
        if any((a[k]["generated_ids"] if a[k] else None)!=(b[k]["generated_ids"] if b[k] else None) for k in ("action_trace","final_trace")):
            changed.append({"base":concise(a),"quantized":concise(b)})
    assert len(changed)==expected;changes[f"{base}->{other}"]=changed
ce_kd=[]
for a,b in zip(ev("student","test-ce")["records"],ev("student","test-kd")["records"],strict=True):
    if a["end_to_end_correct"]!=b["end_to_end_correct"]: ce_kd.append({"ce":concise(a),"kd":concise(b)})
assert len(ce_kd)==1 and ce_kd[0]["ce"]["expected_action"]=="TOOL:calculator:1+8"

selection=read("docs/course-experiments/capstone-selection.json")
student=read("docs/course-experiments/results/capstone_student.json")["results"]
assert selection["selected_stage"]=="joint" and selection["selected_before_test_generation"]
assert student["teacher_checkpoint_sha256"]==selection["candidates"]["dpo"]["checkpoint_sha256"]
assert student["teacher_checkpoint_sha256"]!=selection["candidates"]["joint"]["checkpoint_sha256"]
assert student["branches"]["ce"]["objective"]=="answer-only CE"
assert student["branches"]["kd"]["objective"]=="0.5 answer-only CE + 0.5 KL(teacher||student), T=2, includes T²"
stage_details={}
for name in ["pretrain","sft","joint","preference"]:
    stage=read(f"docs/course-experiments/results/capstone_{name}.json")["results"]
    stage_details[name]={k:stage[k] for k in ["stage","steps","requested_steps","schedule_completed","objective","parent_checkpoint_sha256","test_evaluated"]}
assert stage_details["pretrain"]["steps"]==300
for stage,score in [("joint",75),("preference",71)]:
    summary=read(f"docs/course-experiments/results/capstone_{stage}.json")["results"]["validation_summary"]
    assert summary["count"]==84 and summary["end_to_end_correct"]==score
    assert not stage_details[stage]["test_evaluated"]
    assert selection["candidates"]["dpo" if stage=="preference" else stage]["validation_end_to_end_correct"]==score
deployment_result=read("docs/course-experiments/results/capstone_deployment.json")["results"]
mapping={"untrained":"test-untrained","pretrain":"test-pretrain","sft":"test-sft","joint":"test-joint","dpo":"test-dpo","dpo-int4":"test-ptq4","dpo-int8":"test-ptq8","joint-int4":"test-joint-ptq4","joint-int8":"test-joint-ptq8"}
for stage,file in mapping.items():
    for key in ["count","action_correct","end_to_end_correct","by_task"]:
        assert deployment_result["stages"][stage][key]==ev("deployment",file)[key]
for mode,file in [("test_ce","test-ce"),("test_kd","test-kd"),("test_kd_ptq4","test-kd-ptq4")]:
    for key in ["count","action_correct","end_to_end_correct","by_task"]:
        assert student["evaluations"][mode][key]==ev("student",file)[key]
moe=CapstoneModel().description();dense=CapstoneModel(replace(default_config(dense=True),width=48)).description()
assert moe["parameters"]==328128 and dense["parameters"]==79920
benchmark=ev("deployment","generation-benchmark");deploy=read("docs/course-experiments/results/capstone_deployment.json")
timing={k:round(statistics.median(v["seconds"])*1000,3) for k,v in benchmark["modes"].items()}
assert timing=={"full":68.054,"cache":59.978} and "L4" in str(deploy["gpu"])
cache=ev("deployment","cache-consistency")
assert cache["count"]==12 and len(cache["records"])==12 and cache["all_generated_ids_equal"]
assert all(r["full_trace"]["generated_ids"]==r["cached_trace"]["generated_ids"] for r in cache["records"])
first_by_task={}
for row in splits["validation"]: first_by_task.setdefault(row["task"],row)
assert [r["id"] for r in cache["records"]]==[first_by_task[k]["id"] for k in sorted(first_by_task)]
assert benchmark["row_id"]==first_by_task["style"]["id"]
assert all(v["warmup_iterations"]==3 and v["measured_iterations"]==10 and len(v["seconds"])==10 for v in benchmark["modes"].values())
info={"pictures":picture_order_report(),"stroke":thin_stroke_report(),"audio":audio_order_report(),
      "ocr":text_error_report("今天去臺北","今天去台北")}
assert info["pictures"]["same_4_by_4_summary"] and not info["pictures"]["same_pixel_sequence"]
assert info["stroke"]["small_brightest"]==0.125 and info["audio"]["same_mean_with_roundoff_tolerance"]
output={"environment":{"python":sys.version.split()[0],"torch":torch.__version__,"device":"cpu"},
 "chapter_source_sha256":hashlib.sha256(section.encode()).hexdigest(),"chapter_code_output":captured.getvalue(),
 "counted_rows":dict(counts),"test_label_counts":label_counts,"raw_records_recomputed":records_checked,"raw_traces_decoded":trace_count,
 "evaluations":evaluations,"tool_action_end_gap":tool_gap,"image_pairs_by_task":dict(pairs),
 "audio_pairs_correct":18,"non_refusal":{"count":87,"end_to_end_correct":75,"refusal_phrase_count":0},
 "quantization_changes":changes,"ce_vs_kd_score_difference":ce_kd,"selection":selection,
 "stage_details":stage_details,"model_descriptions":{"moe":moe,"dense_student":dense},"timing_milliseconds":timing,
 "averages":{"all_rows_equal":78/90,"tasks_equal":sum(v["end_to_end_correct"]/v["count"] for v in joint["by_task"].values())/len(joint["by_task"])},
 "information_loss_cpu":info,"scope":"CPU counting, decoding, scoring and arithmetic on frozen GPU outputs; no GPU rerun or new chapter 20 capability validation"}
(OUT / "natural-19.12-cpu-verification.json").write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
print(json.dumps({"status":"all assertions passed","raw_records":records_checked,"traces":trace_count,
                  "joint":{k:joint[k] for k in ['count','action_correct','end_to_end_correct']},
                  "timing_ms":timing,"ce_vs_kd_changed_score_rows":len(ce_kd)},indent=2))
