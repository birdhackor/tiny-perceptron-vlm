import ast
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
import torch

ROOT = Path.cwd()
OUT = ROOT / "docs/technical-reviews/artifacts/phase4-12_10-independent"
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
def function_only(path, name, namespace):
    tree = ast.parse((ROOT/path).read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path)+":"+name, "exec"), namespace)
    return namespace[name]
records_fn = function_only("scripts/course_experiments/modalities.py", "_audio_records", {})
tone = function_only("tiny_perceptron/multimodal.py", "tone", {"torch":torch, "math":math})
tok_tree = ast.parse((ROOT/"tiny_perceptron/data.py").read_bytes())
nodes = [n for n in tok_tree.body if (isinstance(n,ast.ClassDef) and n.name=="ByteTokenizer") or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="SPECIALS" for t in n.targets))]
ns = {};exec(compile(ast.Module(body=nodes,type_ignores=[]),"ByteTokenizer-contract","exec"),ns);tok=ns["ByteTokenizer"]()
d = json.loads((ROOT/"docs/course-experiments/results/audio.json").read_bytes())
splits = records_fn()
report = {"kind":"bounded_existing_data_verification", "environment":{"python":sys.version,"torch":str(torch.__version__),"torch_git_version":str(torch.version.git_version),"device":"cpu","cuda_build":str(torch.version.cuda),"cuda_available":str(torch.cuda.is_available()),"platform":platform.platform()},"original_json_sha256":hashlib.sha256((ROOT/"docs/course-experiments/results/audio.json").read_bytes()).hexdigest(),"inspected_pointers":['/revision', '/device', '/seed', '/torch_version', '/python_version', '/code_sha256/scripts~1course_experiments~1modalities.py', '/code_sha256/tiny_perceptron~1multimodal.py', '/results/data/splits/train/count', '/results/data/splits/train/sha256', '/results/data/splits/train/records', '/results/data/splits/validation/count', '/results/data/splits/validation/sha256', '/results/data/splits/validation/records', '/results/data/splits/test/count', '/results/data/splits/test/sha256', '/results/data/splits/test/records', '/results/test/examples', '/results/test/correct', '/results/test/exact_match', '/results/test/ablation', '/results/test/samples', '/results/blank/examples', '/results/blank/correct', '/results/blank/exact_match', '/results/blank/ablation', '/results/blank/samples', '/results/shuffle/examples', '/results/shuffle/correct', '/results/shuffle/exact_match', '/results/shuffle/ablation', '/results/shuffle/samples'],"training_or_model_evaluation_performed":False}
for split,records in splits.items():
    stored = d["results"]["data"]["splits"][split]
    assert records == stored["records"]
    digest = hashlib.sha256(json.dumps(records,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    assert digest == stored["sha256"] and len(records)==stored["count"]
families = {k:{r["family"] for r in rows} for k,rows in splits.items()}
assert not any(families[a]&families[b] for a,b in [("train","test"),("train","validation"),("validation","test")])
report["split_counts"]={k:len(v) for k,v in splits.items()};report["split_frequency_families"]={k:sorted(v,key=int) for k,v in families.items()};report["frequency_family_overlap"]=False
rows = splits["test"]
assert len(rows)==14
report["test_class_counts"]=dict(Counter(r["answer"] for r in rows))
assert report["test_class_counts"]=={"low":8,"high":6}
for key in ["test","blank","shuffle"]:
    result=d["results"][key];samples=result["samples"];assert len(samples)==14
    for i,s in enumerate(samples):
        assert s["row"]==i and s["target"]==rows[i]["answer"] and s["question"]=="pitch?"
        raw=s["generated_ids"]
        assert tok.eos_id in raw
        raw=raw[:raw.index(tok.eos_id)]
        assert tok.decode(raw)==s["generated"] and (raw==tok.encode(s["target"]))==s["exact_match"]
    correct=sum(s["exact_match"] for s in samples)
    assert result["examples"]==14 and correct==result["correct"] and correct/14==result["exact_match"]
    report[key]={"correct_against_old_target":correct,"count":14,"rate_against_old_target":correct/14,"generated_class_counts":dict(Counter(s["generated"] for s in samples)),"all_eos":all(s["eos"] for s in samples)}
assert report["test"]["correct_against_old_target"]==11
assert report["blank"]["correct_against_old_target"]==6 and report["blank"]["generated_class_counts"]=={"high":14}
shuffled=d["results"]["shuffle"]["samples"];table=[]
for i,s in enumerate(shuffled):
    donor=(i+max(1,len(rows)//2))%len(rows)
    assert s["donor_row"]==donor
    new=rows[donor]["answer"]
    table.append({"row":i,"donor_row":donor,"old_frequency_hz":rows[i]["frequency"],"new_frequency_hz":rows[donor]["frequency"],"old_target":rows[i]["answer"],"new_target":new,"generated":s["generated"],"correct_new":s["generated"]==new,"labels_unchanged":new==s["target"]})
new_correct=sum(r["correct_new"] for r in table);same=sum(r["labels_unchanged"] for r in table)
assert report["shuffle"]["correct_against_old_target"]==3 and new_correct==11 and same==2
report["shuffle"].update(correct_against_new_target=new_correct,rate_against_new_target=new_correct/14,unchanged_label_count=same,unchanged_label_rows=[r["row"] for r in table if r["labels_unchanged"]],donor_table=table)
report["fixed_low_baseline"]={"correct":8,"count":14,"accuracy":8/14}
report["four_frequency_variant"]={"truth":["high" if f>300 else "low" for f in [200,220,440,660]],"constant_high_accuracy":2/4,"at_300_hz":"high" if 300>300 else "low","at_310_hz":"high" if 310>300 else "low","nine_low_of_ten_baseline":9/10}
assert report["four_frequency_variant"]["at_300_hz"]=="low"
wave_checks=[]
for row in rows:
    wave=tone(row["frequency"],seconds=row["seconds"])*(row["amplitude"]/0.5)
    blank=torch.zeros_like(wave)
    assert wave.shape==(round(row["seconds"]*16000),) and bool((blank==0).all())
    wave_checks.append({"frequency_hz":row["frequency"],"duration_seconds":row["seconds"],"samples":wave.numel(),"zero_wave_nonzero_samples":int(torch.count_nonzero(blank))})
report["waveform_contract"]={"sample_rate_hz":16000,"blank_mechanism":"zeros_like; no single tone carrier","checks":wave_checks}
(OUT/"bounded-result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in {"shuffle","waveform_contract","inspected_pointers","environment"}},ensure_ascii=False,indent=2))
print("shuffle",json.dumps({k:v for k,v in report["shuffle"].items() if k!="donor_table"}))
print("waveforms checked",len(wave_checks),"all silence zero")
