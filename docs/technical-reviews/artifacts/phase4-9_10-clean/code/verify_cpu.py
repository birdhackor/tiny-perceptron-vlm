"""Bounded independent checks. Reads named raw-measurement pointers only; no model inference/training."""
import ast
import hashlib
import inspect
import json
import math
import platform
import signal
import sys
from pathlib import Path

signal.alarm(30)
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.alignment import reliability_bins

torch.set_num_threads(1)
OUT = Path(__file__).resolve().parents[1] / "execution"
INPUT = ROOT / "docs/course-experiments/results/encoders.json"
raw = INPUT.read_bytes()
data = json.loads(raw)
reads = {}

def pointer(path):
    value = data
    for key in path.split("/")[1:]:
        value = value[key.replace("~1", "/").replace("~0", "~")]
    reads[path] = value
    return value

def check_close(actual, expected, tolerance=2e-6):
    assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), (actual, expected)

def weighted_ece(rows, n):
    return sum(r["count"] / n * abs(r["accuracy"] - r["confidence"]) for r in rows)

confidence = torch.tensor([0.55, 0.65, 0.85, 0.95])
correct = torch.tensor([1, 0, 1, 0])
original = reliability_bins(confidence, correct, bins=2)
check_close(weighted_ece(original, 4), .25)
changed = reliability_bins(torch.cat([confidence, torch.tensor([.25])]), torch.cat([correct, torch.tensor([1])]), bins=2)
check_close(weighted_ece(changed, 5), .35)
boundary = reliability_bins(torch.tensor([0., .5, 1.]), torch.tensor([1, 0, 1]), bins=2)
assert [r["count"] for r in boundary] == [1, 2]
fine_bin_ece = weighted_ece(reliability_bins(confidence, correct, bins=10), 4)
check_close(fine_bin_ece, .55)
assert fine_bin_ece != weighted_ece(original, 4)
check_close(-math.log(.8), .2231435513142097, 1e-12)
check_close(-math.log(.2), 1.6094379124341003, 1e-12)
brier_one = float((torch.tensor([.8,.2]) - torch.tensor([0.,1.])).square().sum())
check_close(brier_one, 1.28)
z = torch.tensor([[1., 2., -1.], [0., 0., 0.]])
temperature_demo = {str(t): (z/t).softmax(-1).tolist() for t in [.5,1.,2.]}
for t in [.5,1.,2.]:
    assert torch.equal(z.argmax(-1), (z/t).argmax(-1))
check_close(float((z/.5).softmax(-1)[0].max()), .87887824)
assert (z/.5).softmax(-1)[0].max() > z.softmax(-1)[0].max() > (z/2).softmax(-1)[0].max()
assert torch.allclose((z/.5).softmax(-1)[1], (z/2).softmax(-1)[1])

# Execute the exact nested calculation from the source whose run-time hash is in the raw JSON.
source = ROOT / "scripts/course_experiments/modalities.py"
source_raw = source.read_bytes()
tree = ast.parse(source_raw)
node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "confidence_metrics")
metric_code = ast.get_source_segment(source_raw.decode(), node)
(OUT.parent / "code/original-confidence-metrics.py").write_text(metric_code + "\n")
run_hash = pointer("/code_sha256/scripts~1course_experiments~1modalities.py")
assert hashlib.sha256(source_raw).hexdigest() == run_hash
code_version_checks = {}
for file in ["scripts/course_experiments/common.py", "scripts/course_experiments/run.py", "tiny_perceptron/alignment.py", "tiny_perceptron/multimodal.py", "tiny_perceptron/modal_data.py"]:
    run_hash = pointer("/code_sha256/" + file.replace("/", "~1"))
    current_hash = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    if current_hash == run_hash:
        code_version_checks[file] = {"expected_run_sha256":run_hash,"current_sha256":current_hash,"match":"current source"}
    else:
        original_path = OUT.parent / "code/original-revision" / file
        assert hashlib.sha256(original_path.read_bytes()).hexdigest() == run_hash
        code_version_checks[file] = {"expected_run_sha256":run_hash,"current_sha256":current_hash,"match":"saved original revision source", "original_path":str(original_path.relative_to(ROOT))}
provenance = {k: pointer("/"+k) for k in ["schema_version","experiment_id","revision","device","seed","torch_version","python_version","step_scale","evidence_status"]}
results = {}
for modality in ["vision", "audio"]:
    p = "/results/"+modality
    classes = pointer(p+"/classes")
    config = pointer(p+"/config")
    namespace = {"torch": torch, "F": F, "classes": classes}
    exec(compile(metric_code, str(source)+":446-490", "exec"), namespace)
    metric = namespace["confidence_metrics"]
    training = {k: pointer(p+"/training/"+k) for k in ["steps","effective_targets","weights_changed","nonzero_gradient_seen","cpu_smoke"]}
    assert training["steps"] == 250 and training["effective_targets"] == 2000
    assert training["weights_changed"] and training["nonzero_gradient_seen"] and not training["cpu_smoke"]
    manifests = {}
    for split in ["train","validation","test"]:
        rows = pointer(p+"/data/splits/"+split+"/records")
        count = pointer(p+"/data/splits/"+split+"/count")
        digest = pointer(p+"/data/splits/"+split+"/sha256")
        assert count == len(rows)
        assert digest == hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        manifests[split] = rows
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        assert not {r["family"] for r in manifests[a]} & {r["family"] for r in manifests[b]}
    if modality == "audio":
        assert all(r["answer"] == ("high" if r["frequency"] > 300 else "low") for rows in manifests.values() for r in rows)
    grid = pointer(p+"/calibration/temperature_grid")
    assert grid == [.5,.75,1.,1.5,2.,3.,5.]
    v_logits = torch.tensor(pointer(p+"/calibration/validation/logits"), dtype=torch.float32)
    v_labels = torch.tensor(pointer(p+"/calibration/validation/labels"), dtype=torch.long)
    selection = [{"temperature": t, "validation_nll": float(F.cross_entropy(v_logits/t,v_labels))} for t in grid]
    saved_selection = pointer(p+"/calibration/selection")
    for a,b in zip(selection,saved_selection,strict=True):
        assert a["temperature"] == b["temperature"]
        check_close(a["validation_nll"], b["validation_nll"])
    chosen = min(selection,key=lambda r:r["validation_nll"])["temperature"]
    assert chosen == pointer(p+"/calibration/chosen_temperature") == .5
    metrics = {}
    for split in ["validation","test"]:
        sp = p+"/calibration/"+split
        count = pointer(sp+"/count")
        logits = torch.tensor(pointer(sp+"/logits"),dtype=torch.float32)
        labels = torch.tensor(pointer(sp+"/labels"),dtype=torch.long)
        families = pointer(sp+"/families")
        assert count == len(labels) == len(logits) == len(families) == len(manifests[split])
        assert families == [r["family"] for r in manifests[split]]
        assert labels.tolist() == [classes.index(r["answer"]) for r in manifests[split]]
        assert torch.equal(logits.argmax(-1),(logits/chosen).argmax(-1))
        metrics[split] = {}
        for name,temp in [("original",1.),("calibrated",chosen)]:
            observed = metric(logits,labels,temp)
            for key in ["temperature","count","correct","accuracy","mean_confidence","nll","brier","ece","bins","probabilities","predicted_labels","confidence"]:
                saved = pointer(sp+"/"+name+"/"+key)
                if isinstance(saved, float): check_close(observed[key], saved)
                elif key in ["probabilities","confidence"]: assert torch.allclose(torch.tensor(observed[key]),torch.tensor(saved),rtol=2e-6,atol=2e-6)
                elif key == "bins":
                    for x,y in zip(observed[key],saved,strict=True):
                        for field in ["lower","upper","upper_inclusive","count","mean_confidence","accuracy"]:
                            if isinstance(y[field],float): check_close(x[field],y[field])
                            else: assert x[field] == y[field]
                else: assert observed[key] == saved, (modality,split,name,key)
            # Independent formulas additionally check the original implementation.
            probs = (logits/temp).softmax(-1)
            truth = F.one_hot(labels,len(classes))
            check_close(float((probs-truth).square().sum(-1).mean()), observed["brier"])
            check_close(float(-probs[torch.arange(count),labels].log().mean()), observed["nll"])
            metrics[split][name] = {k:observed[k] for k in ["temperature","count","correct","accuracy","mean_confidence","nll","brier","ece","bins"]}
    samples = pointer(p+"/test/samples")
    assert len(samples) == metrics["test"]["original"]["count"]
    assert sum(s["correct"] for s in samples) == metrics["test"]["original"]["correct"]
    predicted = torch.tensor(pointer(p+"/calibration/test/logits")).argmax(-1).tolist()
    for r,s,k in zip(manifests["test"],samples,predicted,strict=True):
        assert (s["family"],s["target"],s["predicted"],s["correct"]) == (r["family"],r["answer"],classes[k],classes[k]==r["answer"])
    results[modality] = {"config":config,"classes":classes,"training":training,"split_counts":{k:len(v) for k,v in manifests.items()},"selection":selection,"chosen_temperature":chosen,"metrics":metrics,"incorrect_test_families":[s["family"] for s in samples if not s["correct"]]}

printed_table = []
for modality,name,ece_digits,brier_digits,expected_ece,expected_brier in [
    ("vision","original",5,6,"0.02858","0.001485"),
    ("vision","calibrated",6,9,"0.000572","0.000000859"),
    ("audio","original",5,5,"0.13186","0.20207"),
    ("audio","calibrated",5,5,"0.14154","0.25337"),
]:
    value=results[modality]["metrics"]["test"][name]
    rendered_ece=format(value["ece"],f".{ece_digits}f")
    rendered_brier=format(value["brier"],f".{brier_digits}f")
    assert rendered_ece==expected_ece and rendered_brier==expected_brier
    printed_table.append({"modality":modality,"variant":name,"ece":rendered_ece,"brier":rendered_brier})
for name,confidence_text,nll_text in [("original","0.8796","0.2776"),("calibrated","0.9273","0.3465")]:
    value=results["audio"]["metrics"]["test"][name]
    assert format(value["mean_confidence"],".4f")==confidence_text
    assert format(value["nll"],".4f")==nll_text

summary = {
    "environment":{"python":platform.python_version(),"torch":str(torch.__version__),"device":"cpu","cuda_available":str(torch.cuda.is_available()),"torch_threads":str(torch.get_num_threads())},
    "input_path":str(INPUT.relative_to(ROOT)),"full_input_sha256":hashlib.sha256(raw).hexdigest(),
    "pointer_count":len(reads),"provenance":provenance,"code_version_checks":code_version_checks,
    "toy":{"original":original,"original_ece":weighted_ece(original,4),"exercise":changed,"exercise_ece":weighted_ece(changed,5),"endpoint_bins":boundary,"ten_bin_ece":fine_bin_ece,"nll_example":[-math.log(.8),-math.log(.2)],"brier_example":brier_one,"temperature_demo":temperature_demo},
    "results":results,"printed_table":printed_table,"scope":"CPU arithmetic recomputation of saved logits/labels and exact metric function; no training, weights, model inference, media download, or generated performance."}
(OUT/"cpu-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
(OUT/"raw-pointer-inspection.json").write_text(json.dumps({"input_path":str(INPUT.relative_to(ROOT)),"full_input_sha256":hashlib.sha256(raw).hexdigest(),"read_pointers":sorted(reads),"selected_raw_values":reads},ensure_ascii=False,indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False,indent=2))
