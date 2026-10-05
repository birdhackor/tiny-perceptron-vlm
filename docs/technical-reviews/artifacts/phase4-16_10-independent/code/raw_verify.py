"""Check only designated original measurement pointers and raw dataset/sample values."""
import hashlib
import json
import random
from pathlib import Path

from scripts.course_experiments.common import records_sha256, text_examples
from tiny_perceptron.data import ByteTokenizer, IGNORE

OUT = Path(__file__).resolve().parents[1]
original = OUT / "raw/efficiency.json"
dataset_file = OUT / "raw/dataset-original.json"
data = json.loads(original.read_bytes())
dataset = json.loads(dataset_file.read_bytes())
expected_sha = next(x["sha256"] for x in data["artifacts"] if x["path"] == "dataset.json")
actual_sha = hashlib.sha256(dataset_file.read_bytes()).hexdigest()
assert actual_sha == expected_sha
print("raw_file_sha256", hashlib.sha256(original.read_bytes()).hexdigest())
print("dataset_sha256", actual_sha)
for split in ("train", "validation", "test"):
    report=data["results"]["dataset"][split]
    assert len(dataset[split])==report["records"]
    assert records_sha256(dataset[split])==report["sha256"]
    print("dataset_split",split,len(dataset[split]),report["sha256"])

examples=text_examples(dataset["train"],mode="sft",max_length=128)
tok=ByteTokenizer()
for record,(_,labels) in zip(dataset["train"],examples,strict=True):
    expected=sum(len(tok.encode(message["content"]))+1 for message in record["messages"] if message["role"]=="assistant")
    assert int((labels!=IGNORE).sum())==expected
sampler=random.Random(data["seed"])
per_step=[sum(int((y!=IGNORE).sum()) for _,y in sampler.choices(examples,k=8)) for _ in range(40)]
assert sum(per_step)==2328
print("sampler_per_step_effective_answer_plus_eos_targets",per_step)
print("sampler_total",sum(per_step),"updates",40,"batch_size",8)
expected_timings={"ordinary":"14.070","activation_checkpoint":"20.735"}
expected_mib={"ordinary":["67.164","75.988","8.824"],"activation_checkpoint":["69.336","74.377","5.041"]}
computed={}
for variant in ("ordinary","activation_checkpoint"):
    value=data["results"]["update_variants"][variant]
    training=value["training"]
    assert training["steps"]==training["requested_steps"]==training["optimizer_updates"]==40
    assert training["skipped_updates"]==0 and training["effective_tokens"]==sum(per_step)
    assert training["batch_size"]==8 and training["micro_batch_sizes"]==[8]
    assert training["peak_memory_allocated_bytes"]-training["memory_allocated_before_bytes"]==training["peak_additional_allocated_bytes"]
    milliseconds=f'{training["warm_step_median_seconds"]*1000:.3f}'
    mib=[f'{training[k]/2**20:.3f}' for k in ("memory_allocated_before_bytes","peak_memory_allocated_bytes","peak_additional_allocated_bytes")]
    assert milliseconds==expected_timings[variant] and mib==expected_mib[variant]
    quality={}
    for split in ("validation","test"):
        held=value["heldout"][split];samples=held["samples"]
        assert held["records"]==len(samples)==len(dataset[split])
        matches=0;ended=0
        for sample,record in zip(samples,dataset[split],strict=True):
            ids=sample["generated_ids"]
            raw=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
            exact=raw==tok.encode(sample["expected"])
            assert sample["expected"]==record["messages"][-1]["content"]
            assert sample["messages"]==record["messages"][:-1]
            assert sample["exact"]==exact and sample["eos"]==(tok.eos_id in ids)
            assert sample["generated"]==tok.decode(raw)
            matches+=int(exact);ended+=int(tok.eos_id in ids)
        assert matches==held["matches"] and matches/len(samples)==held["exact_match"]
        assert ended/len(samples)==held["eos_rate"]
        quality[split] = {"matches":matches,"denominator":len(samples),"exact_match":held["exact_match"],"eos":ended}
    computed[variant]={"update_median_ms":milliseconds,"allocated_before_peak_additional_mib":mib,"quality":quality}
assert computed["ordinary"]["quality"]==computed["activation_checkpoint"]["quality"]
assert data["results"]["activation_checkpoint"]["logit_max_error"]==0
assert data["results"]["activation_checkpoint"]["gradient_max_error"]==0
assert data["results"]["update_variants"]["activation_checkpoint"]["weight_max_error_from_ordinary_after_updates"]==0
print(json.dumps(computed,ensure_ascii=False,indent=2))
print("all raw checks passed; GPU measurements and weight equality are reported original values, not rerun")
