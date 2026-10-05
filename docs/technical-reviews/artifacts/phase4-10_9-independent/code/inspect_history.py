"""Inspect only named raw measurements/config/provenance/samples from the immutable full JSON."""
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parents[1]
path = base / "inputs/contrastive-original.json"
raw = path.read_bytes()
document = json.loads(raw)
pointers = []

def get(pointer):
    pointers.append(pointer)
    value = document
    for key in pointer.strip("/").split("/"):
        value = value[key.replace("~1", "/").replace("~0", "~")]
    return value

output = {"original_sha256": hashlib.sha256(raw).hexdigest(), "provenance": {}, "config": get("/results/config"), "splits": {}, "measurements": {}}
for key in ["schema_version", "experiment_id", "revision", "device", "seed", "torch_version", "python_version", "step_scale"]:
    output["provenance"][key] = get("/" + key)
for key in ["seed", "split_policy"]:
    output["provenance"]["data_" + key] = get("/results/data/" + key)
record_sets = {}
for split in ["train", "validation", "test"]:
    root = "/results/data/splits/" + split
    count = get(root + "/count")
    records = get(root + "/records")
    assert len(records) == count
    record_sets[split] = records
    families = {row["family"] for row in records}
    answers = {row["answer"] for row in records}
    output["splits"][split] = {"count": count, "families": sorted(families), "answers": sorted(answers), "offsets": sorted({row["offset"] for row in records})}
for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert {row["family"] for row in record_sets[left]}.isdisjoint({row["family"] for row in record_sets[right]})

for variant in ["one_way", "two_way"]:
    output["measurements"][variant] = {}
    training = {}
    for key in ["parameters", "trainable_parameters", "steps", "effective_targets", "weights_changed", "nonzero_gradient_seen", "cpu_smoke"]:
        training[key] = get("/results/variants/" + variant + "/training/" + key)
    assert training["steps"] == 250 and training["effective_targets"] == 250 * 6
    output["measurements"][variant]["training"] = training
    for split in ["before", "validation", "test"]:
        root = "/results/variants/" + variant + "/" + split
        samples_i = get(root + "/image_samples")
        samples_t = get(root + "/text_samples")
        n_i, n_t = get(root + "/image_queries"), get(root + "/text_queries")
        a_i, a_t = get(root + "/image_to_text_accuracy"), get(root + "/text_to_image_accuracy")
        assert len(samples_i) == n_i and len(samples_t) == n_t
        assert all(row["correct"] == (row["target"] == row["predicted"]) for row in samples_i)
        assert all(row["correct"] == (row["target"] == row["retrieved"]) for row in samples_t)
        count_i, count_t = sum(row["correct"] for row in samples_i), sum(row["correct"] for row in samples_t)
        assert a_i == count_i / n_i and a_t == count_t / n_t
        # Recheck textual target identities against the raw split records, without rerunning models.
        records = record_sets["test" if split == "before" else split]
        assert all(row["target"] == records[row["row"]]["answer"] for row in samples_i)
        assert all(row["retrieved"] == records[row["image_row"]]["answer"] for row in samples_t)
        output["measurements"][variant][split] = {"image_correct": count_i, "image_queries": n_i, "text_correct": count_t, "text_queries": n_t, "image_accuracy": a_i, "text_accuracy": a_t}

code_hash = get("/code_sha256/scripts~1course_experiments~1modalities.py")
assert hashlib.sha256((base / "inputs/historical-modalities.py").read_bytes()).hexdigest() == code_hash
output["provenance"]["inspected_implementation_sha256"] = code_hash
output["inspected_pointers"] = pointers
output["scope"] = "Recomputed counts, target identities, split-family disjointness and aggregate accuracy from saved raw records only. No model loading, training, GPU execution, inference rerun or embedded notes/scope review fields inspected."
print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
