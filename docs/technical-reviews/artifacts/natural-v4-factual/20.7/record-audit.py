"""Independent fixed-run audit + native tokenizer recomputation on CPU.

Uses original Jinja template/BPE tokenizer and actual existing image dimensions.
Does not run Transformers image pixel preprocessing, model forward or GPU work.
"""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import random
import sys
import time
import torch
from PIL import Image
from jinja2 import Template
from tokenizers import Tokenizer
from tiny_perceptron.natural_assistant import messages_for

started = time.monotonic()
root = Path.cwd()
out = root / "docs/technical-reviews/artifacts/natural-v4-factual/20.7"
research = root / "outputs/natural-v4/factual-research/20.7"
record_root = root / "docs/natural-assistant/evidence/v4-research/full-token-audit/final-current-assembly"
data = root / "outputs/natural-v4/data"
manifest_path = root / "docs/natural-assistant/v4/manifest.json"
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def fingerprint(v):
    return hashlib.sha256(canonical(v)).hexdigest()

manifest = json.loads(manifest_path.read_text())
rows = manifest["rows"]
records_path = record_root / "row-results.jsonl"
records = {r["id"]: r for r in map(json.loads, records_path.read_text().splitlines())}
assert len(rows) == len(records) == 2371
assert {r["id"] for r in rows} == set(records)
template_path = research / "qwen-template.json"
tokenizer_path = research / "qwen-tokenizer.json"
template = Template(json.loads(template_path.read_text())["chat_template"])
tokenizer = Tokenizer.from_file(str(tokenizer_path))
assert tokenizer.truncation is None and tokenizer.padding is None
asset_results = {}
split_counts, split_tokens = Counter(), Counter()
row_tokens = {}
verification_counts = Counter()

# Same arithmetic as original smart_resize, using patch16 * merge2 factor32.
def resized(height, width):
    assert max(height, width) / min(height, width) <= 200
    factor, minimum, maximum = 32, 65536, 524288
    h, w = round(height / factor)*factor, round(width / factor)*factor
    if h*w > maximum:
        beta = math.sqrt(height*width/maximum)
        h = max(factor, math.floor(height/beta/factor)*factor)
        w = max(factor, math.floor(width/beta/factor)*factor)
    elif h*w < minimum:
        beta = math.sqrt(minimum/(height*width))
        h, w = math.ceil(height*beta/factor)*factor, math.ceil(width*beta/factor)*factor
    return h, w

def render(messages, generation_prompt, grids):
    text = template.render(messages=messages, add_generation_prompt=generation_prompt)
    original = text
    for t, h, w in grids:
        assert t*h*w % 4 == 0
        text = text.replace("<|image_pad|>", "<|placeholder|>"*(t*h*w//4), 1)
    text = text.replace("<|placeholder|>", "<|image_pad|>")
    return original, tokenizer.encode(text, add_special_tokens=False).ids

for row in rows:
    r = records[row["id"]]
    assert fingerprint(row) == r["row_sha256"], row["id"]
    sha256_answer = hashlib.sha256(row["answer"].encode()).hexdigest()
    assert sha256_answer == r["answer_utf8_sha256"]
    grids = []
    for asset in r["assets"]:
        name = asset["relative_path"]
        if name not in asset_results:
            p = data / name
            with Image.open(p) as im:
                dimensions = list(im.size)
            asset_results[name] = {"sha256": sha(p), "bytes": p.stat().st_size, "dimensions": dimensions}
        actual = asset_results[name]
        assert actual["sha256"] == asset["sha256"] and actual["bytes"] == asset["bytes"]
        assert actual["dimensions"] == asset["original_dimensions"]
        width, height = actual["dimensions"]
        rh, rw = resized(height, width)
        grids.append([1, rh//16, rw//16])
    assert grids == r["full_visual"]["image_grid_thw"] == r["prompt_visual"]["image_grid_thw"]
    pt, pi = render(messages_for(row, data), True, grids)
    ft, fi = render(messages_for(row, data, assistant=True), False, grids)
    assert ft.startswith(pt) and ft.endswith(row["answer"] + "<|im_end|>\n")
    assert hashlib.sha256(pt.encode()).hexdigest() == r["template_sha256"]["prompt"]
    assert hashlib.sha256(ft.encode()).hexdigest() == r["template_sha256"]["full"]
    assert pi == fi[:len(pi)]
    assert len(pi) == r["prompt_tokens"] and len(fi) == r["full_tokens"] <= 2048
    assert fingerprint([pi]) == r["prompt_input_ids_sha256"]
    assert fingerprint([fi]) == r["full_input_ids_sha256"]
    suffix = fi[len(pi):]
    labels = [-100]*len(pi) + suffix
    assert suffix and suffix == r["supervised_answer_token_ids"]
    assert suffix[-2:] == [151645, 198]  # EOS and template newline.
    assert len(suffix) == r["supervised_tokens"] == r["next_token_shifted_supervised_tokens"]
    assert len(suffix) == r["assistant_template_suffix_tokens"]
    assert fingerprint([labels]) == r["labels_sha256"]
    assert r["status"] == "passed" and all(r["checks"].values())
    verification_counts["recomputed_full_and_prompt_id_hashes"] += 1
    verification_counts["recomputed_labels_and_complete_suffix"] += 1
    verification_counts["recomputed_grid_shape_from_actual_images"] += 1
    row_tokens[row["id"]] = len(suffix)
    split_counts[row["split"]] += 1
    split_tokens[row["split"]] += len(suffix)

training_rows = [r for r in rows if r["split"] == "train"]
assert len(training_rows) == 2077 and split_tokens["train"] == 39436
run_results = []
for run in ["37213067566", "37217452291"]:
    p = root / f"outputs/natural-v4/modal-runs/train-{run}/natural-natural-v4-train-{run}-1/review/adapter/training.json"
    tr = json.loads(p.read_text())
    assert tr["manifest_sha256"] == sha(manifest_path)
    assert tr["model_revision"] == "89644892e4d85e24eaac8bacfd4f463576704203"
    assert tr["completed_steps"] == tr["requested_steps"] == len(tr["history"]) == 2077
    assert tr["seed"] == 42 and tr["gradient_accumulation"] == 2
    assert tr["max_tokens"] == 2048 and tr["max_pixels"] == 524288 and tr["min_pixels"] == 65536
    observed, tokens = Counter(), 0
    for index, step in enumerate(tr["history"]):
        assert step["step"] == index+1 and len(step["row_ids"]) == 2
        assert step["supervised_tokens"] == sum(row_tokens[rid] for rid in step["row_ids"])
        assert all(rid in row_tokens for rid in step["row_ids"])
        for micro, rid in enumerate(step["row_ids"]):
            position = index*2+micro
            permutation = list(range(len(training_rows)))
            random.Random(42 + position//len(training_rows)).shuffle(permutation)
            assert rid == training_rows[permutation[position % len(training_rows)]]["id"]
            observed[rid] += 1
        tokens += step["supervised_tokens"]
    assert len(observed) == 2077 and set(observed.values()) == {2}
    assert sum(observed.values()) == tr["trained_rows"] == 4154
    assert tokens == 39436*2 == 78872
    assert tr["status"] == "completed"
    run_results.append({"run": run, "record_path": p.relative_to(root).as_posix(), "record_sha256": sha(p),
                        "recorded_gpu_environment": {k: tr[k] for k in ["device", "dtype", "gpu_name", "versions"]},
                        "configuration": {k: tr[k] for k in ["seed", "learning_rate", "gradient_accumulation", "min_pixels", "max_pixels", "max_tokens"]},
                        "updates": len(tr["history"]), "row_visits": sum(observed.values()),
                        "unique_train_rows": len(observed), "visits_per_row": sorted(set(observed.values())),
                        "sum_supervised_tokens": tokens, "each_step_row_order_and_target_count_verified": True})
result = {"environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu",
                           "tokenizers": importlib.metadata.version("tokenizers"), "jinja2": importlib.metadata.version("jinja2")},
          "manifest_sha256": sha(manifest_path), "records_sha256": sha(records_path),
          "original_template_sha256": sha(template_path), "original_tokenizer_sha256": sha(tokenizer_path),
          "split_counts": dict(split_counts), "split_supervised_tokens": dict(split_tokens),
          "verification_counts": dict(verification_counts), "unique_image_assets_checked": len(asset_results),
          "max_full_tokens": max(r["full_tokens"] for r in records.values()),
          "derivation": "2077 train rows * 2 visits = 4154; 39436 encoded targets * 2 = 78872. Units are prediction targets including im_end and newline, not characters or independent answers.",
          "training_run_audits": run_results, "wall_seconds": time.monotonic()-started,
          "limits": "Independent CPU native tokenizer/template/hash/grid arithmetic and existing-record validation. No Transformers pixel tensor recomputation, model forward/backward, GPU training, benchmark, or quality replication."}
(out / "record-audit-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
