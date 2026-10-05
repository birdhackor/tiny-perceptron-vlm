"""Bounded offline CPU checks of the actual section and split helper, not training."""
from pathlib import Path
import hashlib
import itertools
import json
import os
import platform
import random
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import split_documents

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
base = ["貓看狗。", "狗看貓。", "鳥看魚。", "魚看鳥。", "貓看狗。"]
parts = split_documents(base, seed=42)
assert base == ["貓看狗。", "狗看貓。", "鳥看魚。", "魚看鳥。", "貓看狗。"]
expected = {
    "train": ["鳥看魚。", "狗看貓。"],
    "validation": ["魚看鳥。"],
    "test": ["貓看狗。"],
}
assert parts == expected
assert parts == split_documents(base, seed=42)
intersections = {f"{a}/{b}": sorted(set(parts[a]) & set(parts[b])) for a,b in itertools.combinations(parts,2)}
assert all(not value for value in intersections.values())
assert sum(map(len,parts.values())) == len(set(base)) == 4
added_duplicate = split_documents(base+["貓看狗。"],seed=42)
added_unique = split_documents(base+["鳥看貓。"],seed=42)
assert added_duplicate == parts
assert sum(map(len, added_unique.values())) == 5
assert {name:len(rows) for name,rows in added_unique.items()} == {"train":3,"validation":1,"test":1}
shared = set("貓看狗。") & set("狗看貓。")
assert shared == set("貓看狗。")
# The specified windows are consecutive, length-four windows of one source.
source = "貓看狗。狗看貓。"
windows = [source[i:i+4] for i in range(len(source)-3)]
assert windows[0] == "貓看狗。" and windows[1] == "看狗。狗"
assert windows[0][1:] == windows[1][:-1] == "看狗。"
assert len(windows[0][1:]) == 3
# Probe the stated limitation: unequal strings sharing a source are not grouped automatically.
variants = ["貓看狗。", "貓看狗！", "狗看貓。", "鳥看魚。"]
witness = None
for seed in range(16):
    split = split_documents(variants, seed=seed)
    owners = {doc:name for name,rows in split.items() for doc in rows}
    if owners["貓看狗。"] == "train" and owners["貓看狗！"] != "train":
        witness = {"seed":seed,"split":split,"family":["貓看狗。","貓看狗！"]}
        break
assert witness is not None
# The section's assertion cannot detect a duplicate only between train and test.
bad = {"train":["貓看狗。"],"validation":["狗看貓。"],"test":["貓看狗。"]}
assert not (set(bad["train"]) & set(bad["validation"]))
assert set(bad["train"]) & set(bad["test"])
for n in range(3,21):
    rows = [f"文件-{i}" for i in range(n)]
    groups = split_documents(rows+rows[:2],seed=42)
    assert all(groups.values())
    assert sum(map(len,groups.values())) == n
    assert all(not set(groups[a])&set(groups[b]) for a,b in itertools.combinations(groups,2))
try:
    split_documents(["同一句", "同一句", "另一句"],seed=42)
except ValueError as e:
    minimum_error = str(e)
else:
    raise AssertionError("Expected minimum-three distinct documents error")
print(json.dumps({
    "base_parts":parts,"all_pair_intersections":intersections,
    "repeat_seed_identical":True,"input_unmodified":True,
    "added_duplicate_total":sum(map(len,added_duplicate.values())),
    "added_unique_total":sum(map(len,added_unique.values())),
    "added_unique_counts":{name:len(rows) for name,rows in added_unique.items()},
    "shared_character_set_example":sorted(shared),
    "adjacent_windows":windows[:2],"overlap":windows[0][1:],"overlap_characters":3,
    "near_duplicate_cross_split_witness":witness,
    "train_validation_only_assertion_misses_train_test_duplicate":True,
    "minimum_three_error":minimum_error,
    "bounded_split_sizes_checked":"3..20 unique documents; added exact duplicates; all three groups nonempty and pairwise disjoint",
    "not_executed":"no model construction, optimizer, training, evaluation score, GPU work or dataset/model download"
},ensure_ascii=False,indent=2))
if __name__ == "__main__":
    environment={"python":platform.python_version(),"executable":sys.executable,"torch":torch.__version__,
        "device":"cpu","cuda_build":str(torch.version.cuda),"cuda_available":str(torch.cuda.is_available()),
        "cwd":str(Path.cwd()),"offline_environment":{k:os.environ.get(k,'') for k in ("CUDA_VISIBLE_DEVICES","HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","TRANSFORMERS_OFFLINE","OMP_NUM_THREADS","MKL_NUM_THREADS")},
        "sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("tiny_perceptron/data.py","docs/technical-reviews/artifacts/phase4-1_2-factual/probe.py")}}
    (Path(__file__).parent/'probe-environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
