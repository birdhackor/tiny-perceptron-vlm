import ast
import hashlib
import json
import platform
import sys
from pathlib import Path
import torch
from tiny_perceptron.natural_concepts import picture_order_report, swapped_picture
import torch.nn.functional as F

root = Path(__file__).resolve().parents[5]
base = Path(__file__).resolve().parents[1]
readme = (base / "sources/fashion-mnist-upstream-README.md").read_text()
assert "28x28 grayscale image" in readme
assert "| 1 | Trouser |" in readme and "| 8 | Bag |" in readme and "| 9 | Ankle boot |" in readme

original_report = picture_order_report()
assert original_report == {"different_pixels":128,"same_4_by_4_summary":True,"same_pixel_sequence":False}
first, swapped = swapped_picture()
changed_grid = first.clone()
changed_grid[0, 8:16, 0:4] = 0
changed_grid[0, 8:16, 8:12] = 1
pool = lambda x: F.adaptive_avg_pool2d(x[None],(4,4))
assert not torch.equal(pool(first),pool(changed_grid))
# Positions are labelled scenario inputs, not trained model predictions.
scenes = [{"bag":0,"boot":1},{"bag":1,"boot":0}]
def answer(scene, subject, reference):
    return "左邊" if scene[subject] < scene[reference] else "右邊"
cases = [answer(scenes[0],"bag","boot"),answer(scenes[1],"bag","boot"),answer(scenes[0],"boot","bag")]
assert cases == ["左邊","右邊","右邊"]
assert set(scenes[0]) == set(scenes[1])
# Each composed sample carries every original source id used to form the scene.
train = [("bag-A","boot-B"),("bag-A/crop","boot-B/scale")]
test = [("bag-C","boot-D"),("bag-C/scale","boot-D/crop")]
family = lambda id: id.split("/")[0]
ids = lambda samples: {family(id) for sample in samples for id in sample}
assert ids(train).isdisjoint(ids(test))
leaky_test = [("bag-A/scale","boot-D")]
assert ids(train) & ids(leaky_test) == {"bag-A"}

result = {
 "environment":{"python":sys.version,"torch":torch.__version__,"numpy":__import__("numpy").__version__,"device":"cpu","threads":torch.get_num_threads()},
 "fashion_spec_parse":{"image_shape":[28,28],"color":"grayscale","labels":{"1":"Trouser","8":"Bag","9":"Ankle boot"},"pixels_per_image":28*28,"scope":"official README bytes only, not dataset decoding"},
 "prerequisite_original_report":original_report,
 "cross_grid_variant":{"pooled_equal":torch.equal(pool(first),pool(changed_grid)),"same_4_by_4_cells":False},
 "designed_relation_answers":{"original_bag_vs_boot":cases[0],"swapped_bag_vs_boot":cases[1],"original_boot_vs_bag":cases[2],"identity_set_same_when_swapped":True,"scope":"deterministic design oracle, no model execution"},
 "source_family_split":{"training_families":sorted(ids(train)),"test_families":sorted(ids(test)),"overlap":sorted(ids(train)&ids(test)),"leaky_variant_overlap":sorted(ids(train)&ids(leaky_test)),"scope":"bounded illustrative contract, no claim that stage-5 data exists"},
 "weights_loaded":False,"training_performed":False,"source_file_sha256":hashlib.sha256((root/"tiny_perceptron/natural_concepts.py").read_bytes()).hexdigest()
}
print(json.dumps(result,ensure_ascii=False,indent=2))
