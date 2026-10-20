"""核對字元視窗、固定視窗模型、資料分母與已記錄的比較。"""

import hashlib
import json
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.train_simple import examples
from tiny_perceptron.data import split_documents, toy_documents
from tiny_perceptron.simple import ContextMLP
from tiny_perceptron.training import seed_everything


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


torch.set_num_threads(2)
torch.set_default_device("cpu")
source_files = [
    "tiny_perceptron/simple.py",
    "tiny_perceptron/data.py",
    "tiny_perceptron/training.py",
    "scripts/train_simple.py",
    "scripts/course_experiments/text.py",
    "scripts/course_experiments/common.py",
    "docs/course-experiments/results/simple_models.json",
    "course/figures/window_training.svg",
]
record = json.loads((ROOT / "docs/course-experiments/results/simple_models.json").read_bytes())
parts = split_documents(toy_documents(), 42)
vocabulary = {c: i + 2 for i, c in enumerate(sorted(set("".join(parts["train"]))))}
report = {
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "dtype": str(torch.get_default_dtype()),
        "threads": str(torch.get_num_threads()),
        "platform": platform.platform(),
    },
    "source_sha256": {name: digest(name) for name in source_files},
    "window_examples": {
        str(length): [(prefix[-length:], answer) for prefix, answer in
                      [("紅色物體是", "圓"), ("藍色物體是", "方")]]
        for length in [1, 3, 4, 5]
    },
    "exercise": {
        "lengths": [len(s) for s in ["紅色的小小物體是", "藍色的小小物體是"]],
        "suffixes": {str(length): [s[-length:] for s in
                                   ["紅色的小小物體是", "藍色的小小物體是"]]
                     for length in [7, 8]},
    },
    "data": {
        "documents": len(toy_documents()),
        "split_counts": {name: len(rows) for name, rows in parts.items()},
        "train_character_count": len(vocabulary),
        "vocabulary": vocabulary,
        "reserved_ids": {"boundary": 0, "unknown": 1},
        "target_counts": {name: sum(len(s) + 1 for s in parts[name])
                          for name in ["train", "validation"]},
        "document_target_lengths": {name: [len(s) + 1 for s in parts[name]]
                                    for name in ["train", "validation"]},
        "split_sha256_matches_record": {},
    },
    "configuration": {
        "seed": 42, "updates": 200, "width": 16,
        "optimizer": "AdamW", "lr": 0.01, "weight_decay": 0.01,
        "gradient_clip_max_norm": 1.0,
        "loss": "cross_entropy; reduction=mean; all character targets plus one EOS per document",
        "test_evaluated": False,
    },
    "runs": {},
}
for split, documents in parts.items():
    payload = "".join(json.dumps({"text": text, "family": text}, ensure_ascii=False) + "\n"
                      for text in documents).encode()
    matched = hashlib.sha256(payload).hexdigest() == record["results"]["data"][split]["sha256"]
    report["data"]["split_sha256_matches_record"][split] = matched
    assert matched, "資料切分與實驗紀錄不一致"

for length in [1, 3, 5]:
    seed_everything(42)
    model = ContextMLP(len(vocabulary) + 2, context=length, width=16)
    batches = {name: examples(parts[name], vocabulary, length) for name in ["train", "validation"]}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        x, y = batches["train"]
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    observed = {}
    sums = {}
    with torch.no_grad():
        for name, (x, y) in batches.items():
            scores = model(x)
            observed[name] = F.cross_entropy(scores, y).item()
            sums[name] = F.cross_entropy(scores, y, reduction="sum").item()
            assert scores.shape == (len(y), 17), "候選分數維度不一致"
            assert x.shape == (len(y), length), "輸入視窗維度不一致"
            assert x.dtype == torch.int64 and y.dtype == torch.int64, "字元編號型別不一致"
    expected = record["results"]["runs"][f"mlp{length}"]
    absolute_error = {name: abs(observed[name] - expected["after_nll_same_post_update_time"][name])
                      for name in observed}
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 577 + 256 * length == expected["parameters"], "參數公式與模型不一致"
    assert max(absolute_error.values()) < 0.000005, "重算代價超出五位小數比較的容忍差"
    x = batches["train"][0]
    with torch.no_grad():
        duplicate_logits = model(torch.stack([x[0], x[0]]))
    report["runs"][str(length)] = {
        "parameters": parameters,
        "parameter_shapes": {name: list(value.shape) for name, value in model.named_parameters()},
        "observed": observed, "nll_sums": sums,
        "recorded_train_validation": {name: expected["after_nll_same_post_update_time"][name]
                                      for name in observed},
        "recorded_table_rounded": {name: format(expected["after_nll_same_post_update_time"][name], ".5f")
                                   for name in observed},
        "observed_table_rounded": {name: format(value, ".5f") for name, value in observed.items()},
        "absolute_errors": absolute_error,
        "same_input_same_scores": torch.equal(duplicate_logits[0], duplicate_logits[1]),
        "boundary_targets": {name: int((y == 0).sum()) for name, (_, y) in batches.items()},
        "first_context": x[0].tolist(),
        "per_character_extra_parameters": 16 * 16,
    }

chart = ET.parse(ROOT / "course/figures/window_training.svg").getroot()
chart_checks = []
for point in chart.findall("{http://www.w3.org/2000/svg}circle"):
    if "data-window" not in point.attrib:
        continue
    length, metric = point.attrib["data-window"], point.attrib["data-metric"]
    value = float(point.attrib["data-value"])
    stored = record["results"]["runs"][f"mlp{length}"]["after_nll_same_post_update_time"][metric]
    expected_x = 66 + (int(length) - 1) * 59
    expected_y = 266 - (value - 0.15) / 0.45 * 146
    assert value == stored, "曲線圖數字與紀錄不一致"
    assert float(point.attrib["cx"]) == expected_x, "曲線圖視窗橫座標不一致"
    assert abs(float(point.attrib["cy"]) - expected_y) < 0.001, "曲線圖代價縱座標不一致"
    chart_checks.append({"window": length, "metric": metric, "value_matches": True,
                         "x_matches": True, "y_error": abs(float(point.attrib["cy"]) - expected_y)})
report["linked_chart"] = chart_checks
report["result"] = "全部斷言通過；完成字元與參數核對、短程 CPU 重算、平均分母及曲線圖座標核對。"
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
