"""保存原始來源收據，並核對圖文、空間對照與可訓練的圖片通路。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.selftrained.train import set_trainable
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def fetch(url, quotes, html=False):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "40", url]
    completed = subprocess.run(command, check=True, capture_output=True)
    raw = completed.stdout
    text = raw.decode("utf-8")
    if html:
        parser = TextOnly()
        parser.feed(text)
        text = " ".join(parser.parts)
    text = " ".join(text.split())
    for quote in quotes:
        assert quote in text, f"來源摘錄不存在：{url}"
    return {
        "url": url,
        "command": command,
        "returncode": completed.returncode,
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "original_bytes": len(raw),
        "original_sha256": sha(raw),
        "excerpts": quotes,
        "full_text_saved": False,
    }, text


def main():
    revision = "b2617bb6d3ffa2e429640350f613e3291e10b141"
    base = f"https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/{revision}/"
    readme, text = fetch(base + "README.md", ["Each example is a 28x28 grayscale image", "| 1 | Trouser |", "| 8 | Bag |", "| 9 | Ankle boot |"])
    size = re.search(r"Each example is a (\d+)x(\d+) grayscale image", text)
    dimensions = list(map(int, size.groups()))
    assert dimensions == [28, 28], "來源影像尺寸不符"
    license_receipt, _ = fetch(base + "LICENSE", ["The MIT License (MIT)", "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software."])
    paper, _ = fetch("https://arxiv.org/html/2304.08485v2", ["we apply a trainable projection matrix", "The model is trained to predict the assistant answers and where to stop"], html=True)
    groups, _ = fetch("https://scikit-learn.org/1.7/modules/cross_validation.html", ["groups that are not represented at all in the paired training fold"], html=True)
    leakage, _ = fetch("https://scikit-learn.org/1.7/common_pitfalls.html", ["information that would not be available at prediction time is used when building the model"], html=True)

    paths = [
        "docs/course-review-20261019/freeze-01/sources/11.17.md",
        "docs/course-review-20261019/freeze-04.json",
        "docs/course-review-20261019/figure-renders-04.json",
        "course/figures/new-11.17-left-right.svg",
        "course/figures/new-11.17-single-object.svg",
        "tiny_perceptron/selftrained/model.py",
        "tiny_perceptron/selftrained/dataset.py",
        "scripts/selftrained/train.py",
        "scripts/selftrained/prepare_vision_ocr.py",
        "docs/selftrained/v2-manifest.json",
        "docs/selftrained/results/v2-final-public-results.json",
    ]
    hashes = {path: sha((ROOT / path).read_bytes()) for path in paths}
    frozen = (ROOT / paths[0]).read_text()
    assert hashes[paths[0]] == "32b2d4d4ce624624cb1527556642f4fb2d75c73457742dc47fe4657177598f93", "凍結原文不符"
    assert hashes[paths[1]] == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412", "凍結清單不符"
    assert not re.search(r"^```python", frozen, re.M), "原文含 Python 區塊"
    receipt = json.loads((ROOT / paths[2]).read_text())
    render_records = [item for item in receipt["records"] if item["page_id"] == "11.17"]
    figure_checks = []
    for record in render_records:
        figure = record["figure"]
        tree = ElementTree.fromstring((ROOT / figure).read_bytes())
        assert hashes[figure] == record["frozen_svg"]["sha256"], "SVG 版本不符"
        assert sha((ROOT / record["png"]["path"]).read_bytes()) == record["png"]["sha256"], "PNG 版本不符"
        image_nodes = tree.findall(".//{http://www.w3.org/2000/svg}image")
        assert not image_nodes, "示意圖含外部影像節點"
        figure_checks.append({"figure": figure, "embedded_image_nodes": len(image_nodes), "png_receipt_matches": True})

    def answer(positions, subject, reference):
        return "左邊" if positions[subject] < positions[reference] else "右邊"

    original = {"包": 170, "短靴": 470}
    swapped = {"包": 470, "短靴": 170}
    answers = [answer(original, "包", "短靴"), answer(swapped, "包", "短靴"), answer(original, "短靴", "包")]
    assert answers == ["左邊", "右邊", "右邊"], "關係對照不符"
    feature_sequence = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    same_mean = torch.equal(feature_sequence.mean(0), feature_sequence.flip(0).mean(0))
    assert same_mean, "交換後的平均摘要不同"

    torch.manual_seed(2718)
    torch.set_num_threads(1)
    config = SelftrainedConfig(vocab_size=64, width=16, layers=1, heads=2, kv_heads=1, ffn_hidden=32, experts=2, top_k=1, max_length=64, backend="manual")
    model = LimitedAssistant(config)
    optimizer = torch.optim.AdamW(set_trainable(model, "joint"), lr=0.001, weight_decay=0.0)
    payload = {"kind": "image", "values": torch.rand(2, 1, 32, 32), "coordinates": torch.tensor([[0.25, 0.5], [0.75, 0.5]])}
    input_ids = torch.tensor([[1, 3, 7, 7, 11, 4, 12, 2]])
    labels = torch.tensor([[-100, -100, -100, -100, -100, 12, 2, -100]])
    names = ["vision_encoder.cnn.0.weight", "vision_encoder.projection.weight", "vision_encoder.symbol_projection.weight", "lm.embedding.weight"]
    parameters = dict(model.named_parameters())
    before = {name: parameters[name].detach().clone() for name in names}
    result = model(input_ids, labels=labels, modalities=[[payload]])
    result["loss"].backward()
    gradients = {name: float(parameters[name].grad.abs().sum()) for name in names}
    assert all(value > 0 for value in gradients.values()), "圖片或文字通路沒有梯度"
    optimizer.step()
    changes = {name: float((parameters[name].detach() - before[name]).abs().max()) for name in names}
    assert all(value > 0 for value in changes.values()), "圖片或文字通路參數未更新"
    with torch.no_grad():
        inference = model(input_ids, modalities=[[payload]])
    assert inference["loss"] is None, "未提供答案仍計算監督損失"
    manifest = json.loads((ROOT / "docs/selftrained/v2-manifest.json").read_text())
    release = json.loads((ROOT / "docs/selftrained/results/v2-final-public-results.json").read_text())
    assert manifest["initialization"] == "random", "資料配套的初始化聲明不符"
    moe = release["architectures"]["moe"]
    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-11-17/verify.py",
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__), "device": "cpu", "dtype": "float32", "platform": platform.platform(), "seed": "2718", "torch_threads": "1"},
        "sources": {"fashion_readme": readme, "fashion_license": license_receipt, "llava": paper, "grouped_validation": groups, "data_leakage": leakage},
        "source_dimensions": dimensions,
        "selected_categories": {"1": "Trouser", "8": "Bag", "9": "Ankle boot"},
        "file_sha256": hashes,
        "figures": figure_checks,
        "relation_control": {"original_positions": original, "swapped_positions": swapped, "answers": answers, "constant_left_correct": sum(x == "左邊" for x in answers), "cases": len(answers), "swapped_one_hot_mean_equal": same_mean},
        "trainable_path": {"config": vars(config), "input_ids": input_ids.tolist(), "labels": labels.tolist(), "image_shape": list(payload["values"].shape), "coordinates": payload["coordinates"].tolist(), "logits_shape": list(result["logits"].shape), "supervised_tokens": int(result["supervised_tokens"]), "loss": float(result["loss"].detach()), "gradient_l1": gradients, "parameter_max_changes": changes, "inference_without_labels": inference["loss"] is None, "scope": "隨機小模型的一次 CPU 更新，只核對數值特徵通路與參數可訓練性；沒有驗收看圖品質。"},
        "released_model_record": {"initialization": moe["capacity"]["initialization"], "selected_source": moe["selected_source"], "training_revision": moe["training_revision"], "safe_weights": moe["safe_weights"], "manifest_initialization": manifest["initialization"], "scope": "讀取現有成品出處紀錄，未下載權重或重跑訓練與評測。"},
        "all_checks_passed": True,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
