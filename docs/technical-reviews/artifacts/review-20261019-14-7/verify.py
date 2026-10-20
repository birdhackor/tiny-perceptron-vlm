"""獨立核對文字卡、因果配對、位置表與旋轉算式。"""

import hashlib
import inspect
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from tiny_perceptron.modern import rope


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


text = "校慶週六上午開始請在操場入口集合"
assert len(text) == 16
records = {}
for length in (8, 12, 16):
    pairs = [(query, key) for query in range(length) for key in range(query + 1)]
    counts = Counter(query - key for query, key in pairs)
    observed = [counts[d] for d in range(16)]
    expected = [max(length - d, 0) for d in range(16)]
    assert observed == expected
    assert len(pairs) == length * (length + 1) // 2
    records[str(length)] = {
        "maximum_distance": max(counts),
        "pairs_by_distance_0_through_15": observed,
        "total_causal_pairs": len(pairs),
        "longest_distance_pairs": [list(pair) for pair in pairs if pair[0] - pair[1] == length - 1],
    }

lookup = {}
for capacity in (8, 12):
    table = nn.Embedding(capacity, 4)
    positions = torch.arange(16)
    errors = []
    for position in range(16):
        try:
            actual = table(torch.tensor(position))
            assert actual.shape == (4,)
        except IndexError as error:
            errors.append({"position": position, "type": type(error).__name__, "message": str(error)})
    assert [item["position"] for item in errors] == list(range(capacity, 16))
    lookup[str(capacity)] = {
        "weight_shape": list(table.weight.shape),
        "num_embeddings": table.num_embeddings,
        "comparison": (positions < table.num_embeddings).tolist(),
        "errors": errors,
    }

positions = torch.arange(16)
x = torch.ones(1, 1, 16, 4)
rotated = rope(x, positions)
expected = torch.empty_like(rotated)
for position in range(16):
    for pair, frequency in enumerate((1.0, 0.01)):
        angle = torch.tensor(position * frequency, dtype=torch.float32)
        expected[..., position, 2 * pair] = angle.cos() - angle.sin()
        expected[..., position, 2 * pair + 1] = angle.sin() + angle.cos()
rotation_error = (rotated - expected).abs().max().item()
assert rotated.shape == (1, 1, 16, 4)
assert torch.isfinite(rotated).all().item()
assert rotation_error < 2e-6

query = torch.tensor([1.0, 2.0, -3.0, 0.5])
key = torch.tensor([-0.5, 4.0, 2.0, -1.0])
dot_product_errors = []
for query_position, key_position in ((7, 0), (15, 0), (15, 8)):
    rq = rope(query.reshape(1, 1, 1, 4), torch.tensor([query_position]))
    rk = rope(key.reshape(1, 1, 1, 4), torch.tensor([key_position]))
    relative = rope(key.reshape(1, 1, 1, 4), torch.tensor([key_position - query_position]))
    left = (rq * rk).sum().item()
    right = (query * relative).sum().item()
    error = abs(left - right)
    assert error < 1e-5
    dot_product_errors.append({"query_position": query_position, "key_position": key_position, "dot": left, "relative_dot": right, "absolute_error": error})

figure = ROOT / "course/figures/rewrite-14-7-position-range.svg"
tree = ElementTree.fromstring(figure.read_bytes())
labels = list(tree.iter("{http://www.w3.org/2000/svg}text"))
characters = [node.text for node in labels if node.attrib.get("font-size") == "36"]
indices = [int(node.text) for node in labels if node.text and node.text.isdecimal()]
assert "".join(characters) == text
assert indices == list(range(16))
blue = [node.text for node in labels if node.text and node.text.isdecimal() and node.attrib.get("fill") == "#246a9e"]
orange = [node.text for node in labels if node.text and node.text.isdecimal() and node.attrib.get("fill") == "#b85119"]
assert blue == [str(i) for i in range(8)]
assert orange == [str(i) for i in range(8, 16)]
receipt_file = ROOT / "docs/course-review-20261019/figure-renders-04.json"
receipt = next(item for item in json.loads(receipt_file.read_bytes())["records"] if item["figure"] == figure.relative_to(ROOT).as_posix())
assert digest(figure) == receipt["frozen_svg"]["sha256"]
assert digest(ROOT / receipt["png"]["path"]) == receipt["png"]["sha256"]

embedding_file = Path(inspect.getfile(nn.Embedding))
embedding_source = inspect.getsource(nn.Embedding)
assert "self.num_embeddings = num_embeddings" in embedding_source
assert "(num_embeddings, embedding_dim)" in embedding_source
assert "F.embedding(" in inspect.getsource(nn.Embedding.forward)

record = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-7/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform()},
    "sentence": {"text": text, "count": len(text)},
    "causal_pairs": records,
    "embedding": lookup,
    "rope": {"shape": list(rotated.shape), "all_finite": True, "reference_formula_max_absolute_error": rotation_error, "formula_tolerance": 2e-6, "relative_dot_products": dot_product_errors, "relative_dot_tolerance": 1e-5},
    "figure": {"svg_path": figure.relative_to(ROOT).as_posix(), "svg_sha256": digest(figure), "characters": characters, "indices": indices, "blue_indices": blue, "orange_indices": orange, "render_receipt": receipt, "receipt_sha256": digest(receipt_file), "render_hash_verified": True},
    "implementation": {"path": "tiny_perceptron/modern.py", "sha256": digest(ROOT / "tiny_perceptron/modern.py")},
    "installed_embedding": {"path": str(embedding_file), "sha256": digest(embedding_file), "inspected_identifiers": ["Embedding.__init__", "Embedding.forward"], "excerpt": "self.num_embeddings = num_embeddings"},
    "result": "所有確定性核對通過；沒有執行訓練或任務品質評測。",
}
print(json.dumps(record, ensure_ascii=False, indent=2))
