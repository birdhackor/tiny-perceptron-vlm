"""Bounded CPU verification for section 16.13; no model, training, or GPU kernel."""
import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import torch

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
source = HERE / "sources/transformers-v4.57.1-masking_utils.py"
source_text = source.read_text()
tree = ast.parse(source_text)
names = {
    "and_masks", "causal_mask_function", "sliding_window_overlay", "chunked_overlay",
    "_legacy_chunked_overlay", "sliding_window_causal_mask_function", "chunked_causal_mask_function",
}
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert {node.name for node in nodes} == names
namespace = {"Callable": Callable, "torch": torch, "_is_torch_greater_or_equal_than_2_6": True}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
inspection = {node.name: [node.lineno, node.end_lineno] for node in nodes}
(HERE / "official-mask-function-locators.json").write_text(json.dumps(inspection, indent=2) + "\n")
n, width = 6, 3
q_idx = torch.arange(n)[:, None]
k_idx = torch.arange(n)[None, :]
masks = {
    "full": namespace["causal_mask_function"](0, 0, q_idx, k_idx),
    "sliding": namespace["sliding_window_causal_mask_function"](width)(0, 0, q_idx, k_idx),
    "chunked": namespace["chunked_causal_mask_function"](width, torch.tensor([0]))(0, 0, q_idx, k_idx),
}
predicates = {
    "full": lambda i, j: j <= i,
    "sliding": lambda i, j: max(0, i - width + 1) <= j <= i,
    "chunked": lambda i, j: j <= i and i // width == j // width,
}
expected_counts = {"full": 21, "sliding": 15, "chunked": 12}
svg = ET.parse(HERE / "inputs/visible-pairs.svg").getroot()
rectangles = list(svg.iter("{http://www.w3.org/2000/svg}rect"))
matrix_results = {}
for panel, (name, mask) in enumerate(masks.items()):
    expected = torch.tensor([[predicates[name](i, j) for j in range(n)] for i in range(n)])
    assert torch.equal(mask, expected)
    assert int(mask.sum()) == expected_counts[name]
    assert not torch.any(torch.triu(mask, diagonal=1))
    figure_mask = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            cells = [r for r in rectangles if r.get("x") == str(69 + 38 * j)
                     and r.get("y") == str(148 + 360 * panel + 38 * i)
                     and r.get("width") == "34" and r.get("height") == "34"]
            assert len(cells) == 1
            assert cells[0].get("fill") in {"#2563eb", "#e2e8f0"}
            figure_mask[i, j] = cells[0].get("fill") == "#2563eb"
    assert torch.equal(figure_mask, mask)
    matrix_results[name] = {
        "row_counts": mask.sum(dim=1).tolist(), "visible_pairs": int(mask.sum()),
        "position_3": torch.where(mask[3])[0].tolist(),
        "position_4": torch.where(mask[4])[0].tolist(),
        "svg_all_36_cells_match": True,
    }
assert sum(range(1, 7)) == 21
assert 1 + 2 + 4 * 3 == 15
assert 2 * sum(range(1, 4)) == 12

# A complete allowed row is still summed when candidate tiles become smaller.
q = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., 1.], [1., 2.], [2., 2.]], dtype=torch.float64)
k = torch.tensor([[.1, 1.], [1., .2], [.3, .5], [.7, .8], [.9, .4], [.6, 1.]], dtype=torch.float64)
v = torch.tensor([[1., 2.], [3., 5.], [7., 11.], [13., 17.], [19., 23.], [29., 31.]], dtype=torch.float64)
scores = q @ k.T / math.sqrt(2)
full_scores = scores.masked_fill(~masks["full"], float("-inf"))
reference = full_scores.softmax(dim=-1) @ v
tile_results = {}
for tile_size in [1, 2, 3, 4, 6]:
    outputs, count = [], 0
    for row in range(n):
        maximum = torch.tensor(float("-inf"), dtype=torch.float64)
        denominator = torch.tensor(0., dtype=torch.float64)
        numerator = torch.zeros(2, dtype=torch.float64)
        for start in range(0, n, tile_size):
            indices = torch.arange(start, min(start + tile_size, n))
            indices = indices[masks["full"][row, indices]]
            if indices.numel() == 0:
                continue
            block = scores[row, indices]
            new_maximum = torch.maximum(maximum, block.max())
            rescale = (maximum - new_maximum).exp()
            weights = (block - new_maximum).exp()
            denominator = denominator * rescale + weights.sum()
            numerator = numerator * rescale + weights @ v[indices]
            maximum = new_maximum
            count += indices.numel()
        outputs.append(numerator / denominator)
    output = torch.stack(outputs)
    error = (output - reference).abs().max().item()
    assert error < 1e-12 and count == 21
    tile_results[str(tile_size)] = {"visible_pairs": count, "max_abs_error": error,
                                    "position_4": output[4].tolist()}

# Execute the actual referenced prior-section fence from the frozen chapter.
import importlib.util
helper = HERE / "inputs/section_facts.py"
spec = importlib.util.spec_from_file_location("section_facts_16_13", helper)
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
prior_bytes, _, prior_line = facts.original_section(HERE / "inputs/frozen-chapter16.md", "16.9")
prior_fences = [f for f in facts.fences(prior_bytes, prior_line) if f["language"] == "python"]
assert len(prior_fences) == 1
original_code = prior_fences[0]["raw"]
(HERE / "inputs/referenced-16.9-original-fence.py").write_bytes(original_code)
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(original_code, "referenced-16.9-original-fence.py", "exec"), {"__name__": "__main__"})
assert "分塊結果 24.2857" in stdout.getvalue() and "完整結果 24.2857" in stdout.getvalue()
(HERE / "referenced-fence-stdout.txt").write_text(stdout.getvalue())

result = {
    "environment": {"python": sys.version, "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version), "device": "cpu",
                    "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
                    "platform": platform.platform(), "torch_threads": str(torch.get_num_threads())},
    "official_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "official_ast_functions": inspection,
    "official_execution_scope": "Only exact AST-selected mask factories, with torch>=2.6 flag true and zero left padding. Transformers package is not installed; no backend integration claimed.",
    "masks": matrix_results, "full_causal_tiling": tile_results,
    "denominators": {"positions": 6, "possible_grid_cells_per_rule": 36,
                     "sliding_width_including_current": 3, "chunk_size": 3, "chunks": 2},
    "referenced_16_9_original_fence": {"sha256": hashlib.sha256(original_code).hexdigest(),
                                      "stdout": stdout.getvalue()},
    "scope": "Rule counts, official pure mask functions, SVG cell consistency and mathematical tiled forward calculation. No GPU timing, kernel test, gradients, parameter updates or model score.",
}
(HERE / "cpu-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
