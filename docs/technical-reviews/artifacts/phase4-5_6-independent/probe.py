"""Bounded CPU checks of exactly the substantive claims in current lesson 5.6."""
import ast
import copy
import hashlib
import json
import math
import platform
import runpy
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.training import learning_rate

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")


def close(observed, expected, tolerance=1e-18):
    assert abs(observed - expected) <= tolerance, (observed, expected, tolerance)


def independent_cosine(step, total, warmup):
    """SGDR Eq.5 with eta_min=1e-4, eta_max=1e-3; phase is exact rational."""
    phase = Fraction(step - warmup, total - warmup)
    return 0.0001 + (0.001 - 0.0001) / 2 * (1 + math.cos(math.pi * float(phase)))


results = {}
fence = (OUT / "original-run/fence-1.py").read_bytes()
parameter = torch.nn.Parameter(torch.tensor([1.0]))
optimizer = torch.optim.SGD([parameter], lr=0.123)
before_parameter = parameter.detach().clone()
before_optimizer = copy.deepcopy(optimizer.state_dict())
namespace = {"__name__": "__main__", "parameter": parameter, "optimizer": optimizer}
exec(compile(fence, "original-fence-1.py", "exec"), namespace)
assert torch.equal(parameter, before_parameter)
assert parameter.grad is None
assert optimizer.state_dict() == before_optimizer
rates = namespace["rates"]
assert len(rates) == 40
for i in range(5):
    close(rates[i], float(Fraction(i + 1, 5000)))
for i in range(5, 40):
    close(rates[i], independent_cosine(i, 40, 5))
assert rates[4] == rates[5] == max(rates) == 0.001
assert all(a >= b for a, b in zip(rates[5:], rates[6:]))
assert min(rates[5:]) > 0.0001
close(learning_rate(40, 40, warmup=5), 0.0001)
close(learning_rate(100, 40, warmup=5), 0.0001)
results["original"] = {
    "count": len(rates), "first_six": rates[:6], "last_three": rates[-3:], "maximum": max(rates),
    "first_cosine_index": 5, "final_training_index": 39,
    "last_phase": "34/35", "rate_at_index_40": learning_rate(40, 40, warmup=5),
    "no_parameter_gradient_or_update": True, "optimizer_state_unchanged": True,
    "original_fence_sha256": hashlib.sha256(fence).hexdigest(),
    "independent_formula": "eta_min +(eta_max-eta_min)/2*(1+cos(pi*(i-w)/(T-w))) for i >=w",
}
ten = [learning_rate(i, 40, warmup=10) for i in range(40)]
thirty = [learning_rate(i, 40, warmup=30) for i in range(40)]
for i in range(10):
    close(ten[i], float(Fraction(i + 1, 10000)))
for i in range(10, 40):
    close(ten[i], independent_cosine(i, 40, 10))
assert ten == thirty
results["warmup_variation"] = {"warmup_10_first_ten": ten[:10], "warmup_10_last_three": ten[-3:],
                               "warmup_30_exactly_equals_10": ten == thirty, "rates_checked": 80}
short = {}
for total, cap in [(1, 1), (2, 1), (3, 1), (4, 1), (7, 1), (8, 2), (40, 10)]:
    observed = [learning_rate(i, total, warmup=30) for i in range(total)]
    close(observed[0], 0.001 / cap)
    for i in range(cap):
        if i < total:
            close(observed[i], 0.001 * (i + 1) / cap)
    if cap < total:
        close(observed[cap], 0.001)
    short[str(total)] = {"expected_effective_warmup": cap, "count": len(observed),
                         "first": observed[0], "last": observed[-1]}
results["warmup_cap_boundaries"] = short
same_index = {str(total): learning_rate(19, total, warmup=5) for total in [40, 100]}
assert same_index["40"] != same_index["100"]
results["same_20th_update_with_different_plan"] = {"zero_based_index": 19, "rates": same_index}

# Read and execute the actual CLI metadata contract and its exact resume comparison,
# without invoking main(), building a model, downloading data, or doing any training.
train_ns = runpy.run_path(str(ROOT / "scripts/train.py"), run_name="fresh_5_6_contract_inspection")
args = train_ns["parser"]().parse_args(["--steps", "40", "--lr", "0.001"])
meta = train_ns["training_metadata"](args)
assert meta["schedule_steps"] == 40 and meta["peak_lr"] == 0.001
train_raw = (ROOT / "scripts/train.py").read_bytes()
tree = ast.parse(train_raw)
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
contract = next(n for n in main.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == "args.resume"
                and any(isinstance(child, ast.For) and ast.unparse(child.iter) == "current_metadata.items()"
                        for child in n.body))
module = ast.Module(body=[contract], type_ignores=[])
ast.fix_missing_locations(module)
contract_code = compile(module, "scripts/train.py:original-resume-metadata-check", "exec")
args.resume = True
exec(contract_code, {"args": args, "current_metadata": meta, "metadata": meta})
args.steps = 100
changed = train_ns["training_metadata"](args)
try:
    exec(contract_code, {"args": args, "current_metadata": changed, "metadata": meta})
except ValueError as error:
    rejected = str(error)
else:
    raise AssertionError("Original CLI must reject changed total plan during --resume")
results["resume_plan_contract"] = {"saved_metadata": meta, "unchanged_plan_accepted": True,
                                   "changed_40_to_100_rejected": rejected,
                                   "original_ast_lines": [contract.lineno, contract.end_lineno],
                                   "scope": "Exact original metadata comparison executed; no training/resume process run"}

svg = (OUT / "inputs/course/figures/rewrite-05-06-learning-rate.svg").read_bytes()
svg_tree = ET.fromstring(svg)
ns = {"svg": "http://www.w3.org/2000/svg"}
line = svg_tree.find("svg:polyline", ns)
points = [tuple(float(part) for part in point.split(",")) for point in line.attrib["points"].split()]
assert len(points) == 40
x_errors = [abs(x - (184 + i * (600 - 184) / 39)) for i, (x, _) in enumerate(points)]
y_errors = [abs(y - (424 - rate / 0.001 * 300)) for (_, y), rate in zip(points, rates)]
assert max(x_errors) <= 0.0051 and max(y_errors) <= 0.0051
texts = ["".join(node.itertext()) for node in svg_tree.findall("svg:text", ns)]
assert {"0.001", "0.0005", "0.0001", "1", "20", "40", "更新次序",
        "1–5步：逐步暖身", "6–40步：平滑向下限靠近", "圖中只計時間表，沒有更新模型。"} <= set(texts)
results["figure_numeric_consistency"] = {"polyline_points": len(points), "x_axis": "one-based update numbers 1..40",
                                        "y_mapping": "y=424-300*lr/0.001", "max_x_error_px": max(x_errors),
                                        "max_y_error_px": max(y_errors), "tolerance_px": 0.0051,
                                        "ticks_lr_y": {"0.001": 124, "0.0005": 274, "0.0001": 394},
                                        "visual_inspection": "Rendered with Inkscape and viewed separately by reviewer"}
env = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
       "torch_git_version": torch.version.git_version, "torch_cuda_build": str(torch.version.cuda),
       "cuda_available": str(torch.cuda.is_available()), "device": "CPU", "torch_threads": str(torch.get_num_threads()),
       "platform": platform.platform(), "cwd": str(Path.cwd()),
       "input_sha256": {"tiny_perceptron/training.py": hashlib.sha256((ROOT / "tiny_perceptron/training.py").read_bytes()).hexdigest(),
                        "scripts/train.py": hashlib.sha256(train_raw).hexdigest()}}
(OUT / "probe-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
(OUT / "probe-environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2))
