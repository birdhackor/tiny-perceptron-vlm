"""Original-owner current-source recheck; no repeat of GPU experiments."""
from pathlib import Path
import hashlib
import inspect
import json
import platform
import re
import subprocess
import torch
from tiny_perceptron.modern import RMSNorm

base = Path("docs/technical-reviews/artifacts/natural-v4-factual/19.10")
out = base / "round2"
old = json.loads((base / "first-own-report.json").read_text())
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def section(path, ident):
    text = Path(path).read_text()
    start = re.search(r"^## " + re.escape(ident) + r"(?:\s|$)", text, re.M)
    following = re.search(r"^## ", text[start.end():], re.M)
    return text[start.start():start.end() + following.start() if following else len(text)]

current = section("course/chapters/19.md", "19.10")
assert current.encode() == (out / "section.raw.md").read_bytes()
assert sha(out / "section.raw.md") == "2e76a2277bfddb6f4cf03e26f852ee9242ac1a970655912294a0f70644c50006"
assert sha("course/chapters/19.md") == "10e73ce8ded3f69d70721ce0a83106eee4e396d10d0f4f0954f3492f247d3a97"
original = (base / "section.raw.md").read_text()
old_sentence = "字嵌入把輸入編號查成向量，正規化則調整一組特徵的數值中心與尺度。"
new_sentence = "字嵌入把輸入編號查成向量。本成品使用[14.2的RMSNorm](14.md#14.2)：用均方根調整一組特徵的數值大小，不先減掉平均值。"
assert original.replace(old_sentence, new_sentence) == current
assert re.findall(r"```python\n(.*?)\n```", original, re.S) == re.findall(r"```python\n(.*?)\n```", current, re.S)

reuse = json.loads((out / "reuse-byte-comparison.json").read_text())
assert not reuse["errors"]
for path, receipt in reuse["checks"].items():
    if "current_sha256" in receipt and "#" not in path:
        assert sha(path) == receipt["current_sha256"]
assert sha(base / "first-own-report.json") == "3be2fa0b001b27c7fa50de7b6afd8feb82c5e2038362a5c9e3c07a22f5c0b1f9"
assert sha(base / "section.raw.md") == "1682819575e0609a0873d5d9e61fffd22bf6dd15a78bbdf288789bf19b384ca6"
for ident, receipt in old["prerequisite_sections"].items():
    path = receipt["source"].split("#")[0]
    assert hashlib.sha256(section(path, ident).encode()).hexdigest() == receipt["sha256"]

torch.set_num_threads(2)
assert torch.__version__ == "2.14.1+cpu" and not torch.cuda.is_available()
prerequisite = section("course/chapters/14.md", "14.2")
assert prerequisite.encode() == (out / "prerequisite-14.2.md").read_bytes()
block = re.search(r"```python\n(.*?)\n```", prerequisite, re.S).group(1)
namespace = {}
exec(compile(block, "course/chapters/14.md#14.2", "exec"), namespace)
x, ln, rms = [namespace[name] for name in ("x", "ln", "rms")]
assert torch.allclose(ln[0], torch.zeros(3), atol=1e-7, rtol=0)
assert torch.allclose(ln[1], torch.tensor([-1.2247, 0, 1.2247]), atol=5e-5, rtol=0)
assert torch.allclose(rms[0], torch.ones(3), atol=2e-6, rtol=0)
assert torch.allclose(rms[1], torch.tensor([0.4629, 0.9258, 1.3887]), atol=5e-5, rtol=0)
assert torch.allclose(RMSNorm(3)(x), rms, atol=1e-7, rtol=1e-7)
shifted_ln = torch.nn.functional.layer_norm(x + 10, (3,), eps=1e-5)
shifted_rms = (x + 10) / torch.sqrt((x + 10).square().mean(-1, keepdim=True) + 1e-5)
assert torch.allclose(shifted_ln, ln, atol=1e-6, rtol=0)
assert not torch.allclose(shifted_rms[1], rms[1], atol=1e-4, rtol=1e-4)
fresh = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu"},
         "current_section_sha256": sha(out / "section.raw.md"), "prior_section_sha256": sha(base / "section.raw.md"),
         "whole_chapter_sha256": sha("course/chapters/19.md"), "only_edit": "Incorrect center/scale explanation replaced by explicit RMSNorm RMS rescaling without mean subtraction, with a valid14.2 link.",
         "new_prerequisite": {"source": "course/chapters/14.md#14.2", "sha256": sha(out / "prerequisite-14.2.md")},
         "new_actual_CPU_probe": {"x": x.tolist(), "LN": ln.tolist(), "RMS": rms.tolist(),
                                  "project_RMSNorm_equals_prerequisite_formula": True,
                                  "LN_after_plus10": shifted_ln.tolist(), "RMS_after_plus10": shifted_rms.tolist(),
                                  "scope": "Fresh small prerequisite mechanism probe; not a new run of the unchanged lesson experiment, GPU training, historical scores or benchmark."},
         "reuse_scope": reuse["execution_reuse_scope"], "byte_checks": len(reuse["checks"]), "all_byte_comparisons_equal": True,
         "original_authority_reinspection": "Personally re-read original1910.07467v1 §4 Eq.(4) and following paragraph, original installedTorch RMSNorm formula/forward, actual capstone norm='rms' and modern.RMSNorm implementation; all original authority bytes unchanged."}
out.joinpath("recheck-results.json").write_text(json.dumps(fresh, ensure_ascii=False, indent=2) + "\n")

figure = Path("course/figures/normalization.svg")
destination = out / "normalization.png"
command = ["inkscape", str(figure), "--export-type=png", "--export-filename=" + str(destination)]
run = subprocess.run(command, capture_output=True, text=True)
run.check_returncode()
receipt = {"source": str(figure), "source_sha256": sha(figure), "render": str(destination), "render_sha256": sha(destination),
           "command": command, "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
out.joinpath("normalization-render-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(fresh, ensure_ascii=False, indent=2))
print(json.dumps(receipt, ensure_ascii=False))
