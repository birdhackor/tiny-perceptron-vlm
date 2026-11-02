"""Save small source, derivation and actual SVG-view receipts; originals stay outside Git."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

from bs4 import BeautifulSoup
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CACHE = Path("/workspace/work/reference/omni-technical-14-19-v1")
PIN = "5c4886908584029761b579af026dcfb627c84070"
COMMAND = "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-14-19-v1/capture_evidence.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open("x") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


sources = []
for title, filename, url, selector, excerpts, remote_sha in [
    ("Qwen2.5-Omni Technical Report", "/workspace/work/reference/qwen25-omni-report-v1.html",
     "https://arxiv.org/html/2503.20215v1", "S2.SS2.SSS0.Px2",
     ["This is achieved by deconstructing the original rotary embedding into three components: temporal, height, and width.",
      "When processing images, the temporal IDs of each visual token remain constant, while distinct IDs are assigned to the height and width components based on the token’s position in the image."],
     "c1ec6bdc527b59733556cdfb438dbf346ff48be408203dd5b571d90900a41ec4"),
    ("Qwen3-Omni Technical Report", "/workspace/work/reference/qwen3-omni-report-v1.html",
     "https://arxiv.org/html/2509.17765v1", "S2.SS3.SSS0.Px2",
     ["TM-RoPE factorizes the conventional rotary position embedding into three distinct dimensions: temporal, height, and width.",
      "Specifically, the temporal, height, and width dimensions are interleaved and assigned 24, 20, and 20 rotary angles, respectively."],
     "a4102e4a626248880d2a99b3fd6a9a7febdf153384f50d43ac1be641ac7a18f1"),
]:
    path = Path(filename)
    section = BeautifulSoup(path.read_text(), "html.parser").find(id=selector).get_text(" ", strip=True)
    for excerpt in excerpts:
        assert excerpt in section
    assert sha(path) == remote_sha
    item = {"title": title, "url": url, "version": "arXiv v1", "accessed_on": "2026-10-29",
            "cache_path": str(path), "sha256": sha(path), "locator": selector,
            "short_excerpts": excerpts,
            "remote_integrity_command": f"curl --fail --location --silent --show-error {url} | sha256sum",
            "remote_sha256_observed_in_this_review": remote_sha}
    if "Qwen3" in title:
        match = re.search(r"interleaved and assigned (\d+), (\d+), and (\d+) rotary angles", section)
        assert match
        counts = list(map(int, match.groups()))
        assert counts == [24, 20, 20]
        item["parsed_temporal_height_width_rotary_angles"] = counts
    sources.append(item)

roformer_pdf = CACHE / "roformer-v5.pdf"
roformer_txt = CACHE / "roformer-v5.txt"
assert "arXiv:2104.09864v5" in roformer_txt.read_text()
sources.append({
    "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
    "url": "https://arxiv.org/pdf/2104.09864v5", "version": "arXiv v5, 8 Nov 2023",
    "accessed_on": "2026-10-29", "cache_path": str(roformer_pdf), "sha256": sha(roformer_pdf),
    "text_cache_path": str(roformer_txt), "text_sha256": sha(roformer_txt),
    "locator": "§3.1 Eq.11; §3.2.1 Eq.12-13 (p.4); §3.2.2 Eq.14-16 (p.5)",
    "inspection_note": "Read the rotation matrix, d/2 independent 2-D subspaces and R(m)^T R(n)=R(n-m); these support the pairwise mechanism, not a universal distance-quality claim.",
    "retrieval_command": "curl --fail --location --silent --show-error https://arxiv.org/pdf/2104.09864v5 --output /workspace/work/reference/omni-technical-14-19-v1/roformer-v5.pdf && pdftotext -layout /workspace/work/reference/omni-technical-14-19-v1/roformer-v5.pdf /workspace/work/reference/omni-technical-14-19-v1/roformer-v5.txt",
})
torch_root = Path(torch.__file__).resolve().parent
for relative, cache, locator, excerpt in [
    ("_torch_docs.py", "torch-docs-pinned.py", "L1919 stack; L3297 cos; L4242 equal; L7909 matmul; L9680 arange; L9898 round; L10336 sin",
     "``True`` if two tensors have the same size and elements, ``False`` otherwise."),
    ("_tensor_docs.py", "tensor-docs-pinned.py", "L2010 flatten; L4164 reshape; L4269 round",
     "Returns a tensor with the same data and number of elements as :attr:`self`"),
    ("nn/functional.py", "functional-pinned.py", "L6370-6404 SDPA reference algorithm; L6490-6498 boolean mask and square causal mask",
     "temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)"),
]:
    path = CACHE / cache
    installed = torch_root / relative
    assert excerpt in path.read_text()
    assert sha(installed) == sha(path), (relative, "installed source differs from official pin")
    sources.append({"title": "PyTorch official " + relative,
                    "url": f"https://github.com/pytorch/pytorch/blob/{PIN}/torch/{relative}",
                    "version": f"PyTorch {torch.__version__}; git {PIN}", "accessed_on": "2026-10-29",
                    "cache_path": str(path), "sha256": sha(path),
                    "installed_path": str(installed), "installed_sha256": sha(installed),
                    "installed_matches_official_pin": True, "locator": locator, "short_excerpt": excerpt})

inspection = {"created_at": datetime.now(UTC).isoformat(), "command": COMMAND,
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
              "sources": sources,
              "result": "Versioned original paper passages read; Qwen3 allocation parsed as [24,20,20]; installed PyTorch docs/source exactly match official git-pin files.",
              "scope": "Original passages establish method/configuration; CPU execution establishes only the toy implementation's behavior."}
save("source-inspection.json", inspection)

png = CACHE / "axis-time-index.png"
svg = ROOT / "course/figures/omni-14.19-axis-time-index.svg"
figure = {"created_at": datetime.now(UTC).isoformat(), "source": str(svg.relative_to(ROOT)),
          "source_sha256": sha(svg), "render_path_outside_git": str(png), "render_sha256": sha(png),
          "render_size_px": list(Image.open(png).size),
          "render_command": "inkscape course/figures/omni-14.19-axis-time-index.svg --export-type=png --export-filename=/workspace/work/reference/omni-technical-14-19-v1/axis-time-index.png --export-dpi=160",
          "renderer": subprocess.check_output(["inkscape", "--version"], text=True).strip(),
          "view_tool": "functions.exec -> tools.view_image; returned image forwarded with image() and visually inspected before this receipt",
          "viewed": True,
          "inspection": ["F5 is 1250 ms / 250 = t5 and serialized index6.",
                         "A1 time shifts 1300 to2300 ms, t5.2 to9.2, with index20 in both cases.",
                         "Three equal-width colored boxes depict feature pairs 0/1, 2/3, 4/5 mapped respectively to t,h,w; they are a schematic partition, not a numeric scale.",
                         "Only the first pair's angle changes in the stated time-shift comparison.",
                         "The bottom box correctly states index20 can read indices0-20; no physical-time causal ordering is drawn.",
                         "All labels, signs and numbers are visible, with no clipped text; the figure has no plotted arrows or metric axes to mis-scale."],
          "result": "Rendered and actually viewed; all numeric labels, pair positions and causal range agree with the frozen lesson and execution."}
save("figure-render.json", figure)
print(json.dumps({"source_inspection": inspection["result"], "figure": figure["result"],
                  "figure_render_sha256": sha(png)}, ensure_ascii=False, indent=2))
