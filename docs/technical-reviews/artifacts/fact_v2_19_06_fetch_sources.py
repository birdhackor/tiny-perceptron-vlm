"""Save original sources and exact reviewed sections for the independent 19.6 audit."""

import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.check_technical_reviews import sections  # noqa: E402

OUT = Path(__file__).parent
PREFIX = "fact_v2_19_06_"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(name, data):
    path = OUT / (PREFIX + name)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


reads = []
for chapter, wanted in [(19, ["19.6", "19.1", "19.4"]), (10, ["10.1", "10.7"]), (12, ["12.5", "12.12"])]:
    for sid, body in sections(ROOT / f"course/chapters/{chapter:02}.md"):
        if sid in wanted:
            path = OUT / (PREFIX + "read_" + sid + ".md")
            path.write_text(body)
            reads.append({"section": sid, "path": str(path.relative_to(ROOT)), "sha256": sha(path)})
save_json("section_reads.json", reads)

git = torch.version.git_version
requests = {
    "torch_functional.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{git}/torch/functional.py",
    "torch_nn_functional.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{git}/torch/nn/functional.py",
    "torch_grad_mode.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{git}/torch/autograd/grad_mode.py",
    "whisper_audio.py": "https://raw.githubusercontent.com/openai/whisper/fc5ded7d9045c693692f13853857c3f8baea3a7b/whisper/audio.py",
    "vqa_original.pdf": "https://arxiv.org/pdf/1612.00837v2",
}


def fetch(item):
    name, url = item
    destination = OUT / (PREFIX + name)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
            result = {"status": response.status, "url": url, "final_url": response.url}
        destination.write_bytes(content)
        result.update(path=str(destination.relative_to(ROOT)), bytes=len(content), sha256=sha(destination))
    except Exception as error:
        result = {"url": url, "error": f"{type(error).__name__}: {error}"}
    return name, result


with ThreadPoolExecutor(max_workers=5) as pool:
    fetched = dict(pool.map(fetch, requests.items()))
pdf = OUT / (PREFIX + "vqa_original.pdf")
if pdf.exists():
    result = subprocess.run(["pdftotext", "-layout", str(pdf), str(pdf.with_suffix(".txt"))], capture_output=True, text=True)
    fetched["pdf_extraction"] = {"returncode": result.returncode, "stderr": result.stderr}
save_json("source_fetch.json", {"date": "2026-10-04", "torch_version": torch.__version__, "git": git, "fetched": fetched})

manifest = json.loads((ROOT / "outputs/reading-time/figures/manifest.json").read_text())
figures = []
for name in ["multimodal_expand_image", "multimodal_audio_axes", "capstone_pipeline"]:
    relative = f"course/figures/{name}.svg"
    row = next(row for row in manifest["records"] if row["source"] == relative)
    assert sha(ROOT / relative) == row["source_sha256"]
    assert sha(ROOT / row["render"]) == row["render_sha256"]
    destination = OUT / (PREFIX + name + ".png")
    shutil.copyfile(ROOT / row["render"], destination)
    figures.append({**row, "own_render": str(destination.relative_to(ROOT)), "own_sha256": sha(destination)})
save_json("figure_provenance.json", figures)
print(json.dumps({"reads": reads, "fetched": fetched, "figures": figures}, ensure_ascii=False, indent=2))
