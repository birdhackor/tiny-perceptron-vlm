"""Anonymous read-only primary source receipts, retaining short excerpts, never full third-party pages."""
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime, UTC
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).parent
SCRATCH = ROOT / "work/transport-readme/sources"
SCRATCH.mkdir(parents=True, exist_ok=True)
revision = "979cdfacc588ad0536f1c64fff96f264571cf054"
specs = [
    ("uv-sync", "https://docs.astral.sh/uv/concepts/projects/sync/"),
    ("hf-v2-tree", f"https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/{revision}/selftrained/v2?recursive=true"),
    ("hf-card", f"https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/{revision}/README.md"),
    ("ci-run", "https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/actions/runs/37071683289"),
    ("ci-jobs", "https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/actions/runs/37071683289/jobs"),
    ("site-readme", "https://birdhackor.github.io/tiny-perceptron-vlm/readme.html"),
    ("site-1-1", "https://birdhackor.github.io/tiny-perceptron-vlm/1.1.html"),
]
for group in ("moe-pretrain", "moe-sft", "moe-joint", "dense-joint"):
    specs.append((group, f"https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/{revision}/selftrained/v2/{group}/inference-manifest.json"))

def obtain(spec):
    name, url = spec
    entry = {"id": name, "url": url, "at": datetime.now(UTC).isoformat(), "authentication": "no Authorization header supplied"}
    try:
        with urlopen(Request(url, headers={"User-Agent": "tiny-perceptron-independent-readme-review"}), timeout=30) as response:
            raw = response.read(4 * 1024 * 1024)
            entry.update(status=response.status, final_url=response.url, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        (SCRATCH / (name + ".txt")).write_bytes(raw)
        text = raw.decode("utf-8")
        if name == "uv-sync":
            plain = re.sub(r"<[^>]+>", " ", text)
            plain = re.sub(r"\s+", " ", plain)
            terms = ("frozen", "Optional dependencies", "extra", "exact", "run", "sync")
            entry["excerpts"] = {term: [plain[max(0, m.start()-90):m.end()+260] for m in list(re.finditer(re.escape(term), plain, re.I))[:3]] for term in terms}
        elif name.startswith("site-"):
            entry["features"] = {
                "local_mathjax": bool(re.search(r'src=["\'][^"\']*javascripts/mathjax/tex-chtml.js', text)),
                "external_mathjax": bool(re.search(r'src=["\']https?://[^"\']*(?:mathjax|MathJax)', text)),
                "colab_links": re.findall(r'https://colab.research.google.com/[^"\'\s<>]+', text)[:2],
                "download_links": re.findall(r'href=["\']([^"\']*\.ipynb)["\']', text)[:2],
                "cpu_output_present": "5.770" in text or "輸出" in text or "output" in text,
                "search_present": "search" in text,
                "code_copy_present": "content.code.copy" in text or "md-clipboard" in text,
                "palette_present": "palette" in text,
            }
        elif name == "hf-card":
            entry["excerpts"] = [line for line in text.splitlines() if any(x in line.lower() for x in ("license:", "selftrained/v2", "mit", "inference", "optimizer"))][:22]
        else:
            value = json.loads(raw)
            if name == "ci-run":
                entry["fields"] = {k: value[k] for k in ("id", "head_sha", "status", "conclusion", "html_url")}
            elif name == "ci-jobs":
                entry["jobs"] = [{k: j[k] for k in ("id", "name", "status", "conclusion", "html_url")} for j in value["jobs"]]
            elif name == "hf-v2-tree":
                entry["files"] = value
            else:
                entry["manifest"] = {k: v for k, v in value.items() if k != "asset_sha256"}
                entry["manifest"]["asset_sha256_entry_count"] = len(value.get("asset_sha256", {}))
    except Exception as error:
        entry.update(error_type=type(error).__name__, error=str(error))
    return entry

with ThreadPoolExecutor(max_workers=4) as executor:
    receipts = list(executor.map(obtain, specs))
(OUT / "network-results.json").write_text(json.dumps({"reviewer_task": "/root/tech_transport_readme_20261016", "receipts": receipts}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps([{k: v for k, v in e.items() if k in ("id", "status", "error", "features")} for e in receipts], ensure_ascii=False))
