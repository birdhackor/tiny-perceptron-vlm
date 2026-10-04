"""Independent 19.12 original-source retrieval; no model training or Git writes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
COMMIT = "1df335318bda03fd771807f66976953231d5a00b"
files = ["docs/course-experiments/results/" + n + ".json" for n in
         ["capstone_deployment", "capstone_student", "capstone_pretrain", "capstone_sft", "capstone_joint", "capstone_preference"]]
files += ["docs/course-experiments/capstone-selection.json", "tiny_perceptron/capstone.py",
          "scripts/course_experiments/capstone_student.py", "tiny_perceptron/capstone_quantization.py"]
files += [str(p.relative_to(ROOT)) for sub in ["deployment", "student"]
          for p in (ROOT / "docs/course-experiments/capstone-evidence" / sub).glob("*.json")
          if p.name != "review-manifest.json"]

def retrieve(pair):
    name, url = pair
    try:
        with urllib.request.urlopen(url, timeout=35) as r:
            raw = r.read()
            status = r.status
        record = {"name": name, "url": url, "http_status": status,
                  "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if name in files:
            local = (ROOT / name).read_bytes()
            record["local_sha256"] = hashlib.sha256(local).hexdigest()
            record["matches_local_bytes"] = raw == local
        else:
            suffix = ".pdf" if name.startswith("paper-") else ".txt"
            temporary = Path("/tmp") / ("natural-19.12-" + name + suffix)
            temporary.write_bytes(raw)
            if suffix == ".pdf":
                text_file = temporary.with_suffix(".txt")
                subprocess.run(["pdftotext", "-layout", str(temporary), str(text_file)], check=True)
                record["temporary_text"] = str(text_file)
            else:
                retained = OUT / ("natural-19.12-" + name + ".txt")
                retained.write_bytes(raw)
                record["snapshot"] = str(retained.relative_to(ROOT))
        return record
    except Exception as e:
        return {"name": name, "url": url, "error": str(e)}

urls = [(p, "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/" + COMMIT + "/" + p)
        for p in files]
urls += [("paper-kd", "https://arxiv.org/pdf/1503.02531v1"),
         ("paper-moe", "https://arxiv.org/pdf/1701.06538v1"),
         ("paper-instruct", "https://arxiv.org/pdf/2203.02155v1"),
         ("paper-dpo", "https://arxiv.org/pdf/2305.18290v1"),
         ("qwen-config", "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/config.json"),
         ("qwen-readme", "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/README.md"),
         ("whisper-config", "https://huggingface.co/openai/whisper-small/raw/973afd24965f72e36ca33b3055d56a652f456b4d/config.json")]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    records = list(pool.map(retrieve, urls))
output = {"accessed_on": "2026-10-04", "github_commit": COMMIT, "records": records}
(OUT / "natural-19.12-original-source-receipt.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"retrieved": sum("error" not in r for r in records), "total": len(records),
                  "mismatches": [r["name"] for r in records if r.get("matches_local_bytes") is False],
                  "errors": [r for r in records if "error" in r]}, indent=2))
