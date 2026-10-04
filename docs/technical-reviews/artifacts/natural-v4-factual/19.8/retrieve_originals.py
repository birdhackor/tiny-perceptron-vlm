"""Fetch only original documents and official frozen run records for this review."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime, hashlib, json, shutil, urllib.request

ROOT = Path(__file__).resolve().parents[5]
RAW = ROOT / "outputs/natural-v4/factual-research/19.8"
ART = Path(__file__).resolve().parent
COMMIT = "1df335318bda03fd771807f66976953231d5a00b"
URLS = {
    "ppo.pdf": "https://arxiv.org/pdf/1707.06347v2",
    "r1.pdf": "https://arxiv.org/pdf/2501.12948v1",
    "mixtral.pdf": "https://arxiv.org/pdf/2401.04088v1",
    "switch.pdf": "https://jmlr.org/papers/volume23/21-0998/21-0998.pdf",
    "torch-module.py": "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/module.py",
}
PROJECT = [
    "docs/course-experiments/results/capstone_joint.json",
    "docs/course-experiments/results/capstone_preference.json",
    "docs/course-experiments/results/capstone_deployment.json",
    "docs/course-experiments/results/capstone_student.json",
    "docs/course-experiments/results/posttraining.json",
    "docs/course-experiments/capstone-evidence/joint/validation.json",
    "docs/course-experiments/capstone-evidence/dpo/validation.json",
    "docs/course-experiments/capstone-evidence/dpo/train-report.json",
    "docs/course-experiments/capstone-evidence/deployment/data.json",
    "docs/course-experiments/capstone-evidence/deployment/test-joint-ptq4.json",
    "docs/course-experiments/capstone-evidence/deployment/test-joint-ptq8.json",
    "scripts/course_experiments/capstone.py",
    "scripts/course_experiments/capstone_student.py",
    "scripts/course_experiments/capstone_deployment.py",
]
for path in PROJECT:
    URLS["official/" + path] = f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{COMMIT}/{path}"

def fetch(item):
    name, url = item
    target = RAW / name
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Independent factual review; original documents only"})
    with urllib.request.urlopen(request, timeout=50) as response:
        data = response.read()
        final_url = response.url
    target.write_bytes(data)
    record = {"path": str(target.relative_to(ROOT)), "url": url, "final_url": final_url,
              "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
              "retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if name.startswith("official/"):
        local = ROOT / name.removeprefix("official/")
        record["current_repository_same_bytes"] = local.exists() and local.read_bytes() == data
    return record

records = []
for version, name in [("2203.02155v1", "instructgpt.pdf"), ("2305.18290v3", "dpo.pdf")]:
    cached = ROOT / f"outputs/natural-v4/factual-original-cache/{version}.pdf"
    target = RAW / name
    shutil.copyfile(cached, target)
    data = target.read_bytes()
    records.append({"path": str(target.relative_to(ROOT)), "url": f"https://arxiv.org/pdf/{version}",
                    "version": version, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "retrieval_method": "Original PDF bytes from permitted originals-only cache; no donor analysis read"})
with ThreadPoolExecutor(max_workers=5) as pool:
    records.extend(pool.map(fetch, URLS.items()))
(ART / "retrieval-receipt.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2))
