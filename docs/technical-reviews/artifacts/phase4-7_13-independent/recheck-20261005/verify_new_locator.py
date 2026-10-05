"""Recheck the edited original lesson and original raw baseline report.

This run verifies the new locator only; earlier CPU fence/variation results are
reused after byte comparisons. No training, model, data collection, or upload.
"""
import concurrent.futures
import hashlib
import json
import platform
import re
import sys
import urllib.request
from pathlib import Path

R = Path(__file__).resolve().parent
A = R.parent
ROOT = A.parents[3]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


raw = (ROOT / "course/chapters/07.md").read_bytes()
headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
index = next(i for i, h in enumerate(headings) if h[0].startswith(b"## 7.13 "))
section = raw[headings[index].start():headings[index + 1].start()]
section.decode("utf-8")
(R / "new-section.md").write_bytes(section)
old_section = (A / "section.md").read_bytes()
old_wording = "也就是[7.12](#7.12)那條從零練900次的模型"
new_wording = "也就是[完整實測報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/sft.json)中從零練900次的模型"
assert old_section.replace(old_wording.encode(), new_wording.encode()) == section
assert old_wording.encode() not in section
assert sha(section) == "5ec078d9359734e99936f3b9ee64314622d53a89acec493a4198009256ef3784"

fence = re.search(rb"```python\r?\n(.*?)```", section, re.S)[1]
assert fence == (A / "fence-1.py").read_bytes()
figures = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", section.decode())
assert not figures

prior_report = json.loads((ROOT / "docs/technical-reviews/7.13.json").read_bytes())
assert prior_report["verdict"] == "revise"
prior_report_bytes = (ROOT / "docs/technical-reviews/7.13.json").read_bytes()
assert sha(prior_report_bytes) == "7695cbcb1e14f77ad3e887ad22dede3bfdf0c05cc4e2162638d48a0df1460e86"
(R / "own-initial-revise-report.json").write_bytes(prior_report_bytes)
retained_artifacts = []
for artifact in prior_report["artifacts"]:
    path = ROOT / artifact["path"]
    digest = sha(path.read_bytes())
    assert digest == artifact["sha256"]
    retained_artifacts.append({"id": artifact["id"], "path": artifact["path"], "sha256": digest})

unchanged_current_code = []
for name in ["scripts/course_experiments/common.py", "scripts/course_experiments/text.py",
             "scripts/course_experiments/compression.py", "scripts/prepare_data.py",
             "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/training.py"]:
    current = (ROOT / name).read_bytes()
    previous = (A / "inputs/current" / name).read_bytes()
    assert current == previous
    unchanged_current_code.append({"path": name, "sha256": sha(current)})

local_report = (ROOT / "docs/course-experiments/results/sft.json").read_bytes()
assert local_report == (A / "inputs/results/sft.json").read_bytes()
(R / "current-original-sft.json").write_bytes(local_report)

blob_url = "https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/sft.json"
raw_url = "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/main/docs/course-experiments/results/sft.json"


def fetch(item):
    name, url = item
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "tiny-perceptron-independent-factual-review"})
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read(2 * 1024 * 1024)
            (R / name).write_bytes(body)
            return {"file": name, "url": url, "final_url": response.url, "status": response.status,
                    "bytes": len(body), "sha256": sha(body), "content_type": response.headers.get("Content-Type")}
    except Exception as error:
        return {"file": name, "url": url, "error": f"{type(error).__name__}: {error}"}


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    network = list(pool.map(fetch, [("github-blob.html", blob_url), ("github-original-sft.json", raw_url)]))

result = json.loads(local_report)
payload = result["results"]
assert result["experiment_id"] == "sft" and result["seed"] == 42
assert result["revision"] == "a253d1262bf5f361f9ac4e19232ae752f0ecc7a3"
assert payload["checkpoint"] == payload["training"]["checkpoint"] == "model.pt"
assert payload["training"]["steps"] == 900
test = payload["after"]["test"]
assert test["records"] == len(test["samples"]) == 10
recomputed = []
for sample in test["samples"]:
    ids = sample["generated_ids"]
    content_ids = ids[:ids.index(2)] if 2 in ids else ids
    expected_ids = [b + 8 for b in sample["expected"].encode("utf-8")]
    exact = content_ids == expected_ids
    assert exact is sample["exact"]
    assert (2 in ids) is sample["eos"]
    recomputed.append(exact)
assert sum(recomputed) == test["matches"] == 5
assert sum(recomputed) / len(recomputed) == test["exact_match"] == 0.5
model_artifact = next(x for x in result["artifacts"] if x["path"] == "model.pt")
assert model_artifact["sha256"] == "b4184ed88631596b40b9d8c9b2db79f80e11404da308d3d760ed0505fae9801f"

remote = next(n for n in network if n["file"] == "github-original-sft.json")
if "error" not in remote:
    remote_bytes = (R / remote["file"]).read_bytes()
    remote_payload = json.loads(remote_bytes)
    assert remote_payload["results"]["training"]["checkpoint"] == "model.pt"
    assert remote_payload["results"]["training"]["steps"] == 900
    assert remote_payload["results"]["after"]["test"]["matches"] == 5
    assert remote_payload["results"]["after"]["test"]["records"] == 10
    assert remote_payload["revision"] == result["revision"]
    remote["identical_to_local_original_bytes"] = remote_bytes == local_report
    remote["baseline_result_identical"] = remote_payload["results"]["after"]["test"] == test

receipt = {
    "kind": "edited_locator_recheck", "accessed_on": "2026-10-05",
    "environment": {"python": platform.python_version(), "executable": sys.executable,
                    "device": "cpu; standard-library-only locator recheck", "cwd": str(ROOT)},
    "source_sha256": sha(section), "previous_source_sha256": sha(old_section),
    "exact_single_substitution_confirmed": True, "new_wording": new_wording,
    "original_fence_sha256": sha(fence), "figure_references": figures,
    "previous_report_sha256": sha(prior_report_bytes),
    "old_C10_status_preserved": next(c for c in prior_report["claims"] if c["id"] == "C10")["status"],
    "retained_artifacts": retained_artifacts, "unchanged_current_code": unchanged_current_code,
    "new_target": {"blob_url": blob_url, "raw_url": raw_url, "local_report_sha256": sha(local_report),
                   "experiment_revision": result["revision"], "checkpoint": "sft/model.pt", "checkpoint_sha256": model_artifact["sha256"],
                   "updates": 900, "seed": 42, "test_matches": sum(recomputed), "test_denominator": len(recomputed)},
    "network_inspection": network,
    "reused_cpu_evidence": "Original fence and bounded variations/denominator replay were not rerun; exact fence, relevant implementation, input report and all artifact hashes were checked unchanged.",
    "scope": "New original lesson and actual raw JSON locator inspection only. No training, weights, new dataset/model, GPU, paid action, or upload.",
}
(R / "verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k not in ["retained_artifacts", "unchanged_current_code"]}, ensure_ascii=False, indent=2))
