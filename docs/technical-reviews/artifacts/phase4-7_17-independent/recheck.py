"""Bounded revised-link and original-result check; no models or training."""
import hashlib
import json
import platform
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


whole = (ROOT / "course/chapters/07.md").read_bytes()
headings = list(re.finditer(rb"(?m)^## [^\r\n]+", whole))
i = next(i for i, h in enumerate(headings) if h[0].startswith(b"## 7.17 "))
body = whole[headings[i].start():headings[i + 1].start() if i + 1 < len(headings) else len(whole)]
assert sha(body) == "a3dce865d853a7a12bf3a99b7bda55e0175ab48b9ee7948c90ffbcebdf0ee6e0"
old = (HERE / "original/section.md").read_bytes().decode()
substitutions = {
    "食谱": "食譜", "介绍": "介紹", "制作": "製作", "并": "並",
    "继續": "繼續", "评測": "評測", "礼貌": "禮貌", "成绩": "成績",
    "[7.12實測](#7.12)": "[小世界實測報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/sft.json)",
}
counts = {}
for before, after in substitutions.items():
    counts[before] = old.count(before)
    old = old.replace(before, after)
assert old.encode() == body
(HERE / "revised-section.md").write_bytes(body)
fences = re.findall(rb"```python\n(.*?)```", body, re.S)
assert len(fences) == 1 and fences[0] == (HERE / "original/fence-1.py").read_bytes()
svg_path = ROOT / "course/figures/rewrite-07-17-prepost-routes.svg"
assert svg_path.read_bytes() == (HERE / "original/figure.svg").read_bytes()
initial = json.loads((HERE / "initial-report.json").read_bytes())
reused = {}
for artifact in initial["artifacts"]:
    path = ROOT / artifact["path"]
    actual = sha(path.read_bytes())
    assert actual == artifact["sha256"], artifact["id"]
    reused[artifact["id"]] = actual

url = "https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/sft.json"
assert f"[小世界實測報告]({url})" in body.decode()
raw_url = "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/main/docs/course-experiments/results/sft.json"
local = (ROOT / "docs/course-experiments/results/sft.json").read_bytes()
assert local == (HERE / "original/sft-result.json").read_bytes()
with urllib.request.urlopen(raw_url, timeout=20) as response:
    remote = response.read(2 * 1024 * 1024)
    remote_status = response.status
assert remote_status == 200 and remote == local
data = json.loads(local)
assert data["revision"] == "a253d1262bf5f361f9ac4e19232ae752f0ecc7a3"
d = data["results"]
branches = [
    ("results.pretrain_then_sft.before_sft.test", d["pretrain_then_sft"]["before_sft"]["test"], 1),
    ("results.pretrain_then_sft.after_sft.test", d["pretrain_then_sft"]["after_sft"]["test"], 7),
    ("results.after.test", d["after"]["test"], 5),
]
first_questions = None
checks = {}
for locator, report, expected in branches:
    questions = [(s["messages"], s["expected"]) for s in report["samples"]]
    if first_questions is None:
        first_questions = questions
    assert questions == first_questions
    matched = ended = 0
    for sample in report["samples"]:
        ids = sample["generated_ids"]
        eos = 2 in ids
        content = ids[:ids.index(2)] if eos else ids
        exact = content == [b + 8 for b in sample["expected"].encode("utf-8")]
        assert exact == sample["exact"] and eos == sample["eos"]
        matched += int(exact)
        ended += int(eos)
    denominator = len(report["samples"])
    assert matched == expected == report["matches"]
    assert denominator == 10 == report["records"]
    assert matched / denominator == report["exact_match"]
    assert ended == 10 and report["eos_rate"] == 1.0
    assert sum(len(s["expected"].encode()) + 1 for s in report["samples"]) == report["effective_tokens"] == 69
    checks[locator] = {"matches": matched, "question_denominator": denominator,
                      "eos_count": ended, "answer_byte_plus_eos_targets": 69}
assert d["data"]["train"]["records"] == 45
assert d["training"]["steps"] == d["pretrain_then_sft"]["sft"]["steps"] == 900
assert d["pretrain_then_sft"]["pretraining"]["steps"] == 250

print(json.dumps({
    "revised_section_sha256": sha(body),
    "initial_section_sha256": initial["source_sha256"],
    "exact_allowed_text_substitution_counts": counts,
    "direct_link": url, "original_bytes_url": raw_url,
    "original_https_status": remote_status, "original_json_sha256": sha(local),
    "original_json_revision": data["revision"],
    "remote_local_retained_original_json_identical": True,
    "recomputed_branches": checks, "train_questions": 45,
    "additional_text_updates": 250,
    "unchanged_fence_sha256": sha(fences[0]),
    "unchanged_figure_sha256": sha(svg_path.read_bytes()),
    "reused_initial_artifact_sha256": reused,
    "environment": {"python": platform.python_version(), "device": "cpu; Python only",
                    "torch": "not imported", "TLS": "default urllib verified HTTPS"},
    "scope": "Full revised section personally reread; bounded new-link/remote original JSON/raw-token recount executed. Initial cost CPU run and SVG render/view reused after hash verification, not rerun.",
    "tolerance": "Exact integer counts, byte identity and token-list equality.",
}, ensure_ascii=False, indent=2))
