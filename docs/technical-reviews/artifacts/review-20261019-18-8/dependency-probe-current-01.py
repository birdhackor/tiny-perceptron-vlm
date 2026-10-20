"""核對來源變動、既有證據與相關程式行為。"""

import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import IGNORE
from tiny_perceptron.posttraining import ppo_clipped_objective


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


original_path = "docs/course-review-20261019/reviews/rv1019_technical_18-8/technical-18-8-original-01.json"
original_bytes = (ROOT / original_path).read_bytes()
assert sha(original_bytes) == "a341b80c476e2ed4e184ffae30d80ebce6e05e5993e9f13b0353890243243232"
original = json.loads(original_bytes)
source_audit = []
for item in original["sources"]:
    if item["kind"] == "repository_code":
        actual = sha((ROOT / item["path"]).read_bytes())
        source_audit.append({"id": item["id"], "path": item["path"], "old_sha256": item["sha256"], "current_sha256": actual, "unchanged": actual == item["sha256"]})
assert [x["path"] for x in source_audit if not x["unchanged"]] == ["course/chapters/13.md"]
artifact_audit = []
for item in original["artifacts"]:
    actual = sha((ROOT / item["path"]).read_bytes())
    assert actual == item["sha256"]
    artifact_audit.append({"id": item["id"], "path": item["path"], "sha256": actual, "unchanged": True})
old = git("show", "5ef56bc756f3:course/chapters/13.md")
current = (ROOT / "course/chapters/13.md").read_bytes()
committed = git("show", "3af7f13ef959:course/chapters/13.md")
assert sha(old) == "6a17de3cd906373d429c795e39c1c62fdb61a2e00effa3316f9d088906625131"
assert sha(current) == "e1e11b7d73dcf9ca366e196ccebccbedbbc50d651ddd7e80665eefa8755076f3"
assert committed == current
pattern = rb"(?ms)^## 13\.14 .*?(?=^## |\Z)"
old_section = re.search(pattern, old).group()
current_section = re.search(pattern, current).group()
assert old_section == current_section
section = (ROOT / "docs/course-review-20261019/freeze-01/sources/18.8.md").read_bytes()
current_lesson = re.search(rb"(?ms)^## 18\.8 .*?(?=^## |\Z)", (ROOT / "course/chapters/18.md").read_bytes()).group()
assert section == current_lesson
freeze = (ROOT / "docs/course-review-20261019/freeze-05.json").read_bytes()
assert sha(freeze) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
page = next(p for p in json.loads(freeze)["pages"] if p["page_id"] == "18.8")
assert page["source_sha256"] == sha(section) and page["figures_sha256"] == {}

logits = torch.tensor([[[0., 2., 0.], [2., 0., 0.], [0., 0., 2.]]])
labels = torch.tensor([[IGNORE, 0, 2]])
valid = labels != IGNORE
safe = labels.masked_fill(~valid, 0)
selected = logits.log_softmax(-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
rounded = [round(value, 4) for value in selected[0].tolist()]
assert rounded == [-2.2395, -0.2395, -0.2395]
ratio = torch.tensor([.7, 1., 1.3, .7, 1., 1.3], requires_grad=True)
advantage = torch.tensor([1., 1., 1., -1., -1., -1.])
old_log = torch.full((6,), .5).log()
result = ppo_clipped_objective(old_log + ratio.log(), old_log, advantage, clip_range=.2)
expected_clipped = ratio.clamp(.8, 1.2) * advantage
expected_surrogate = torch.minimum(ratio * advantage, expected_clipped)
assert torch.allclose(result["clipped"], expected_clipped)
assert torch.allclose(result["surrogate"], expected_surrogate)
(-result["surrogate"].sum()).backward()
assert torch.allclose(ratio.grad, torch.tensor([-1., -1., 0., 0., 1., 1.]))

receipt_path = "docs/technical-reviews/artifacts/review-20261019-18-8/primary-source-receipts.json"
receipts = json.loads((ROOT / receipt_path).read_bytes())
torch_matches = []
for source in receipts["sources"]:
    if source["id"].startswith("primary_torch_"):
        tail = source["url"].split("/v2.14.1/torch/", 1)[1]
        actual = sha((Path(torch.__file__).parent / tail).read_bytes())
        assert actual == source["response_sha256"]
        torch_matches.append({"id": source["id"], "installed_sha256": actual, "matches_prior_official_response": True})

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-8/dependency-probe-current-01.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
    "revision": git("rev-parse", "HEAD").decode().strip(),
    "original_report": {"path": original_path, "sha256": sha(original_bytes)},
    "source_sha256": sha(section),
    "current_section_matches_freeze": True,
    "source_freeze_sha256": sha(freeze),
    "source_audit": source_audit,
    "artifact_audit": artifact_audit,
    "original_issues": original["issues"],
    "policy_reference_section": {"unchanged": True, "sha256": sha(current_section), "title": "獎勵模型、價值模型與兩種基準各做什麼？"},
    "dependency_diff_command": "git diff 5ef56bc756f3 3af7f13ef959 -- course/chapters/13.md",
    "dependency_diff": git("diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/13.md").decode(),
    "dependency_change_execution": {
        "rounding": {"input": selected[0].tolist(), "output": rounded, "interpretation": "取到小數四位後收成新清單；本例文案與程式相符。"},
        "clipping": {"ratio": ratio.detach().tolist(), "advantage": advantage.tolist(), "clipped": result["clipped"].tolist(), "surrogate": result["surrogate"].tolist(), "ratio_gradient": ratio.grad.tolist(), "source": inspect.getsource(ppo_clipped_objective), "interpretation": "先裁切 ratio、再乘 advantage，最後與未裁切乘積取較小值。"}
    },
    "official_torch_sources_still_match_installed": torch_matches,
    "scope": "只核對本節主張所依據的來源變動；沒有把兩處旁支文案的核對當成整章審閱。"
}
print(json.dumps(output, ensure_ascii=False, indent=2))
