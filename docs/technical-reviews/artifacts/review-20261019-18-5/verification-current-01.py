"""複核高溫近似與平方補償的界線，並確認可沿用的證據版本。"""

import hashlib
import io
import json
import math
import platform
import re
import subprocess
import tarfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-18-5"
REPORT = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_18-5/technical-18-5-original-01.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def gradient(teacher, student, temperature):
    teacher = torch.tensor(teacher, dtype=torch.float64)
    student = torch.tensor(student, dtype=torch.float64, requires_grad=True)
    p = (teacher / temperature).softmax(0)
    q = (student / temperature).softmax(0)
    loss = -(p * (student / temperature).log_softmax(0)).sum()
    loss.backward()
    assert torch.allclose(student.grad, (q.detach() - p) / temperature, atol=1e-14, rtol=1e-11)
    return {
        "temperature": temperature,
        "teacher_probability": p.tolist(),
        "student_probability": q.detach().tolist(),
        "probability_difference_norm": (q.detach() - p).norm().item(),
        "gradient_norm": student.grad.norm().item(),
        "scaled_gradient_norm": student.grad.norm().item() * temperature**2,
    }


assert sha(REPORT.read_bytes()) == "bc5ee8d13f842d779b7cc0f1e85a5a8bed329092aac9a5f9f0590e457a5c1730"
original = json.loads(REPORT.read_bytes())
previous_probe = json.loads((BASE / "verification.json").read_bytes())
result = {
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "dtype": "torch.float64",
    },
    "preserved_files": [],
}
entries = [{"path": str(REPORT.relative_to(ROOT)), "sha256": sha(REPORT.read_bytes())}]
entries += original["artifacts"]
entries += [s for s in original["sources"] if s["kind"] == "repository_code"]
entries += [{"path": path, "sha256": digest} for path, digest in previous_probe["source_files"].items()]
seen = set()
for entry in entries:
    path = entry["path"]
    if path in seen:
        continue
    seen.add(path)
    actual = sha((ROOT / path).read_bytes())
    assert actual == entry["sha256"]
    result["preserved_files"].append({"path": path, "sha256": actual, "unchanged": True})

snapshot = ROOT / "docs/course-review-20261019/freeze-05/sources/18.5.md"
raw = snapshot.read_bytes()
assert sha(raw) == "3563f4c44de406a2bb60fd1414937f436c36f811200b1fff66c796483af673c0"
chapter = (ROOT / "course/chapters/18.md").read_bytes()
start = re.search(rb"(?m)^## 18\.5 ", chapter).start()
following = re.search(rb"(?m)^## ", chapter[start + 1:])
end = start + 1 + following.start() if following else len(chapter)
assert chapter[start:end] == raw
old = (ROOT / "docs/course-review-20261019/freeze-01/sources/18.5.md").read_bytes()
pattern = rb"```python\n(.*?)\n```"
assert re.findall(pattern, old, re.S) == re.findall(pattern, raw, re.S)
result["section"] = {
    "source_sha256": sha(raw),
    "live_matches_snapshot": True,
    "python_fences_unchanged": True,
    "fence_count": len(re.findall(pattern, raw, re.S)),
}
diff_command = ["git", "diff", "--no-index", "docs/course-review-20261019/freeze-01/sources/18.5.md", "docs/course-review-20261019/freeze-05/sources/18.5.md"]
diff = subprocess.run(diff_command, cwd=ROOT, capture_output=True, text=True)
assert diff.returncode == 1
result["diff"] = {"command": diff_command, "exit_code": diff.returncode, "stdout": diff.stdout, "stderr": diff.stderr}

teacher = [10, 0, 0]
student = [20, 0, 0]
rows = [gradient(teacher, student, t) for t in (1, 4, 100, 1000, 10000)]
centered_difference = [((student[i] - sum(student) / 3) - (teacher[i] - sum(teacher) / 3)) / 3 for i in range(3)]
limit = math.sqrt(sum(x*x for x in centered_difference))
result["temperature_scaling"] = {
    "teacher_logits": teacher,
    "student_logits": student,
    "teacher_logit_range": max(teacher) - min(teacher),
    "student_logit_range": max(student) - min(student),
    "high_temperature_limit_vector": centered_difference,
    "high_temperature_scaled_norm_limit": limit,
    "rows": rows,
}
errors = []
for row in rows[2:]:
    error = abs(row["scaled_gradient_norm"] - limit) / limit
    errors.append(error)
    row["relative_error_from_limit"] = error
assert errors[0] > errors[1] > errors[2]
assert errors[2] < 0.001
assert rows[1]["gradient_norm"] > rows[0]["gradient_norm"]
assert rows[1]["probability_difference_norm"] > rows[0]["probability_difference_norm"]
result["temperature_scaling"]["gradient_ratio_t4_over_t1"] = rows[1]["gradient_norm"] / rows[0]["gradient_norm"]
result["temperature_scaling"]["scaled_gradient_ratio_t4_over_t1"] = rows[1]["scaled_gradient_norm"] / rows[0]["scaled_gradient_norm"]
shifted = gradient([x+1000 for x in teacher], [x-5000 for x in student], 1000)
assert abs(shifted["scaled_gradient_norm"] - rows[3]["scaled_gradient_norm"]) < 1e-9
result["temperature_scaling"]["shifted_logit_check"] = shifted
result["lesson_example"] = [gradient([4, 1, 0], [0, 0, 0], t) for t in (1, 2, 4)]

command = ["curl", "-fsSL", "--max-time", "45", "https://arxiv.org/src/1503.02531v1"]
response = subprocess.run(command, check=True, capture_output=True).stdout
with tarfile.open(fileobj=io.BytesIO(response)) as archive:
    tex = archive.extractfile("distillation11.tex").read().decode()
excerpt = "If the temperature is high compared with the magnitude of the logits, we can approximate:"
assert excerpt in tex
result["original_source_receipt"] = {
    "command": command,
    "sha256": sha(response),
    "bytes": len(response),
    "version": "arXiv:1503.02531v1，2015-03-09",
    "accessed_on": "2026-10-20",
    "locator": "Distillation；Matching logits is a special case of distillation；公式 (2)–(4)",
    "excerpt": excerpt,
    "inspection_note": "重新讀取梯度精確式、高溫展開條件、中心化與平方補償的原文；全文只留在記憶體，僅保存本段短摘錄。",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
