"""保存教材依賴差異、沿用證據核對與本節重新執行的收據。"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.model import ModelConfig, TinyLM


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, check=False, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    receipt = {"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode != 0:
        raise RuntimeError(json.dumps(receipt, ensure_ascii=False))
    return receipt


original_path = Path("docs/course-review-20261019/reviews/rv1019_technical_16-1/technical-16-1-current-01.json")
original_bytes = original_path.read_bytes()
assert digest(original_bytes) == "a5673ab0dea02c7589749c16aa3d4611893b2c0edc0a291f4c1a7b27e5c9eda0"
original = json.loads(original_bytes)
assert original["issues"] == []

repo_checks = []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    raw = (ROOT / source["path"]).read_bytes()
    current = digest(raw)
    same = current == source["sha256"]
    assert same or source["path"] == "course/training.md"
    repo_checks.append({"id": source["id"], "path": source["path"], "original_sha256": source["sha256"], "current_sha256": current, "unchanged": same})

artifact_checks = []
for artifact in original["artifacts"]:
    current = digest((ROOT / artifact["path"]).read_bytes())
    assert current == artifact["sha256"]
    artifact_checks.append({"id": artifact["id"], "path": artifact["path"], "sha256": current, "unchanged": True})

old_cmd = command(["git", "show", "ecf6dec71280:course/training.md"])
new_cmd = command(["git", "show", "c6b9339d2595:course/training.md"])
old = old_cmd["stdout"].encode("utf-8")
new = new_cmd["stdout"].encode("utf-8")
assert digest(old) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
assert digest(new) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
assert new == (ROOT / "course/training.md").read_bytes()
diff = command(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])


def measurement_block(raw):
    title = "### 選讀：效率實驗的量測條件".encode("utf-8")
    start = raw.index(title)
    end = raw.index(b"\n</details>", start)
    return raw[start:end]


old_block = measurement_block(old)
new_block = measurement_block(new)
assert old_block == new_block
prior_evidence_path = OUT / "recheck-current-01.json"
prior_evidence_bytes = prior_evidence_path.read_bytes()
assert digest(prior_evidence_bytes) == "2bc579f71f4662af93978413788158288713c56980d047aeb6e1d6ee2aae628a"
assert new_block.decode("utf-8") == json.loads(prior_evidence_bytes)["dependency_relevant_block"]["text"]
block_text = new_block.decode("utf-8")
assert "../docs/course-experiments/results/efficiency.json" in block_text
assert "../docs/course-experiments/results/flash_probe.json" in block_text
assert "推論先暖機，再在計時兩側等待 GPU 完成" in block_text
assert "保存起點記憶體、過程峰值與新增量" in block_text

frozen_path = Path("docs/course-review-20261019/freeze-01/sources/16.1.md")
frozen = frozen_path.read_bytes()
raw_chapter = (ROOT / "course/chapters/16.md").read_bytes()
section = re.search(rb"(?ms)^## 16\.1 .*?(?=^## |\Z)", raw_chapter).group()
assert section == frozen
assert digest(section) == original["source_sha256"]
freeze_path = Path("docs/course-review-20261019/freeze-07.json")
freeze_sha = digest(freeze_path.read_bytes())
assert freeze_sha == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
freeze_page = next(page for page in json.loads(freeze_path.read_bytes())["pages"] if page["page_id"] == "16.1")
assert freeze_page["source_sha256"] == digest(section)
assert freeze_page["figures_sha256"] == {}

helper = command([".venv/bin/python", "docs/review-tools/section_facts.py", "course/chapters/16.md#16.1", "--execute"])
summary = json.loads(helper["stdout"])
assert summary["source_sha256"] == digest(section)
assert summary["execution_exit_code"] == 0
assert summary["svg_references"] == 0
run_dir = Path(summary["artifact_directory"])
assert run_dir.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
copy_files = {}
for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json", "fence-1.py"):
    raw = (run_dir / name).read_bytes()
    copy_files[name] = {"sha256": digest(raw), "text": raw.decode("utf-8")}
assert "分數形狀 (1, 3, 264)" in copy_files["stdout.txt"]["text"]
assert "參數bytes 24416" in copy_files["stdout.txt"]["text"]

snapshots = json.loads((OUT / "original-source-excerpts.json").read_bytes())["sources"]
profiler_snapshot = next(s for s in snapshots if s["id"] == "profiler_util")
import torch.autograd.profiler_util
installed_sha = digest(Path(torch.autograd.profiler_util.__file__).read_bytes())
assert installed_sha == profiler_snapshot["original_sha256"]
assert torch.version.git_version == "5c4886908584029761b579af026dcfb627c84070"

torch.set_num_threads(1)
profiles = []
for length in (3, 12):
    torch.manual_seed(42)
    model = TinyLM(ModelConfig(width=8)).eval()
    ids = torch.arange(1, length + 1)[None]
    with torch.no_grad():
        for _ in range(3):
            model(ids)
        with profile(activities=[ProfilerActivity.CPU]) as profiler:
            for _ in range(5):
                result = model(ids)
    rows = {e.key: {"calls": e.count, "self_cpu_us": e.self_cpu_time_total, "cpu_total_us": e.cpu_time_total, "cpu_avg_us": e.cpu_time} for e in profiler.key_averages()}
    assert result["logits"].shape == (1, length, 264)
    assert rows["aten::mm"]["calls"] == 25
    assert rows["aten::addmm"]["calls"] == 10
    assert rows["aten::index_select"]["calls"] == 10
    assert rows["aten::_softmax"]["calls"] == 5
    assert rows["aten::gelu"]["calls"] == 5
    assert rows["aten::repeat_interleave"]["calls"] == 10
    assert all(abs(e["cpu_avg_us"] * e["calls"] - e["cpu_total_us"]) < 1e-8 for e in rows.values())
    parameters = [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype), "numel": p.numel(), "element_size": p.element_size()} for n, p in model.named_parameters()]
    assert sum(p["numel"] for p in parameters) == 6104
    assert sum(p["numel"] * p["element_size"] for p in parameters) == 24416
    profiles.append({"length": length, "seed": 42, "warmup_calls": 3, "measured_forwards": 5, "logits_shape": list(result["logits"].shape), "parameters": parameters, "events": rows})

receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-1/recheck-current-02.py",
    "status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform(), "cuda_available": str(torch.cuda.is_available())},
    "previous_report": {"path": str(original_path), "sha256": digest(original_bytes)},
    "original_issue_count": len(original["issues"]),
    "source_sha256": digest(section), "source_matches_current_and_freeze": True,
    "source_freeze": {"path": str(freeze_path), "sha256": freeze_sha}, "figure_sha256": {},
    "repository_source_checks": repo_checks, "reused_artifact_checks": artifact_checks,
    "dependency_diff_receipt": diff,
    "dependency_revision_receipts": [{"command_argv": old_cmd["command_argv"], "exit_code": old_cmd["exit_code"], "file_sha256": digest(old)}, {"command_argv": new_cmd["command_argv"], "exit_code": new_cmd["exit_code"], "file_sha256": digest(new)}],
    "dependency_relevant_block": {"title": "選讀：效率實驗的量測條件", "original_sha256": digest(old_block), "current_sha256": digest(new_block), "unchanged": True, "text_reference": {"path": "docs/technical-reviews/artifacts/review-20261019-16-1/recheck-current-01.json", "sha256": "2bc579f71f4662af93978413788158288713c56980d047aeb6e1d6ee2aae628a", "locator": "dependency_relevant_block.text"}, "supports_claim_ids": ["c23"]},
    "changed_dependency_passages": [
        {"locator": "這三題只是抽查", "impact_on_lesson": "改正多模態練習的驗證範圍與固定實驗入口；本節沒有用該段支持任何主張。", "substantive_lesson_claims_changed": False},
        {"locator": "vision.pt、audio.pt、joint.pt 的文字能力檢查", "impact_on_lesson": "改寫多模態檢查點的文字退步對照方法；本節引用的效率量測方法區塊不在這一段。", "substantive_lesson_claims_changed": False},
    ],
    "helper_execution": helper, "helper_summary": summary, "copied_files": copy_files,
    "installed_profiler_source_sha256": installed_sha, "profiles": profiles,
    "claim_recheck": [{"claim_id": c["id"], "status": "verified", "basis": "凍結原文未變；沿用來源與artifact逐檔SHA核對相符。" + ("依賴的效率量測方法區塊另與舊版逐位元組比較一致。" if c["id"] == "c23" else "") } for c in original["claims"]],
    "limitations": ["沒有重跑GPU或長時間訓練；先前GPU紀錄的外推限制保留。", "training.md中兩段多模態練習說明的變動不屬於本節的實質主張，這裡只核對其對本節證據的影響。"],
}
with (OUT / "recheck-current-02.json").open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"status": "passed", "unchanged_lesson_claims": len(original["claims"]), "reused_artifacts_verified": len(artifact_checks), "measurement_block_unchanged": old_block == new_block, "helper_exit_code": summary["execution_exit_code"], "profiles": [{"length": p["length"], "shape": p["logits_shape"], "mm": p["events"]["aten::mm"]["calls"], "addmm": p["events"]["aten::addmm"]["calls"]} for p in profiles]}, ensure_ascii=False))
