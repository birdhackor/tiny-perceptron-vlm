"""核對複查依據的指紋、章檔差異與既有主張的適用範圍。"""
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def section(raw, identifier):
    match = re.search(rb"(?ms)^## " + re.escape(identifier.encode()) + rb" .*?(?=^## |\Z)", raw)
    assert match is not None
    return match.group()


report_path = "docs/course-review-20261019/reviews/rv1019_technical_19-10/technical-19-10-original-02.json"
original_raw = (ROOT / report_path).read_bytes()
assert sha(original_raw) == "f280b5ba1a76878ef33d29904ce85c3e93c7cfbc7cc3d63eb8137b760f7f6e98"
original = json.loads(original_raw)
changed = {"course/chapters/17.md", "course/chapters/18.md", "course/chapters/19.md"}
record = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-10/recheck-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "original_report": {"path": report_path, "sha256": sha(original_raw)},
    "previous_issues": original["issues"],
    "repository_source_checks": [],
    "reused_artifact_checks": [],
    "source_diffs": [],
    "current_model_arithmetic": {},
}
assert original["issues"] == []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    path = source["path"]
    raw = (ROOT / path).read_bytes()
    current = sha(raw)
    equal = current == source["sha256"]
    assert equal == (path not in changed)
    record["repository_source_checks"].append({"id": source["id"], "path": path, "previous_sha256": source["sha256"], "current_sha256": current, "unchanged": equal})
for artifact in original["artifacts"]:
    actual = sha((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    record["reused_artifact_checks"].append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})
for path in sorted(changed):
    before = git("show", "5ef56bc756f3:" + path)
    after = git("show", "3af7f13ef959:" + path)
    current = (ROOT / path).read_bytes()
    assert after == current
    previous = next(s for s in original["sources"] if s.get("path") == path)
    assert sha(before) == previous["sha256"]
    command = ["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", path]
    diff = git(*command[1:]).decode()
    item = {"path": path, "command_argv": command, "before_sha256": sha(before), "current_sha256": sha(current), "diff": diff}
    if path.endswith("17.md"):
        marker = "**量化（quantization）**"
        old = next(p for p in before.decode().split("\n\n") if marker in p)
        new = next(p for p in current.decode().split("\n\n") if marker in p)
        assert old == new
        item.update(affected_claim_ids=["c12"], supporting_passage=new, supporting_passage_unchanged=True, conclusion="量化的少位元、同形狀與同參數數量定義未改；to 的副本條件修正不推翻本頁依據。")
    elif path.endswith("18.md"):
        markers = ["蒸餾（distillation）指", "把模型取名student"]
        passages = []
        for marker in markers:
            old = next(p for p in before.decode().split("\n\n") if marker in p)
            new = next(p for p in current.decode().split("\n\n") if marker in p)
            assert old == new
            passages.append(new)
        item.update(affected_claim_ids=["c13"], supporting_passages=passages, supporting_passages_unchanged=True, conclusion="教師文字／分布訊號與先選較小學生的說明未改；尺寸與溫度等段落的修正不擴張本頁的蒸餾主張。")
    else:
        page = original["lesson_id"]
        result_page = "19.12"
        old_page, new_page = section(before, page), section(current, page)
        old_result, new_result = section(before, result_page), section(current, result_page)
        assert old_page == new_page
        assert old_result == new_result
        assert "| 測試任務 |" in new_result.decode() and "| Dense |" in new_result.decode()
        item.update(affected_claim_ids=["c18"], reviewed_page_unchanged=True, result_section_unchanged=True, result_section_sha256=sha(new_result), result_heading=new_result.decode().splitlines()[0], result_document="docs/selftrained/results/v2-final-public-results.json", conclusion="本頁及其Dense結果連結目的地均逐位元組未變；章內其他改句不影響本頁主張。")
    record["source_diffs"].append(item)
current_section = section((ROOT / "course/chapters/19.md").read_bytes(), original["lesson_id"])
frozen_section = (ROOT / "docs/course-review-20261019/freeze-01/sources/19.10.md").read_bytes()
assert current_section == frozen_section
assert sha(current_section) == original["source_sha256"]
record["current_section_sha256"] = sha(current_section)
record["changed_sentences_in_reviewed_page"] = []
public = json.loads((ROOT / "docs/technical-reviews/artifacts/review-20261019-19-10/public-verification.json").read_bytes())
for path, expected in public["repository_inputs"].items():
    assert sha((ROOT / path).read_bytes()) == expected
record["public_verification_repository_inputs_unchanged"] = True
torch.set_num_threads(1)
for architecture in ("moe", "dense"):
    prior = public["models"][architecture]
    model = LimitedAssistant(SelftrainedConfig(**prior["config"]))
    count = sum(p.numel() for p in model.parameters())
    count_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    assert count == prior["parameters"]
    assert count_bytes == prior["unique_fp32_bytes"]
    assert model.lm.embedding.weight is model.lm.output.weight
    record["current_model_arithmetic"][architecture] = {"parameters": count, "fp32_bytes": count_bytes, "fp32_mib": count_bytes / 1024**2, "ideal_two_byte_bytes": count * 2, "dtype": sorted({str(p.dtype) for p in model.parameters()})}
results = json.loads((ROOT / "docs/selftrained/results/v2-final-public-results.json").read_bytes())
assert "dense" in results["architectures"] and "final_test" in results["architectures"]["dense"]
record["dense_result_exists"] = True
record["limitations"] = "固定公開權重及外部原始來源沿用已核對且SHA未變的證據；本次未重下載權重、重訓或重跑品質評測，也未全面重審其他章節的所有改句。"
print(json.dumps(record, ensure_ascii=False, indent=2))
