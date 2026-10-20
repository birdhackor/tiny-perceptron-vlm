import copy
import difflib
import hashlib
import html
import inspect
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from tiny_perceptron.lesson_products import evaluation

ARTIFACT = Path(__file__).parent
REPORT_PATH = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_19-18/technical-19-18-original-01.json"
OLD_PAGE = ROOT / "docs/course-review-20261019/freeze-01/sources/19.18.md"
NEW_PAGE = ROOT / "docs/course-review-20261019/freeze-05/sources/19.18.md"
FREEZE_PATH = ROOT / "docs/course-review-20261019/freeze-05.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


original = read_json(REPORT_PATH)
assert sha(REPORT_PATH.read_bytes()) == "69363f66d465fedde81030db6260f3c6fdc0ea14fe92752a9a5b8808d314aa96"
old_raw, new_raw = OLD_PAGE.read_bytes(), NEW_PAGE.read_bytes()
assert sha(old_raw) == original["source_sha256"]
assert sha(new_raw) == "3d68676803c802348319b8557f64de7e88f9e34229b318452c1dae4a53cfe591"
assert sha(FREEZE_PATH.read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
freeze = read_json(FREEZE_PATH)
page = next(p for p in freeze["pages"] if p["page_id"] == "19.18")
chapter_path = ROOT / page["source"]
section = re.search(r"^## 19\.18 .*?(?=^## |\Z)", chapter_path.read_text(), flags=re.M | re.S).group()
assert section.encode() == new_raw
assert page["figures_sha256"] == {}
assert new_raw.count(b"```") == 0

retained_sources = []
for source in original["sources"]:
    if source.get("path"):
        path = ROOT / source["path"]
        actual = sha(path.read_bytes())
        assert actual == source["sha256"], source["path"]
        retained_sources.append({"source_id": source["id"], "path": source["path"], "sha256": actual, "unchanged": True})
retained_artifacts = []
for artifact in original["artifacts"]:
    path = ROOT / artifact["path"]
    actual = sha(path.read_bytes())
    assert actual == artifact["sha256"], artifact["path"]
    retained_artifacts.append({"artifact_id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})

old_lines, new_lines = old_raw.decode().splitlines(), new_raw.decode().splitlines()
changes = []
for tag, i, j, k, l in difflib.SequenceMatcher(a=old_lines, b=new_lines).get_opcodes():
    if tag != "equal":
        changes.append({"operation": tag, "before": old_lines[i:j], "after": new_lines[k:l]})
assert len(changes) == 1

results_path = ROOT / "docs/lesson-products/results/final-report-v1.json"
results = read_json(results_path)
latency = results["latency"]
fields = {
    "requests", "integrated_mean_seconds", "integrated_median_seconds",
    "integrated_p95_seconds", "c_mean_seconds", "direct_b_mean_seconds",
}
for measurement in latency["measurements"].values():
    assert set(measurement) == fields
    assert measurement["requests"] == 72
assert len(latency["sample_ids"]) == 24

sample = [{"id": str(i), "state": "固定探針輸入"} for i in range(24)]
probe_calls = Counter()


class BProbe:
    device = torch.device("cpu")

    def solve(self, state, depth):
        return {"seconds": 0.001}


def controlled_auto(b, c, state, effort):
    probe_calls[effort] += 1
    measured_index = probe_calls[effort] - 3
    return {"seconds": max(measured_index, 0) / 1000, "c_seconds": 0.0001}


torch.set_num_threads(2)
with patch.object(evaluation, "auto_solve", side_effect=controlled_auto):
    probe = evaluation.latency_report(BProbe(), object(), sample)
for effort, value in probe["measurements"].items():
    assert probe_calls[effort] == 75
    assert value["requests"] == 72
    assert abs(value["integrated_mean_seconds"] - 0.0365) < 1e-12
    assert abs(value["integrated_median_seconds"] - 0.0365) < 1e-12
    assert value["integrated_p95_seconds"] == 0.068
    assert set(value) == fields
    assert not any(isinstance(v, (list, dict)) for v in value.values())

count = 72
index = int(0.95 * (count - 1))
rank = index + 1
above = count - rank
assert index == 67 and rank == 68 and above == 4

url = "https://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm"
curl_command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "30", url]
response = subprocess.run(curl_command, check=True, capture_output=True).stdout
source_text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", response.decode())))
excerpt = "Order statistics provide a way of estimating proportions of the data that should fall above and below a given value"
assert excerpt in source_text
assert "There is not a standard univerally accepted way" in source_text

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-18/verify-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "frozen_page": {"old_path": str(OLD_PAGE.relative_to(ROOT)), "old_sha256": sha(old_raw), "path": str(NEW_PAGE.relative_to(ROOT)), "sha256": sha(new_raw), "freeze_path": str(FREEZE_PATH.relative_to(ROOT)), "freeze_sha256": sha(FREEZE_PATH.read_bytes()), "current_section_equals_snapshot": True, "code_fences": 0, "figures": {}},
    "original_report": {"path": str(REPORT_PATH.relative_to(ROOT)), "sha256": sha(REPORT_PATH.read_bytes())},
    "changed_blocks": changes,
    "unchanged_claim_ids": [c["id"] for c in original["claims"] if c["id"] != "c48"],
    "retained_sources": retained_sources,
    "retained_artifacts": retained_artifacts,
    "historical_latency_fields": {effort: sorted(value) for effort, value in latency["measurements"].items()},
    "historical_latency_measurements": latency["measurements"],
    "historical_raw_prediction_audit": {"artifact_path": "docs/technical-reviews/artifacts/review-20261019-19-18/verification-output.json", "artifact_sha256": next(a["sha256"] for a in original["artifacts"] if a["id"] == "a_run"), "finding": "原審閱已核對完整公開JSONL的SHA並掃描全檔，無時間欄位；該artifact及所引來源SHA本次均未變。"},
    "latency_probe": {"scope": "受控輸入測試實際latency_report的聚合與回傳；數字由探針提供，不是本機速度或歷史實測。", "measured_seconds": "每模式先暖機三次，再提供0.001至0.072的72個不同值。", "calls_per_mode_with_warmup": dict(probe_calls), "returned_measurements": probe["measurements"], "raw_measurements_returned": False, "minimum_maximum_returned": False},
    "p95_rank_calculation": {"code_expression": "sorted(measurements)[int(0.95 * (len(measurements) - 1))]", "count": count, "zero_based_index": index, "one_based_rank": rank, "larger_rank_positions": above, "fraction_of_positions_after_selected_percent": above / count * 100, "interpretation": "72筆時取升序第68個，其後4個排序位置占5.5556%；因此『最慢5%邊界附近』是合適的近似說法。相等值、有限樣本及不同分位數方法不保證恰好5%的數值嚴格較大。"},
    "percentile_source": {"url": url, "accessed_on": datetime.now().astimezone().date().isoformat(), "version": "NIST/SEMATECH e-Handbook網站讀取快照，以完整回應SHA-256識別", "curl_command": curl_command, "full_response_sha256": sha(response), "full_response_bytes": len(response), "locator": "Definition of percentiles；Estimation of percentiles；Note that there are other ways of calculating percentiles in common use", "excerpt": excerpt, "inspection_note": "讀取百分位數的排序／上下尾部意義及樣本估計方法差異；本文使用『附近』而非聲稱有限樣本上尾恰為5%。只保存短摘錄及收據，原始完整HTML僅在記憶體。"},
    "source_excerpt": inspect.getsource(evaluation.latency_report),
    "source_file_sha256": sha((ROOT / "tiny_perceptron/lesson_products/evaluation.py").read_bytes()),
    "scope_limits": "沿用SHA未變的原始預測重算、權重計數與原始來源檢查。本次新增的是摘要回傳與p95取值的確定性探針，未重訓、未重現歷史CPU延遲，也沒有補造歷史逐次量測。",
}
output = ARTIFACT / "verification-current-01.json"
with output.open("x") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"artifact": str(output.relative_to(ROOT)), "sha256": sha(output.read_bytes()), "changed_blocks": len(changes), "retained_sources": len(retained_sources), "retained_artifacts": len(retained_artifacts), "probe": "pass", "p95_rank": rank, "fraction_after_rank_percent": above / count * 100}, ensure_ascii=False))
