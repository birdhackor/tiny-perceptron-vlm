"""Bounded original-report location check after full 7.9 reread; no training."""
import hashlib
import json
import platform
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer

base = Path(__file__).parent
sha = lambda b: hashlib.sha256(b).hexdigest()
old = (base / "section.md").read_bytes()
new = (base / "followup-section.md").read_bytes()
old_clause = "；[7.12](#7.12)會列出成績。"
new_clause = "；[完整實測報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/sft.json)保留了逐題紀錄。"
assert old.count(old_clause.encode()) == 1
assert new == old.replace(old_clause.encode(), new_clause.encode())
assert sha(new) == "2aa093dff15ba9cff880625d32246f599642985d1ec129cdc68785009a6cf64e"
old_meta = json.loads((base / "extraction.json").read_text())
new_meta = json.loads((base / "followup-extraction.json").read_text())
assert old_meta["python_fences"] == new_meta["python_fences"]
assert old_meta["bootstrap"]["bootstrap_sha256"] == new_meta["bootstrap"]["bootstrap_sha256"]
assert old_meta["figure_sha256"] == new_meta["figure_sha256"] == {}
previous = json.loads((base / "bounded-checks.stdout.json").read_text())
for path, digest in previous["hashes"].items():
    assert sha((ROOT / path).read_bytes()) == digest

commit = json.loads((base / "followup-main-commit.json").read_text())["sha"]
assert commit == "6b8fb865b6f4f5f1c1649420520232f9911c1227"
blob_url = "https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/sft.json"
raw_main_url = "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/main/docs/course-experiments/results/sft.json"
raw_pinned_url = "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/" + commit + "/docs/course-experiments/results/sft.json"
requests = []
responses = {}
for name, url, bound in [
    ("followup-github-sft-page.html", blob_url, 1500000),
    ("followup-github-sft-main.json", raw_main_url, 200000),
    ("followup-github-sft-pinned.json", raw_pinned_url, 200000),
]:
    request = urllib.request.Request(url, headers={"User-Agent": "Codex-independent-factual-audit/7.9"})
    with urllib.request.urlopen(request, timeout=15) as response:
        content = response.read(bound + 1)
        assert len(content) <= bound
        (base / name).write_bytes(content)
        responses[name] = content
        requests.append({"url": url, "final_url": response.url, "status": response.status,
                         "bytes": len(content), "sha256": sha(content), "snapshot": name})

page = responses["followup-github-sft-page.html"].decode()
title = re.search(r"<title>(.*?)</title>", page, re.S)[1]
assert "sft.json at main" in title
report_bytes = responses["followup-github-sft-main.json"]
assert report_bytes == responses["followup-github-sft-pinned.json"]
assert report_bytes == (base / "sft-original-results.json").read_bytes()
assert report_bytes == (ROOT / "docs/course-experiments/results/sft.json").read_bytes()
assert sha(report_bytes) == previous["existing_result_recount"]["input_sha256"]
document = json.loads(report_bytes)
results = document["results"]
assert results["training"]["steps"] == 900 and results["checkpoint"] == "model.pt"
tok = ByteTokenizer()
splits = {}
for split, expected_count in [("validation", 5), ("test", 10)]:
    report = results["after"][split]
    rows = report["samples"]
    assert len(rows) == report["records"] == expected_count
    count_eos = count_exact = 0
    checked_rows = []
    for number, row in enumerate(rows):
        ids = row["generated_ids"]
        ended = tok.eos_id in ids
        visible = ids[:ids.index(tok.eos_id)] if ended else ids
        exact = visible == tok.encode(row["expected"])
        assert tok.decode(visible) == row["generated"]
        assert ended == row["eos"] and exact == row["exact"]
        count_eos += int(ended)
        count_exact += int(exact)
        checked_rows.append({"locator": "results.after." + split + ".samples[" + str(number) + "]",
                             "expected": row["expected"], "generated": row["generated"],
                             "generated_ids": ids, "eos": ended, "exact": exact})
    assert count_eos == expected_count
    assert count_exact == report["matches"]
    assert count_exact / expected_count == report["exact_match"]
    assert count_eos / expected_count == report["eos_rate"]
    splits[split] = {"records": expected_count, "EOS_count": count_eos, "exact_count": count_exact,
                     "content_errors": expected_count - count_exact, "checked_rows": checked_rows}
first = splits["validation"]["checked_rows"][0]
assert first["expected"] == "circle" and first["generated"] == "cire"
assert first["generated_ids"] == [107, 113, 122, 109, 2]
result = {
    "kind": "followup_original_report_location_check", "accessed_on": "2026-10-05",
    "environment": {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                    "device": "cpu", "TLS": "urllib verified default HTTPS context"},
    "full_reread_scope": "Reviewer personally read complete new7.9 including original fence, exercise and supplement before this check",
    "old_source_sha256": sha(old), "new_source_sha256": sha(new),
    "exact_isolated_clause_change": {"old": old_clause, "new": new_clause, "verified": True},
    "new_link_page_title": title, "requests": requests,
    "public_repository_head_commit": commit, "recorded_experiment_code_revision": document["revision"],
    "report_sha256": sha(report_bytes), "same_as_previously_audited_original_report": True,
    "report_authority": "Original project-author experiment output at its stated repository path; supports this recorded run, not general model capabilities",
    "training_steps_locator": "results.training.steps", "training_steps": results["training"]["steps"],
    "model_branch": results["checkpoint"], "splits": splits,
    "reused_execution": {"original_fence_rerun": False, "bounded_CPU_contract_rerun": False,
                         "reason": "Exact fence/bootstrap/figure and inspected implementation hashes unchanged; old raw JSON byte-identical",
                         "new_execution": "Actual HTTPS location/byte checks and all15 original raw sample recounts performed now"},
    "review_judgment": "The new direct link accurately locates the claimed per-question records; prior erroneous7.12 promise is resolved."
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
