"""複查權重刻度的來源變動、既存證據與當次執行結果。"""

from datetime import datetime, UTC
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    start = re.search(rb"(?m)^## " + re.escape(identifier.encode()) + rb" ", raw).start()
    next_header = re.search(rb"(?m)^## ", raw[start + 1:])
    end = start + 1 + next_header.start() if next_header else len(raw)
    return raw[start:end]


def capture(argv):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, check=True)
    return result, {"command_argv": argv, "exit_code": result.returncode,
                    "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}


def main():
    run = Path(sys.argv[1]).resolve()
    assert run.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
    original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_17-5/technical-17-5-original-01.json"
    original_raw = original_path.read_bytes()
    assert digest(original_raw) == "028b52ff9a7c616336b7279f4053c62aa34f097712839189e564710bcf39a859"
    original = json.loads(original_raw)
    assert original["issues"] == []
    inherited = []
    for item in original["artifacts"]:
        actual = digest((ROOT / item["path"]).read_bytes())
        assert actual == item["sha256"]
        inherited.append({"path": item["path"], "sha256": actual, "matches_original": True})
    chapter = (ROOT / "course/chapters/17.md").read_bytes()
    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/17.5.md").read_bytes()
    assert digest(chapter) == "3a1053817afd9934216d8b7eed62dd129ed230af0d56d4e546f52493c46486c4"
    assert section(chapter, "17.5") == frozen
    old_result, old_receipt = capture(["git", "show", "5ef56bc756f3:course/chapters/17.md"])
    new_result, new_receipt = capture(["git", "show", "3af7f13ef959:course/chapters/17.md"])
    assert digest(old_result.stdout) == "4328608fc1cfb9f9ac6febf838df9b9f60c2e9ffe1912e957fe3fc55824e365a"
    assert chapter == new_result.stdout
    comparisons = {}
    for identifier in ("17.2", "17.5"):
        old_section = section(old_result.stdout, identifier)
        current_section = section(chapter, identifier)
        assert old_section == current_section
        comparisons[identifier] = {"sha256": digest(current_section), "bytes_unchanged": True}
    _, diff_receipt = capture(["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/17.md"])
    repo_sources = []
    for item in original["sources"]:
        if item["kind"] == "repository_code":
            current_hash = digest((ROOT / item["path"]).read_bytes())
            if item["id"] == "s_repo":
                assert current_hash == item["sha256"]
            repo_sources.append({"id": item["id"], "path": item["path"],
                                 "previous_sha256": item["sha256"], "current_sha256": current_hash})
    _, probe_receipt = capture([str(ROOT / ".venv/bin/python"), str(ART / "verification.py")])
    probe_output = json.loads(probe_receipt["stdout"])
    assert probe_output == json.loads((ART / "verification-output.json").read_bytes())
    weight = torch.zeros(64, 32, dtype=torch.float32, device="cpu")
    to_results = []
    for dtype in (torch.float32, torch.float16, torch.int8):
        converted = weight.to(dtype)
        same = converted is weight
        assert same == (dtype == torch.float32)
        assert converted.dtype == dtype and converted.device == weight.device
        to_results.append({"dtype": str(dtype), "same_object": same,
                           "same_storage_pointer": converted.data_ptr() == weight.data_ptr(),
                           "numel": converted.numel(), "element_size": converted.element_size(),
                           "total_bytes": converted.numel() * converted.element_size()})
    url = "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.to.html"
    response, source_command = capture(["curl", "-fLsS", "--max-time", "30", url])
    source_raw = response.stdout
    article = None
    for block in re.findall(r"<script[^>]*type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>", source_raw.decode(), re.S):
        value = json.loads(block)
        if isinstance(value, dict) and "articleBody" in value:
            article = value["articleBody"]
            break
    if article is None:
        article = html.unescape(re.sub("<[^>]+>", " ", source_raw.decode()))
    article = re.sub(r"\s+", " ", article)
    needle = "then self is returned"
    assert needle in article
    pos = article.index(needle)
    passage = article[max(0, pos - 180):pos + 800]
    print("官方文件核對段落：", passage)
    source_receipt = {"url": url, "version": "PyTorch 2.14 API documentation",
                      "accessed_on": datetime.now(UTC).date().isoformat(),
                      "command_argv": source_command["command_argv"], "exit_code": response.returncode,
                      "retrieved_bytes": len(source_raw), "retrieved_sha256": digest(source_raw),
                      "excerpt": needle,
                      "inspection_note": "核對相同 dtype/device 回傳 self、否則建立轉換副本；copy 預設 False，本章呼叫沒有另設 copy。"}
    copied = []
    for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json"):
        raw = (run / name).read_bytes()
        stem, suffix = name.rsplit(".", 1)
        destination = ART / ("section-" + stem + "-current-01." + suffix)
        with destination.open("xb") as handle:
            handle.write(raw)
        copied.append({"path": str(destination.relative_to(ROOT)), "sha256": digest(raw)})
    metadata = json.loads((run / "extraction.json").read_bytes())
    assert metadata["source_sha256"] == digest(frozen) and metadata["figure_sha256"] == {}
    result = {
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "device": "cpu", "dtype": "torch.float32"},
        "original_report": {"path": str(original_path.relative_to(ROOT)), "sha256": digest(original_raw)},
        "original_issue_count": len(original["issues"]), "inherited_artifacts": inherited,
        "repository_sources": repo_sources, "section_comparisons": comparisons,
        "chapter_diff": diff_receipt,
        "commit_blobs": [{"command_argv": old_receipt["command_argv"], "sha256": digest(old_result.stdout)},
                         {"command_argv": new_receipt["command_argv"], "sha256": digest(new_result.stdout)}],
        "probe_execution": probe_receipt, "probe_output_matches_original": True,
        "tensor_to_results": to_results, "tensor_to_source": source_receipt,
        "section_extraction": {key: metadata[key] for key in ("source", "source_sha256", "source_file_sha256", "figure_sha256")},
        "copied_execution_files": copied,
        "claim_recheck": [{"claim_id": item["id"], "status": "verified", "text_unchanged": True}
                          for item in original["claims"]],
    }
    destination = ART / "recheck-current-01.json"
    with destination.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"path": str(destination.relative_to(ROOT)), "sha256": digest(destination.read_bytes()),
                      "claims": len(result["claim_recheck"]), "probe_matches_original": True,
                      "tensor_to_results": to_results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
