"""核對既有判斷引用的原文、證據及聲音特徵路線是否仍相容。"""

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, h in enumerate(headings) if h[0].startswith(b"## " + identifier.encode() + b" "))
    end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
    return raw[headings[index].start():end]


def command(argv):
    done = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True)
    return {"argv": argv, "stdout": done.stdout.decode(), "stderr": done.stderr.decode(), "exit_code": done.returncode}


def main():
    import torch
    import torch.nn.functional as F
    from tiny_perceptron.multimodal import log_mel, tone
    from tiny_perceptron.natural_concepts import audio_order_report
    from tiny_perceptron.selftrained.model import AudioEncoder

    torch.set_num_threads(1)
    original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_12-14/technical-12-14-original-01.json"
    original_raw = original_path.read_bytes()
    assert sha(original_raw) == "0e32232cb5ed0210c322f1a2d92b275dd79db2eaed3c81dec68bfdefbf9fb9bc"
    original = json.loads(original_raw)
    assert original["issues"] == []
    code_path = BASE / "verify_facts.py"
    previous_code = next(a for a in original["artifacts"] if a["id"] == "verification_code")
    assert sha(code_path.read_bytes()) == previous_code["sha256"]
    execution = command([sys.executable, str(code_path)])
    local = json.loads(execution["stdout"])
    artifact_checks = []
    for artifact in original["artifacts"]:
        actual = sha((ROOT / artifact["path"]).read_bytes())
        assert actual == artifact["sha256"]
        artifact_checks.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
    source_checks = []
    for source in original["sources"]:
        if source["kind"] != "repository_code":
            continue
        actual = sha((ROOT / source["path"]).read_bytes())
        unchanged = actual == source["sha256"]
        assert unchanged or source["path"] == "course/chapters/12.md"
        source_checks.append({"id": source["id"], "path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual, "unchanged": unchanged})

    older = command(["git", "show", "5ef56bc756f3:course/chapters/12.md"])
    newer = command(["git", "show", "3af7f13ef959:course/chapters/12.md"])
    diff = command(["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/12.md"])
    current = (ROOT / "course/chapters/12.md").read_bytes()
    assert newer["stdout"].encode() == current
    assert sha(older["stdout"].encode()) == "2f03c718edf6357579c572ca3ecba5596a5e50a8d5684648d4f2e4e79061bddc"
    assert sha(current) == "ad9b58fa5af0bfa776321634a7fe266b7ed32232fece728f562fded292d24092"
    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/12.14.md").read_bytes()
    assert section(current, "12.14") == section(older["stdout"].encode(), "12.14") == frozen
    freeze_path = ROOT / "docs/course-review-20261019/freeze-05.json"
    freeze_raw = freeze_path.read_bytes()
    assert sha(freeze_raw) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
    page = next(p for p in json.loads(freeze_raw)["pages"] if p["page_id"] == "12.14")
    assert page["source_sha256"] == sha(frozen)
    sections = []
    for number in ("12.11", "12.13", "12.14", "12.15", "12.16"):
        before, after = section(older["stdout"].encode(), number), section(current, number)
        sections.append({"section": number, "before_sha256": sha(before), "after_sha256": sha(after), "unchanged": before == after})

    order = audio_order_report()
    assert order["same_time_sequence"] is False
    assert order["same_mean_with_roundoff_tolerance"] is True
    wave = torch.cat([tone(440, seconds=0.1), tone(880, seconds=0.1)])
    frames = log_mel(wave).T
    means = frames.mean(0)
    assert means.shape == (16,)
    assert torch.allclose(means, frames.flip(0).mean(0))
    torch.manual_seed(0)
    encoder = AudioEncoder(64).eval()
    shapes = []
    with torch.no_grad():
        for length in (173, 600, 1000):
            mel = torch.randn(length, 40)
            shortened = encoder.temporal(mel.T[None]).shape[-1]
            features, _ = encoder(mel)
            assert features.shape == (16, 64)
            weights = F.adaptive_avg_pool1d(torch.eye(shortened)[None], 16)[0].T
            intervals = []
            for index, row in enumerate(weights):
                start, end = math.floor(index * shortened / 16), math.ceil((index + 1) * shortened / 16)
                support = row.nonzero().flatten().tolist()
                assert support == list(range(start, end))
                assert torch.allclose(row[start:end], torch.full((end-start,), 1/(end-start)))
                intervals.append([start, end])
            overlap = any(intervals[i][1] > intervals[i+1][0] for i in range(15))
            assert overlap
            shapes.append({"input_frames": length, "shortened": shortened, "features_shape": list(features.shape), "pool_intervals": intervals, "adjacent_intervals_overlap": overlap})
    return {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-14/verify-current-01.py",
        "environment": local["environment"],
        "result": "全部斷言通過；本節原文未變，原證據未變，現行直接聲音路線仍由特徵接入。",
        "original_report": {"path": str(original_path.relative_to(ROOT)), "sha256": sha(original_raw), "issues": original["issues"]},
        "repository_source_checks": source_checks,
        "original_artifact_checks": artifact_checks,
        "local_verifier_execution": execution,
        "local_verifier_result": local,
        "chapter_diff": diff,
        "chapter_versions": {"before": {"argv": older["argv"], "sha256": sha(older["stdout"].encode())}, "after": {"argv": newer["argv"], "sha256": sha(current)}},
        "section_comparisons": sections,
        "current_source_sha256": sha(frozen),
        "current_freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha(freeze_raw), "page": page},
        "dependency_observations": {"audio_order_report": order, "mean_log_features": means.tolist(), "encoder_and_pool_checks": shapes},
        "additional_input_sha256": {name: sha((ROOT/name).read_bytes()) for name in ("tiny_perceptron/multimodal.py", "tiny_perceptron/selftrained/model.py")},
        "scope": "相鄰段落的補充檢查只用於確認原文導航與證據依賴，沒有代替那些小節的完整審閱；不執行訓練、不使用真錄音、不載入成熟模型。",
    }


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
