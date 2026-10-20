"""保存新版課文差異、執行輸出及修訂語句的獨立核對結果。"""

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.selftrained.model import AudioEncoder


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_hashed(path):
    raw = (ROOT / path).read_bytes()
    return {"path": path, "sha256": sha(raw)}


old_report_path = "docs/course-review-20261019/reviews/rv1019_technical_12-15/technical-12-15-original-01.json"
old_report = json.loads((ROOT / old_report_path).read_text())
assert sha((ROOT / old_report_path).read_bytes()) == "5f74e780f67d337d4540e62b9b6679df4ec75688d3c72de0c2c09220a4cd149b"
run = ROOT / sys.argv[1]
command = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-15/verify-current-01.py " + sys.argv[1]
old_path = "docs/course-review-20261019/freeze-01/sources/12.15.md"
new_path = "docs/course-review-20261019/freeze-05/sources/12.15.md"
old_raw, new_raw = [(ROOT / path).read_bytes() for path in (old_path, new_path)]
assert sha(new_raw) == "ad0b0adcef37c637dc319d1aeb976c8c0c9a5f38f1ad05926a7ca99e5f3133b2"
chapter_raw = (ROOT / "course/chapters/12.md").read_bytes()
current = re.search(rb"(?ms)^## 12\.15 .*?(?=^## |\Z)", chapter_raw)[0]
assert current == new_raw
assert (run / "section.md").read_bytes() == new_raw
argv = ["git", "diff", "--no-index", old_path, new_path]
diff = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
assert diff.returncode == 1
assert not diff.stderr
result = {
    "command": command,
    "accessed_on": datetime.now(timezone.utc).date().isoformat(),
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git": torch.version.git_version, "device": "cpu"},
    "original_report": read_hashed(old_report_path),
    "old_source": read_hashed(old_path),
    "source_freeze": read_hashed("docs/course-review-20261019/freeze-05.json"),
    "source": read_hashed(new_path),
    "current_section_equals_freeze": True,
    "diff": {"command_argv": argv, "exit_code": diff.returncode, "stdout": diff.stdout, "stderr": diff.stderr},
    "reused_files": [],
    "section_execution": {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/12.md#12.15" --execute',
        "copied_files": {name: {"sha256": sha((run / name).read_bytes()), "text": (run / name).read_text()} for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "section.md", "fence-1.py")},
    },
}
execution = json.loads(result["section_execution"]["copied_files"]["execution.json"]["text"])
assert execution["exit_code"] == 0
assert execution["source_sha256"] == sha(new_raw)
for entry in old_report["sources"] + old_report["artifacts"]:
    if "path" in entry:
        actual = read_hashed(entry["path"])
        actual.update(id=entry["id"], previous_sha256=entry["sha256"], unchanged=actual["sha256"] == entry["sha256"])
        assert actual["unchanged"]
        result["reused_files"].append(actual)
previous = json.loads((ROOT / "docs/technical-reviews/artifacts/review-20261019-12-15/verification.json").read_text())
assert previous["environment"]["torch_git"] == torch.version.git_version
result["previous_environment_compatible"] = True
result["figure"] = previous["figure_receipt"]
for item in (result["figure"]["frozen_svg"], result["figure"]["png"]):
    assert sha((ROOT / item["path"]).read_bytes()) == item["sha256"]
result["current_figure"] = read_hashed("course/figures/new-12.15-direct-audio.svg")
assert result["current_figure"]["sha256"] == result["figure"]["frozen_svg"]["sha256"]
result["changed_statements"] = [
    {"text": "最後沿時間順序切出 16 個區間，各自取平均（`adaptive_avg_pool1d`；區間邊界取整，相鄰區間可能重疊）。", "judgment": "成立；以有序且可重疊的區間描述 adaptive average pooling，不再聲稱互斥等分。", "evidence": "pooling_probe；official_pooling_source；AudioEncoder.forward"},
    {"text": "打字版也答錯，先查文字核心", "judgment": "成立；文字路徑已失敗時先核對共同文字核心是排查起點，不是唯一故障歸因。", "evidence": "diagnosis_cases；LimitedAssistant._embeddings/generate；沿用的模組映射原始研究及反例"},
    {"text": "打字版答對、語音版答錯，先查語音入口與接頭，也要確認核心能用語音送來的特徵", "judgment": "成立；同時保留聲音路徑與核心適配的可能原因，沒有用文字成功保證核心能處理語音表示。", "evidence": "diagnosis_cases；沿用的 diagnosis_counterexample"},
]
text = new_raw.decode()
for changed in result["changed_statements"]:
    assert changed["text"] in text

torch.set_num_threads(1)
torch.manual_seed(0)
encoder = AudioEncoder(64)
result["pooling_probe"] = []
for frames in (1, 2, 63, 64, 65, 173, 600, 1000):
    mel = torch.randn(frames, 40)
    with torch.no_grad():
        length = encoder.temporal(mel.T[None]).shape[-1]
        tokens, scores = encoder(mel)
    assert length == math.ceil(frames / 4)
    assert tuple(tokens.shape) == (16, 64)
    assert tuple(scores.shape) == (3,)
    intervals = [(i * length // 16, ((i + 1) * length + 15) // 16) for i in range(16)]
    weights = F.adaptive_avg_pool1d(torch.eye(length)[None], 16)[0].T
    expected = torch.zeros_like(weights)
    for i, (start, end) in enumerate(intervals):
        expected[i, start:end] = 1 / (end - start)
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    result["pooling_probe"].append({"input_shape": list(mel.shape), "temporal_length": length, "output_shape": list(tokens.shape), "score_shape": list(scores.shape), "intervals_half_open": intervals, "max_weight_error": float((weights - expected).abs().max())})

url = "https://raw.githubusercontent.com/pytorch/pytorch/" + torch.version.git_version + "/aten/src/ATen/native/AdaptivePooling.h"
curl_argv = ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", "40", url]
fetched = subprocess.run(curl_argv, capture_output=True, check=True)
raw = fetched.stdout
original = raw.decode()
start = original.index("inline int64_t start_index")
end_marker = "return 1 + ((a + 1) * c - 1) / b;\n}"
end = original.index(end_marker, start) + len(end_marker)
assert sha(raw) == previous["external_sources"]["adaptive_source"]["original_sha256"]
result["official_pooling_source"] = {"url": url, "version": "PyTorch " + str(torch.__version__) + ", commit " + torch.version.git_version, "accessed_on": result["accessed_on"], "command_argv": curl_argv, "exit_code": fetched.returncode, "original_sha256": sha(raw), "locator": "start_index/end_index", "excerpt": original[start:end]}

result["diagnosis_cases"] = []
for name, core_weights, audio_sign in (("核心未適配語音表示", [1., -1.], 1.), ("聲音路徑方向相反", [1., 1.], -1.), ("文字路徑也失敗", [-1., -1.], 1.)):
    rows = []
    weights = torch.tensor(core_weights)
    for intent in (-1, 1):
        typed = torch.tensor([float(intent), 0.])
        audio = torch.tensor([0., float(intent) * audio_sign])
        rows.append({"intent": intent, "typed_prediction": int(torch.sign(weights @ typed)), "audio_prediction": int(torch.sign(weights @ audio))})
    result["diagnosis_cases"].append({"case": name, "core_weights": core_weights, "audio_sign": audio_sign, "rows": rows})
assert all(x["typed_prediction"] == x["intent"] and x["audio_prediction"] != x["intent"] for case in result["diagnosis_cases"][:2] for x in case["rows"])
result["diagnosis_scope"] = "這些透明小例只核對排查建議沒有排除任何必要候選元件，不證明本專案實際模型發生這些故障，也不宣稱此排查次序對所有系統最佳。"
result["result"] = "新版原文與現行小節逐位元組相同；小節程式成功，來源及舊 artifacts 未變；池化權重與有序區間公式精確一致；排查建議涵蓋核心適配。所有斷言通過。"
output = Path(__file__).with_name("verification-current-01.json")
with output.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": sha(output.read_bytes()), "environment": result["environment"], "result": result["result"]}, ensure_ascii=False))
