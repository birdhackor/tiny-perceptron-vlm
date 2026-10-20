"""複查章首主張、相依文件變更與既有證據的完整性。"""

from datetime import datetime
import hashlib
from html.parser import HTMLParser
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import unicodedata
from zoneinfo import ZoneInfo

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = ROOT / "docs/technical-reviews/artifacts/review-20261019-chapter-10"
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_chapter-10/technical-chapter-10-original-01.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


class OriginalText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.ignore = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.ignore += 1
        if tag == "img" and not self.ignore:
            self.parts.append(dict(attrs).get("alt", ""))

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.ignore -= 1

    def handle_data(self, data):
        if not self.ignore:
            self.parts.append(data)


def receipt(url, version, excerpts):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=True)
    parser = OriginalText()
    parser.feed(result.stdout.decode())
    text = " ".join("".join(parser.parts).split())
    for excerpt in excerpts:
        assert excerpt["quotation"] in text, "支持主張的摘錄必須出現在實際讀取的原文"
    return {
        "url": url, "version": version,
        "accessed_on": datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat(),
        "command": command, "returncode": result.returncode,
        "full_response_sha256": sha(result.stdout), "full_response_bytes": len(result.stdout),
        "checked_original": True, "excerpts": excerpts,
        "storage": "完整第三方內容僅在記憶體讀取；只保存來源指紋與短摘錄。",
    }


assert sha(ORIGINAL.read_bytes()) == "3e471eac23f331cb8dbabf6949c0f31cfe6644a110bd43accfa12af57b8e519d", "原始報告必須保持不變"
original = json.loads(ORIGINAL.read_bytes())
assert original["issues"] == [], "原始報告的問題清單必須如實讀取"
original_integrity = []
for item in original["artifacts"]:
    actual = sha((ROOT / item["path"]).read_bytes())
    assert actual == item["sha256"], "沿用證據的內容必須保持不變"
    original_integrity.append({"path": item["path"], "sha256": actual, "unchanged": True})
self_check = ARTIFACTS / "self-check-01.txt"
assert sha(self_check.read_bytes()) == "3de7ec140185a0a6a5e78032907b8cb3064da5540473d48442e2030483ad4845", "先前自我檢查輸出必須保持不變"
original_integrity.append({"path": self_check.relative_to(ROOT).as_posix(), "sha256": sha(self_check.read_bytes()), "unchanged": True})
freeze_path = ROOT / "docs/course-review-20261019/freeze-05.json"
assert sha(freeze_path.read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba", "複查凍結清單必須相符"
freeze = json.loads(freeze_path.read_bytes())
page = next(item for item in freeze["pages"] if item["page_id"] == "chapter-10")
source = (ROOT / page["snapshot"]).read_bytes()
assert sha(source) == original["source_sha256"] == page["source_sha256"], "凍結導言必須維持同一份原文"
assert page["figures_sha256"] == {}, "凍結導言不含圖"

changed_paths = ["course/chapters/10.md", "course/glossary.md"]
expected_hashes = [
    "a90293b8deaec91d8b634525c1ff6174d55420cab8dfb7b13b7bd906e01877eb",
    "6985f5f3137fe8fa3937b3e03d8cafb985e6f8787a4e6db675199a650514e037",
]
diffs = []
for path, expected in zip(changed_paths, expected_hashes, strict=True):
    command = ["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", path]
    diff = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    old_bytes = subprocess.run(["git", "show", f"5ef56bc756f3:{path}"], capture_output=True, check=True).stdout
    new_bytes = subprocess.run(["git", "show", f"3af7f13ef959:{path}"], capture_output=True, check=True).stdout
    current_bytes = (ROOT / path).read_bytes()
    assert sha(new_bytes) == sha(current_bytes) == expected, "指定變更後的版本必須等於現行來源"
    prior = next(item for item in original["sources"] if item.get("path") == path)
    assert sha(old_bytes) == prior["sha256"], "指定變更前的版本必須等於原始核對來源"
    diffs.append({"path": path, "command": command, "returncode": 0, "diff": diff,
                  "old_sha256": sha(old_bytes), "current_sha256": sha(current_bytes)})

command = ["uv", "run", "--no-sync", "python", str((ARTIFACTS / "audit.py").relative_to(ROOT))]
rerun = subprocess.run(command, capture_output=True, text=True, check=True)
runtime = json.loads(rerun.stdout)
assert runtime["freeze"]["current_introduction_equals_snapshot"], "現行章首必須仍等於凍結原文"
for source_record in original["sources"]:
    if source_record["kind"] == "repository_code" and source_record["path"] not in changed_paths:
        assert sha((ROOT / source_record["path"]).read_bytes()) == source_record["sha256"], "其他沿用程式來源必須保持不變"

normalize = {}
for name, tensor in [
    ("integer", torch.tensor([[0, 3], [2, 0]])),
    ("floating", torch.tensor([[0.0, 3.0], [2.0, 0.0]])),
    ("complex", torch.tensor([[1 + 0j, 0 + 1j]])),
]:
    try:
        result = F.normalize(tensor, dim=-1)
        normalize[name] = {"dtype": str(tensor.dtype), "accepted": True,
                           "real": result.real.tolist(),
                           "imaginary": result.imag.tolist() if result.is_complex() else None}
    except RuntimeError as error:
        normalize[name] = {"dtype": str(tensor.dtype), "accepted": False,
                           "exception_type": type(error).__name__, "message": str(error)}
assert not normalize["integer"]["accepted"] and normalize["floating"]["accepted"] and normalize["complex"]["accepted"], "整數限制與浮點／複數行為必須符合實際介面"
images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
texts = torch.tensor([[0.0, 3.0], [2.0, 0.0]])
scores = F.normalize(images, dim=-1) @ F.normalize(texts, dim=-1).T
assert scores.tolist() == [[0.0, 1.0], [1.0, 0.0]] and scores.argmax(-1).tolist() == [1, 0], "交換候選後必須符合預期配對"
normalize["swapped_candidates"] = {"scores": scores.tolist(), "selected_columns": scores.argmax(-1).tolist()}
normalize["implementation"] = {"function": "torch.nn.functional.normalize", "path": inspect.getsourcefile(F.normalize),
    "source_sha256": sha(Path(inspect.getsourcefile(F.normalize)).read_bytes()),
    "inspected_expression": "denom = input.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(input)"}
assert normalize["implementation"]["inspected_expression"] in inspect.getsource(F.normalize), "dtype 錯誤必須對應實際的 norm 路徑"

unicode_examples = []
for text in ["A", "é", "中", "𠀀", "👍🏽"]:
    unicode_examples.append({"text": text, "codepoints": [f"U+{ord(c):04X}" for c in text],
        "names": [unicodedata.name(c) for c in text], "codepoint_count": len(text),
        "utf8_hex": text.encode("utf-8").hex(), "utf8_bytes": len(text.encode("utf-8"))})
assert unicode_examples[-1]["codepoints"] == ["U+1F44D", "U+1F3FD"]
assert [item["utf8_bytes"] for item in unicode_examples[:4]] == [1, 2, 3, 4]
surrogate_results = []
for cp in [0xD800, 0xDFFF]:
    try:
        chr(cp).encode("utf-8")
        raise AssertionError("標準 UTF-8 不可接受代理碼點")
    except UnicodeEncodeError as error:
        surrogate_results.append({"codepoint": f"U+{cp:04X}", "exception_type": type(error).__name__, "reason": error.reason})

logit = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
old_probability, advantage, epsilon, step_size = 0.5, 1.0, 0.2, 20.0
ratio = logit.sigmoid() / old_probability
objective = torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
objective.backward()
with torch.no_grad():
    new_logit = logit + step_size * logit.grad
    new_probability = new_logit.sigmoid().item()
    new_ratio = new_probability / old_probability
assert new_ratio > 1 + epsilon, "裁切目標本身沒有將更新後機率比限制在區間內"
ppo = {"formula": "min(r*A, clip(r, 1-epsilon, 1+epsilon)*A)", "old_probability": old_probability,
       "advantage": advantage, "epsilon": epsilon, "initial_logit": logit.item(),
       "initial_objective": objective.item(), "initial_gradient": logit.grad.item(),
       "gradient_ascent_step_size": step_size, "new_logit": new_logit.item(),
       "new_probability": new_probability, "new_ratio": new_ratio,
       "ratio_exceeds_clipping_interval": True,
       "scope": "單一標量的確定性反例，驗證裁切不構成硬性更新限制；未做 RL 訓練或效能比較。"}

originals = [
    receipt("https://docs.pytorch.org/docs/2.14/generated/torch.linalg.vector_norm.html", "PyTorch 2.14 官方文件", [
        {"locator": "torch.linalg.vector_norm／支援 dtype", "quotation": "Supports input of float, double, cfloat and cdouble dtypes."}]),
    receipt("https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/", "The Unicode Standard 17.0.0，Conformance", [
        {"locator": "Unicode Encoding Forms／Unicode scalar value", "quotation": "Any Unicode code point except high-surrogate and low-surrogate code points."},
        {"locator": "UTF-8 encoding form", "quotation": "one to four bytes in length"}]),
    receipt("https://www.unicode.org/reports/tr29/tr29-47.html", "Unicode Standard Annex #29，Revision 47，Unicode 17.0.0", [
        {"locator": "Grapheme Cluster Boundaries／emoji modifier sequences", "quotation": "Do not break within emoji modifier sequences or emoji zwj sequences."}]),
    receipt("https://spinningup.openai.com/en/latest/algorithms/ppo.html", "OpenAI Spinning Up PPO 文件，latest；實際內容以回應 SHA-256 鎖定", [
        {"locator": "Key Equations／You Should Know", "quotation": "it is still possible to end up with a new policy which is too far from the old policy"}]),
]
output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-10/audit-current-01.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
                    "device": "cpu", "unicodedata": unicodedata.unidata_version, "training_steps": "0"},
    "original_report": {"path": ORIGINAL.relative_to(ROOT).as_posix(), "sha256": sha(ORIGINAL.read_bytes()), "issue_count": len(original["issues"])},
    "original_artifact_integrity": original_integrity,
    "freeze": {"path": freeze_path.relative_to(ROOT).as_posix(), "sha256": sha(freeze_path.read_bytes()),
               "source_sha256": sha(source), "figure_sha256": page["figures_sha256"]},
    "diffs": diffs,
    "rerun": {"command": command, "returncode": rerun.returncode, "stderr": rerun.stderr, "result": runtime},
    "dependency_change_verification": {"normalize": normalize, "unicode": {"examples": unicode_examples, "surrogates": surrogate_results}, "ppo": ppo},
    "original_sources": originals,
    "result": "所有章首主張與相依變更核對通過；先前報告與證據未變，未進行訓練或遠端寫入。",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
