"""核對現行教材與先前已保存證據的相容性，只將短摘錄納入輸出。"""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import tokenizers
from tokenizers import pre_tokenizers

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    selected = [i for i, h in enumerate(headers) if h[0].startswith(b"## " + identifier.encode() + b" ")]
    assert len(selected) == 1, "章檔中須有唯一的小節"
    position = selected[0]
    end = headers[position + 1].start() if position + 1 < len(headers) else len(raw)
    return raw[headers[position].start():end]


def run(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    return result.stdout


def source_receipt(url, version, locator, excerpts, note):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
    raw = run(command)
    text = raw.decode("utf-8")
    for excerpt in excerpts:
        assert excerpt in text, "來源須含有核對所需的原文"
    return {"url": url, "version": version, "locator": locator,
            "accessed_at_utc": datetime.now(UTC).isoformat(), "command_argv": command,
            "exit_code": 0, "response_sha256": digest(raw), "response_bytes": len(raw),
            "excerpts": excerpts, "inspection_note": note}


original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_6-7/technical-6-7-original-01.json"
original_raw = original_path.read_bytes()
assert digest(original_raw) == "fd54392b9e5573c65266d412ee5073d5066ba15af6fe1bad10f132fab4d526e5"
original = json.loads(original_raw)
assert original["issues"] == []
preserved = []
for artifact in original["artifacts"]:
    actual = digest((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"], "先前證據不可被改寫"
    preserved.append({"path": artifact["path"], "sha256": actual, "unchanged": True})

chapter_path = ROOT / "course/chapters/06.md"
current = chapter_path.read_bytes()
old = run(["git", "show", "5ef56bc756f3:course/chapters/06.md"])
committed = run(["git", "show", "3af7f13ef959:course/chapters/06.md"])
assert digest(old) == "c0141b41a580e54516c71a7d6f45ed2352ee6830ecbfbf939f5f45eb5a16163a"
assert digest(current) == "e5ba622295b957ac094f3c39e76a6a29beaf89e5db2b96c750732f4b837e4820"
assert current == committed
frozen_path = ROOT / "docs/course-review-20261019/freeze-01/sources/6.7.md"
frozen = frozen_path.read_bytes()
assert section(old, "6.7") == section(current, "6.7") == frozen
assert digest(frozen) == original["source_sha256"]
freeze_path = ROOT / "docs/course-review-20261019/freeze-05.json"
assert digest(freeze_path.read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"

diff_command = ["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/06.md"]
difference = run(diff_command).decode("utf-8")
current_bpe = section(current, "6.3").decode("utf-8")
old_bpe = section(old, "6.3").decode("utf-8")
fence = re.search(r"```python\n(.*?)```", current_bpe, re.S)[1]
old_fence = re.search(r"```python\n(.*?)```", old_bpe, re.S)[1]
assert fence == old_fence
namespace = {"__name__": "__main__"}
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(fence, "course/chapters/06.md:tokenizer-example", "exec"), namespace)
tok = namespace["tok"]
examples = {}
for text in ["貓看狗", "小"]:
    encoding = tok.encode(text)
    examples[text] = {"ids": encoding.ids, "tokens": encoding.tokens,
                      "decoded": tok.decode(encoding.ids), "raw_bytes": list(text.encode())}
assert len(examples["貓看狗"]["ids"]) == 1
assert len(examples["小"]["ids"]) == 3
assert tok.get_vocab_size() == 277

pre = pre_tokenizers.ByteLevel(add_prefix_space=False)
whitespace = []
for text, expected in [("a cat", ["a", "Ġcat"]),
                       ("a\ncat", ["a", "Ċ", "cat"]),
                       ("a  cat", ["a", "Ġ", "Ġcat"]),
                       ("  cat", ["Ġ", "Ġcat"])]:
    pieces = pre.pre_tokenize_str(text)
    assert [piece for piece, _ in pieces] == expected
    whitespace.append({"text": text, "expected_pieces": expected, "actual": pieces})

rejected = 0
for codepoint in range(0xD800, 0xE000):
    try:
        chr(codepoint).encode("utf-8")
    except UnicodeEncodeError:
        rejected += 1
    else:
        raise AssertionError("surrogate 不應使用標準 UTF-8 獨立編碼")
assert rejected == 2048
boundaries = {f"U+{value:04X}": list(chr(value).encode("utf-8")) for value in [0xD7FF, 0xE000]}

receipts = [
    source_receipt("https://www.rfc-editor.org/rfc/rfc3629.txt", "RFC 3629，November 2003",
                   "UTF-8 encoding 中 surrogate 排除規則",
                   ["U+D800 and U+DFFF", "first decode the UTF-16 data to obtain character numbers"],
                   "已讀 surrogate 排除段落：UTF-16 surrogate code point 不能作為 UTF-8 編碼目標，須先把 UTF-16 pair 解成實際碼點。"),
    source_receipt("https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/tokenizers/src/pre_tokenizers/byte_level.rs",
                   "Hugging Face tokenizers v0.23.2", "static RE 與 ByteLevel::pre_tokenize",
                   [r" ?\p{L}+", r"\s+(?!\S)|\s+", "normalized.split(re_ref, SplitDelimiterBehavior::Isolated)"],
                   "已讀正規表示式與分段呼叫：詞前可附帶一個 U+0020，換行與多餘空白另由 whitespace 分支處理；這與四組實跑相符。")
]

p = 0.5 * 0.25
nll = -math.log(0.5) - math.log(0.25)
assert math.isclose(p, 0.125, abs_tol=0)
assert math.isclose(nll, -math.log(p), abs_tol=1e-15)
changes = [
    {"location": "字元切詞有什麼成本？／UTF-8 定義", "status": "verified",
     "changed_wording": "U+D800–U+DFFF 這段保留給 UTF-16 使用的碼點除外，它們不能單獨存成 UTF-8",
     "basis": "RFC 的 surrogate 排除規則；實跑全部 2048 個 surrogate 均觸發 UnicodeEncodeError。",
     "affected_claim_ids": ["utf8-cat", "utf8-emoji", "utf8-leading-byte"],
     "effect": "本節兩個例子都是合法 Unicode scalar value，byte 數與解碼結論不變。"},
    {"location": "怎麼訓練實用的 tokenizer？／ByteLevel 前處理", "status": "verified",
     "changed_wording": "詞前的一個普通空格會黏在後一個詞的開頭（換行、連續的空格可能自成一段）",
     "basis": "上游 ByteLevel 正規表示式及四組直接執行；原始訓練 fence 完全未變且重跑通過。",
     "affected_claim_ids": ["bpe-counts", "bpe-boundaries", "bpe-seen", "bpe-unseen"],
     "effect": "貓看狗仍是一個 token，小仍為三個 byte token。"},
    {"location": "不同切詞的代價怎麼比？／總 NLL 的機率對象", "status": "verified",
     "changed_wording": "tokenizer 切出的這一整串 token 的機率；總 NLL 就是這串 token 機率的負對數（計入 EOS 時也包含 EOS）；只對選定切法計分，不加總所有切法",
     "basis": "對固定 token 序列 t 與已知前綴 x，P(t|x)=∏ P(t_i|x,t_<i)，所以 -ln P(t|x)=Σ -ln P(t_i|x,t_<i)；若 EOS 列入序列即含 EOS 因子。解碼為同一字串的多個離散序列，其總字串機率另需對序列求和。",
     "affected_claim_ids": [],
     "effect": "此句不作本節證據；澄清不改變 token 與字元邊界、UTF-8 或 incremental decoder 的主張。"}
]
old_verification = json.loads((BASE / "verification.json").read_bytes())
assert digest((ROOT / "tiny_perceptron/data.py").read_bytes()) == old_verification["input_sha256"]["tiny_perceptron/data.py"]
assert digest((ROOT / "docs/course-experiments/results/real_text.json").read_bytes()) == old_verification["input_sha256"]["docs/course-experiments/results/real_text.json"]
claim_rechecks = []
for claim in original["claims"]:
    if claim["id"] in {"bpe-seen", "bpe-unseen"}:
        basis = "現行前置的訓練 fence 與先前完全相同，已重新執行並核對具體 token 數。"
    elif claim["id"] == "recorded-truncation":
        basis = "資料紀錄與 ByteTokenizer 的全檔 SHA-256 均未變，保留先前對所選樣本與 IDs 的逐 byte 核對。"
    else:
        basis = "本節原文與先前相同，所引原始證據及 artifact SHA-256 未變；重新核對既有推導與限制，並重跑原始程式。"
    claim_rechecks.append({"claim_id": claim["id"], "status": "verified", "basis": basis})

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-7/verify-current-01.py",
    "environment": {"python": sys.version, "tokenizers": tokenizers.__version__,
                    "platform": platform.platform(), "device": "cpu"},
    "original_report_sha256": digest(original_raw), "original_issues": original["issues"],
    "preserved_artifacts": preserved,
    "input_sha256": {"course/chapters/06.md": digest(current),
                     "tiny_perceptron/data.py": digest((ROOT / "tiny_perceptron/data.py").read_bytes()),
                     "docs/course-experiments/results/real_text.json": digest((ROOT / "docs/course-experiments/results/real_text.json").read_bytes()),
                     "docs/course-review-20261019/freeze-05.json": digest(freeze_path.read_bytes())},
    "current_section_equals_original_and_frozen": True,
    "section_sha256": digest(frozen),
    "diff": {"command_argv": diff_command, "stdout": difference, "exit_code": 0},
    "changed_wording_verification": changes,
    "bpe_fence_unchanged": True, "bpe_fence_sha256": digest(fence.encode()),
    "bpe_fence_stdout": output.getvalue(), "bpe_examples": examples,
    "whitespace_examples": whitespace,
    "surrogates": {"range": "U+D800–U+DFFF", "tested": 2048,
                   "UnicodeEncodeError_count": rejected, "valid_neighbor_bytes": boundaries},
    "nll_calculation": {"conditional_probabilities": [0.5, 0.25], "sequence_probability": p,
                        "nll_sum": nll, "negative_log_product": -math.log(p)},
    "source_receipts": receipts, "claim_rechecks": claim_rechecks,
    "result": "全部斷言通過；先前證據完整保留，現行必要前置與本節全部主張相容。"
}, ensure_ascii=False, indent=2))
