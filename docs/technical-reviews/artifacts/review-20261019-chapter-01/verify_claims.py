"""核對導言的概念來源、章內操作與凍結版本，僅保存小段外部摘錄。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return result.stdout, command


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, value):
        if not self.hidden and value.strip():
            self.parts.append(value.strip())


def section(raw, title):
    match = re.search(r"^## \S+ " + re.escape(title) + r"\n", raw, re.M)
    assert match, "找不到指定標題"
    rest = raw[match.end():]
    return re.split(r"^## ", rest, maxsplit=1, flags=re.M)[0]


def run_first_block(raw, title):
    body = section(raw, title)
    code = re.findall(r"```python\n(.*?)```", body, re.S)[0]
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "<教材程式區塊>", "exec"), namespace)
    return namespace, {"title": title, "code_sha256": digest(code.encode()), "stdout": output.getvalue()}


def main():
    accessed = datetime.now(timezone.utc).isoformat()
    revision = "119a6930b670bced5800b6b03ec4b8cb6b8ff4ec"
    rnn_url = f"https://gist.githubusercontent.com/karpathy/d4dee566867f8291f086/raw/{revision}/min-char-rnn.py"
    rnn_raw, rnn_command = fetch(rnn_url)
    rnn = rnn_raw.decode()
    assert "def lossFun(inputs, targets, hprev):" in rnn
    assert "def sample(h, seed_ix, n):" in rnn
    rnn_selected = [
        "    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars",
        "    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars",
        "    ix = np.random.choice(range(vocab_size), p=p.ravel())",
        "    x[ix] = 1",
        "    ixes.append(ix)",
    ]
    for excerpt in rnn_selected:
        assert excerpt in rnn, "原始碼摘錄與來源不符"

    ime_url = "https://learn.microsoft.com/en-us/globalization/input/traditional-chinese-ime"
    ime_raw, ime_command = fetch(ime_url)
    parser = VisibleText()
    parser.feed(ime_raw.decode())
    ime = "\n".join(parser.parts)
    ime_excerpt = "After you select a character, the associate phrase window might offer text candidates related to what you typed."
    assert ime_excerpt in ime, "官方文件摘錄與來源不符"
    assert "2024-06-20" in ime
    excerpts = {
        "accessed_at_utc": accessed,
        "sources": [
            {
                "id": "rnn",
                "url": rnn_url,
                "version": revision,
                "original_sha256": digest(rnn_raw),
                "retrieval_command": rnn_command,
                "locator": "lossFun 的前向計算與 sample 的抽樣迴圈",
                "excerpts": rnn_selected,
                "inspection": "已讀完整原始碼；前向計算以輸入字與歷史狀態算下一字分數，sample 將抽出的 ID 設為下一輪輸入。這個來源採 RNN，不證明接字表能讀長前文。",
            },
            {
                "id": "ime",
                "url": ime_url,
                "version": "頁面 Last updated on 2024-06-20；實際 HTML 以 original_sha256 識別",
                "original_sha256": digest(ime_raw),
                "retrieval_command": ime_command,
                "locator": "Use the Bopomofo IME；associate phrase window",
                "excerpts": [ime_excerpt],
                "inspection": "已讀官方文件的輸入流程、聯想詞候選與啟用設定說明；僅支持存在此類候選功能，未聲稱某個輸入法實際列出貓後的看或吃。",
            },
        ],
    }
    (ARTIFACTS / "source-excerpts.json").write_text(json.dumps(excerpts, ensure_ascii=False, indent=2) + "\n")

    frozen_path = ROOT / "docs/course-review-20261019/freeze-01/sources/chapter-01.md"
    freeze_path = ROOT / "docs/course-review-20261019/freeze-01.json"
    frozen = frozen_path.read_bytes()
    assert digest(frozen) == "b5c6dc1958e67315b9c1d7e5cd8948d8abea62ac3dd7fa79bbadea573d8fde98"
    assert digest(freeze_path.read_bytes()) == "71bd28f633f6b1ae3fce6e6279239f261893b42055e70974b142d25b9a45b5c0"
    chapter_path = ROOT / "course/chapters/01.md"
    chapter = chapter_path.read_text()
    introduction = re.split(r"^## ", chapter, maxsplit=1, flags=re.M)[0]
    assert introduction.encode() == frozen
    assert "```python" not in frozen.decode()
    assert not re.findall(r"!\[[^\]]*\]\([^)]+\)", frozen.decode())

    encoded, encode_run = run_first_block(chapter, "文字怎麼變成可還原的編號？")
    assert encoded["restored"] == "貓看狗，狗看貓。"
    assert encoded["ids"] == [3, 2, 1, 4, 1, 2, 3, 0]
    shifted, shift_run = run_first_block(chapter, "「猜下一字」該配哪個答案？")
    pairs = list(zip(shifted["x"], shifted["y"], strict=True))
    assert pairs == [("貓", "看"), ("看", "狗"), ("狗", "。")]
    table, table_run = run_first_block(chapter, "怎樣把接字偏好存成可調的數字？")
    scores = table["scores"]
    assert scores.weight.shape == (3, 3)
    assert table["output"].tolist() == [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
    x, target = torch.tensor([0]), torch.tensor([1])
    before_weights = scores.weight.detach().clone()
    before_probabilities = scores(x).softmax(-1).detach().clone()
    before_loss = F.cross_entropy(scores(x), target)
    before_loss.backward()
    with torch.no_grad():
        scores.weight -= 0.1 * scores.weight.grad
    after_probabilities = scores(x).softmax(-1).detach().clone()
    after_loss = F.cross_entropy(scores(x), target)
    assert not torch.equal(before_weights, scores.weight)
    assert after_probabilities[0, 1] > before_probabilities[0, 1]
    assert after_loss < before_loss
    generated, generation_run = run_first_block(chapter, "下一字怎樣接成一段新文字？")
    assert len(generated["output"]) == 13
    assert generated["current"] == generated["chars"].index(generated["output"][-1])
    assert all(char in generated["chars"] for char in generated["output"])
    glossary = (chapter_path.parent / "../glossary.md").resolve()
    assert glossary.is_file()
    assert "字表（vocabulary" in glossary.read_text()

    local_paths = [chapter_path, ROOT / "tiny_perceptron/simple.py", frozen_path, freeze_path, glossary]
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-01/verify_claims.py",
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "accessed_at_utc": accessed,
        "source_retrievals": [{"id": item["id"], "command": item["retrieval_command"], "sha256": item["original_sha256"], "result": "HTTPS 讀取成功；摘錄逐字核對成功"} for item in excerpts["sources"]],
        "freeze": {"source_sha256": digest(frozen), "manifest_sha256": digest(freeze_path.read_bytes()), "live_introduction_equals_snapshot": True, "python_block_count": 0, "figure_count": 0},
        "local_files": {path.relative_to(ROOT).as_posix(): digest(path.read_bytes()) for path in local_paths},
        "executed_blocks": [encode_run, shift_run, table_run, generation_run],
        "next_character_pairs": pairs,
        "table_update": {"input": "貓", "target": "看", "learning_rate": 0.1, "updates": 1, "dtype": str(scores.weight.dtype), "before_probabilities": before_probabilities.tolist(), "after_probabilities": after_probabilities.tolist(), "before_loss": before_loss.item(), "after_loss": after_loss.item(), "other_rows_unchanged": torch.equal(before_weights[1:], scores.weight[1:])},
        "generated_text": "".join(generated["output"]),
        "generation_scope": "使用教材手動指定的機率表與固定 CPU generator；僅驗證循環接字，不證明訓練品質或跨環境相同抽樣。",
        "glossary_target": glossary.relative_to(ROOT).as_posix(),
        "result": "全部斷言通過；未執行長時間訓練。",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
