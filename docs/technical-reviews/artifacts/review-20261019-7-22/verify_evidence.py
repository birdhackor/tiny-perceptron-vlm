"""核對標準序列的配對、Python 語法及外部原始來源；全文只留在記憶體。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


class TextReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag in ("p", "section", "h1", "h2", "h3", "h4", "h5", "div", "li", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1
        if tag in ("p", "section", "h1", "h2", "h3", "h4", "h5", "div", "li", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def run_example(code, tokens):
    tree = ast.parse(code)
    tree.body[0].value = ast.parse(repr(tokens), mode="eval").body
    ast.fix_missing_locations(tree)
    namespace = {}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(tree, "frozen-example", "exec"), namespace)
    rows = [{"module": 0, "given_prefix": tokens[:2], "target": tokens[2]}]
    for depth in (1, 2):
        rows.append({"module": depth, "given_index": 1 + depth,
                     "given": tokens[1 + depth], "target_index": 2 + depth,
                     "target": tokens[2 + depth]})
    return {"tokens": tokens, "t": 1, "rows": rows, "stdout": stream.getvalue()}


def fetch(url, excerpts, locators, inspection):
    command = ["curl", "-fLsS", "--max-time", "45", url]
    result = subprocess.run(command, capture_output=True, check=True)
    parser = TextReader()
    parser.feed(result.stdout.decode("utf-8"))
    original = "".join(parser.parts)
    flat = re.sub(r"\s+", " ", original)
    for excerpt in excerpts:
        assert excerpt in flat, "原始來源找不到預期摘錄"
    for locator in locators:
        assert locator in original, "原始來源找不到預期定位"
    return {
        "command_argv": command,
        "url": url,
        "accessed_at": datetime.now(UTC).isoformat(),
        "curl_exit_code": result.returncode,
        "curl_stderr": result.stderr.decode(),
        "download_bytes": len(result.stdout),
        "download_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "locators_checked": locators,
        "short_excerpts": excerpts,
        "inspection_note": inspection,
        "full_text_saved": False,
    }


def main():
    root = Path(__file__).resolve().parents[4]
    directory = Path(__file__).resolve().parent
    freeze = root / "docs/course-review-20261019/freeze-01/sources/7.22.md"
    raw = freeze.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "7ab4f061cc08bc6c816df52aed93e069fa75683b36ab5c12423ed840be3d6c27"
    code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
    baseline = run_example(code, ["〈B〉", "雨", "停", "了", "〈E〉"])
    exercise = run_example(code, ["〈B〉", "雨", "又", "停", "〈E〉"])
    copied = json.loads((directory / "section-execution.json").read_text())
    assert baseline["stdout"] == copied["copied_outputs"]["stdout.txt"], "原始輸出與重跑結果不符"
    assert [row["target"] for row in baseline["rows"]] == ["停", "了", "〈E〉"], "原句配對不符"
    assert [row["target"] for row in exercise["rows"]] == ["又", "停", "〈E〉"], "練習配對不符"
    assert [row["given"] for row in exercise["rows"][1:]] == ["又", "停"], "練習前文不符"
    for case in (baseline, exercise):
        for previous, current in zip(case["rows"], case["rows"][1:]):
            assert current["given"] == previous["target"], "相鄰模組未對齊"
            assert current["target_index"] == current["given_index"] + 1, "目標位移不符"

    namespace = {}
    exec("value = 7  # raise RuntimeError('不應執行')\ntext = '#'\n", namespace)
    assert (namespace["value"], namespace["text"]) == (7, "#"), "註解語法與字串混淆"
    given, target = "停", "了"
    inputs, labels, valid = [0, 0], [0, 0], [False, False]
    inputs[:2], labels[:2], valid[:2] = [1, 2], [2, 3], [True, True]
    assert (given, target) == ("停", "了")
    assert (inputs, labels, valid) == ([1, 2], [2, 3], [True, True])

    sources = []
    sources.append(fetch(
        "https://arxiv.org/html/2412.19437v2",
        ["its embedding layer is shared with the main model.", "each token will predict one additional token."],
        ["MTP Modules.", "MTP Training Objective.", "MTP in Inference.", "4.2 Hyper-Parameters", "4.5.1 Ablation Studies for Multi-Token Prediction"],
        "已讀 Multi-Token Prediction 全節、Model Hyper-Parameters、MTP 消融段與 Multi-Token Prediction Evaluation。模組以先前深度特徵和偏移 token 的 embedding 經各自 RMSNorm、串接、投影及 Transformer，預測再後一項；輸出頭與 embedding 各自和主模型共用。D=1；推論可移除模組或用於 speculative decoding。消融的模型與語料規模不能證明本課小模型必然改善。"
    ))
    sources.append(fetch(
        "https://arxiv.org/html/2404.19737v1",
        ["a shared unembedding matrix"],
        ["2 Method", "Memory-efficient implementation", "Inference"],
        "已讀 Method 的 next-token 與 multi-token 目標、獨立分支公式及 Inference。多分支由同一前文的 trunk 表徵出發，分支各自處理再接共用 unembedding；一般 next-token 目標仍使用資料前文。這支持正文對平行分支與共用輸出矩陣的比較。"
    ))
    sources.append(fetch(
        "https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html",
        ["using the real target outputs as each next input"],
        ["Training the Model", "Teacher forcing", "Evaluation"],
        "已讀 Training the Model 的 teacher forcing 定義，並核對 DecoderRNN.forward 與 AttnDecoderRNN.forward：下一步輸入來自 target_tensor；Evaluation 改讀自己的預測。只引用術語與訓練／推論前文的差異，不外推此 RNN 範例的品質或收斂主張到 MTP。"
    ))
    sources.append(fetch(
        "https://docs.python.org/3.13/reference/lexical_analysis.html",
        ["Comments are ignored by the syntax."],
        ["Comments", "string literal"],
        "已讀 Comments：字串外的 # 開始註解，到該實體行末結束，語法忽略註解。教材中的 # 正在此條件內；字串內的 # 不屬於這項說明。"
    ))
    sources.append(fetch(
        "https://docs.python.org/3.13/reference/simple_stmts.html",
        ["the items are assigned, from left to right, to the corresponding targets."],
        ["Assignment statements", "target_list", "iterable"],
        "已讀 Assignment statements 的 expression list 先求值與 target list 分配規則。兩個名稱接兩個值，以及三個切片目標接三個值，皆為同一 unpacking assignment 語法；不是宣稱所有左側寫入在時間上原子發生。"
    ))
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-22/verify_evidence.py",
        "environment": {"python": sys.version, "device": "cpu", "platform": platform.platform(),
                        "curl": subprocess.run(["curl", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]},
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "examples": {"baseline": baseline, "exercise": exercise},
        "syntax_checks": {"comment_ignored": namespace["value"], "hash_in_string": namespace["text"],
                          "unpacked_pair": [given, target], "unpacked_slices": [inputs, labels, valid]},
        "original_sources": sources,
        "result": "原始輸出、練習配對、註解、unpacking 與五個原始來源的定位及短摘錄檢查皆通過。",
        "scope": "程式核對字串、整數索引與語法；外部下載核對定位與短摘錄。科學主張仍由審閱者讀原文後判斷，不以字串命中代替技術審閱。",
    }
    with (directory / "supplementary-execution.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["result"])
    print("原句目標：" + "／".join(row["target"] for row in baseline["rows"]))
    print("練習目標：" + "／".join(row["target"] for row in exercise["rows"]))


if __name__ == "__main__":
    main()
