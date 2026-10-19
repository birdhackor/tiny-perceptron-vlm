"""使用 curl 閱讀原始來源，只保存短摘錄、版本與取得收據。"""

import hashlib
import json
import re
import subprocess
from datetime import datetime


def fetch(url):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", url]
    result = subprocess.run(command, capture_output=True, check=True)
    return result.stdout, command


def article(raw):
    match = re.search(r'"articleBody":\s*("(?:[^"\\]|\\.)*")', raw.decode())
    return json.loads(match[1])


examples_commit = "acc295dc7b90714f1bf47f06004fc19a7fe235c4"
tutorials_info, _ = fetch("https://api.github.com/repos/pytorch/tutorials/commits/main")
tutorials_commit = json.loads(tutorials_info)["sha"]
entries = [
    ("multinomial", "https://docs.pytorch.org/docs/2.14/generated/torch.multinomial.html", "PyTorch 2.14", "torch.multinomial：分布、權重與 generator 參數", "the values as weights), but must be non-negative, finite and have a non-zero sum.", "每列提供抽樣權重，向量輸入抽一個索引；generator 是可指定的偽隨機產生器。"),
    ("generator", "https://docs.pytorch.org/docs/2.14/generated/torch.Generator.html", "PyTorch 2.14", "Generator.manual_seed 與 device", "Sets the seed for generating random numbers. Returns a torch.Generator object.", "Generator 可獨立保存隨機狀態，manual_seed 設定種子並回傳該產生器。CPU 範例的 device 為 cpu。"),
    ("reproducibility", "https://docs.pytorch.org/docs/2.14/notes/randomness.html", "PyTorch 2.14；頁面更新日期 2026-05-14", "Reproducibility 開頭及 PyTorch random number generator", "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms.", "同環境並排除其他非決定性時可重現同一亂數序列；相同種子不保證不同版本與 CPU/GPU 完全一致。"),
    ("item", "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.item.html", "PyTorch 2.14", "Tensor.item API", "Returns the value of this tensor as a standard Python number.", "單元素 tensor 可轉成 Python number；本例抽樣回傳整數 tensor，所以 item 給 int。"),
    ("generation", f"https://raw.githubusercontent.com/pytorch/examples/{examples_commit}/word_language_model/generate.py", examples_commit, "生成迴圈的 is_transformer_model 分支", "input = torch.cat([input, word_tensor], 0)", "官方語言模型生成範例依當前輸入計算下一字權重、抽樣，再接回下一輪輸入；此步驟不需要真實後文作目標。"),
    ("ngram", f"https://raw.githubusercontent.com/pytorch/tutorials/{tutorials_commit}/beginner_source/nlp/word_embeddings_tutorial.py", tutorials_commit, "An Example: N-Gram Language Modeling", "P(w_i | w_{i-1}, w_{i-2}, \\dots, w_{i-n+1} )", "官方教學將 n-gram 分布條件化於最近 n−1 個字；bigram 因而只保留前一字。多字 context 提供原先沒有的可區分輸入，但未保證品質。"),
]
receipts = []
for identifier, url, version, locator, quote, inspected in entries:
    raw, command = fetch(url)
    text = article(raw) if "/docs/2.14/" in url else raw.decode()
    assert quote in text, identifier
    if identifier == "generation":
        assert "word_idx = torch.multinomial(word_weights, 1)[0]" in text
    if identifier == "multinomial":
        assert "generator (torch.Generator, optional)" in text
    receipts.append({"id": identifier, "url": url, "version": version, "retrieved_at": datetime.now().astimezone().isoformat(), "full_response_sha256": hashlib.sha256(raw).hexdigest(), "response_bytes": len(raw), "command_argv": command, "curl_exit_code": 0, "locator": locator, "excerpt": quote, "inspection_note": inspected, "checked_original": True})
print(json.dumps({"retention": "原始全文只經過記憶體，未寫入 repo；此檔僅保留短摘錄與取得收據。", "sources": receipts}, ensure_ascii=False, indent=2))
