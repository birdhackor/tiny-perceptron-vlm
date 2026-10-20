"""以 curl 讀原始來源，只保存收據與支持主張的短摘錄。"""

import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime


def clean(raw):
    text = raw.decode("utf-8")
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", text, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))


entries = [
    {
        "id": "sequence_kd", "url": "https://arxiv.org/html/1606.07947v4",
        "version": "arXiv:1606.07947v4，2016-09-22，HTML 原文",
        "locator": "Sequence-Level Knowledge Distillation：Word-Level Knowledge Distillation 與 Sequence-Level Knowledge Distillation",
        "excerpt": "train the student network with cross-entropy on this new dataset.",
        "inspection_note": "讀取原文的 word-level 與 sequence-level 目標公式、模式近似及三步訓練流程。教師生成文字可直接作學生 NLL 的目標；這不等於取得各位置完整候選分布。文中實驗為 NMT，不把其效益數字外推至本頁玩具模型。",
    },
    {
        "id": "gsm8k", "url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/README.md",
        "version": "openai/grade-school-math commit 3101c7d5072418e28b9008a6636bde82a006892c，README.md",
        "locator": "Dataset Details；Socratic Dataset",
        "excerpt": "GSM8K consists of 8.5K high quality grade school math problems created by human problem writers.",
        "inspection_note": "讀取 Dataset Details 的題目與自然語言答案欄位，以及 Socratic Dataset 對 contractor-provided 解答步驟的說明。一般資料原始解答與模型生成的 Socratic subquestions、example_model_solutions 是分開的來源。課文對原始答案的來源敘述成立。",
    },
    {
        "id": "pytorch_ce", "url": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html",
        "version": "PyTorch 2.14 文件；本機執行 2.14.1",
        "locator": "CrossEntropyLoss：class-index target 公式與 ignore_index",
        "excerpt": "Specifies a target value that is ignored and does not contribute to the input gradient.",
        "inspection_note": "核對 class-index CE 為目標類別的負 log probability，遮罩位置不參與 loss；有效位置的 mean 分母不含被忽略項。來源不是候選庫摘要。",
    },
    {
        "id": "pytorch_source", "url": "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/loss.py",
        "version": "PyTorch git 5c4886908584029761b579af026dcfb627c84070，與本機 torch.version.git_version 相同",
        "locator": "CrossEntropyLoss class docstring：class indices 與 probability targets 的兩種公式",
        "excerpt": "This criterion computes the cross entropy loss between input logits",
        "inspection_note": "讀取 CrossEntropyLoss 原始碼中的兩種 target 公式及 ignore_index 規約。hard label 和 probability target 不是同一種輸入；可由同一組 hard labels 推得相同的 loss 函數，但單一輸出 token 無法唯一決定教師 softmax。",
    },
    {
        "id": "python_json", "url": "https://docs.python.org/3.13/library/json.html",
        "version": "Python 3.13 官方文件；本機 3.13.16",
        "locator": "Encoders and Decoders：dumps、JSONEncoder 的型別對照表與 ensure_ascii",
        "excerpt": "If ensure_ascii is false, all characters will be output as-is, except for the characters that must be escaped:",
        "inspection_note": "讀取 dumps 產生 JSON 字串、None 對應 null、ensure_ascii 控制非 ASCII 字元輸出的規約；實跑本頁程式另外核對中文與 null。",
    },
    {
        "id": "jsonlines", "url": "https://jsonlines.org/",
        "version": "JSON Lines 格式說明，讀取日現行網頁",
        "locator": "Each Line is a Valid JSON Value；Line Terminator",
        "excerpt": "Each Line is a Valid JSON Value",
        "inspection_note": "讀取 UTF-8、逐行合法 JSON value 及換行符要求。本頁每筆字典 dumps 成 JSON、各寫一行符合 JSONL；示範程式本身只有 print，沒有建立資料檔。",
    },
]
for entry in entries:
    command = ["curl", "-fsSL", entry["url"]]
    completed = subprocess.run(command, capture_output=True, check=True)
    normalized = clean(completed.stdout)
    if entry["excerpt"] not in normalized:
        raise AssertionError("原文找不到預定短摘錄：" + entry["id"])
    entry.update(command=command, exit_code=completed.returncode, accessed_on=datetime.now(UTC).date().isoformat(),
                 original_sha256=hashlib.sha256(completed.stdout).hexdigest(), original_bytes=len(completed.stdout),
                 checked_original=True)
print(json.dumps(entries, ensure_ascii=False, indent=2))
