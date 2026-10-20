"""核對對話編碼、移位、內容邊界與 loss 遮罩；不進行訓練。"""

import ast
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tiny_perceptron.attention import attention_mask
from tiny_perceptron.data import IGNORE, ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum


def require(condition, message):
    if not condition:
        raise AssertionError(message)


torch.set_num_threads(1)
torch.manual_seed(0)
tok = ByteTokenizer()
messages = [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "2"}]
x, y = render_chat(messages, tok)
expected_x = [1, 3, 57, 51, 57, 69, 71, 2, 4, 58]
expected_y = [IGNORE] * 8 + [58, 2]
require(x.tolist() == expected_x and y.tolist() == expected_y, "對話輸入或標籤與逐項推導不符")
require(x.dtype == y.dtype == torch.int64, "token 與標籤應為整數")
default_x, default_y = render_chat(messages)
require(torch.equal(x, default_x) and torch.equal(y, default_y), "預設 tokenizer 的結果不符")

snapshot = ROOT / "docs/course-review-20261019/freeze-01/sources/7.1.md"
fences = snapshot.read_text(encoding="utf-8").split("```python\n")
excerpt = fences[2].split("```", 1)[0]
namespace = {"torch": torch, "IGNORE": IGNORE, "ByteTokenizer": ByteTokenizer}
exec(compile(excerpt, str(snapshot), "exec"), namespace)
require(ast.dump(ast.parse(excerpt)) == ast.dump(ast.parse(inspect.getsource(render_chat))), "教材節錄與目前函式不符")
excerpt_x, excerpt_y = namespace["render_chat"](messages, tok)
require(torch.equal(x, excerpt_x) and torch.equal(y, excerpt_y), "教材節錄執行結果不符")

variant = [{"role": "user", "content": "2+1=?"}, messages[1]]
vx, vy = render_chat(variant, tok)
changed = (x != vx).nonzero().flatten().tolist()
require(changed == [2] and vx[2].item() == 58 and torch.equal(y, vy), "只改問題時的差異不符")
corrected = [variant[0], {"role": "assistant", "content": str(2 + 1)}]
cx, cy = render_chat(corrected, tok)
require(cy[cy != IGNORE].tolist() == [59, 2], "正確算術答案的有效標籤不符")

invalid = [
    [{"role": "other", "content": "文字"}],
    [{"role": "assistant", "content": 2}],
    [{"role": "user", "content": "只有問題"}],
    [],
]
errors = []
for example in invalid:
    try:
        render_chat(example, tok)
    except ValueError as error:
        errors.append(str(error))
    else:
        raise AssertionError("不合法或無監督答案的資料應報錯")

turns = [
    {"role": "system", "content": "規則"},
    {"role": "user", "content": "<assistant>"},
    {"role": "assistant", "content": "甲"},
    {"role": "user", "content": "乙"},
    {"role": "assistant", "content": ""},
]
ids, targets = [tok.bos_id], [IGNORE]
role_ids = {"system": 7, "user": 3, "assistant": 4}
for message in turns:
    encoded = list(message["content"].encode("utf-8"))
    content = [byte + 8 for byte in encoded] + [2]
    ids.extend([role_ids[message["role"]], *content])
    targets.extend([IGNORE, *(content if message["role"] == "assistant" else [IGNORE] * len(content))])
mx, my = render_chat(turns, tok)
require(mx.tolist() == ids[:-1] and my.tolist() == targets[1:], "多輪角色或移位與獨立推導不符")
require(all(value >= 8 for value in tok.encode("<assistant>")), "內容文字不能產生角色 token")
require(len(tok.encode("甲")) == 3, "UTF-8 中文的 byte 數不符")

config = ModelConfig(width=16, heads=1, max_length=64)
model = TinyLM(config)
logits = model(x[None])["logits"]
logits.retain_grad()
per_position = F.cross_entropy(logits[0], y, ignore_index=IGNORE, reduction="none")
total, count = loss_sum(logits, y[None])
direct = F.cross_entropy(logits[0, -2:], y[-2:], reduction="sum")
require(logits.shape == (1, 10, 264), "每個位置都應產生詞表分數")
require(count.item() == 2 and torch.equal(total, direct), "loss 只應計入有效目標")
require(per_position[:8].eq(0).all().item(), "忽略位置的逐項 loss 應為零")
total.backward()
require(logits.grad[0, :8].eq(0).all().item(), "忽略位置的 logits 梯度應為零")
require(model.embedding.weight.grad[57].abs().sum().item() > 0, "問題仍應能透過注意力影響答案 loss")
allowed = attention_mask(torch.arange(10), torch.arange(10))[0, 0]
require(allowed[8, 2:7].all().item(), "答案預測位置應可讀取問題 token")
require(not allowed[2, 8].item(), "問題位置不能讀取未來的助手 token")
try:
    model.embedding(torch.tensor([-100]))
except IndexError as error:
    embedding_error = type(error).__name__
else:
    raise AssertionError("IGNORE 不應是合法 embedding ID")

result = {
    "執行環境": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "seed": "0"},
    "來源雜湊": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py"]},
    "問答": {"原始內容": messages, "X": x.tolist(), "Y": y.tolist(), "有效目標": y[y != IGNORE].tolist(), "dtype": str(x.dtype)},
    "教材函式": {"AST相等": True, "執行結果相等": True},
    "問題變體": {"X": vx.tolist(), "Y": vy.tolist(), "變更位置": changed},
    "正確算術變體": {"答案": "3", "X": cx.tolist(), "有效目標": cy[cy != IGNORE].tolist()},
    "無效輸入錯誤": errors,
    "多輪": {"messages": turns, "X": mx.tolist(), "Y": my.tolist(), "有效目標": my[my != IGNORE].tolist(), "內容角色字樣均為普通byte": True},
    "loss": {"logits_shape": list(logits.shape), "有效目標數": count.item(), "逐項loss": per_position.detach().tolist(), "loss總和": total.item(), "只取有效位置的loss總和": direct.item(), "忽略位置logits梯度最大值": logits.grad[0, :8].abs().max().item(), "問題token的embedding梯度絕對值總和": model.embedding.weight.grad[57].abs().sum().item(), "IGNORE查表錯誤類型": embedding_error},
    "注意力": {"assistant位置可讀問題": allowed[8, 2:7].tolist(), "問題位置可讀未來assistant": allowed[2, 8].item()},
    "結果": "全部檢查通過；僅核對格式與計算機制，未測訓練品質。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
