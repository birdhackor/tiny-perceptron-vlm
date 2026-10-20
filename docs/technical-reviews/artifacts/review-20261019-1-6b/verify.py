import contextlib
import hashlib
import io
import json
import pathlib
import platform
import re

import torch
from torch import nn


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_default_device("cpu")
torch.manual_seed(0)
root = pathlib.Path(__file__).resolve().parents[4]
frozen = root / "docs/course-review-20261019/freeze-01/sources/1.6.md"
text = frozen.read_text()
code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
namespace = {}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(code, str(frozen), "exec"), namespace)
scores = namespace["scores"]
output = namespace["output"]
expected = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
assert torch.equal(scores.weight, expected), "手動填表結果不符"
assert torch.equal(output, expected[[0, 2]]), "查表結果不符"
assert list(output.shape) == [2, 3], "查表形狀不符"
assert namespace["inputs"].dtype == torch.int64, "索引型別不符"
assert isinstance(scores.weight, nn.Parameter), "權重未登記為參數"
assert scores.weight.requires_grad and scores.weight.is_leaf, "參數梯度狀態不符"
assert scores.weight.grad_fn is None and scores.weight.grad is None, "填表留下非預期求導狀態"
assert type(output.grad_fn).__name__ == "EmbeddingBackward0", "查表運算記錄不符"
before = scores.weight.detach().clone()
scores(namespace["inputs"])
assert torch.equal(scores.weight, before), "預設查表意外改動權重"
with torch.no_grad():
    untracked = scores(namespace["inputs"])
assert untracked.grad_fn is None and not untracked.requires_grad, "停用記錄時仍有運算圖"
assert torch.is_grad_enabled(), "離開區塊後梯度模式未恢復"
baseline = {
    "權重": scores.weight.detach().tolist(),
    "輸出": output.detach().tolist(),
    "形狀": list(output.shape),
    "索引型別": str(namespace["inputs"].dtype),
    "權重參數型別": type(scores.weight).__name__,
    "requires_grad": scores.weight.requires_grad,
    "grad_fn": type(output.grad_fn).__name__,
    "查表未更新": torch.equal(scores.weight, before),
    "原始標準輸出": captured.getvalue(),
}
with torch.no_grad():
    scores.weight[2, 0] = 3.0
exercise = scores(namespace["inputs"])
assert torch.equal(exercise, torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]])), "練習結果不符"
assert torch.equal(scores.weight[:2], before[:2]), "修改狗列意外影響其他列"

probe = nn.Embedding(3, 3)
with torch.no_grad():
    probe.weight.copy_(expected)
    probe.weight[0, 0] = -4.0
assert probe(torch.tensor([0])).tolist() == [[-4.0, 2.0, 0.0]], "負分數未原樣保留"
assert probe.weight[0].sum().item() == -2.0, "分數意外被正規化"
zero_target = torch.ones(2, 3)
pointer = zero_target.data_ptr()
returned = zero_target.zero_()
assert returned is zero_target and pointer == zero_target.data_ptr(), "清零未就地修改"
assert zero_target.count_nonzero().item() == 0, "清零未完成"

trainable = nn.Embedding(3, 3)
with torch.no_grad():
    trainable.weight.copy_(expected)
inputs = torch.tensor([0])
answers = torch.tensor([1])
logits = trainable(inputs)
loss = nn.functional.cross_entropy(logits, answers)
initial = trainable.weight.detach().clone()
loss.backward()
assert torch.equal(initial, trainable.weight), "求導意外改動權重"
gradient = trainable.weight.grad.detach().clone()
assert gradient[0, 1] < 0 and gradient[0, 0] > 0 and gradient[0, 2] > 0, "答案梯度方向不符"
assert torch.equal(gradient[1:], torch.zeros_like(gradient[1:])), "未使用列出現梯度"
with torch.no_grad():
    trainable.weight.add_(gradient, alpha=-0.1)
after_loss = nn.functional.cross_entropy(trainable(inputs), answers)
assert after_loss.item() < loss.item(), "這次微小更新未降低代價"
assert inputs.tolist() == [0] and answers.tolist() == [1], "參數更新改動了字的索引"

previous_path = root / "course/chapters/01.md"
chapter = previous_path.read_text()
previous = re.search(r"^## 1\.5 .*?(?=^## |\Z)", chapter, re.M | re.S).group()
material = re.search(r'^text = "(.*?)"$', previous, re.M).group(1)
vocabulary = sorted(set(material))
assert len(vocabulary) == 7, "前一個接字表的字表大小不符"
assert scores.weight.numel() == 3 * 3 == 9, "示範參數格數不符"
assert len(vocabulary) ** 2 == 49, "較大字表格數不符"
assert max([0, 2, 0]) == 2 and [0, 2, 0].index(2) == 1, "候選分數排序不符"

contexts = ["顏色=", "形狀="]
extended_vocabulary = sorted(set("".join(contexts)))
lookup = {char: index for index, char in enumerate(extended_vocabulary)}
last_ids = torch.tensor([lookup[context[-1]] for context in contexts])
extended = nn.Embedding(len(extended_vocabulary), len(extended_vocabulary))
extended_output = extended(last_ids)
assert last_ids[0] == last_ids[1], "相同末字未使用相同索引"
assert torch.equal(extended_output[0], extended_output[1]), "相同末字未取同列"

next_path = root / "course/chapters/02.md"
next_text = next_path.read_text()
next_section = re.search(r"^## 2\.1 .*?(?=^## |\Z)", next_text, re.M | re.S).group()
next_code = re.search(r"```python\n(.*?)```", next_section, re.S).group(1)
next_namespace = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(next_code, str(next_path), "exec"), next_namespace)
assert isinstance(next_namespace["embedding"], nn.Embedding), "後續教材未使用同一查表介面"
assert next_namespace["x"].tolist() == [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]], "中間特徵查表不符"
assert 2 * next_namespace["x"][0, 0, 0].item() + next_namespace["x"][0, 0, 2].item() == 2.0, "後續手算候選分數不符"

print(json.dumps({
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "device": "cpu", "dtype": str(scores.weight.dtype), "seed": "0"},
    "原始程式": baseline,
    "練習輸出": exercise.detach().tolist(),
    "負分數與列總和": {"分數": probe(torch.tensor([0])).detach().tolist(), "總和": probe.weight[0].sum().item()},
    "原地清零": {"同一物件": returned is zero_target, "同一儲存位置": pointer == zero_target.data_ptr(), "結果": zero_target.tolist()},
    "依答案調整的機制探針": {"答案索引": answers.tolist(), "梯度": gradient.tolist(), "更新前代價": loss.item(), "更新後代價": after_loss.item(), "說明": "單次固定輸入的機制驗證，不支持一般訓練品質結論"},
    "格數": {"示範字表": 3, "示範格數": scores.weight.numel(), "前一接字表字表": vocabulary, "較大字表格數": len(vocabulary)**2},
    "同末字查表": {"前文": contexts, "擴充字表": extended_vocabulary, "末字索引": last_ids.tolist(), "結果相同": torch.equal(extended_output[0], extended_output[1]), "說明": "結構性的例子需用含等號的字表；三字示範字表本身不含等號"},
    "後續教材": {"查表類別": type(next_namespace["embedding"]).__name__, "中間特徵": next_namespace["x"].detach().tolist(), "候選狗手算分數": 2.0},
    "讀取檔案": [{"path": str(path.relative_to(root)), "sha256": digest(path)} for path in [frozen, previous_path, next_path]],
    "結果": "所有斷言成立"
}, ensure_ascii=False, indent=2))
