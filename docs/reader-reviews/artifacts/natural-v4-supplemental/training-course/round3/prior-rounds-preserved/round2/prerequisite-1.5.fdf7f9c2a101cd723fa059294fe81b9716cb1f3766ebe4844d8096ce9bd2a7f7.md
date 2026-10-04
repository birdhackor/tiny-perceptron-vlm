## 1.5 前一個字提供什麼資訊？

在「貓看狗。狗吃魚。貓看鳥。」裡，貓後面兩次都是「看」，狗後面則可能是句號或「吃」。如果只用 [1.4](#1.4) 的整體頻率，問貓後面接什麼與問狗後面接什麼，答案分配完全一樣。我們現在多看一件事：目前是哪個字，再從它曾經接過的字中估計下一字。這不是理解句子，只是比完全不看前文多了一條線索。

相鄰的兩個字叫 bigram。把所有候選字排成一張表，列是目前字，欄是下一字；例如「貓→看」出現一次，就在貓那列、看那欄加一。固定看貓那列，只用這列的總次數來除，就得到「已知目前是貓時」下一字的機率。這叫條件機率，寫 `p(看|貓)`，豎線讀作「在貓這個條件下」。表格軸與 tensor 的操作見 [W.3](../first-steps.md#W.3)，字與 ID 的約定見 [1.1](#1.1)。

若某組合沒出現過，直接計數會給它零機率；資料太少時，這未必合理。我們先在每格放 1，等於每個候選預留一份小支援，再加入真正的次數，這叫加一平滑（smoothing）。假設字表共 V 個字，某列實際觀察 N 次，分母便是 N+V。V 只是字表大小，不是 Python 內建名稱。

```python
import torch

text = "貓看狗。狗吃魚。貓看鳥。"
chars = sorted(set(text))
ids = [chars.index(char) for char in text]
counts = torch.ones(len(chars), len(chars))
for current, answer in zip(ids[:-1], ids[1:]):
    counts[current, answer] += 1
probability = counts / counts.sum(dim=1, keepdim=True)
row = chars.index("貓")
print(chars)
print(counts[row])
print(probability[row])
assert torch.allclose(probability.sum(dim=1), torch.ones(len(chars)))
```

`chars.index(char)` 找到字在字表中的位置，完成編碼。`torch.ones` 建立初值全為 1 的表；切片配對與 [1.3](#1.3) 相同。`sum(dim=1, keepdim=True)` 將每列的欄位相加，卻保留一欄的形狀，好讓每列除以自己的總數。這種讓一欄數字配到整列的運算叫廣播（broadcasting），這裡只需看它為每列提供同一個分母。

字表順序是 `['。','吃','狗','看','貓','魚','鳥']`，V=7。貓那列除「看」為 3 外，其餘都是 1：初始的 1 加上兩次觀察，總數為 9。因此「看」機率是 3/9，其餘各是 1/9；不是百分之百認為貓後面只會接看。`allclose` 檢查每列和在小數容忍範圍內都為 1。表中其他列各用自己的分母，不能把整張表除以一個總數。

下圖只展開貓那一列，橫向的七欄是候選下一字。橘框標出「看」這欄：初始1加兩次觀察得到3；下面同欄的3/9就是它的機率。其他欄各1/9，整列加起來才是1。這讓你看見平滑留下的支援，並不是把未見組合當成實際觀察過。

![貓這一列的計數和平滑機率](../figures/foundations_bigram_row.svg)

練習只將 `torch.ones(...)` 改成 `torch.full((len(chars), len(chars)), 0.1)`，表示每格初始支援為 0.1。先算貓那列總數應是 `2+7×0.1=2.7`，「看」應是 `2.1/2.7≈0.778`，其他各約 0.037，再執行核對。降低平滑讓已見組合更佔優勢，但仍給未見組合一點機會。

