## 1.5 知道前一字，怎樣改變下一字的機率？

讀這段材料：「貓看狗。狗吃魚。貓看鳥。」其中貓後面兩次都是「看」。如果知道目前是貓，應比完全不讀前文更偏向看；問題是怎樣把這條關係記成可計算的表。

一對相鄰字叫 **bigram**。我們讓列代表目前字、欄代表下一字：「貓→看」每出現一次，貓列、看欄就加 1。再讓每列除以自己的總數，得到「目前是這個字時」的下一字機率。這叫**條件機率**；`p(看|貓)` 讀作「在目前是貓的條件下，下一字為看的機率」。

材料少時，沒看過的組合不一定不可能。我們先給每格 1，再加入觀察次數，這叫**加一平滑**。本例字表有七種字，因此貓列的總數是七份起始支援加兩次觀察，共 9。

下面用數值工具庫 PyTorch 保存與運算這張表；它裝數字的容器叫 tensor，這裡是一張七列七欄的表。

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

`chars.index(char)` 查字的 ID。`counts[current,answer]` 按目前字與答案選中一格，與上一節的全部字計數不同。字表為 `['。','吃','狗','看','貓','魚','鳥']`，貓列的「看」格是 1+2=3，其餘六格都是 1。

![貓這一列的計數和平滑機率](../figures/foundations_bigram_row.svg)

圖中橘格的 3/9≈0.333 就是 `p(看|貓)`，其餘各 1/9。它不等於 1，因為我們仍給未見候選留一點機會。`sum(dim=1,keepdim=True)` 把每列的欄加起來並保留一欄；除法便讓同一列每格共用自己的分母。每列機率加總為 1，整張表卻不是總和為 1：它包含七道不同的「目前字」問題。

練習把每格起始值改成 0.1，也就是 `torch.full((len(chars),len(chars)),0.1)`。貓列總數變 `2+7×0.1=2.7`，看的機率為 `2.1/2.7≈0.778`。先算再跑，看看為什麼支援減少會讓已見組合更佔優勢。

<details>
<summary>補充與重做</summary>

列、欄和 `dim` 見 [W.3](../first-steps.md#W.3)。讓一欄分母配到整列的運算叫廣播。若字表大小是 V、某列觀察次數是 N，加一平滑的分母是 N+V；不是把整張表除以全部觀察數。

</details>

