# 不訓練模型的局部數值核對

本輪以 Python 標準函式庫在 CPU 執行固定資料與手算公式，全部局部檢查通過。實際數值、輸入與範圍保存在 [numerical-demo.json](numerical-demo.json)。沒有模型實例、已訓練權重、`optimizer.step()`、GPU、成品生成或能力評測；這些結果不能當作 MTP 改善品質或加速的證據。

## 目標與遮罩

原始序列為：

`BOS / USER / Q / EOS-U / ASSISTANT / A / B / EOS-A`

只監督 A、B 和 assistant 的 EOS；問題仍存在於輸入。沿用現成 head-1 的有效預測起點，較遠目標只能停在同一個連續 assistant 目標區間。`None` 是研究輸出的忽略記號；正式 trainer 的忽略值為 `-100`。

| 未來距離 | 輸入位置 ASSISTANT 的目標 | 輸入位置 A 的目標 | 輸入位置 B 的目標 | 有效數 |
| --- | --- | --- | --- | ---: |
| 1 | A | B | EOS-A | 3 |
| 2 | B | EOS-A | 忽略 | 2 |
| 3 | EOS-A | 忽略 | 忽略 | 1 |
| 4 | 忽略 | 忽略 | 忽略 | 0 |

在尾端追加兩個 PAD 後，原來的所有目標與有效數不變，PAD 對應目標全部忽略。另將 `A / EOS-A` 與 `B / EOS-B` 兩份獨立文件裝填：head-1 只留下各自 EOS，距離 2 不留下跨文件目標。文件隔離與 loss mask 各自必要，這段沒有執行注意力。

以下是可離線重做的純資料核對。沒有修改 repository 的 encoder 或訓練程式：

```python
def targets(raw, span, valid, documents, distance):
    labels1 = [
        raw[i + 1]
        if (
            i + 1 < len(raw)
            and valid[i]
            and valid[i + 1]
            and span[i + 1] is not None
            and documents[i] == documents[i + 1]
        )
        else None
        for i in range(len(raw))
    ]
    result = [None] * len(raw)
    for t in range(len(raw)):
        if labels1[t] is None or t + distance >= len(raw):
            continue
        inside = range(t + 1, t + distance + 1)
        if all(
            valid[j]
            and documents[j] == documents[t]
            and span[j] == span[t + 1]
            for j in inside
        ):
            # labels1 已對齊下一 token，所以只再偏移 distance - 1。
            result[t] = labels1[t + distance - 1]
            assert result[t] == raw[t + distance]
    return result


raw = ["BOS", "USER", "Q", "EOS-U", "ASSISTANT", "A", "B", "EOS-A"]
span = [None, None, None, None, None, 0, 0, 0]
counts = []
for distance in (1, 2, 3, 4):
    base = targets(raw, span, [True] * 8, [0] * 8, distance)
    padded = targets(
        raw + ["PAD", "PAD"], span + [None, None],
        [True] * 8 + [False, False], [0] * 10, distance,
    )
    assert base == padded[:8] and padded[8:] == [None, None]
    counts.append(sum(v is not None for v in base))
assert counts == [3, 2, 1, 0]

packed = ["A", "EOS-A", "B", "EOS-B"]
doc = [0, 0, 1, 1]
assert targets(packed, doc, [True] * 4, doc, 1) == ["EOS-A", None, "EOS-B", None]
assert targets(packed, doc, [True] * 4, doc, 2) == [None] * 4
print(counts)
```

距離 4 無有效目標的情況只展示零分母，沒有把它加入下面的三組 loss。真正配置必須定義拒絕空 head 或如何記錄其零貢獻及係數；不能默默除以零，也不能在不同 batch 改變平均係數卻不記錄。

## 固定機率與 loss

每組只給正確目標的固定機率，不讓模型產生它們。交叉熵取 `-log(p)`：

| 距離 | 正確目標機率 | CE 加總 | 依該 head 有效數平均 |
| --- | --- | ---: | ---: |
| 1 | 0.5、0.25、0.8 | 2.302585 | 0.767528 |
| 2 | 0.4、0.6 | 1.427116 | 0.713558 |
| 3 | 0.7 | 0.356675 | 0.356675 |

示範約定為「正常目標的平均，加上 λ 乘兩個輔助 head 平均的平均」，取手寫 λ＝0.3：

`0.767528 + 0.3 × (0.713558 + 0.356675) / 2 = 0.928063`

全部六個有效配對等權重平均則為 0.681063，三個 head 等權重平均為 0.612587。數字不同，是因為每個目標得到的份量不同；不能在不同 reduction 間比較 loss 大小後宣稱某種方法學得更好。本例每個 head 的分母是有效數，與 DeepSeek Eq.24 的 `T` 分母不同；λ 也沒有經過本課配方選擇。

```python
import math

probabilities = [[0.5, 0.25, 0.8], [0.4, 0.6], [0.7]]
sums = [sum(-math.log(p) for p in row) for row in probabilities]
means = [s / len(row) for s, row in zip(sums, probabilities)]
print(means)
print(means[0] + 0.3 * (means[1] + means[2]) / 2)
print(sum(sums) / 6, sum(means) / 3)
```

## 共用量收到兩種梯度，不等於已經訓練

另取共用標量 `z=1`，兩個二元分類 head 的 logit 分別為 `z` 與 `-z`，答案都取類別 1。這只是可求導的兩個映射，不是 Transformer MTP。

正常 loss 對 z 的導數為 −0.268941；另一個 loss 的導數為 +0.731059，乘 0.3 後為 +0.219318。合計導數為 −0.049623847781；中心差分得到 −0.049623847775，誤差低於 `1e-8`。這說明共同量會收到多路貢獻，而不同目標也可能競爭；沒有更新任何參數。

```python
import math

z, weight = 1.0, 0.3
sigmoid = lambda x: 1 / (1 + math.exp(-x))
loss = lambda x: math.log1p(math.exp(-x)) + weight * math.log1p(math.exp(x))
analytic = (sigmoid(z) - 1) + weight * sigmoid(z)
epsilon = 1e-5
finite_difference = (loss(z + epsilon) - loss(z - epsilon)) / (2 * epsilon)
assert abs(analytic - finite_difference) < 1e-8
print(analytic, finite_difference)
```

## 草稿驗證與理論成本

手寫草稿 `[A, X, C]`，主模型對三個提議前綴的最高分答案分別手寫為 `[A, B, D]`。第一個 A 通過，第二個 X 被拒絕，C 也丟棄；本區塊輸出 `[A, B]`。這只驗證候選取捨規則，沒有神經前向或 KV cache 回退實作。

另用兩候選的 sampling 算例核對 Leviathan §2.3 的校正規則。目標分布 `p=[0.2,0.8]`、草稿 `q=[0.6,0.4]`；各候選接受率為 `[1/3,1]`，總拒絕機率 0.4，`max(0,p-q)` 正規化為 `[0,1]`。合成後的輸出邊際機率重新得到 `[0.2,0.8]`，差低於 `1e-12`。這是固定分布的精確算式，沒有量到模型接受率。

```python
p, q = [0.2, 0.8], [0.6, 0.4]
accept = [min(1, pi / qi) for pi, qi in zip(p, q)]
residual = [max(0, pi - qi) for pi, qi in zip(p, q)]
total = sum(residual)
residual = [v / total for v in residual]
reject = sum(qi * (1 - ai) for qi, ai in zip(q, accept))
marginal = [qi * ai + reject * ri for qi, ai, ri in zip(q, accept, residual)]
assert all(abs(a - b) < 1e-12 for a, b in zip(p, marginal))
print(marginal)
```

最後，沿用 Leviathan 的理想化假設：一個草稿 token、額外並行驗證不增加主模型時間、生成足夠長且不計尾端停止。在假設接受率 α＝0.8 時，預期每輪產生 `1+α=1.8` 個 token；若 draft 成本為單步主模型的 `c=0.1`，理論比值為 `1.8/1.1≈1.636`。若 `c=1`，同樣接受率卻只有 `1.8/2=0.9`，會較慢。這些是手寫假設的結果，不能標成模型實測 TPS 或部署速度。

由這組算例可以教會目標、遮罩、係數、共享梯度及驗證機制；未核驗的項目仍是 MTP 神經實作、訓練效果、真實接受率、cache 回退及硬體效能。教材要引用成熟來源的用途，不需要為這些局部機制新增 GPU 訓練。
