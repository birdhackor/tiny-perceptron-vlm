## 5.15 一次比較略好就有進步嗎？

方法A三次得0.70、0.72、0.69，B得0.71、0.71、0.73，同次差值+0.01、-0.01、+0.04。只比各自最高值，會遮住第二次B反而較差；先保留全部結果，再問改善是否穩定。

初始化、抽batch、生成可能用亂數。種子是重現起點的編號，不是品質排名；下面只換初始化種子，觀察輸入特徵表。程式中的 `model.embedding.weight` 就是這張數字表，每個文字 ID 對應一排可調特徵；這次用標準差概括整張表初值的起伏。

```python
import torch
from tiny_perceptron.model import TinyLM, ModelConfig

for seed in [1, 2, 3]:
    torch.manual_seed(seed)
    model = TinyLM(ModelConfig(width=8))
    print("種子", seed, "輸入特徵表初值標準差", model.embedding.weight.std().item())
torch.manual_seed(1)
a = TinyLM(ModelConfig(width=8))
torch.manual_seed(1)
b = TinyLM(ModelConfig(width=8))
print("同種子重建輸入特徵表相同", torch.equal(a.embedding.weight, b.embedding.weight))
```

輸入種子1、2、3，架構不動。`torch.manual_seed` 重設PyTorch的隨機起始狀態，隨後所有亂數初始化受它控制；三個標準差一般都接近1，卻不完全相同。標準差含義見 [3.5](03.md#3.5)。最後兩次建模都在之前重設seed1，所以輸入特徵表應逐格相同，列印True。這段觀察初始化，不是品質測量。

假設實際測得方法A三個成績0.70、0.72、0.69，B為0.71、0.71、0.73；同種子差是+0.01、-0.01、+0.04。直接報B最高0.73比A最高0.72高，看不見第二次B反而較差。保存全部值與範圍，讓人看出波動與平均趨勢，才知道「略好」有多穩定。

真比較可讓兩方法用同組種子，保存每次品質與配對差值。相同種子也不必有相同batch：架構消耗亂數次數不同，後面抽樣可能錯開，必要時分開模型和資料亂數。少數重複只給局部波動線索，題數、家族與嘗試數也影響結論。

練習只刪除建立b之前第二次 `torch.manual_seed(1)`，先預測a建立時已經消耗亂數，b會接着往下抽，最後比較通常False，再執行核對。再次設seed纔會從同起點重建；保留同一個整數變量名稱，並不會自動讓隨機狀態回頭。

<details>
<summary>補充：實作約定與原始紀錄</summary>

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

