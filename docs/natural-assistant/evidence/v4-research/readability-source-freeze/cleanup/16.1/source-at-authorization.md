## 16.1 最慢或最佔空間的是哪裡？

模型跑完用了兩秒，究竟慢在注意力、特徵處理，還是程式第一次初始化？只看總時間，像只知道一餐準備了多久，卻不知道該改善哪個步驟。開始改效率以前，先量出成本的位置。本章標題的BKM指Best Known Method，也就是在明確條件下目前已知的做法；條件變了，應重新量測，不能把某個選項名稱當成速度保證。

前置是[一層模型的組成](04.md#4.5)與[矩陣使用量不等於耗時](15.md#15.11)。這裡使用已備妥的TinyLM小型文字模型：輸入token ID，經查表、注意力與FFN後，替每個位置輸出詞表分數。profiler（效能剖析器）會記錄各運算的時間與呼叫次數，幫我們找到下一個值得研究的瓶頸。

```python
import torch
from torch.profiler import profile, ProfilerActivity
from tiny_perceptron.model import TinyLM, ModelConfig

torch.set_num_threads(1)
model = TinyLM(ModelConfig(width=8)).eval()
ids = torch.tensor([[1, 2, 3]])
with torch.no_grad():
    for _ in range(3):
        model(ids)
    with profile(activities=[ProfilerActivity.CPU]) as p:
        for _ in range(5):
            result = model(ids)
print("分數形狀", tuple(result["logits"].shape))
print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=5))
print("權重bytes", sum(v.numel() * v.element_size() for v in model.parameters()))
```

輸入一列三個ID，預設詞表264項，第一行應為 `(1,3,264)`，可確認確實執行了整個模型而非空迴圈。先跑三次暖機，避免把第一次初始化當成穩定成本；再記錄五次向前計算，`key_averages`依運算名稱彙整。這裡只開CPU活動，不會替你量到GPU工作。

表中的Self CPU表示運算自身花的時間，不包含子運算；CPU total還包括它呼叫的子運算。按Self排序有助找直接耗時的工作，但不能把全部total列加起來，因為父子可能重複計數。`# of Calls`是記錄期間呼叫次數：有些線性層在模型裡出現多次，五輪後次數自然比5大。`aten::`前綴是PyTorch底層運算名稱，例如矩陣乘法與softmax；在此TinyLM中，`aten::index_select`對應字與位置的查表，`aten::softmax`或`aten::_softmax`對應注意力分數轉權重，`aten::gelu`則對應FFN中的啟動函數。這樣才能由名稱追到模型哪一段；表的排序可能隨本機量測改變。

最後一行的`numel()`回報參數表有幾個元素，`element_size()`回報每個元素占幾bytes，兩者相乘才是該表的數字bytes，再沿所有參數表相加。此設定6104個FP32參數、每個4bytes，應得24416bytes；它與表中的微秒是不同單位。執行時還有中間特徵、生成用的cache；訓練更有梯度與優化器狀態，所以權重大小不能代表最高記憶體占用。profiler本身也會增加開銷，主要用於定位，總耗時應另以重複計時核對。

本章的GPU實測也先暖機，再同步量九次、取中位數。中位數先把時間由小到大排序：九次採樣取第5個；若留下偶數次採樣，則取中間兩個的平均，例如 `[1,2,3,20]` 的中位數是 `(2+3)/2=2.5`。一般平均數則把全部採樣相加再除以次數。訓練另略過前三步取更新中位數，不能把兩種時間混在同一欄。

後面以MiB列記憶體，1 MiB是2²⁰ bytes。量測還要說清楚開始時已占多少、過程新增多少；本章各項實驗的起點、軟體與記憶體範圍集中在[T.8的效率量測方法](../training.md#T.8)，需要重跑或追查數字時再閱讀。

練習把ids改成 `torch.arange(1,13)[None]`，也就是一段十二個token。先預測分數形狀變成 `(1,12,264)`、權重bytes不變，再重跑比較前五項時間。短CPU例子可能排序不穩定，應多跑幾輪；你要找的是隨序列長度變大的成本，而不是記住某台機器某次的第一名。

