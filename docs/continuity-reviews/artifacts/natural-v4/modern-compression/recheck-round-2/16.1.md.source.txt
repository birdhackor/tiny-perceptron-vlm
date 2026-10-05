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

表中的Self CPU表示運算自身花的時間，不包含子運算；CPU total還包括它呼叫的子運算。按Self排序有助找直接耗時的工作，但不能把全部total列加起來，因為父子可能重複計數。`# of Calls`是記錄期間呼叫次數：有些線性層在模型裡出現多次，五輪後次數自然比5大。`aten::`前綴是PyTorch底層運算名稱，不是模型層的名稱。

讀到陌生列時，先分清它在做哪種工作。`new_zeros`建立裝滿零的數字容器；`empty`、`new_empty`、`empty_like`準備儲存空間，尚未填好數值。`view`、`unsqueeze`與`as_strided`處理形狀或讀取同一份儲存的方式。它們也有執行成本，但排名高不能直接解讀成「注意力算太多」。這個寬度只有8的短例子，準備容器與呼叫運算的開銷就可能佔很大比例。

再找能連回已學內容的列：`index_select`用於字與位置的查表；`softmax`或`_softmax`把注意力分數轉成權重；`gelu`對應FFN中的啟動函數。`mm`、`matmul`與`addmm`則涉及矩陣乘法，可能出現在注意力、FFN或輸出層。`key_averages`會把不同部位的同名運算合在一起，所以這張表先幫我們定位「哪種工作值得再查」，還不能唯一指出哪一層。本節先練習判讀這幾類列，不要求猜出每個底層名稱的唯一來源。

最後一行的`numel()`回報參數表有幾個元素，`element_size()`回報每個元素占幾bytes，兩者相乘才是該表的數字bytes，再沿所有參數表相加。此設定6104個32位元浮點數（FP32）參數、每個4bytes，應得24416bytes；它與表中的微秒是不同單位。執行時還有中間特徵、生成用的cache；訓練更有梯度與優化器狀態，所以權重大小不能代表最高記憶體占用。profiler本身也會增加開銷，主要用於定位，總耗時應另以重複計時核對。

本章的GPU實測也先暖機，再同步量九次、取中位數。中位數先把時間由小到大排序：九次採樣取第5個；若留下偶數次採樣，則取中間兩個的平均，例如 `[1,2,3,20]` 的中位數是 `(2+3)/2=2.5`。一般平均數則把全部採樣相加再除以次數。訓練另略過前三步取更新中位數，不能把兩種時間混在同一欄。

後面以MiB列記憶體，1 MiB是2²⁰ bytes。量測還要說清楚開始時已占多少、過程新增多少；本章各項實驗的起點、軟體與記憶體範圍集中在[T.8的效率量測方法](../training.md#選讀效率實驗的量測條件)，需要重跑或追查數字時再閱讀。

練習保留同一個model，把ids改成 `torch.arange(1,13)[None]`，也就是一段十二個token。先預測分數形狀變成 `(1,12,264)`、權重bytes不變，再重跑暖機與profile兩個區塊。原輸入和新輸入各重複三輪，按相同運算名稱比較Self CPU時間與呼叫次數，不拿兩張表的第一名互比。想看的列未出現時，把`row_limit=5`改成20，找查表、softmax或矩陣乘法列。

如果某一列在較長輸入下多次都花更久，就值得再查它處理了多少位置；如果時間起伏比差異更大，結論就是這份短例子尚未顯示穩定差距。完成練習只需核對形狀、權重bytes，並用表中一種可辨認的工作說明這兩種結果之一。這樣得到的是下一步量測的線索，不是對整個模型瓶頸的定論。

