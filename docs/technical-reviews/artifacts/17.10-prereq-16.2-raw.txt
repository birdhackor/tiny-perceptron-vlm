## 16.2 讀提示和生成下一字有何差別？

輸入一段提示後，模型先等待一會才吐出第一個字，接著逐字續寫。這兩種等待來自不同工作：一開始要處理整段提示，後面每次只處理剛加入的token。前者稱prefill，後者稱decode。這裡token是文字切分後的一個生成單位，可能是字、詞片段或一個字的部分，[不一定是一個完整字](06.md#6.7)；程式的ID代表這些單位。若只用一個「每秒幾字」平均值，可能掩蓋長提示造成的首字等待，或每一步生成隨前文變長而變慢的情況。

前置是[注意力查詢與被查詢](03.md#3.3)、[取回內容](03.md#3.4)與[下一字分數](04.md#4.6)。Query是目前位置提出的查詢，Key供比對，Value是比對後取回的內容。prefill對提示的每個位置都有Query；decode只對剛加入的位置做新的Query，但仍要讀前面所有Key和Value。把已算好的K、V保存起來供後續使用，叫KV cache（快取）。

假設提示ID是 `[1,2,3,4]`，prefill產生四個位置的分數，其中最後一個位置用來選擇下一個token。為了觀察形狀，下面直接假定接著選到ID5，把它送入decode。此例不宣稱隨機模型真的會選5，也不測語言品質。`eval()`把模型切換成推理模式；`torch.no_grad()`暫停記錄訓練用的反向梯度，因為這裡只看向前形狀，不需要保存反向計算。兩者用途不同，可回顧[eval與梯度](05.md#5.17)。

```python
import torch
from tiny_perceptron.model import TinyLM, ModelConfig

model = TinyLM(ModelConfig(width=8)).eval()
prefix = torch.tensor([[1, 2, 3, 4]])
with torch.no_grad():
    prefill = model(prefix)
    decode = model(torch.tensor([[5]]), cache=prefill["cache"])
print("prefill分數", tuple(prefill["logits"].shape))
print("decode分數", tuple(decode["logits"].shape))
print("第一層K形狀", tuple(prefill["cache"][0][0].shape), tuple(decode["cache"][0][0].shape))
```

預設詞表264項，前兩行應為 `(1,4,264)` 與 `(1,1,264)`：一段提示的四個位置，對比一個新位置。`prefill["cache"]`存每層的一對K、V，`[0][0]`取第一層的K。其形狀從 `(1,1,4,8)`變成 `(1,1,5,8)`，四軸依序是一段輸入、一個注意力頭、前文長度、每頭八個特徵。新Query只有一格，保存的前文卻從四格增加到五格。

所以decode不是與prefill相同長度的向前計算，也不是只看新token、完全忽略前文。前文投影可以重用，注意力比對與加權取回仍須讀cache；生成越長，cache越大，每一步需要讀的內容也可能增加。下一節會核對這條快取路徑與重算整段是否輸出相同。

量服務體驗時，可以分開報TTFT（Time To First Token，從送出請求到第一個token的等待時間）與後續decode的tokens/s。TTFT可能包含文字切分、資料傳入及排隊，不等於本例的prefill呼叫時間；也應註明提示長度、輸出長度與同時處理的請求數，才能比較。

實測設備是NVIDIA L4 GPU，使用提示 `Once upon a time, a girl`；提示前加一個BOS（表示序列開始的特殊token ID），所以含BOS共25個輸入位置，固定產生12個新ID。這裡改用四頭MHA（Multi-Head Attention，多頭注意力）：[每個頭](03.md#3.7)使用自己一組Q、K、V尋找關聯，同一層並排做四組比對，與上面只看單頭形狀的小例子是不同設定。四頭MHA的prefill呼叫中位數2.421毫秒，整段重算路徑29.034毫秒、快取路徑27.915毫秒。後兩個數字都包含prefill，沒有單獨量「第一個token之後」的decode吞吐，也沒有文字切分、傳輸或排隊，所以不是服務TTFT。

這個提示不是屬性問答格式。EOS是表示序列結束的特殊token ID，正常生成選到它就停止。為了固定計算量，速度測試即使選到EOS也繼續跑滿12步，原始ID與可讀文字都留在報告；正常留出問答評估則遇EOS停止。不能把這段固定成本測試的文字當成語言品質，也不能把12除以總秒數標成純decode的tokens/s。

練習只把prefix改成六個ID，仍只送一個新ID進decode。先預測分數形狀 `(1,6,264)`與 `(1,1,264)`，K長度6與7，再執行核對。你應能指出哪些工作隨提示變長，哪些軸在每步生成時仍保持一格。

