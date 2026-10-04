## 18.14 蒸餾後再量化能省多少？

先選較小學生，讓它學教師，再把學生權重存成低位元，可以同時利用兩種省成本方式。但它們改動的東西不同：[學生縮小](18.md#18.2)減少需要保存的數字，蒸餾希望改善小學生品質；[量化](17.md#17.9)減少部分數字每格的位元，並可能增加還原誤差。只比較教師與最終量化學生，會看不出哪一步帶來品質變化。

實驗應保留四組：教師、同架構的普通SFT學生、蒸餾學生、量化蒸餾學生。前兩種學生架構相同，權重bytes通常相同，品質差異才主要反映學習訊號；後兩組可以隔離量化的影響。每組都在同一獨立資料上量正確率、格式與行為，再一起列成本，不能只有teacher與最終版兩個端點。

下面只檢查結構儲存，不執行任何蒸餾。教師架構寬16、兩層，學生寬8、一層；量化副本換掉學生的Linear，其他嵌入與正規化保留浮點。算原始權重數字與buffer，讓我們先知道這條組合路線在此模型能省多少。

```python
import copy
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.quantization import replace_linear_layers

teacher = TinyLM(ModelConfig(width=16, layers=2))
student = TinyLM(ModelConfig(width=8, layers=1))
quantized = replace_linear_layers(copy.deepcopy(student), bits=4)


def payload_bytes(model):
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors)


for name, model in (("教師結構", teacher), ("學生結構", student), ("量化學生結構", quantized)):
    print(name, "bytes", payload_bytes(model))
print("量化學生/教師比例", round(payload_bytes(quantized) / payload_bytes(teacher), 4))
```

輸出應是67840、24416、15680bytes，最後比例約0.2311。學生架構先減少參數，四位元轉換再縮到15680；它沒有把完整學生縮成八分之一，因為一些層沒量化、scale與bias仍需要儲存。`payload_bytes`同時算parameters和buffers，避免轉換後量化權重登記成buffer就從報告消失。對每個tensor，`numel()`數它有多少元素，`element_size()`給每個元素佔幾個bytes，兩者相乘才是這張表的儲存量；最後把各表相加。這些是數字payload，不是磁碟檔案或執行峰值。

三個模型都隨機初始化，名稱中的「結構」提醒你：此例只確認成本組合，沒有宣稱學生已學會教師。正式流程應先執行蒸餾、保存與驗證該學生，再對同一份權重做PTQ，最後重跑[品質與行為保留](17.md#17.15)。訓練過程的教師算力與生成資料成本也要另算，不能因為部署時只放學生就忽略它們。

現在可用[18.10](18.md#18.10)的實際寬32學生補上這四個階段。每版用共同十道屬性題、69個真值目標，最後一版直接從已訓練KL學生量化，沒有新增更新：

| 正式屬性版本 | 模型tensor bytes | 私有原檔bytes | 答對／10題 | 回答NLL |
| --- | ---: | ---: | ---: | ---: |
| 寬64教師 | 566,272 | 587,913 | 5 | 0.5058 |
| 寬32 CE學生 | 134,528 | 153,024 | 4 | 0.4393 |
| 寬32 CE+KL學生 | 134,528 | 153,099 | 4 | 0.5037 |
| 同一KL學生packed4 | 64,160 | 76,287 | 3 | 0.5960 |

架構縮小後，數值儲存先變教師的約23.76%；packed4再變同一學生的約47.69%，合計是教師的約11.33%。但最後一步又少答對一題：藍圓形的`color?`，KL學生答`blue`，量化後答`ble`。四版都EOS10/10、沒有非法控制ID，仍有品質退步。檔案大小除了數值還有格式、metadata與浮點檔的隨機狀態，上表沒有optimizer；`*-training.pt`另存訓練狀態，公開移除狀態後的匯出大小也要重新量。

MoE那條路同樣保留Dense CE、Dense KL與KL後packed4：後者數值64,160 bytes，最後52篇NLL由2.3890升至2.4057，仍低於CE的2.4601；三版原文續段匹配都0/52且仍重複。這個結果同時保留了較小儲存、少許分布收益與未解決的生成問題，沒有讓最後的壓縮比遮住它們。packed推論仍完整反量化到FP32，這組沒有量專用低位元kernel的速度或隔離執行RAM。

蒸餾損失、量化權重的平均絕對差（MAE：逐格比較還原數值與原值，取絕對差後平均）與最終答案品質是不同指標。若四位元讓行為明顯改變，可以先比較八位元，或另設保留敏感層浮點的實驗。本例只量化權重，沒有量化中間特徵，也沒有執行[17.12的特徵範圍校準](17.md#17.12)；若後來要量化中間特徵，才需另設計那條校準流程，不能只改本例bits就說完成。這些取捨要由測試回答支持，不能只因最終byte比最大教師小就宣布成功。

練習只把bits改成8。先預測學生原版仍24416、量化後比15680大，實跑應17120bytes；教師67840不變。這只改了第二條壓縮路線，不會自動改學生架構，也不會補回尚未執行的蒸餾訓練。
