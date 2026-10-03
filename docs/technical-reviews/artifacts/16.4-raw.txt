## 16.4 KV cache 怎麼縮小？

如果每個注意力頭都保存一份Key、Value，長對話會讓快取占用持續增加。可以保留多個Query頭，讓幾個Query頭共用同一組K、V嗎？前置是[多頭注意力](03.md#3.7)與[KV快取](16.md#16.3)：每個頭用一小組特徵做比對，K、V按每層、每頭、每個過去位置保存。多個Query頭共用較少的KV頭，叫Grouped-Query Attention（GQA，分組查詢注意力）。

例如模型寬度16，Query有四頭，每頭四項特徵。FP32是32位元浮點數，每個數占4 bytes。若KV也有四頭，三個位置、一段輸入，用FP32保存K和V的數字需 `2×4×3×4×4=384` bytes：第一個2是K與V各一份，最後一個4是每個FP32數字的bytes。若四個Query頭改共用一個KV頭，只有頭數由4變1，快取變成96 bytes。Query仍然各自計算，並不是把四個頭全部合成同一個查詢。

下圖的「頭0至3」是注意力頭編號，不是位置編號。左側每個Query讀自己的KV，右側四個Query都讀同一組；連線表示讀取關係，每組仍保存三個位置，96 bytes的計算條件與上述相同。

![四個Query頭保留，KV快取由四組減為一組](../figures/architecture_gqa_cache.svg)

```python
import torch
from tiny_perceptron.model import TinyLM, ModelConfig

ids = torch.tensor([[1, 2, 3]])
with torch.no_grad():
    for kv_heads in (4, 1):
        model = TinyLM(ModelConfig(width=16, heads=4, kv_heads=kv_heads)).eval()
        cache = model(ids)["cache"]
        size = sum(t.numel() * t.element_size() for pair in cache for t in pair)
        print("KV頭數", kv_heads, "K形狀", tuple(cache[0][0].shape), "bytes", size)
```

此例只有一層。`cache`中的每個pair是一層的K、V。`t.numel()`數出tensor的元素個數，`t.element_size()`給每元素占用的bytes，相乘就是該tensor的數字儲存量；迴圈再把各層的K、V兩份加總。K形狀分別是 `(1,4,3,4)`與 `(1,1,3,4)`，bytes為384與96。這個實作保存未擴張的KV；比對時讓Query讀共享內容，不把四份重複副本存進cache，否則儲存收益會消失。

一般分組設定要求Query頭數能被KV頭數整除，例如四個Query、兩個KV，每組有兩個Query；全部Query共用一個KV的情況也常稱Multi-Query Attention（MQA）。投影權重的形狀與共享方式會改變，不能隨意把訓練好的四頭KV重排成一頭，就宣稱原模型功能保持不變。

快取縮四倍也不代表整個模型記憶體縮四倍：權重、其他中間特徵仍然存在。它不保證實際耗時或答案品質改善，需要在相容架構上訓練、核對快取路徑與量測。此程式只是同一輸入的結構與儲存比較，兩個模型權重各自初始化，沒有做品質競賽。

正式模型寬度64、兩層、四個Query頭，每頭16項特徵。先把25個位置的提示一次讀入，這個階段叫[prefill](16.md#16.2)；產生的FP32快取，四個KV頭實測25,600 bytes，一個KV頭為6,400 bytes，正好四分之一。後者是MQA端點，報告沿用`gqa`作組別名；沒有另外訓練兩個KV頭的中間設定。

下面的代價把模型對每個標準答案ID所給的機率p換成 `-log(p)`，再沿實際計分的目標取平均；機率越高，代價越低，算法可先回顧[機率與猜錯代價](../first-steps.md#W.4)。EOS是表示回答結束的特殊ID。有效目標就是納入計分的下一個標準ID：本表只計助手回答與EOS，不把問題或補齊佔位算入。例如標準回答 `[7,8,EOS]`有三個目標，若三個正確ID各得到機率0.5，單項代價都約0.6931，三項相加除以3仍為0.6931。這個平均衡量各個答案位置的預測，完整答案匹配則要求整段生成ID都正確。

| 設定 | 總參數 | 100次續訓後驗證／最後檢查代價 | 完整答案匹配：驗證／最後檢查 |
| --- | ---: | --- | --- |
| 四Query、四KV | 141,568 | 0.97333／0.31474 | 2/5／6/10 |
| 四Query、一KV | 129,280 | 0.16176／0.11392 | 3/5／6/10 |

代價只計助手回答與EOS，兩側分母為36／69個有效目標；匹配則按5／10題的完整原始答案ID核對，最多生成32個新ID。兩組各更新100次、計11,278個訓練目標，都正常EOS結束。這輪一KV組的代價較低，最後檢查答對數卻相同，不能把兩種品質指標互換。

還要看改動：起點是[T.4](../training.md#T.4)的單頭SFT模型。SFT是監督式微調，用示範問題與正確回答繼續訓練模型。「可複製的表」指字表、位置表、FFN、正規化係數及投影矩陣等模型權重中形狀相同的部分，並非上面的結果表；但改為四Query頭已改變比對方式。一KV組的新K/V投影則以固定隨機起點重新初始化。[GQA原論文第2.1–2.2節](https://arxiv.org/pdf/2305.13245v3)採用KV投影平均後再訓練，我們沒有重現它的轉換配方，也沒有證明原模型功能原封不動。快取確實省bytes；短提示總生成中位數由四KV快取的27.915毫秒到一KV的27.105毫秒，是這個小形狀的一次觀察，不能推成長對話或所有後端的普遍加速。

PyTorch提供SDPA（Scaled Dot-Product Attention，縮放點積注意力）函式介面，負責以Q、K比對、轉成權重，再取回V。後端是執行這套計算的不同實作選項；例如math用一般張量運算組合，其他選項可用專門的GPU運算。支援範圍也要帶版本：[2.14的SDPA文件GQA段落](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)列出Flash、cuDNN、math及NVIDIA CUDA memory-efficient這些實作選項，但仍有頭數整除與輸入限制。本組GQA訓練／快取用手寫後端，下一個SDPA剖析只測四KV的MHA，沒有驗證上述所有GQA後端都可用。

練習只把ids延長成六個ID，先預測K的位置軸加倍、bytes變成768與192，比例仍四倍，再執行核對。你應能分辨「前文變長」與「KV頭數減少」分別改了哪個儲存維度。

