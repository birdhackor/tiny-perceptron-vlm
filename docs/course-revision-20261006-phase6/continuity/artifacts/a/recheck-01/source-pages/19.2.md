## 19.2 為什麼成品選 MoE，卻仍保留 Dense 路線？

助理讀到「App」或一個商品向量時，每一層都要用前饋網路（FFN）轉換當前特徵。Dense 每層使用一組前饋網路；本章 MoE 每層備有四組 expert，由 router 選出兩組，按權重合併它們的輸出。expert 是可訓練的前饋網路，不是事先寫好「專門看圖」或「專門聽聲音」的程式。

成品採 MoE，是把[第15章](15.md)的路由機制放進同一位多模態助理，觀察多組規則怎麼共同學習。Dense 則提供同寬、同層數、同注意力與感知入口的另一條完整訓練路線，讓我們有具體對象理解成本和行為的取捨。

下面建立與公開成品**相同結構的隨機模型**，只數參數，不訓練也不生成答案。`parameters` 數整個助理儲存的參數；`language_parameters` 只數文字核心；最後一項是每個位置共享的文字參數加上所選 expert 的結構計數。

```python
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig

for architecture in ("moe", "dense"):
    config = SelftrainedConfig(
        vocab_size=550,
        width=256,
        layers=4,
        heads=4,
        kv_heads=2,
        ffn_hidden=512,
        experts=4,
        top_k=2,
        max_length=512,
        architecture=architecture,
    )
    model = LimitedAssistant(config)
    report = model.description()
    print(architecture, "全部", report["parameters"], "文字", report["language_parameters"])
    print("每位置的邏輯啟用文字參數", report["active_language_parameters_per_token"])
    print("嵌入與輸出共用同一張表", model.lm.embedding.weight is model.lm.output.weight)
```

兩次最後一行都會印出 `True`。這稱為 tied weights：輸入字元嵌入與輸出候選分數使用同一張權重表，數參數時只算一次。注意力、這張表及其他共同零件，也由各位 expert 共用。

| 相同結構的計數 | MoE：四組選兩組 | Dense：一組 |
| --- | ---: | ---: |
| 整個助理的參數 | 5,447,107 | 2,288,067 |
| 其中的文字核心參數 | 5,140,224 | 1,981,184 |
| 每位置的邏輯啟用文字參數 | 3,036,928 | 1,981,184 |
| 三個感知入口及接頭的參數合計 | 306,883 | 306,883 |

「邏輯啟用」用來說明只選部分 expert，沒有計入真正運算次數、分派花費或執行時記憶體。所有四組 expert 仍要存放；同一段輸入的不同位置也可能選不同組。本章 MoE 一個位置用兩組 FFN，Dense 用一組，所以這個配對不是同計算量或同記憶體的比較。

兩條路線的最終訓練目標與選定歷程也不同，不能把得分差直接歸因於 MoE。保留 Dense 的用途，是對照兩份真實成品並看清差異；要做隔離架構因素的實驗，還需要按[15.13](15.md#15.13)另行配對條件。上面的程式確認結構與計數，模型已學會什麼則由生成考題核對。

<details>
<summary>補充：550個編號從哪裡來？</summary>

本章沿用字元 tokenizer：普通文字逐字轉成編號，另外保留角色、模態和結束等控制編號，合計550個。普通字表只由訓練文字建立，並加入事先指定的ASCII格式字元與12個讀字字元；驗證和最後考卷沒有拿來擴充字表。推論時不在字表內的字會變成未知字元，不能把這份小字表當作完整中文詞彙。實作見[tokenizer.py](../../tiny_perceptron/selftrained/tokenizer.py)，公開配置與計數見[成品紀錄](../../docs/selftrained/results/v2-final-public-results.json)。

</details>

