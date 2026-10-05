## 19.12 完成整合後，怎樣回答「這個模型會什麼」？

「會看圖、會聽聲音、會用工具」還太籠統。最後交付時，應能指著同一個成品說：哪些圖片、哪些字、哪些聲音與哪種問題，在什麼未見材料上通過了哪個判準。總分只是一個摘要，每項都有自己的分母。

| 任務 | 必須同時檢查 | 需要保留的對照 |
| --- | --- | --- |
| 有限文字與指令 | 內容、範圍、格式、完整結束 | 新問法、改條件、資料不足 |
| 三類商品及位置 | 類別與所問物件／關係 | 原圖隔離、左右互換、移除／錯配圖 |
| 12字指定框短串 | 字元、整串、範圍、字序、換行 | 同圖換框、未見組合、字體／版面 |
| 三類真人語音意圖 | 聲音支持的意思、答覆、共享歷史 | 留出錄音、改述、移除／錯配聲音 |
| 計算器往返 | 請求參數、真執行、回填後答案 | 新數字／問法、工具關閉、回放返回值 |

這是一張待填的驗收矩陣，不替尚未完成的能力填分數。若來源沒有 speaker ID，則保留「未驗收新說話者」；若只測已知字體，就只聲明有限字卡。收斂範圍時先在驗證份決定，最後考卷不能用來挑最好看的模型或題目。

舊資料的短例示範分母怎樣從實際題目來：

```python
from collections import Counter

from tiny_perceptron.capstone import build_dataset

splits, _ = build_dataset()
counts = Counter(row["task"] for row in splits["test"])
for task, count in sorted(counts.items()):
    print("最後檢查", task, "分母", count)
print("合計", sum(counts.values()), "這只是考題數量，不是答對數量")
```

`Counter` 只數舊 test 的任務題數，合計90，沒有模型回答。每題還要核對實際輸入、生成與停止。所有題直接平均會讓題多的任務佔較大比重，不能讓它遮住讀字或語音未通過。

完成有限主線後，可以繼續擴大來源，也可以到第20章研究成熟 Qwen／Whisper 的應用。後者接手已有權重，不是這份共同 MoE 的接續訓練，也不替主線補能力分數。說清能力從哪裡來，才能知道下一步該改入口、資料還是回答。

練習用人工分數比較兩種平均：若一類答對12／12、另一類0／6，按題合計是12／18；若每類任務各佔一半，則是 `(1+0)/2`。說出兩者差別，並想想交付時為何要先約定各項權重。這些是假設分數，不是模型成績。

<details>
<summary>補充：舊合成成品如何界定能力</summary>

舊推薦 joint 成品只學幾種合成圖形、兩群純音、窄範圍句型與一個加法工具；它仍不能可靠辨識留出的形狀，也會讀錯計算器結果。舊任務沒有訓練或驗收一般照片理解、中文讀字與真人語音辨識，也沒有網路搜尋或通用安全與信心校準保證。它不替主文待填的新驗收矩陣提供分數。

先分清兩個判準。`action_correct` 要求首段正常產生 EOS，且完整文字等於 `expected_action`，包含 TOOL 的工具名、有順序的參數，以及 DIRECT／ASK 的內容。例如真值 `DIRECT:square`、模型 `DIRECT:circle`，雖然標記合法且正常結束，仍算錯。`end_to_end_correct` 還要求最後答案符合真值；工具題須先執行，再讀回並由模型回答，請求全對仍可能整題錯。

下面只摘同一個 joint 版本的四項任務，分母都是實際考題數；這四列不是完整90題總表：

| 舊任務 | 首段完整輸出吻合 | 整題正確 | 結果支持到哪裡 |
| --- | --- | --- | --- |
| 計算器完整迴圈 | 12／12 | 10／12 | 兩次工具算對、模型讀回答錯，例子見[19.7](19.md#19.7)。 |
| 原始圖片顏色 | 9／9 | 9／9 | 原題都為green，常數答案也能全對；不能單靠此欄說用到了圖片。 |
| 原始圖片形狀 | 0／9 | 0／9 | 綠色方形全部答circle；換圖後答對也不使原圖通過。 |
| 單獨音高 | 6／6 | 6／6 | low／high各3題，只辨純音，不能稱真人語音理解。 |

常數基線與資料隔離見[19.3](19.md#19.3)，換圖／換聲的成對檢查見[19.6](19.md#19.6)。少量模板拒絕也只支持本題庫的窄規則，不等於所有正常請求都不會誤拒。能力、速度和儲存是不同證據；快取省時不會改變這些能力判準。

推薦 joint 在最後考卷開封前已由驗證份選定，決定見[選擇紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-selection.json)。[完整90題紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint.json)與[原始資料](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/data.json)用同一個題目 `id` 連起來，可核對實際輸入、生成、結束及評分。最後考卷不再拿來重新選模型。

小 Dense 學生使用事先選定的 DPO 分支作教師，並非推薦 joint 的蒸餾版。學生的能力要另查，不能沿用 joint 的換圖／換聲成對結果當作已通過；量化同分也不保證逐 token 相同。具體反例見[19.10](19.md#19.10)，全部學生條件與逐題檔見[學生實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_student.json)。

</details>

