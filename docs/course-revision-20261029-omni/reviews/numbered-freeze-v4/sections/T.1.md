## T.1 先決定要拿哪些例子教模型

一筆屬性問答可以是「顏色=紅；形狀=圓；形狀？」→「圓」。同樣的材料若改問顏色，答案就要改成「紅」。先說清楚你要教的輸入、要求與答案，才知道資料是否適合。文字接續則不同：整篇文章本身提供下一個文字單位，沒有另外一份問答標籤。

先用三筆材料看分組：

| 題號 | 輸入 | 答案 | 家族 |
| --- | --- | --- | --- |
| A | `color=red;shape=square;pitch=low;shape?` | square | red:square:low |
| B | `color=red;shape=square;pitch=low;describe` | square | red:square:low |
| C | `color=blue;shape=circle;pitch=high;shape?` | circle | blue:circle:high |

家族是同一組屬性衍生出來的題目；切分資料時，同一家族整組放在同一邊。這套產生器（[scripts/prepare_data.py](../scripts/prepare_data.py) 的 `attributes-sft`）把 `describe` 定為回答形狀，所以 A、B 只是同一組屬性換問法。可以把 A、B 一起用來教，把 C 的整個家族留作新題；把 A 教過再拿 B 當新素材，就會高估能力。實際切分方法見[5.10](chapters/05.md#5.10)。

初次練習可用程式產生的小資料。需要公開的故事、對話，或圖片、聲音素材時，再查看資料包：

```bash
.venv/bin/python scripts/fetch_training_assets.py --list
.venv/bin/python scripts/fetch_training_assets.py --asset tinystories
```

`--list` 只讀專案裡的清單，列出名稱、筆數、大小與授權，不需要 Git LFS。第二行下載 TinyStories 起步包（357,896 bytes）。照 [W.1](first-steps.md#W.1) 的寫法 clone 下來時，專案裡的資料包只是 Git LFS 的指標檔；這時工具會先執行 `git lfs pull`，從網路取得實際檔案，電腦沒裝 Git LFS 就會報錯中止。所以先照 [W.1](first-steps.md#W.1) 安裝，並用 `git lfs version` 確認。取得檔案後，工具會自動核對大小與 SHA-256，再解包到 `data/training/`（`--output` 可改位置）。下載不更新模型。打開幾筆，檢查 `text` 或 `messages` 等欄位，再安排訓練、驗證與測試資料。每包的固定來源、授權和內容指紋見[資料說明](../assets/training/README.md)。

圖片的裁切、錄音的加噪版本，也要跟原始素材放在同一側。先核對圖像尺寸、聲音取樣率與任務答案，不能以為檔案能打開就適合模型。第 19 章的商品圖、字卡與語音需求另有自己的資料清單，第 20 章的資料見[第 20 章資料指引](../docs/natural-assistant/v4/DATA.md)；它們都和這些起步實驗分開記錄。

練習替另一組屬性寫兩個問法，標出答案與共同家族，再寫一組不同屬性的新題。先把「教過什麼」和「準備檢查什麼」分清楚。

