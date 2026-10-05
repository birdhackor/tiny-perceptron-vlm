## 19.3 同一題的不同版本，怎麼避免跑進兩份考卷？

同一張包的商品圖，縮小後做單物件題，再放到左格做關係題，會得到多筆練習。但若原圖先進訓練、其裁剪又進最後考卷，就不能稱最後素材全新。資料家族是同一原始素材及其衍生版本；先把家族分好，再製作題目，才能避免這種重複。

文字把原問題與改述放同一組；商品圖把原影像及所有配對、縮放綁在一起；字卡把字串、版面與增強家族一起處理；錄音把同一來源及裁剪綁在一起。訓練份用來更新，驗證份用來選設定，最後份在定版後核對。三種用途分開，不能把已看過的最後題重新稱為新考卷。

以下短例讀取舊合成資料的切分，保留同一機制：每份取家族集合，再用 `&` 查交集。

```python
from collections import Counter

from tiny_perceptron.capstone import build_dataset

splits, manifest = build_dataset(seed=42)
families = {name: {row["family"] for row in rows} for name, rows in splits.items()}
for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
    print(first, second, "重疊家族", len(families[first] & families[second]))
for name, rows in splits.items():
    print(name, "題數", len(rows), "任務", dict(Counter(row["task"] for row in rows)))
print("第一筆訓練問題", splits["train"][0]["user"])
audio_test = [row for row in splits["test"] if row["task"] == "audio"]
labels = Counter(row["audio"]["pitch"] for row in audio_test)
print("最後音高標籤", dict(labels))
print("完全不聽、固定答high的基準", labels["high"], "/", len(audio_test))
for task in ("image_color", "image_shape"):
    baseline = manifest["task_majority_baselines"]["test"][task]
    print(task, "固定答案基準", baseline["majority_label"], baseline["correct"], "/", baseline["count"])
```

前三行的重疊家族應為0。`Counter` 另數任務和答案分佈；純音最後題有 high、low 各3題，一律答 high 只能3/6。相反地，這份舊考卷的原圖全是綠色方形，所以固定答 green 或 square 都能9/9。這不是新資料，也不是模型得分，而是在提醒我們：零家族交集之外，還要檢查實際輸入與常數答案捷徑。

新商品圖先按原始 ID 分割，再合成左右場景；交換位置的成對題不能跨份。語音的 MInDS-14 schema 沒有 speaker ID，來源列不重複還不能宣稱新說話者測試。若另蒐集有 speaker／session 的錄音，再按那些群組分割。測試輸入也不應帶來源轉寫、意圖名或含標籤的檔名，否則模型可能沿旁邊的文字找到答案。

<details>
<summary>補充：舊資料版本與可查證紀錄</summary>

資料版本`capstone-small-world-v2`、固定種子42，共有552筆訓練、84筆驗證與90筆最後檢查。數字以同一組數字對為家族；圖片以顏色／形狀組合為家族，藍色圓形只進驗證、綠色方形只進最後檢查，該組合的位移、明暗和聯合題都一起走。訓練仍包含每一種基本顏色與形狀，因此考的是未見組合，不是新顏色或新形狀。短資料、缺資訊、拒絕與照抄按情境編號一起切。

數字家族`1,2`的六種衍生記錄只在最後考卷，不在訓練或驗證；它們包括`1+2等於多少？`、反向順序的`請算2加1。`、計算器關閉、概念說明與回填結果。因此成品試用的1加2案例能按一組預先留出的題核對，而不只是挑一筆訓練題展示。

純音題先按基頻家族切，再保持低／高音各自8個訓練家族、1個驗證家族、1個最後檢查家族，每個家族的三種頻率變體一起走。最後六題因此是3題low、3題high：完全不聽聲音、一律回答high，只會3／6。程式中「完全不聽、固定答high」那行算的就是這個人工固定基準，不是模型預測。若考卷只有high，固定答案就能全對；先看標籤分佈與基準，才知道高分是否真的需要聽聲音。

圖音聯合的18道最後題則包含9個low、9個high，文字問題都相同，仍需保持圖片、只換音訊，再依換入素材更新音高真值核對；單純「分數有變」不能證明聽對，詳見[12.10的換聲測試](12.md#12.10)。

還有一個容易漏掉的限制：原始圖片最後題全部是綠色方形，所以固定回答green的顏色基準與固定回答square的形狀基準都會9／9。這兩項就算模型全對，也不能單獨證明它看了圖片。上面末兩行讀取資料清單的固定答案基準；正式成品還要看[19.6](19.md#19.6)的換顏色、換形狀成對測試。

數字題使用訓練已出現的兩種問法，考新的數字家族；圖片與音訊則按上面的素材規則留出。本輪沒有另做陌生措辭考卷，因此不能從這些成績推論一般中文提問都能理解。若想擴充這項檢查，可回到[B.7的陌生問法失敗](0B.md#B.7)看需要另外準備什麼資料。

正式實驗沿用這份凍結資料；[完整資料清單](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/data.json)存了所有726筆原問答與素材生成規格，沒有要求讀者只靠輸入編號猜問題。各份資料的SHA-256指紋開頭依序是訓練`e0a419727952`、驗證`afedd4dc84ce`、最後檢查`aef4b7a876ff`，完整指紋在清單的`manifest.sha256`。跨階段與比較方法，訓練、驗證、最後檢查各自的指紋應分別保持不變；不是說三份不同用途的資料彼此相同。

本輪還在CPU重跑[實際輸入不交叉的檢查](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/tests/test_capstone.py)：使用真正的前文編號、圖片張量與聲學摘要的bytes計指紋，三對資料份之間的家族交集與實際輸入交集均為0。這是資料規則檢查，不是模型能力成績；它排除完全相同輸入跨份，沒有保證語意、模板或常數答案也不相似。清單記錄每項常數基線，就是為了保留這個限制。

</details>

