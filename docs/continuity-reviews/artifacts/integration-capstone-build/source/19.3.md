## 19.3 同一題的不同版本，怎麼避免跑進兩份考卷？

把「1+2等於多少」換成「請算1加2」，再把計算器開關改一下，會得到好幾筆記錄。但它們仍圍繞同一組數字。若一筆用來練習，另一筆拿來宣稱沒見過的考題，分數可能只是在量記憶。整合資料的第一步因此不是大量製造句子，而是先決定哪些記錄屬於同一家族。

前置是[資料家族切分](05.md#5.12)、[訓練／驗證／測試](05.md#5.10)與[偏好對](13.md#13.1)。family是同一原始素材或題目的一群衍生記錄；train供更新，validation供選設定，test最後檢查。本章先切家族，再製作不同問法、工具狀態、素材變體與偏好回答。每個階段沿用同一份切分，不能到了偏好訓練又把測試題收回來。

`build_dataset`回傳兩份資料：`splits`是「份名→記錄清單」的字典，例如`splits["train"]`是一串訓練記錄；`manifest`是份名、題數、指紋與固定答案基準的摘要。每筆記錄叫`row`，有`family`家族、`task`任務、`user`問題、`answer`示範與`image`、`audio`素材規格。例如第一筆訓練題是「3+6等於多少？」；它的`task`是calculator，沒有圖片或聲音，因此那兩欄是`None`。

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

輸入是固定種子42的合成資料規則。`splits.items()`依序交給迴圈一對`name, rows`：`name`是份名，`rows`是那份記錄清單。`families`也是字典，但每個份名這次對應不重複的家族集合；內層`row["family"]`從各筆記錄取家族，集合`set`把重複名稱合併。`&`取兩個集合共同的名稱，因此前三行重疊應為0。接著`Counter(row["task"] ...)`數各任務有幾筆，最後一行`["train"][0]["user"]`依序選訓練份、第一筆記錄、問題文字；不是模型生成。

`audio_test`用方括號留下test中`task == "audio"`的記錄，仍是一份清單。`labels`則用`Counter`把每筆的`audio`規格裡的`pitch`轉成「音高→筆數」摘要，所以`labels["high"]`會得到3，`len(audio_test)`會得到6。最後的`baseline`來自另一層摘要：在`manifest`先取`task_majority_baselines`，再取`test`，再取當前`task`。這時它是一份含`majority_label`固定答案、`correct`固定答能對幾題、`count`總題數的字典。程式只是讀規則來算基準，沒有呼叫模型。

零家族重疊還不夠：不同家族也可能製造完全相同的實際輸入。例如兩道聯合題的音高不同，抽掉音訊後，圖片單獨題可能完全相同。因此正式檢查還要比對使用者文字、system設定與真正送入的媒體內容，避免只把家族換名字就宣佈無洩漏。對已知類別的新組合、不同素材變體與陌生問法，也要分開命名；沒有看過某種圖片組合，不等於發現全新顏色概念。

資料版本`capstone-small-world-v2`、固定種子42，共有552筆訓練、84筆驗證與90筆最後檢查。數字以同一組數字對為家族；圖片以顏色／形狀組合為家族，藍色圓形只進驗證、綠色方形只進最後檢查，該組合的位移、明暗和聯合題都一起走。訓練仍包含每一種基本顏色與形狀，因此考的是未見組合，不是新顏色或新形狀。短資料、缺資訊、拒絕與照抄按情境編號一起切。

數字家族`1,2`的六種衍生記錄只在最後考卷，不在訓練或驗證；它們包括`1+2等於多少？`、反向順序的`請算2加1。`、計算器關閉、概念說明與回填結果。因此成品試用的1加2案例能按一組預先留出的題核對，而不只是挑一筆訓練題展示。

純音題先按基頻家族切，再保持低／高音各自8個訓練家族、1個驗證家族、1個最後檢查家族，每個家族的三種頻率變體一起走。最後六題因此是3題low、3題high：完全不聽聲音、一律回答high，只會3／6。程式中「完全不聽、固定答high」那行算的就是這個人工固定基準，不是模型預測。若考卷只有high，固定答案就能全對；先看標籤分布與基準，才知道高分是否真的需要聽聲音。

圖音聯合的18道最後題則包含9個low、9個high，文字問題都相同，仍需保持圖片、只換音訊，再依換入素材更新音高真值核對；單純「分數有變」不能證明聽對，詳見[12.10的換聲測試](12.md#12.10)。

還有一個容易漏掉的限制：原始圖片最後題全部是綠色方形，所以固定回答green的顏色基準與固定回答square的形狀基準都會9／9。這兩項就算模型全對，也不能單獨證明它看了圖片。上面末兩行讀取資料清單的固定答案基準；正式成品還要看[19.6](19.md#19.6)的換顏色、換形狀成對測試。

數字題使用訓練已出現的兩種問法，考新的數字家族；圖片與音訊則按上面的素材規則留出。本輪沒有另做陌生措辭考卷，因此不能從這些成績推論一般中文提問都能理解。若想擴充這項檢查，可回到[B.7的陌生問法失敗](0B.md#B.7)看需要另外準備什麼資料。

正式實驗沿用這份凍結資料；[完整資料清單](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/data.json)存了所有726筆原問答與素材生成規格，沒有要求讀者只靠輸入編號猜問題。各份資料的SHA-256指紋開頭依序是訓練`e0a419727952`、驗證`afedd4dc84ce`、最後檢查`aef4b7a876ff`，完整指紋在清單的`manifest.sha256`。跨階段與比較方法，訓練、驗證、最後檢查各自的指紋應分別保持不變；不是說三份不同用途的資料彼此相同。

本輪還在CPU重跑[實際輸入不交叉的檢查](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/tests/test_capstone.py)：使用真正的前文編號、圖片張量與聲學摘要的bytes計指紋，三對資料份之間的家族交集與實際輸入交集均為0。這是資料規則檢查，不是模型能力成績；它排除完全相同輸入跨份，沒有保證語意、模板或常數答案也不相似。清單記錄每項常數基線，就是為了保留這個限制。

練習只把種子42改成43。預測家族重疊仍為0，但哪些家族分到各份會改變，再執行核對。正式實驗不能看哪個種子分數漂亮就換考卷；教材的固定資料版本與指紋會讓這種變更能被查出。

