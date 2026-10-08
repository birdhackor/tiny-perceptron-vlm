## 11.16 字都出現了，為什麼仍可能讀錯順序？

下面告示先寫「今日休館」，再寫「明日開放」。希望按橫排規則先上後下、每行左到右，讀成「今日休館／明日開放」。如果先念第二行，字的種類一樣，訊息順序仍變了。

![作者製作兩行告示，按上行今日休館、下行明日開放的順序讀](../figures/rewrite-11-line-order.svg)

再把「今天去臺北」當作參考答案，把「今天去台北」當作預測結果，看逐字判準。編輯距離是將預測變成參考的最少插入、刪除或替換；CER 再除以參考字數。

```python
from tiny_perceptron.natural_concepts import text_error_report

reference = "今天去臺北"  # 參考答案
prediction = "今天去台北"  # 預測結果
report = text_error_report(reference, prediction)  # 先參考、後預測
print("完全相同", report["exact"])
print("最少編輯次數", report["edits"])
print("參考字數與CER", report["reference_characters"], report["cer"])
```

輸出 False、1、以及 5 與 0.2：參考五個字、一字替換。這裡按 Unicode 字元計，不按 UTF-8 bytes；日常異體字同義，也不等於逐字照抄相同。

若要接受異體字，事先訂比較前的正規化規則，並保留未轉換結果；不能看到錯字才改規則。漏末尾北，參考分母仍 5；前後字序不同也用整串比較，不能只數全部字有沒有出現。

有無文字、逐字與讀序分別核對。空白圖不應憑常見句子補字；直排、多欄的規則則另定，不用本例橫排約定硬套。練習交換兩行，即使忽略換行，為何仍不等於原有序字串？下一節再把有限物件與短字詞任務安排清楚。

<details>
<summary>回顧與查證</summary>

可回顧：[11.13整串與字元錯誤](11.md#11.13)。

</details>

