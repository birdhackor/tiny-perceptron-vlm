## 20.10 讀懂意思，和逐字抄寫，是同一項能力嗎？

告示是「今天去臺北」，回答「今天去台北」可能保留意思，卻沒有逐字抄對。OCR從圖片辨識文字；用它做原樣轉寫時，字形差異就是需要檢查的內容。

```python
from tiny_perceptron.natural_concepts import text_error_report

report = text_error_report("今天去臺北", "今天去台北")
print("完整相同", report["exact"])
print("最少編輯次數", report["edits"])
print("參考字數", report["reference_characters"])
print("CER", report["cer"])
```

完整相同是False，最少一次替換，參考5字，所以CER=1/5=0.2。CER即最少插入、刪除、替換次數除以參考字元數；漏掉北，分母仍是原來5字。這裡按Unicode字元，不按UTF-8 bytes。

核對前先約定哪些整理允許。NFKC可把全形Ａ換成A，並不把臺改成台。是否忽略空白、標點或接受繁簡必須先訂，原樣結果也保留。要求繁中解釋，不代表能自行把圖片上的簡體轉寫改成繁體。

沒有字的圖要另測：補上一句熟悉文字也是錯誤；參考空串沒有字數可除，不能直接報普通CER。模糊到讀不清則屬於有字但缺可讀內容。合成文件、真實招牌、短行與整頁各有不同難點，得分應跟實際材料範圍一起說。

