## 20.11 字都認對，為什麼整頁仍可能讀錯？

購物單上行牛奶、下行麵包。答「麵包／牛奶」保留了所有字，卻交換兩行。讀整頁需要把所選文字按指定版面次序連起來。

![兩行購物單的內容相同，交換後完整轉寫不同。](../figures/natural_reading_order.svg)

```python
from collections import Counter

reference = "牛奶\n麵包"
predicted = "麵包\n牛奶"
print("字元種類與數量相同", Counter(reference) == Counter(predicted))
print("完整字串相同", reference == predicted)
print("參考行序", reference.splitlines())
print("預測行序", predicted.splitlines())
```

`Counter`把字元與換行都放進字袋，種類和數量相同，所以True；完整字串不同，所以False。`splitlines()`則顯示兩種行序。若去掉所有空白再比較，會把換行要求一起丟掉。

這個橫排例子約定上到下、每行左到右。直排、多欄或不同招牌區域要另訂讀序，不能看到答案後才選一種方便得分的規則。裁切可以放大字形，但必須保留原位置，否則看清一行也不知道它原在頁首還是頁尾。

完整版面驗收應留下文字區域、內容與順序。單區OCR或整理後字串相同不替多欄讀序背書；能力卡把這些用途分開，才能知道下一張考卷缺哪種材料。

