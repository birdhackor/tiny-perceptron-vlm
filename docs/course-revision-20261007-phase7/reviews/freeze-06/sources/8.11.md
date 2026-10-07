## 8.11 交換位置後，裁判仍選同一份內容嗎？

回答「2+2=4」放在A，「2+2=5」放在B；交換後，A變成錯答、B變成正答。如果裁判兩次都選A，字母雖相同，選中的內容卻已改變。我們要檢查的是回答身分，而非畫面位置。

給兩份內容固定名稱correct、wrong，再用清單決定顯示順序。下面故意寫一個只選第一項的裁判，讓**位置偏差**清楚可見；這是人工反例，不是任何真實模型的實測。

```python
answers = {"correct": "2+2=4", "wrong": "2+2=5"}
original_order = ["correct", "wrong"]
swapped_order = original_order[::-1]


def choose_first(order):
    return order[0]


first = choose_first(original_order)
second = choose_first(swapped_order)
print("原順序選", first, answers[first])
print("交換後選", second, answers[second])
print("內容身分一致", first == second)
```

`[::-1]`把順序反轉，`choose_first`只讀清單第一項。輸出先選correct，再選wrong，內容身分一致為`False`。合理的內容判斷則應兩次都選correct，即使顯示字母由A變B。

實際測試保存原問題、同一評分規則、每次順序與選中的回答名稱，才能把字母映回內容。若同時改了題目或判準，差異就不再只來自位置。允許平手時也應先寫好記錄方法。

練習把函式改成`return "correct"`。兩次都選同一內容，第三行變`True`。它仍是我們手寫的理想規則，但提供了對照：交換位置的檢查應辨別什麼。

