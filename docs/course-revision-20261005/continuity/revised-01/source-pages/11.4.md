## 11.4 同一張圖片，換問題後應怎麼改答案？

同一張紅圓圖，問顏色希望答「紅」，問形狀希望答「圓形」。圖片沒變，問題決定要取哪項資訊；這才是圖片問答，而不只是每次念出完整描述。

![同一紅圓合成圖，顏色問題對應紅，形狀問題對應圓形](../figures/rewrite-11-same-image-questions.svg)

```python
records = [
    {"image": "red_circle", "question": "什麼顏色？", "answer": "紅"},
    {"image": "red_circle", "question": "什麼形狀？", "answer": "圓形"},
]
for row in records:
    print(row["image"], row["question"], "→", row["answer"])
print("同圖", records[0]["image"] == records[1]["image"])
print("答案不同", records[0]["answer"] != records[1]["answer"])
```

兩筆記錄的 image 身份相同、答案不同，兩項比較都印 True。image 欄的 `red_circle` 是資料名稱，不是已送進模型的像素；實際資料要同時保存圖或可重建它的生成條件，不能只教問答文字便宣稱模型看圖。

訓練時把圖、問題與正確短答保持成對，回答位置算代價。若生成器有隨機選擇，也記種子、版本與參數，才能重建材料。例如分成訓練組與留出檢查組，同一原圖的所有問答都要放在同一組；否則圖片可能已在訓練中出現，卻把換問法的題目當成新圖檢查。

再固定問題、換圖片：紅圓與藍圓都問顏色，答案應分別紅與藍。同圖多問檢查選擇範圍，同問換圖檢查圖片作用。若只有紅圓多問，模型可能靠「顏色題答紅」猜中。

回答「紅色圓形」雖包含正確詞，對只要顏色的問題仍多答了內容。先訂短答規則，再評內容與範圍；練習加一筆「同時給顏色與形狀」，才把目標改成「紅色圓形」。

<details>
<summary>回顧與查證</summary>

可回顧：[11.3圖片描述目標](11.md#11.3)、[7.1對話資料](07.md#7.1)。

原實驗的資料切分、更新設定與逐題結果見[既有實報](../../docs/course-experiments/results/vqa.json)；重做入口見[實驗說明](../../docs/course-experiments/README.md)。這是上述有限任務的歷史紀錄。

</details>

