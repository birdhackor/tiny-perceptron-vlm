## 20.3 只調整一小份權重，為什麼還要載入大模型？

LoRA修正包像幾張便條，使用時原書仍需在桌上。可更新的數字少，不等於整套權重少。本章 Qwen3-VL-2B-Instruct 圖文底座實際有2,127,532,032個參數；它是上游已訓練的另一份 Dense 模型，並未接續第19章的隨機MoE。

先用乘法估這份底座的數值儲存：

```python
parameters = 2_127_532_032
for label, bytes_per_number in [("32-bit", 4), ("16-bit", 2)]:
    weight_bytes = parameters * bytes_per_number
    print(label, "權重數字bytes", weight_bytes, "約GiB", round(weight_bytes / 1024**3, 2))
```

32-bit 每個數字4 bytes，共8,510,128,128 bytes，約7.93 GiB；16-bit每個2 bytes，約3.96 GiB。GiB以1024³ bytes計。這段沒有載入模型，也不包含圖片特徵、對話快取、工作空間和語音模型。

訓練另有梯度、更新器與中間結果，不能看見4 GiB權重就推論4 GiB顯示卡能訓練。圖片細節與長歷史也增加計算。裝置實測範圍見[訓練指引](../../docs/natural-assistant/v4/TRAINING.md)，乘法估算只回答數字要佔多少位元組。

Dense 在此沒有 expert 路由；選它是接手成熟圖文能力的具體配置，不是 Dense 普遍優於 MoE。Whisper則另負責轉寫。後面比較LoRA時，底座與語音入口各自固定，才能把變化歸到實際改過的地方。

