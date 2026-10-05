examples = [
    {"transcript": "你好", "pitch": "low"},
    {"transcript": "你好", "pitch": "high"},
]
print("轉寫相同", examples[0]["transcript"] == examples[1]["transcript"])
print("音高相同", examples[0]["pitch"] == examples[1]["pitch"])
for row in examples:
    print("只看轉寫", row["transcript"], "實際標籤", row["pitch"])
