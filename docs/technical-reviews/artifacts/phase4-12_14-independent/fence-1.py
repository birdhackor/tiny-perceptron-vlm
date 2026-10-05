from tiny_perceptron.natural_concepts import speech_stages

report = speech_stages("我不吃辣", "我不吃拉", "選清淡的湯麵。", "選清淡的湯麵。")
print("聽寫字元錯誤率", report["transcription"]["cer"])
print("兩路回答完全相同", report["answers_identical"])
print("相同是否足以證明正確", report["answers_correct"])
