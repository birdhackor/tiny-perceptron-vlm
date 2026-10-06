measurements = {
    "微調前": {"A舊任務": None, "B新任務": None},
    "只訓練B後": {"A舊任務": None, "B新任務": None},
}
a_questions = ["紅色物體是圓。請接續：紅色物體是", "藍色物體是方。請接續：藍色物體是"]
b_questions = ["1+1=?", "2+1=?"]
print("固定A題", a_questions)
print("固定B題", b_questions)
print("待填實測", measurements)
