examples = [
    {"回答": "4", "正確": True, "貼切比喻": False},
    {"回答": "4，就像兩雙筷子共有四根。", "正確": True, "貼切比喻": True},
    {"回答": "5，數字如星光流動。", "正確": False, "貼切比喻": False},
]
for row in examples:
    print(row["回答"], row["正確"], row["貼切比喻"])
