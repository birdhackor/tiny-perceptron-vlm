question = "用一句話解釋1+2"
demonstration = {"question": question, "answer": "把1個和2個合在一起共有3個，所以1+2=3。"}
original_preference = {"question": question, "chosen": "3", "rejected": "答案是3喔！"}
revised_preference = {"question": question, "chosen": demonstration["answer"], "rejected": "3"}
print("新的問題", demonstration["question"])
print("新的示範", demonstration["answer"])
print("原偏好兩篇", original_preference["chosen"], original_preference["rejected"])
print("修訂的偏好", revised_preference["chosen"], "勝過", revised_preference["rejected"])
for origin, description in [("真人比較", "人類回饋"), ("程式驗算", "程式規則的可驗證回饋"), ("另一模型評分", "模型回饋")]:
    print(origin, "來源是", description)
