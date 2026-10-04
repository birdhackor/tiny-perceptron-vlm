demonstration = {"question": "只回數字：1+2=?", "answer": "3"}
preference = {"question": demonstration["question"], "chosen": "3", "rejected": "答案是3喔！"}
feedback = {"question": demonstration["question"], "sampled_answer": "4", "reward": 0}
print("示範給什麼", demonstration["answer"])
print("偏好比較什麼", preference["chosen"], "勝過", preference["rejected"])
print("作答後得到什麼", feedback["sampled_answer"], feedback["reward"])
