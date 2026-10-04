answers = ["3", "我已經仔細思考過，答案是4。"]
scores = [len(answer) for answer in answers]
chosen = max(range(len(answers)), key=lambda index: scores[index])
print("評分規則挑中", answers[chosen])
print("符合只回數字且算對", answers[chosen] == str(1 + 2))
