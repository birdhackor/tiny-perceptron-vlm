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
