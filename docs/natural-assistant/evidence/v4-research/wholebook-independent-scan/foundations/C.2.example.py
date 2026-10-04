steps = [(2, 2, 5), (5, -1, 4)]
truth = 2 + 2
valid = [a + b == claimed for a, b, claimed in steps]
linked = [steps[i][0] == steps[i - 1][2] for i in range(1, len(steps))]
print("等式算對", valid)
print("接上前步", linked)
print("最後答案對", steps[-1][2] == truth)
