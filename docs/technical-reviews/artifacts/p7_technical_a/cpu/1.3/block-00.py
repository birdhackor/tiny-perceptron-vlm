s = list("貓看狗。")
x = s[:-1]
y = s[1:]
print(x)
print(y)
for current, answer in zip(x, y, strict=True):
    print("看到", current, "下一字", answer)
assert len(x) == len(y) == 3
