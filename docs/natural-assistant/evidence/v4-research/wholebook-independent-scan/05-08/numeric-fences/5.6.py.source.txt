from tiny_perceptron.training import learning_rate

rates = [learning_rate(i, 40, peak=0.001, warmup=5) for i in range(40)]
print("前六步", rates[:6])
print("最後三步", rates[-3:])
assert max(rates) <= 0.001 + 1e-9
