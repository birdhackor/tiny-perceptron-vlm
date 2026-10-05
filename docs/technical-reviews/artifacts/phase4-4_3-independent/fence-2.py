first = layer(x)
x[0, 1] += 100
shifted = layer(x)
print(torch.allclose(first[0, 0], shifted[0, 0]))
print(torch.allclose(first[0, 1], shifted[0, 1]))
