first = layer(x)
x[0, 1] += 100
moved = layer(x)
print(torch.allclose(first[0, 0], moved[0, 0]))
print(torch.allclose(first[0, 1], moved[0, 1]))
