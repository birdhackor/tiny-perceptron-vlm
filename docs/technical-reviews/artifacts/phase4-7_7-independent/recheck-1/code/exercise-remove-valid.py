actual = model(ids, positions=positions)["logits"][:, 2:]
print("有效位置最大差", (base - actual).abs().max().item())
assert torch.allclose(base, actual, atol=1e-6, rtol=0.0)
