import torch

glyphs = {"0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"]}
for label, rows in glyphs.items():
    image = torch.tensor([[int(c) for c in row] for row in rows], dtype=torch.float32)
    print("文字標籤", label, "像素形狀", tuple(image.shape))
    print(image)
