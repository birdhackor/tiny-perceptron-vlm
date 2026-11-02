import torch
import torch.nn.functional as F

torch.set_num_threads(1)
table = torch.tensor([0.0, 0.5, -0.25, 1.0])
kernel = torch.tensor([1.0, 0.5, 0.25, 0.125])
hop = 2


class Decoder:
    def __init__(self):
        self.tail = torch.zeros(2)

    def push(self, ids):
        pieces = []
        for code in ids:
            contribution = table[code] * kernel
            # 只發出已完整的前 hop 個；後兩個留待下一碼相加。
            pieces.append(contribution[:hop] + self.tail)
            self.tail = contribution[hop:].clone()
        return torch.cat(pieces)

    def finish(self):
        # 本例定義 N 個碼恰好輸出 2N 個樣本；尾部丟棄。
        self.tail.zero_()


ids = torch.tensor([1, 3, 2, 1])
decoder = Decoder()
first = decoder.push(ids[:2])  # 此時沒有送入後兩碼
saved = decoder.tail.clone()
second = decoder.push(ids[2:])
streamed = torch.cat((first, second))
decoder.finish()

# 整段參考只用於核對，並不是串流路徑。
full = F.conv_transpose1d(
    table[ids][None, None], kernel[None, None], stride=hop
).flatten()[:hop * len(ids)]
reset_second = Decoder().push(ids[2:])
future_changed = Decoder().push(torch.tensor([1, 3, 0, 3]))

print("首塊", first.tolist(), "保存尾部", saved.tolist())
print("第二塊", second.tolist())
print("逐塊／整段最大差", (streamed - full).abs().max().item())
print("重設後第二塊", reset_second.tolist())
print("重設造成最大差", (second - reset_second).abs().max().item())
print("改未來碼後舊prefix最大差", (first - future_changed[:4]).abs().max().item())
assert torch.equal(streamed, full)
assert not torch.equal(second, reset_second)
assert torch.equal(first, future_changed[:4])
