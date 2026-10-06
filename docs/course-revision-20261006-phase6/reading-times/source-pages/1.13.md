## 1.13 沒更新參數，第二次梯度為什麼變大？

w=2，代價 w²，每次求導應得 `2*w=4`。如果 w 沒改，第二次不應突然更敏感；但 PyTorch 的 `.grad` 可能由 4 變 8，原因是它儲存的是**累積值**。

回顧兩個動作：前向算出代價，`backward()` 算敏感度並放入 `.grad`；更新則另用敏感度改參數。這裡故意只做前兩個動作。把 `.grad` 想成收集桶，每次倒入新梯度，不會自動清空。

```python
import torch
from torch import nn

w = nn.Parameter(torch.tensor(2.0))
(w.square()).backward()
first = w.grad.clone()
(w.square()).backward()
second = w.grad.clone()
w.grad = None
(w.square()).backward()
print(first.item(), second.item(), w.grad.item(), w.item())
assert second.item() == 2 * first.item()
```

`nn.Parameter` 把 w 標成可調參數並預設追蹤梯度。每次 `w.square()` 都重新做一份前向計算；`clone()` 儲存當次讀到的梯度副本。最後印 `4、8、4、2`：

| 動作 | 這次新梯度 | 桶中的 `.grad` | w |
| --- | ---: | ---: | ---: |
| 第一次求導 | 4 | 4 | 2 |
| 未清除，第二次求導 | 4 | 8 | 2 |
| 設 `.grad=None` 後再求導 | 4 | 4 | 2 |

w 一直是 2，所以變的是記錄方式。若每輪只想用當前資料更新，應先清舊梯度，再算新代價、求導與更新。管理更新的工具叫**最佳化器**；之後會用它的 `zero_grad()` 清梯度、`step()` 更新參數。

練習註釋掉 `w.grad=None`。第三次桶裡變 12，w 仍 2；舊斷言只檢查前兩次，所以仍透過。讀取全部輸出，才能知道清除是否如預期。

<details>
<summary>補充與重做</summary>

這不是對同一個 loss 物件連續求導：PyTorch 預設第一次後釋放部分計算圖材料。本例每次重新計算 w²，不需保留舊圖。刻意累積數份小批資料的梯度也是合理做法，見 [16.6](16.md#16.6)；它需要明確安排清除與更新時機。

</details>

