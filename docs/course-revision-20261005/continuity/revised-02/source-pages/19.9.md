## 19.9 接上較快的做法，先檢查什麼？

把前五個位置一次算完，與先算前三個、再接後兩個，最後位置應看見同一段前文。KV cache 只是重用已算的 Key、Value；若遮罩或位置接錯，快一點也不能當成同一個回答器。先核對計算含義，再量時間。

固定的五個 ID 是純文字機制輸入，core 是隨機模型。

```python
import torch

from tiny_perceptron.capstone import CapstoneModel

torch.manual_seed(42)
core = CapstoneModel().language
core.eval()
ids = torch.tensor([[1, 21, 22, 23, 24]])
with torch.no_grad():
    full = core(ids)["logits"][:, -1]
    cache = core(ids[:, :3])["cache"]
    cached = core(ids[:, 3:], cache=cache)["logits"][:, -1]
print("最大分數差", (full - cached).abs().max().item(), "接近", torch.allclose(full, cached, atol=1e-4, rtol=1e-4))
```

第一路全算，第二路保存前三格，再接後二格，比較最後264項候選分數。`allclose` 應為 True 或在指定誤差範圍接近；這不是新成品的速度或能力成績。

接圖片、聲音時，應先編碼素材，與完整前文一起做首次計算，再沿用快取生成新 token；不能每步又重新處理素材卻稱已省掉那份工作。驗證要保留原始生成 ID 與停止原因，同時分辨可接受的浮點尾差和實際輸出改變。

SDPA 是注意力運算介面，後端取決於裝置與形狀，不保證是 FlashAttention。MoE 的 Python 分派也不等於專用稀疏運算核心。完成一致性核對後，再在具體硬體、同題與同生成設定下量速度、記憶體；快取不會替模型增加未學會的長文能力。

練習把分段方式從前三格、後兩格改成前兩格、後三格：將`ids[:, :3]`改為`ids[:, :2]`，`ids[:, 3:]`也同步改為`ids[:, 2:]`。先預測兩路仍讀相同五個ID、比較同一最後位置的264項分數，應在原容差內接近，再執行核對。只改其中一處會漏讀或重複位置，便不是同一段前文的對照。

<details>
<summary>補充：舊joint權重的快取與計時</summary>

正式部署另載入推薦joint，從固定驗證資料每項任務取第一題，共12題，選取時沒有先看生成結果。每題最多生成16個新編號，完整保留這次生成的ID序列；其中可能因上限截斷，所以這是機制對照，不是又一次完整能力評分。full與cache的原始ID全部一致，同一歷史上的每步logits也全部在`atol=1e-4, rtol=1e-4`範圍內接近。[完整快取對照](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/cache-consistency.json)保留兩條生成路徑與每步分數差，包含有圖片和聲音的任務。

確認一致後，才量一個事先選好的短問句：「照抄數字15，只要答案。」同一個joint、同一張NVIDIA L4、FP32，兩路各暖機3次，再量10次；兩路都生成`DIRECT:15`與EOS，共10個新編號，每次原始ID都相同。GPU具體型號記在[部署總實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_deployment.json)的`gpu`欄；下方逐次計時檔的`cuda:0`只表示裝置編號，不能單從它推斷型號。

| 同一句短生成 | 10次實測的中位時間 |
| --- | --- |
| full，每步重算前文 | 68.054毫秒 |
| cache，沿用前文KV | 59.978毫秒 |

這個cache在該短句較快，但不是任意長度、任意硬體的速度保證。計時包含前文準備、裝置傳輸、貪婪生成與解碼，GPU在每次呼叫前後同步；不含載入權重、啟動、上傳下載或檔案核驗。這次生成對照沒有量記憶體峯值，不能把19.2前向／反向的峯值貼過來稱成快取記憶體。[逐次生成計時實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/generation-benchmark.json)保留所有暖機與測量輸出、選題規則與範圍。

</details>

