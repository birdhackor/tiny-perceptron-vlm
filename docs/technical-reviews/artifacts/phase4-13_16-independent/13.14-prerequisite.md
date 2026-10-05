## 13.14 評分員、估計員與兩種基準各做什麼？

選卡前，估計這題平均能拿0.4分；選卡後，評分員給這次回答1分。前者叫**價值模型（value model）**，也叫critic；它在這個單步任務只看題目情境，預測尚未選卡前的平均結果。後者reward model看情境與已選卡，評這次回答。兩者要回答的問題不同。

教critic時，這次觀察到的1分作目標，用平方代價`(預測−目標)²`。多次不同選擇的樣本，才有機會教出平均估計，不能看一次1就認定每次都能拿1。

```python
import torch

value = torch.tensor([0.4], requires_grad=True)
target = torch.tensor([1.0])
value_loss = (value - target).square().mean()
value_loss.backward()
print("估計員代價與梯度", round(value_loss.item(), 4), round(value.grad.item(), 4))

```

代價0.36，梯度−1.2；下降更新會增加估計值，朝這次分數靠近。這裡只計梯度，沒有更新網路。PPO使用的是更新前保存的critic值計優勢，之後critic另學估計。

![機制示意：五種角色各自的輸入、輸出與固定時間](../figures/rewrite-13-model-roles.svg)

圖中policy給選卡機率，reward評已選卡，critic估平均；old保存本輪收集時的機率，reference保留整段後訓練起點。評分員先教好後固定，critic與policy在PPO期間更新。這是四個小網路加每輪舊紀錄，不代表一定載入五個完整大語言模型。

本配方另對policy加入偏離reference的代價，叫**KL**。若目前兩卡分佈`p=[0.8,0.2]`，參考`q=[0.5,0.5]`，`KL(p||q)=Σ p_i log(p_i/q_i)`約0.1927；i是卡號，Σ表示把所有卡的項加起來。分佈相同時為0。這裡全部卡可枚舉，才是完整求和；沒有把一張抽樣卡當全部分佈。

練習把value改1.4，平方代價0.16、梯度+0.8，下降方向改為降低估計。再把p改成q，KL變0。這兩種檢查各自核對估計與偏移，不是回答已進步的證明。

<details>
<summary>補充：來源、完整設定與既有實測紀錄</summary>

完整數值核對如下，`exact_kl`接收候選分數，先用softmax還原分布，因此指定機率要先取log。

```python
import torch
from tiny_perceptron.posttraining import exact_kl

value = torch.tensor([0.4], requires_grad=True)
target = torch.tensor([1.0])
value_loss = (value - target).square().mean()
value_loss.backward()
print("估計員代價與梯度", round(value_loss.item(), 4), round(value.grad.item(), 4))

policy_logits = torch.tensor([[0.8, 0.2]]).log()
reference_logits = torch.tensor([[0.5, 0.5]]).log()
print("相對固定參考的KL", round(exact_kl(policy_logits, reference_logits).item(), 4))
```

參考約束通常把這個偏離量乘上一個係數，作為額外代價：很高的獎勵值得不值得付出很大的偏移，需要一起考慮。[DPO論文第3節公式3](https://arxiv.org/pdf/2305.18290v3)呈現了「偏好獎勵−相對參考的KL代價」這個目標。本章PPO小實驗的KL另外加入策略代價，價值模型仍估計獎勵模型分數；這是明確的有限選項配方，不等同所有LLM都採同一種逐token回饋。

</details>

