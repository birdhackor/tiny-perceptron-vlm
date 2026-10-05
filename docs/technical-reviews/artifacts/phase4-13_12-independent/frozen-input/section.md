## 13.12 同一批作答重用時，如何記住它來自哪個策略？

某次選中的卡，收集時的機率0.2，更新後變0.4，它已是原來兩倍常見。另一次的卡由0.2變0.1，變一半常見。重用這批紀錄時，要保留作答當時的機率，才能知道已經改了多少。

收集這批迴答時的策略叫**舊策略（old policy）**。同題、同張已選卡的`新機率÷舊機率`叫**ratio（機率比）**；1表示未改，大於1更常，小於1更少。

```python
import torch

old_probability = torch.tensor([0.2, 0.2])
new_probability = torch.tensor([0.4, 0.1])
old_log_probability = old_probability.log().detach()
new_log_probability = new_probability.log()
ratio = (new_log_probability - old_log_probability).exp()
print("更新前後的機率比", [round(value, 4) for value in ratio.tolist()])
```

兩格是兩次作答，不是一份兩卡機率分佈，所以不要求加成1。log相減再取exp等於相除，輸出`[2,0.5]`。實作常保存log機率，便於處理很小的值。`detach()`讓舊log不接受梯度；但歷史值仍需真正保存，不能每步拿新策略重算來替代。

ratio乘13.11固定的優勢，是待提高的更新目標。例如優勢0.6、ratio2，這項是1.2，並非品質提高兩倍。下一節會裁切這項鼓勵，避免同一舊樣本在有利方向上一直推。

old與reference也不同：old對照本輪收集那一刻，下輪重新收集時換新紀錄；reference對照整個後訓練起點，可以全程不動。兩者都叫「舊模型」會混淆用途。

練習只把第一個新機率改0.3，ratio變1.5；第二格仍0.5。若錯用新機率作分母，兩格都變1，恰好把要看的變化抹掉。

<details>
<summary>補充：來源、完整設定與既有實測紀錄</summary>

[PPO原論文第3節公式6–7與演算法1](https://arxiv.org/pdf/1707.06347)使用相對收集時策略的機率比，一批作答可重用數個更新回合；新的收集輪才更新old紀錄。

</details>

