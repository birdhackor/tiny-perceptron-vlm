## 13.15 選擇、評分、更新，可以串起來嗎？

現在把前面幾節接成一次訓練：模型先選一張回答卡，評分員給分，估計員提供舊基準，PPO用這份固定紀錄更新選擇。這裡每次只選整篇預寫候選，不會一個字接一個字地寫答案；它示範有限選項的PPO流程，不能把成績當成語言模型的算術、中文理解或開放問答能力。

前置是[13.11的優勢](#13.11)、[13.13的裁切](#13.13)與[13.14的四種角色](#13.14)。先用未訓練的小網路跑一次，追蹤「誰真的變了」。情境四格是第一個數除10、第二個數除10、是否要求解釋、是否缺購買數量；後兩格用1表示有、0表示沒有。所以`[0.1,0.2,0,0]`是1與2的只回數字題。這些條件已由程式直接提供，不需要模型讀懂原句。

```python
import copy
import torch
from tiny_perceptron.posttraining import (
    FiniteResponsePolicy,
    FiniteRewardModel,
    FiniteValueModel,
    bandit_advantage,
    exact_kl,
    ppo_clipped_objective,
)

torch.manual_seed(42)
context = torch.tensor([[0.1, 0.2, 0.0, 0.0]])
policy, critic = FiniteResponsePolicy(), FiniteValueModel()
reward_model = FiniteRewardModel().requires_grad_(False)
reference = copy.deepcopy(policy).requires_grad_(False)
optimizer = torch.optim.SGD(list(policy.parameters()) + list(critic.parameters()), lr=0.1)

with torch.no_grad():
    old_distribution = torch.distributions.Categorical(logits=policy(context))
    action = old_distribution.sample()
    old_log_probability = old_distribution.log_prob(action)
    reward = reward_model(context).gather(1, action[:, None]).squeeze(1)
    advantage = bandit_advantage(reward, critic(context))

new_logits = policy(context)
new_log_probability = torch.distributions.Categorical(logits=new_logits).log_prob(action)
terms = ppo_clipped_objective(new_log_probability, old_log_probability, advantage)
value_loss = (critic(context) - reward).square().mean()
loss = terms["policy_loss"] + 0.1 * exact_kl(new_logits, reference(context)).mean() + 0.5 * value_loss
before = next(policy.parameters()).detach().clone()
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("選中的卡", action.item(), "更新前ratio", terms["ratio"].item())
print("策略參數改變", not torch.equal(before, next(policy.parameters())))
print("評分員有梯度", any(p.grad is not None for p in reward_model.parameters()))
print("輸入與選卡分數形狀", list(context.shape), list(new_logits.shape))
print("固定參考有梯度", any(p.grad is not None for p in reference.parameters()))
```

`FiniteResponsePolicy`輸出四張卡的分數，`Categorical`把分數轉成可以抽選的分布，`sample`抽一張；編號0到3依序是短答、帶算式的句子、錯數字、求補資訊。`gather`從評分員的四格中取已選卡的分數，`squeeze`移除多餘的一格軸。收集時在`no_grad`裡保存舊機率、回饋與基準；更新時重新計算可求梯度的新機率。參考與評分員用`requires_grad_(False)`凍結，優化器只收到策略與critic的參數。

`SGD`是隨機梯度下降更新器，這裡用本次資料求出的梯度更新；`lr=0.1`是學習率，每個已登記參數減去0.1乘自己的梯度。`zero_grad()`先清掉上次留下的梯度，`backward()`計算這次總代價的梯度，`step()`才真的改動參數。總代價中的另外兩個係數，0.1乘KL、0.5乘估計員平方誤差，是調整這兩項影響的權重，與更新器的學習率各有不同用途。

這次固定seed42，輸出選卡0、更新前ratio1.0、策略改變True、評分員有梯度False。新增的兩行顯示輸入與選卡分數形狀都是`[1,4]`，代表一題、四格情境或四張卡；固定參考有梯度是False。ratio仍1是因為尚未做第一步更新，這一刻沒有觸發裁切；不能說模型已被硬限制在20%內。這些網路都隨機初始化，評分員也還沒學好，所以本例只驗證流程與凍結，不證明參數改動有助於任務。實際評分員必須先按[13.10](#13.10)學比較。

完整CPU實驗則真的先教選卡與評分，再接PPO。兩個數都在1到10，55個無序加數家族先分44個訓練、5個驗證、6個最後檢查；每家族各有三種條件，分別132、15、18題。只回數字的44筆示範，教148參數的選卡網路60次，每批32題。此起點刻意用0.2的label smoothing：大部分目標仍給正卡，其他卡留少量目標份額，避免探索時它們的機率過早接近0。這是本實驗固定的示範配方，不是所有SFT都必須如此。

241參數的評分員用660對訓練偏好，更新300次、每批64對，共抽用19,200對；97參數的critic在PPO階段學分數估計。PPO收集120批，每批64次真實選卡，共7,680次；每批重用三回合，所以策略與critic各更新360次。重用不算新作答，報告另記23,040次被更新流程讀取的動作紀錄。這個選卡實驗沒有token訓練，`effective_tokens=0`有它的單位理由，不表示沒有有效學習資料。

```bash
python -m scripts.course_experiments.run --experiment posttraining --device cpu
```

先依[W.1](../first-steps.md#W.1)啟用專案Python環境，再在專案根目錄執行。完整實驗含DPO分支，CPU兩個執行緒的這次總耗時約2.12秒；PPO段約0.66秒，未計另一台機器安裝環境的時間。[公開實測報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/posttraining.json)保存設定、分組、候選、每題機率、更新紀錄與權重指紋。最後18題的完整要求，起點6題通過，PPO後12題通過；哪些成功、哪些仍失敗，下一節逐項看。

練習只把context第三格改成1，表示要求一句話說明。先預測輸入仍四格、輸出仍四張卡、參考和評分員仍無梯度，再執行核對。抽到哪張卡可能改變，也不保證變成正確的1號卡；因為短程式仍用未訓練網路。用同一句話區分「流程接通」與「已學會依要求選擇」，就是這次練習的驗收。

