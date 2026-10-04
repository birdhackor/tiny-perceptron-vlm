## 13.11 分數怎樣讓回答更常出現？

假設模型挑出一張回答卡，評分員給了1分。光收到這個數字，下一次的選擇不會自己改變。訓練還要決定：這次選擇比平常好多少？若這個題型平常只有0.4分，1分算不錯；如果平常都能拿1分，同樣1分就沒有超出預期。

模型依問題給各張回答卡機率，這個「怎樣選」的規則叫**policy，策略**。[1.7](01.md#1.7)已讓我們把候選分數轉成機率；這裡候選是整篇預寫回答，不是下一個文字token。估計這題平常能拿多少分的數字叫**baseline，基準值**。將實際分數減去更新前的基準，就得到**advantage，優勢估計**：正值表示比預期好，負值表示比預期差。

前置是[13.10的評分員](#13.10)與[1.11的參數更新](01.md#1.11)。我們先不用網路，拿兩次作答和相同基準做減法。

```python
import torch
from tiny_perceptron.posttraining import bandit_advantage

rewards = torch.tensor([1.0, 0.0])
old_values = torch.tensor([0.4, 0.4], requires_grad=True)
advantage = bandit_advantage(rewards, old_values)
print("優勢估計", [round(value, 2) for value in advantage.tolist()])
print("更新策略時還追蹤基準梯度嗎", advantage.requires_grad)
```

輸入rewards是兩次作答的分數，old_values是作答時保存的舊基準。函式計算`reward−old_value`，輸出`[0.6,−0.4]`。最後是False：這一輪把優勢當作固定評語，讓策略代價用它來更新回答選擇，不同時沿這份計算去改動基準。基準網路另外用自己的分數預測代價學習，會在[13.14](#13.14)拆開。

更新的直覺是：比預期好的那次選擇，應更常被選；比預期差的，應少一些。[C.7](0C.md#C.7)用REINFORCE展示過「分數×所選動作的log機率」如何形成梯度。這裡接著學PPO的更新方式：仍根據固定的優勢估計，但會比較更新前後的選擇機率，並處理一次改太多的問題。PPO全名是proximal policy optimization，近端策略最佳化；它是一種更新方法，不是新的語言模型架構。

本章的任務一次只選一張卡，隨後就拿分並結束，叫**contextual bandit，依情境做一次選擇的任務**。因此基準可以直接估計這次選擇的預期分數，優勢用上面的減法。真正逐token寫回答是一串相連的選擇：早期的字還會影響後面的前文與分數。它需要另外處理多步回饋與各位置的優勢估計，不能把本節兩個數字的減法原樣當成完整LLM訓練。

reward也不一定是答對1、答錯0。獎勵模型的數字可能為負，還可能加入偏離參考策略的代價；其定義與尺度要保存。這次1與0是人工例子，目的是讓「拿到0也會比預期差」看得清楚，不是本章實跑網路每次都剛好給這兩種分數。

練習只把兩個舊基準都改成0.9，先預測優勢變`[0.1,−0.9]`，再執行核對。兩次回答的評分沒有改，但超出預期的程度變了；最後說明為何不能用advantage的正負直接代替答案正誤。

## 13.4 為什麼要有參考模型？

你想讓新模型比原本更偏愛簡潔正確的回答，需要一把固定尺。若尺也跟著每次更新伸縮，就難以分清改了多少。SFT（supervised fine-tuning，監督式微調）先用理想助手回答教基礎能力；DPO（direct preference optimization，直接偏好最佳化）再用同題兩篇回答的偏好，調整相對原模型的回答傾向。這種偏好訓練保留一份原來對話模型，叫reference參考模型；正在更新的那份叫policy策略模型，它負責之後實際回答。

前置是[13.3整段log分數](13.md#13.3)與[11.2凍結](11.md#11.2)。先算policy的「較佳分數減較差分數」，再減掉reference原本的同一差。這叫相對margin差距。例子中參考為-4與-3，差-1；更新後策略為-3與-3，差0，相對進步就是0−(-1)=1。我們看的是偏好相對原基準增強多少。

```python
import copy
import torch
from tiny_perceptron.model import TinyLM, ModelConfig

policy = TinyLM(ModelConfig(width=8))
reference = copy.deepcopy(policy).eval().requires_grad_(False)
before = reference.embedding.weight.clone()
with torch.no_grad():
    policy.embedding.weight.add_(0.1)
scores = reference(torch.tensor([[1, 2]]))["logits"]
print("參考未跟著改", torch.equal(before, reference.embedding.weight))
print("參考不記梯度", scores.requires_grad)
```

輸入是一個小隨機模型，用來檢查複製與凍結機制，不是已訓好的偏好基模。`deepcopy`複製整份獨立權重，不能只是`reference=policy`，那會指向同一對象。`requires_grad_(False)`凍結參考參數，`eval()`設評估行為，`no_grad`只包住人為加0.1的原地修改，避免把這個測試動作記成待求導計算。參考分數的計算放在區塊外，讓一般梯度記錄仍開著；此時輸入是整數ID、不求梯度，輸出是否需要梯度就能核對參考參數的凍結狀態。

我們人為將policy的embedding每項加0.1，輸出第一行True：參考沒有跟著變；第二行False：即使計算時沒有用`no_grad`關閉記錄，已凍結的參考分數仍不需要梯度。若漏掉`requires_grad_(False)`，這個輸出會變True；不能用包在`no_grad`裡得到的False替代凍結核對。實際DPO開始時通常複製已完成SFT的模型，整個偏好階段保留參考不變，並且不把它的參數放入policy優化器。

固定尺不表示參考所有回答都好。它用來控制新模型相對原行為的改變範圍，偏好資料仍需指出哪些方向更好；錯誤參考可以被新資料糾正。

正式比較使用[8.3的`content.pt`](08.md#8.3)作固定參考，它在七道留出加法原本零題答對，不是已通過的算術老師。訓練前複製策略與參考，兩種beta設定都從這份相同起點開始。beta是正的縮放係數：偏好代價會先把前面算出的相對差距乘上它，再決定要推動多少。它不是學習率；此處先知道它是另一項訓練設定即可，[13.6](13.md#13.6)會用相同差距逐項觀察它的作用。參考數字指紋在更新後保持不變。這只能證明固定尺沒有跟著動，不能替新策略的回答保證真值。

還要分清「相對進步」與「已經排在前面」。beta 0.1版本在驗證題`3+5=?`上，較佳`8`減較差`9`的log分數，從參考的約−10.89346變成−8.44789，相對提高2.44557。但差距仍為負，表示這兩個完整候選中，錯的9仍有較高機率。相對改善可以是從更偏向錯答，移到稍微沒那麼偏向錯答，並不等於正確候選已勝出。

參考在訓練起點與policy相同，故每個偏好對的相對差距初始為0，即使它們本來更常生成rejected也一樣。這讓目標關注相對偏好的改變，而不是要求兩個原分數本來已經一樣。更新過程中參考只負責再算同題兩篇答案的固定基準，不參與後面的梯度步。若每一步重新複製policy當reference，相對差距會不斷被歸零，失去一開始設定的固定約束意義。代碼裡的before是矩陣快照，clone複製數字；它和deepcopy整模型層級不同，卻都在避免引用同一份可變內容造成假核對。

練習只把那一行的`copy.deepcopy(policy)`替換成`policy`，保留後面的`.eval().requires_grad_(False)`，因此整行成為`reference = policy.eval().requires_grad_(False)`。先預測凍結會連policy也凍結，人為加0.1後第一行變False，再重跑核對，因為兩個名稱其實指同一份矩陣。這個反例說明複製、凍結與不收錄優化器是不同但相配的檢查。

