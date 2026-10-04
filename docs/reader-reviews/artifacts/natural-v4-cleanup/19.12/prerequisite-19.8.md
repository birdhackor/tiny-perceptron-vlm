## 19.8 後訓練方法很多，成品需要全部依序跑嗎？

假設一位已能回答的助理，總會在答案後加一句「祝你愉快」。我們希望它在「只要答案」時短一點，手上又有兩種都保留同一事實的回答，這就適合拿偏好對教它更常選簡短版本。這是偏好對的設計例子；本輪joint在固定照抄驗證題上本來已經簡短，不能先假定它還需要改善。也不需要為了學偏好，先額外造一個獎勵模型，再把每種強化學習演算法全跑一遍。

前置是[後訓練地圖](07.md#7.18)、[DPO相對參考模型](13.md#13.4)、[偏好不等於事實](13.md#13.8)、[本成品的MoE／Dense](19.md#19.2)與[接續訓練](19.md#19.4)。joint是前一站把文字、圖片與聲音一起練的版本；MoE每個位置選少數expert，Dense學生則用較小的同一套規則。這裡的能力矩陣是逐任務的成績表，路由平衡項則是鼓勵expert分工的額外誤差。這些名稱不是另外幾個未見過的助理。

SFT先以示範建立回答方式；DPO在一對回答中調整相對偏好。RLHF是使用人類回饋學習的一類流程，PPO是可用於強化學習更新的方法；它們不是所有成品必須按SFT→RLHF→PPO→DPO依序通關的四站。

本輪從joint完成的權重複製一份固定reference，再用合成的「簡短／加祝福」偏好對做DPO比較分支。reference是本次比較的基準，不隨policy一起更新；policy是正在更新的模型。偏好對不來自真人標註，因此不能稱已經完成RLHF。

```python
import torch

from tiny_perceptron.capstone import CapstoneModel, build_dataset, frozen_reference, preference_loss, preference_pairs

torch.manual_seed(42)
splits, _ = build_dataset()
pair = preference_pairs(splits["train"])[0]
print("同一問題", pair["row"]["user"])
policy = CapstoneModel()
reference = frozen_reference(policy)
loss, _ = preference_loss(policy, reference, [pair])
print("較喜歡", pair["chosen"], "較不喜歡", pair["rejected"])
print("相同起點的DPO誤差", round(loss.item(), 4), "參考可更新", any(p.requires_grad for p in reference.parameters()))
```

`build_dataset`回傳資料切分與清單，`_`略過清單；`preference_pairs`只從訓練記錄製作偏好對，`[0]`選第一對。這對的原問題是「原題：3+6。計算器回報：9。請回答。」；chosen為`DIRECT:9`，rejected為`DIRECT:9，祝你愉快！`。`DIRECT:`是直接回答的動作前綴，不是數學答案的一部分；兩篇都保留9，差別是加不加祝福，因此能核對同題、同事實的偏好。

這段建立隨機起點，沒有載入正式joint權重，也沒有更新。policy與reference同起點，回答對的相對差互相抵消，所以DPO誤差應約0.6931；reference可更新應為False。較短回答已經比較容易生成時，也可能有較高完整序列機率；DPO用reference差值作基準，不能只從一個未調整的長短機率差宣稱偏好訓練成功。

PPO的直覺、獎勵與reference／old policy區別，在[13.11–13.15](13.md#13.11)另用有限回答候選練習。那是獨立概念支線；沒有把它偷偷算成主成品完成了一輪token語言模型PPO。我們選DPO來整合，是為了用已有可檢查偏好對示範真實續訓；是否改善終端回答、是否傷到圖片或工具能力，仍由同一份矩陣檢查。

這個分支真的更新了100步，但用的是混合目標：DPO項的beta為0.1，加上係數0.2的示範回答交叉熵，還有係數0.01的路由平衡項。重新練習示範是希望保留原任務的做法，並不保證其他能力不退步；不能把這條配方描述成只跑純DPO。步數與目標名稱見[DPO實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_preference.json)，實際係數與更新方式見[固定訓練程式](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/scripts/course_experiments/capstone.py)。

程式用`frozen_reference`複製父模型，關掉reference的梯度並切到評估模式；更新器只更新policy。上方CPU程式可以檢查梯度開關，原始碼與單元測試也檢查這個機制。不過正式GPU報告沒有記錄reference更新前後的完整張量指紋，這裡不宣稱做過那種執行期比對。報告第一筆批次的DPO項約0.6931，最後一筆約0.0000460；那是不同訓練批次，不是同一組獨立偏好題的前後比較。訓練目標變小，不能直接當成學生看到的回答變好。兩筆誤差保留在[完整DPO訓練紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/dpo/train-report.json)的`history`欄位。

真正的驗證讓這個取捨變得具體：短回答仍是3／3，沒有觀察到生成成績提升；圖音聯合卻從18／18掉到14／18，全題庫從75／84掉到71／84。因此，在尚未查看最後檢查題之前，本輪推薦的MoE成品選joint權重，DPO權重保留為之後交付、可核對的比較分支。這不是說DPO普遍有害，而是這份資料、這個小模型與這次固定配方沒有提供採用它的理由。前後總分可並排核對[joint實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_joint.json)與[DPO實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_preference.json)，實際回答則見[joint逐題檔](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/joint/validation.json)與[DPO逐題檔](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/dpo/validation.json)。

整合成品的意思是把適合的做法接起來，再依證據選擇；不是證明每種方法都跑過，就一律把最後的更新當成最佳成品。後面的量化已對推薦的joint版本重新儲存、載入與評估。小Dense學生的比較實驗也已完成，仍使用事先選定的DPO分支當教師；這是向那個教師學習，不能改稱推薦joint成品的蒸餾版。完整比較與保留失敗見[19.10](19.md#19.10)，教師檔案指紋見[學生實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_student.json)。

練習把最後一行的`any(p.requires_grad for p in reference.parameters())`改成`any(p.requires_grad for p in policy.parameters())`，同時把顯示字串「參考可更新」改成「policy可更新」。先預測布林結果由False變True，再執行；只改顯示字串不會改檢查對象。這仍只是檢查梯度開關，沒有訓練policy。參照與被訓練者有同樣的結構、甚至一開始有相同數字，角色卻不同；靠名字無法代替實際凍結檢查。

