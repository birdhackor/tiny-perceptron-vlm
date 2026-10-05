## 19.8 後訓練方法很多，成品需要全部依序跑嗎？

助理已能給出 `DIRECT:9`，也可能多加「祝你愉快」。若題目只要答案，兩篇都保留9，簡短版較符合需求，這才是一對有清楚差別的偏好材料。若一篇把9改成8，就不能把事實錯誤與風格偏好混在一起教。

SFT 用示範建立作答；DPO 用同題兩篇答案的相對偏好調整模型。RLHF 是利用人類回饋的一類流程，PPO 是其中可採用的更新方法，並不是 SFT、RLHF、PPO、DPO 四站必須全部依序跑完。共同成品只使用有資料、有用途且經分項驗證支持的選項。

下面的 policy 與 reference 都從同一隨機起點建立，借舊回填題看兩者的角色。

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

選到的問題是「原題：3+6。計算器回報：9。請回答」。chosen 是 `DIRECT:9`，rejected 是加祝福版。reference 固定作比較基準，policy 才可更新；此例沒有更新，初始 DPO loss 約0.6931，reference 的梯度開關為 False。

即使真的訓練偏好，還要看原有圖片、聲音與工具是否保留。舊合成任務的 DPO 分支使驗證整題從75/84降到71/84，簡短回答沒有改善，因此仍選先前的 joint。這個例子教的是「較晚的檢查點也可能較差」，不是 DPO 普遍無用；新成品也不能預先規定最後一定採偏好分支。

<details>
<summary>補充：舊偏好分支的更新配方與選擇</summary>

這個分支真的更新了100步，但用的是混合目標：DPO項的beta為0.1，加上係數0.2的示範回答交叉熵，還有係數0.01的路由平衡項。重新練習示範是希望保留原任務的做法，並不保證其他能力不退步；不能把這條配方描述成只跑純DPO。步數與目標名稱見[DPO實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_preference.json)，實際係數與更新方式見[固定訓練程式](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/scripts/course_experiments/capstone.py)。

程式用`frozen_reference`複製父模型，關掉reference的梯度並切到評估模式；更新器只更新policy。上方CPU程式可以檢查梯度開關，原始碼與單元測試也檢查這個機制。不過正式GPU報告沒有記錄reference更新前後的完整張量指紋，這裡不宣稱做過那種執行期比對。報告第一筆批次的DPO項約0.6931，最後一筆約0.0000460；那是不同訓練批次，不是同一組獨立偏好題的前後比較。訓練目標變小，不能直接當成學生看到的回答變好。兩筆誤差保留在[完整DPO訓練紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/dpo/train-report.json)的`history`欄位。

真正的驗證讓這個取捨變得具體：短回答仍是3／3，沒有觀察到生成成績提升；圖音聯合卻從18／18掉到14／18，全題庫從75／84掉到71／84。因此，在尚未查看最後檢查題之前，本輪推薦的MoE成品選joint權重，DPO權重保留為之後交付、可核對的比較分支。這不是說DPO普遍有害，而是這份資料、這個小模型與這次固定配方沒有提供採用它的理由。前後總分可並排核對[joint實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_joint.json)與[DPO實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_preference.json)，實際回答則見[joint逐題檔](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/joint/validation.json)與[DPO逐題檔](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/dpo/validation.json)。

整合成品的意思是把適合的做法接起來，再依證據選擇；不是證明每種方法都跑過，就一律把最後的更新當成最佳成品。後面的量化已對推薦的joint版本重新儲存、載入與評估。小Dense學生的比較實驗也已完成，仍使用事先選定的DPO分支當教師；這是向那個教師學習，不能改稱推薦joint成品的蒸餾版。完整比較與保留失敗見[19.10](19.md#19.10)，教師檔案指紋見[學生實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_student.json)。

</details>

