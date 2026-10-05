## 19.4 怎麼確認下一階段真的接著上一階段學？

先學文字、再接圖片時，我們希望是同一位助理多學了一個入口。若第二站重新建立隨機核心，或載入別章獨立模型，即使檔名都叫 `model.pt`，也不是接續同一成品。每站因此要記下實際讀入的父檢查點，並比較加入新材料前後的各項回答。

共同核心先從隨機權重學有限文字，接著以示範練回答，再用自己訓練的感知入口學圖音條件答覆與工具往返。感知元件可先各自學習、暫時凍結；「凍結」表示此站不更新它，不能省略它原本的權重來源。偏好或壓縮則按需求另接分支，不必變成最後必經站。

先用下面局部 Dense 例子看**哪些位置被計分**。Dense 省去 expert 分派，讓目標對齊容易看；它不是共同 MoE 的權重，也沒有在此更新。

```python
import torch

from tiny_perceptron.capstone import CapstoneModel, build_dataset, default_config, prepare_batch
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import masked_loss

torch.manual_seed(42)
splits, _ = build_dataset()
row = next(row for row in splits["train"] if row["task"] == "style")
print("問題", row["user"], "示範回答", row["answer"])
model = CapstoneModel(default_config(dense=True))
for pretrain in (True, False):
    batch, labels = prepare_batch([row], pretrain=pretrain)
    result = model(**batch)
    loss = masked_loss(result["logits"], labels)
    print("預訓練" if pretrain else "SFT", "有效目標", int((labels != IGNORE).sum()), "誤差有限", bool(loss.isfinite()))
```

選到的問題是「照抄數字29，只要答案」，示範 `DIRECT:29`。預訓練把短文字的下一 byte 都當目標；SFT 讓問題與前文仍可讀，只計回答與 EOS。有效目標數在此依次為43、10。IGNORE 表示不計這格的代價，不是把它從上下文刪掉。兩種目標不同，不能直接比較這兩個 loss 的大小來判哪種教法好。

真正接續時，檢查父檔內容指紋、架構、tokenizer 與資料版本，再用同一組分項驗證題查看舊能力是否保留。載入正確父檔證明來源連續，答對新題才是能力證據；兩項都需要，不能互相取代。

<details>
<summary>補充：舊合成任務的接續階段證據</summary>

正式流程使用固定資料版本`capstone-small-world-v2`、種子42與NVIDIA L4。前三站沿同一核心接續訓練，DPO則從joint接出比較分支；每站都評估同一份84題驗證資料。

| 階段 | 從哪裡接續 | 這一站練什麼 | 完成更新 | 驗證整題正確 |
| --- | --- | --- | ---: | ---: |
| 預訓練 | 隨機權重 | 不帶角色的文字續寫 | 300 | 0／84 |
| SFT | 預訓練權重 | 文字對話中的助理回答 | 1,400 | 42／84 |
| joint | SFT權重 | 文字、圖片、聲音與聯合示範 | 600 | 75／84 |
| DPO比較分支 | joint權重 | 比較偏好回答，並重練示範 | 100 | 71／84 |

前三站都在交叉熵之外加入係數0.01的路由平衡項，避免少數專家獨佔工作。各站的計分位置不同；DPO重練示範的交叉熵目標數，更沒有包含偏好回答的全部評分計算，不能把它與前三站直接當成等成本預算。

表中的「整題正確」要求完整動作字串、內容或參數與EOS符合預期；工具題還要讀回結果並回答正確。預訓練為什麼0／84？像把題目和答案讀熟，還沒有練習「老師問完後，輪到我用指定格式回答」。這只說明本輪文字續寫尚未學會對話協定，不能推成所有文字續寫能力都為零。

SFT的42／84則由42題文字全部正確、42題模態全部錯誤組成，不是每種能力都有一半成功率。joint後文字仍42／42，模態變33／42；九次失敗全是圖片形狀，[19.6](19.md#19.6)會拆開看。DPO反而降到71／84，因此「最後更新的檔案」不必然是較好的成品，選擇理由見[19.8](19.md#19.8)。

怎樣確認這些階段真的接在一起？以SFT接入joint為例，比較[SFT實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_sft.json)的`results.inference_export.sha256`，與[joint實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_joint.json)的`results.parent_checkpoint_sha256`；應比較完整64字元，不能只看檔名或指紋開頭。本輪兩者相同，其他接續階段也通過同樣核對。這證明載入了指定父檔，能力則另由考題確認。從隨機權重開始的第一站沒有父檔，欄位為空值。

完整條件、檔案大小與計時保留在[預訓練實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_pretrain.json)、上述SFT／joint實報及[DPO實報](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_preference.json)。逐站驗證用來決定成品，當時尚未開啟最後90題；定版後的各項能力統一見[19.12](19.md#19.12)。推論檔與完整續訓狀態的用途則在[19.11](19.md#19.11)分開說明。

</details>

