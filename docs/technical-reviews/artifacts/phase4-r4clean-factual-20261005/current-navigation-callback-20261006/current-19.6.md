## 19.6 圖片與聲音怎麼交給同一位助理？

同樣問「右邊是什麼」，左包右短靴應答短靴，交換成左短靴右包應答包。問題不變，答案跟圖走，才能支持圖片被用到。把圖換掉卻仍答原物件，或只看文字就答對，都值得檢查輸入旁邊是否藏了類別標籤。

主線計畫讓圖片入口讀 Fashion-MNIST 三類商品像素，再把特徵投影成共同 MoE 能讀取的特徵向量，接進問題前面的輸入序列。這裡是把圖片內容交給核心，不是指定物件在圖中的左右或上下。單物件題先驗收類別；兩物件自製場景再驗收左右或上下，類別與位置交叉安排。這不是任意自然照片辨識。

讀字另有指定範圍。下圖的上框是「入口」、下框是「出口」；指定下框時答案應只有「出口」。座標可以由程式裁切，但座標不能包含答案。12字「大小上下左右開關入出人口」組成2–4字短串，字形、範圍、字序與換行各自核對；固定框的成功不等於學會到處偵測文字。

![示意字卡包含入口與出口兩區，指定下框時只讀出口。](../figures/rewrite-19-selected-text.svg)

語音入口將保留隨時間變化的聲學特徵，由自行訓練的 encoder／投影交給同一核心。MInDS-14 中文先選地址、App錯誤、卡片問題三種意圖，輸出有限提示或澄清；推論時不附測試錄音的逐字稿。打字先說「兩點回答」，後以聲音問 App 問題時，核心仍應使用那份歷史；這才是共享對話。意圖答覆不是聽寫，也不是已查到真實銀行帳戶。

<details>
<summary>補充：舊合成圖音的接頭短例</summary>

下面的舊配方把16×16 RGB圖平均成4×4色塊，得到48項；純音做16帶[log-mel特徵](12.md#12.7)，再平均時間，得到16項。平均時間會丟掉語句順序，適合舊高低純音示範，不能直接當真人語音入口。下面挑一筆同時問顏色與音高的舊聯合題，只核對接頭。

```python
import torch

from tiny_perceptron.capstone import CapstoneModel, build_dataset, modality_tensors, prepare_batch

splits, _ = build_dataset()
row = next(row for row in splits["train"] if row["task"] == "joint")
print("問題", row["user"], "示範回答", row["answer"])
image, audio = modality_tensors(row)
print("實際圖片形狀", tuple(image.shape), "聲學摘要形狀", tuple(audio.shape))
model = CapstoneModel()
batch, labels = prepare_batch([row])
with torch.no_grad():
    result = model(**batch)
print("回答分數形狀", tuple(result["logits"].shape), "與目標位置對齊", result["logits"].shape[:2] == labels.shape)
```

這段建立隨機模型，只查素材、分數與 labels 的位置對齊。圖是 `(3,16,16)`，聲學摘要 `(16,)`，分數最後有264個候選。`no_grad()` 沒有更新，尺寸正確也不是辨識成功。它與本節保留時間序列的新聲音任務有明確差別。

</details>

新入口接好後，同時比較正常素材、移除素材、錯配素材與成對替換。看圖題依換入圖重給真值，聲音題依換入意圖重給真值；再核對共同核心是否維持原來的文字和格式行為。一次聯合回答正確不能替所有感知分項背書。

<details>
<summary>補充：新有限任務的資料來源</summary>

Fashion-MNIST 的[固定版本官方README](https://github.com/zalandoresearch/fashion-mnist/blob/b2617bb6d3ffa2e429640350f613e3291e10b141/README.md)列出28×28灰階商品與類別，[LICENSE](https://github.com/zalandoresearch/fashion-mnist/blob/b2617bb6d3ffa2e429640350f613e3291e10b141/LICENSE)明列MIT。MInDS-14 的[固定版本官方資料卡](https://huggingface.co/datasets/PolyAI/minds14/blob/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md)提供真人意圖錄音；它的中文schema沒有speaker ID或現成助理答覆，作者答覆需要另行撰寫與核對。完整範圍、字型與切分條件見[主線來源筆記](../../docs/course-revision-20261005/sources/selftrained-capstone.md)。字卡圖是教材作者製作的紙上素材。

</details>

<details>
<summary>補充：舊合成素材的成對檢查</summary>

固定題庫若全部是同一種顏色，始終答同一字也可能滿分；聯合題只問顏色與音高，也沒有驗到形狀。舊成品因此另做換素材檢查：原圖是事先留出的綠色方形，顏色由green換成blue，形狀由square換成circle；聯合題只換顏色、保持音高。每筆重新生成圖片畫素與模型回答，再按換入素材重給真值。

| 推薦joint的圖片對照 | 原素材答對 | 換後按新真值答對 | 原／換兩題都答對 |
| --- | --- | --- | --- |
| 單獨顏色 | 9／9 | 9／9 | 9／9對 |
| 單獨形狀 | 0／9 | 9／9 | 0／9對 |
| 圖音聯合，只換顏色 | 18／18 | 18／18 | 18／18對 |

形狀換後全對仍不能說模型懂了形狀：原圖與換後都答circle，只有換後碰巧正確，成對0／9保留了這個失敗。顏色與聯合題在原圖和換色後都答對，才支持本小世界裡回答依圖片顏色改變。再保持圖片、將聯合題的low／high音訊互換，原題與換後的18對也皆正確。這些觀察仍只涵蓋合成圖形、兩群純音與固定問句，沒有驗收真人語音或新物件主線。

資料與回答可並排核對[原始題庫](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint.json)、[換圖記錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint-image-swaps.json)、[圖片成對紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint-image-pairs.json)與[換聲紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint-audio-swaps.json)。正式檢查後保留形狀失敗，沒有依最後考卷換配方重訓。

</details>

