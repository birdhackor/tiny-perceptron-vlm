## C.7 Reward 怎麼改變輸出機率？

只在答3與答4之間選，兩個分數都0，概率各0.5。這次人為選答4，reward=1，平常參考baseline=0.5，所以比參考好0.5。希望提高這個動作的機率，但不是一次變成必選。

policy是輸出概率分佈；reward減baseline叫advantage。令代價 `-(reward-baseline)*logp(action)`，其中logp(action)表示「所選動作的機率取log」，是公式符號，程式會用`log_softmax(0)[action]`取得這個值。優勢為正時，降低代價會提高所選動作log機率；為負時相反。這是策略梯度的最小例子。

程式把答3、答4依序放在索引0、1，因此action=1代表本次選答4。`logits`是兩份答案的分數，`softmax(0)`把它們轉成兩項機率，`log_softmax(0)`則給這兩項機率的log值。`requires_grad=True`讓PyTorch記錄分數參與的計算，`backward()`將代價對各分數的梯度存到`.grad`，再用步幅0.1相減；求梯度與更新的區別可回看[1.12](01.md#1.12)。

```python
import torch

logits = torch.tensor([0.0, 0.0], requires_grad=True)
action = 1
reward, baseline = 1.0, 0.5
before = logits.detach().softmax(0)
loss = -(reward - baseline) * logits.log_softmax(0)[action]
loss.backward()
with torch.no_grad():
    updated = logits - 0.1 * logits.grad
    after = updated.softmax(0)
print("梯度", logits.grad.tolist())
print("新分數", updated.tolist())
print("前機率", before.tolist())
print("後機率", [round(p, 6) for p in after.tolist()])
```

梯度 `[0.25,-0.25]`，步幅0.1相減後新分數約 `[-0.025,0.025]`，概率成 `[0.487503,0.512497]`。reward與baseline是固定數字，沒有對驗證器求導。`detach()`保留舊值比較，`no_grad()`裏真的算出更新後分數；不是隻算backward就聲稱更新。

![順序固定為答3與答4，更新分數後答4機率由0.5稍升至0.512497。](../figures/rewrite-C-policy-update.svg)

這裏沒有採樣或訓練語言模型，action人工固定，直接調兩個分數。baseline是比較本次回饋的參考，不是正確答案；扣掉它，才能分清高於或低於參考的更新方向，角色見[13.11](13.md#13.11)。正式策略要先依分佈抽動作、取得回饋，再更新產生分數的參數；適當的baseline可以減少抽樣造成的更新波動，這個固定動作的短例沒有量測這項效果。由可驗真值提供回饋的訓練常稱可驗證獎勵強化學習（Reinforcement Learning with Verifiable Rewards，RLVR），會改權重，與[C.5保持權重的作答計算](0C.md#C.5)不同。

只把reward改0，advantage=-0.5，動作1機率反而降到約0.487503。即使單次機率按期待變化，也要在新題檢查整體行為與評分漏洞；舊嚴格reward策略更新後新題全錯，弱reward則學會列舉。真正更新和訓練回饋變高，都不能取代新題能力。

<details>
<summary>補充：舊有限策略訓練、原結果與選讀重做</summary>

正式組另外訓練一個小策略，直接讀三個0至5的整數。例如數字2用六格`[0,0,1,0,0,0]`表示，只有它對應的一格是1，其餘是0，這叫one-hot；三個數各用六格，再接在一起就是18格輸入。可學的層把它們混合，再算出17種動作的分數；線性層怎麼混合可回看[2.3](02.md#2.3)。前16種動作分別答0至15，最後一種列舉全部候選，兩種評分方式見[C.6](0C.md#C.6)。這個小策略與前面的兩個語言模型分開。

正式策略先依機率分佈抽動作，將動作ID換成上述文字，再按[C.6](0C.md#C.6)的規約取得回饋：嚴格reward要求只交一個正確整數；弱reward只要求文字中的任一數字吻合，因而會獎勵列舉全部數字。REINFORCE是這裡所用的策略梯度方法，用抽到動作的log機率與相對回饋估計更新方向。正式組把梯度交給AdamW更新產生分數的參數，方法見[5.4](05.md#5.4)與[5.5](05.md#5.5)；主例則只用人工固定的動作，直接調兩個分數。

兩支從同一份隨機起點學習，分別使用嚴格與弱reward，並在未參與訓練的24道新題檢查。嚴格支線更新後，新題的最高機率動作仍全錯，例如`(0+3)+3=?`應答6，卻選7。弱支線在每題抽16次、合計384個動作時，全都選列舉，代理分384/384、嚴格正確0/384。這是24道不同題目的重複抽樣，不是384道新題；訓練回饋變高與真正更新參數，都不能取代新題能力。

這個策略選預定動作再映成文字，沒有自回歸生成token，也沒有替前面的兩個語言模型做RL。完整配方、原始動作與檢查記錄見[正式報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/reasoning.json)。可見步驟、多數一致和高reward，都不能單獨證明輸出文字反映內部推理。

練習只把reward改成0，baseline仍0.5。先預測advantage變負、梯度改為 `[-0.25,0.25]`，更新後動作1機率下降到約0.487503，再執行核對。請用「本次比參考差」說明方向，而不是把所有非負reward都理解成應該提高機率。

**選讀：重做本附錄的固定實驗。** 先按[W.1](../first-steps.md#W.1)在專案根目錄準備Python環境，再依[Git LFS官方指引](https://git-lfs.com/)安裝資料下載工具；本課的按需下載步驟見[資料包說明](../../assets/training/README.md#取得資料)。以下入口使用固定公開資料包，完整重做短答與步驟的監督式微調（SFT，也就是學習示範回答）、多候選評估，以及有限策略的兩種回饋比較：

```bash
git lfs install --local
git lfs pull --include='assets/training/gsm8k-v1.tar.gz' --exclude=''
python scripts/course_experiments/run.py --experiment reasoning --device cpu
```

CPU執行完整排程，有自己的CUDA環境才改`--device cuda`；上文正式數字來自NVIDIA L4顯示卡。結果保存於`outputs/course-experiments/course-v1/reasoning/`，可用報告中的資料切分與各項分母核對。語言模型和有限動作策略是不同產物：`policy-*.pt`不是TinyLM權重，不能交給語言模型的推論入口。入口還包含公開長紀錄的資料通路檢查；本附錄的算術能力分數只使用上文合成的留出題。

</details>
