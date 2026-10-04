## T.7 先檢查回答能力，再學較偏好的回答

先讀[13.1的同題比較](chapters/13.md#13.1)與[13.4的參考模型](chapters/13.md#13.4)。偏好資料不是另一份只有標準答案的題庫：它讓同一問題有兩個候選，並記錄較合適的一個。DPO 是用這類比較調整回答傾向的方法。

先沿這條路讀懂同題比較、固定參考，再用同樣問題查看訓練前後生成了什麼。本節先列一份本機探索配方，再對照作者已完成的正式GPU實驗；想重做時要分清兩份設定，單純理解則可以直接看本文的候選與實測結果。核心練習之後的選卡回饋、整合成品與較小模型都是選讀，各自獨立，不用全部跑完才算理解DPO。

```bash
.venv/bin/python scripts/prepare_data.py --kind preference
.venv/bin/python scripts/train.py --task dpo --checkpoint checkpoints/style.pt --data data/generated/preference/train.jsonl --train --steps 200 --output checkpoints/preferred.pt
```

這條命令需要 [T.5](#T.5) 已產生的 `style.pt`。偏好資料可用共用 `prompt` 搭配 `chosen`、`rejected`，分別表示偏好的回答與另一個回答；也可以保存兩份完整對話。先打開一對例子，確認比較真的是同一問題，不是題目難度不同。產生器例如把`0+5=?`的`chosen`寫成`5`、`rejected`寫成`6`，這套玩具偏好是按加法真值選擇，沒有教出更廣泛的人類偏好。`--train`開啟更新，`--steps 200`表示這階段更新200次，不能將步數當成通過證據。

參考模型是開始這一階段時保留、不更新的副本。它提供原來的回答傾向作比較，不是替每一題保證真值的老師。SFT可先建立回答起點，但保存了一份SFT檔案不等於基本能力已通過；開始DPO前仍需核對同一組新題。如果起點就不會回答，偏好排序進步也未必能補足缺少的能力。

訓練後保留[T.5](#T.5)的原測驗，再加偏好對測驗。先在`data/generated/preference/validation.jsonl`挑一筆，讀`prompt`並遮住兩個候選名稱，按加法真值判斷。預設第一筆是`0+5=?`，理想答案5；可先取得訓練前後對同題的生成文字：

```bash
.venv/bin/python scripts/infer.py checkpoints/style.pt --chat --prompt "0+5=?" --tokens 32 --temperature 0
.venv/bin/python scripts/infer.py checkpoints/preferred.pt --chat --prompt "0+5=?" --tokens 32 --temperature 0
```

終端直接印助手回答。逐題保存提問、理想答案及兩份實際文字；例如兩份都答5表示本題基本正確沒有退步，不能證明偏好改善，前答5後答6則表示本題退步。換成其餘驗證題時，兩條命令的提問必須一起換成同一筆`prompt`。同時沿T.5再檢查原有風格與安全題，這些逐題證據才是結果，不是200步已成功。檢查是否更符合目標，也檢查答案數字、格式與誠實是否退步；如果只是回答變長，而評分者喜歡長文，不能當成洞察提升。

正式課程實驗另從本頁`style/content.pt`出發，與上面`checkpoints/style.pt`的200步探索配方分開。它本來在留出加法是0／8、0／7，不能先當成已會回答。64道算式按交換加數家族切成49／8／7對，提問仍是原本的`2+2=?`等形式；兩支分別用beta 0.1與1，固定學習率0.001、每批八對，各250次更新，種子42。寬度64、兩層、上下文128格的策略與參考有相同起點，參考完全凍結，數字指紋核對未變。

先完成T.5的正式`style`實驗，再取UltraFeedback小包，重做這一組：

```bash
.venv/bin/python scripts/fetch_training_assets.py --asset ultrafeedback-dpo
.venv/bin/python -m scripts.course_experiments.run --experiment dpo --device cuda
.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/dpo/model.pt --chat --prompt "4+2=?" --tokens 32 --temperature 0 --device cuda --json
```

`outputs/course-experiments/course-v1/dpo/model.pt`與`beta1.pt`保存兩種beta的策略；各自的`*-reference.pt`保留固定參考，`data/manifest.json`與三側JSONL保留偏好對。兩支各讀到9,448個有效回答目標，合計兩篇候選且包含EOS，不計提問。`result.json`與[公開完整報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/dpo.json)分開保存兩類證據：`preference`逐對列完整chosen／rejected的log分數、參考與策略差距；`arithmetic`列全部留出題的實際生成、原始ID及EOS。

前者的`chosen_higher_absolute_probability`是較佳候選比分給定較差候選高的題數，`relative_preference_improved`是較佳／較差的差距相對參考提高的題數，兩者都以`records`作分母。後者的`matches/records`才核對自由生成的內容，`eos_rate`另看正常結束。上面`4+2=?`實測生成`8`，雖然beta 0.1對此題的候選6給了比7更高的完整機率。[13.4](chapters/13.md#13.4)、[13.5](chapters/13.md#13.5)解釋相對差距、二選一排序與自由生成為何不同；[13.6的表](chapters/13.md#13.6)也保留兩支最後加法皆零題答對的結果。

這組還保存`format-model.pt`：同題兩篇都算對，只偏好沒有`; answer complete`附加句的回答，更新200次；相對排序改善7／7，實際生成仍0／7，見[13.7](chapters/13.md#13.7)。自然資料支線則另建小模型，100對自行切80／10／10，問題和答案各取最多120個UTF-8 byte，先80步SFT、再80步DPO。那一支只報片段候選比較，沒有自由生成品質成績，截短後也未重新人工標偏好；其限制見[13.2](chapters/13.md#13.2)與[13.9](chapters/13.md#13.9)。這次正式比較沒有評完整風格、安全與反附和能力，不能從加法偏好分數替它們填上通過。

練習使用開頭產生的`data/generated/preference/validation.jsonl`，找`0+5=?`這筆。先遮住`chosen`、`rejected`名稱，只看兩個候選`5`與`6`，按加法真值選較合適的回答，再揭開名稱核對。還沒產生資料也可以用本文這一對在紙上完成，不必先訓練。接著說出為什麼候選排序不能代替生成驗收；若標註與規則有分歧，先修規則或資料，不要讓模型替你解決目標還沒定義的問題。格式偏好與自然資料包則可作延伸，先各自說清楚判準。

接下來是**選讀其他路線**。第一條先教一個獎勵模型替回答打分，再讓選擇模型作答、得到回饋、更新。這裡用的**PPO**是依回饋與更新前後機率比較調整選擇的方法，會裁切過大的鼓勵；它不像DPO直接從兩篇回答的比較更新。想拆開看，讀過[13.1–13.2](chapters/13.md#13.1)後，到[13.10](chapters/13.md#13.10)學評分員，再追13.11–13.16，最後在[13.17](chapters/13.md#13.17)比較兩條分支。

這份CPU實驗另有**價值估計員**，估計這題平常可能拿多少分，作為判斷此次回饋高低的基準；前面的固定參考則保存原模型的回答傾向，兩者用途不同。實驗真的訓練評分員、估計員與選擇模型，但每次只選一張預寫回答卡，沒有逐步生成文字。**RLHF**指用真人回饋做強化學習，例如先由真人比較回答，再用這些比較指引更新；本例的標籤由作者規則交給程式套用，沒有真人比較資料。它可以教角色與更新機制，不能稱已完成語言模型的RLHF。

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment posttraining --device cpu
```

這個選讀命令不需要上面的`style.pt`或另下載權重；它從固定小網路建立示範起點，再分成PPO與DPO兩支。配置、候選、資料家族與逐題結果見[公開報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/posttraining.json)，讀法見[13.15](chapters/13.md#13.15)。逐步生成語言模型的完整PPO與多步回饋仍是後續延伸。[C.7](chapters/0C.md#C.7)另有一種依所選動作機率與回饋更新有限策略的小實驗；計分單位與本例不同，權重不能互換。

第二條選讀是[19.8的整合成品對照](chapters/19.md#19.8)。它把DPO接在真正生成文字的模型上：由文字、圖片與聲音一起訓練的聯合任務階段複製固定參考，另開DPO分支，再與原模型比較。第13章的選卡PPO教更新機制，第19章的DPO檢查接續後的實際回答，兩份權重與評估各自獨立。成品目前選聯合任務階段作推薦試用起點，DPO保留作對照，完整能力表見[19.12](chapters/19.md#19.12)。

第三條選讀是[19.10的較小模型](chapters/19.md#19.10)。它以第19章的DPO比較分支作教師，從相同起點、用相同資料順序，分別只學示範與加入教師分布。兩個較小模型每次使用相同的一組完整計算零件，稱為Dense；加入教師分布，是同時模仿較大模型對下一個文字單位的預測機率，這種教法叫蒸餾。正式訓練已完成，最後90題分別通過62題與61題，這次蒸餾沒有勝過示範訓練。教師選擇與推薦的聯合任務成品不同，不能把學生稱作那份成品的縮小版；詳見[學生報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/results/capstone_student.json)。

