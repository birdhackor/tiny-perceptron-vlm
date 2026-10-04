第 1 章：從猜下一個字開始
第 2 章：讓模型看更長的上下文
第 3 章：Attention，自己決定看哪裡
第 4 章：第一個 Dense Transformer 語言模型
第 5 章：訓練與評估形成一個流程
第 6 章：自己的中英文 tokenizer
第 7 章：從續寫文字到回答問題
第 8 章：風格、個性與指令遵循
第 9 章：誠實、安全與適當回應
第 10 章：圖片如何進入文字模型
第 11 章：對齊、問答與語言能力保留
第 12 章：沿用相同介面加入聲音
第 13 章：把回答偏好變成訓練
第 14 章：逐項理解現代文字模型的設計
第 15 章：從 Dense 到 Mixture of Experts（MoE）
第 16 章：由量測帶出效率 BKM
第 17 章：量化，用更少位元表示模型
第 18 章：蒸餾，讓小模型向教師學習
第 19 章：把零件組成一位小小助理
第 20 章：做一位能看圖、讀字與聽問題的助理
本章導讀
20.1 照片、打字與說話，怎麼交給同一位助理？
20.2 不重新訓練，怎麼先開啟成品？
20.3 只調整一小份權重，為什麼還要載入大模型？
20.4 怎樣的練習，才能教到我們想要的能力？
20.5 換個問法，為什麼仍可能是同一道舊題？
20.6 保留底座，怎麼只學一小份修正？
20.7 題目和答案都在輸入裡，模型究竟練哪一段？
20.8 訓練跑完後，怎麼決定可以交付哪個版本？
20.9 回答出現正確名詞，就算看懂照片了嗎？
20.10 讀懂意思，和逐字抄寫，是同一項能力嗎？
20.11 字都認對，為什麼整頁仍可能讀錯？
20.12 助理答錯時，怎麼分清聽錯與回錯？
20.13 交給別人使用時，要一起帶走哪些東西？
操作指引
資料與來源
自行重訓
A：上下文學習與 RAG
B：從結構化回答到工具使用
C：推理步驟、驗證與測試時算力
目錄
1. 先確認 GPU 環境
2. 取得固定資料，確認真正的資料目錄
3. 準備底座，固定各個元件版本
4. 保存配方，開始自己的新訓練
5. 檢查完成狀態；必要時從自己的檢查點續訓
6. 先驗證用途，再選定自己的版本
7. 固定選定版本，再做最後測試
8. 使用自己的模型，或換一台 GPU 執行
首頁
教材
第 20 章：做一位能看圖、讀字與聽問題的助理
用自己的 GPU 重做照片與中文問答微調¶

本版教材依驗證保留原底座，不加LoRA，選擇理由見20.8。這份指引帶你重做已實跑的微調比較：從指定底座建立自己的候選修正，照片、中文讀字及聊天題一起練習，再和原底座比較新題回答。只有語言注意力的LoRA參數更新；建立候選與最後採用候選，是兩個分開的決定。

語音仍先由現成辨識器聽寫，本次不重新訓練它。真人錄音留給驗證兩站流程，因此不必把語音轉寫當成這次 LoRA 的訓練成果。理解範圍可先讀20.5 的家族切分、20.6 的修正分支與20.7 的答案計分。

以下使用 Linux、Python 3.12 與自己的 NVIDIA GPU，指令都在專案根目錄執行。需要另外預留安裝、下載與 GPU 運算時間；短 Python 概念例子不會代替這次實際微調。

1. 先確認 GPU 環境¶

依學生操作指引第 1、2 步取得程式、建立 .venv-natural，安裝 CUDA 12.8 的 PyTorch 與 torchvision 配套，再安裝 requirements-natural.txt。保留前面小實驗使用的 .venv，不在兩套環境之間混裝。

訓練路線使用 bfloat16，先檢查設備：

.venv-natural/bin/python -c "import torch; print('GPU 可用：', torch.cuda.is_available()); print('支援 bfloat16：', torch.cuda.is_bf16_supported())"


兩項應為 True。如果找不到 GPU，先處理驅動與套件；如果換精度或圖片預算，就把它記成自己的另一份配方，再驗證結果。推論的小份 LoRA 檔很小，訓練仍須載入完整底座、保存梯度和更新器狀態，不能用檔案大小推算設備最低容量。

本章這份完整訓練使用NVIDIA L4與bfloat16，配套為PyTorch 2.8.0＋CUDA 12.8、torchvision 0.23.0＋CUDA 12.8、Transformers 4.57.6及PEFT 0.18.1。學習率為0.00003，模型處理預算為65,536–524,288 pixels，序列上限2,048，每次更新累積兩題；下面的數字只描述這套配方：

量測項目	實際記錄	從哪裡量到哪裡？
訓練核心記錄時間	1,708.814秒，約28.48分鐘	讀完並核對資料清單後、載入模型前開始，包含模型載入、更新迴圈及中途檢查點存檔；在最終張量與凍結樣本核對、最終存檔之前結束。
Python訓練子程序時間	1,728.676秒，約28.81分鐘	從匯入套件到這次訓練程序結束，包含清單讀取、模型載入、訓練、最終核對與存檔；不含外層容器與工作流程準備、事先取得資料或後續權重備份。
PyTorch已分配GPU記憶體峰值	5,719,892,480 bytes，約5.327 GiB	載入模型後重設峰值計數，再量這次訓練；含當時仍駐留的權重與訓練緩衝。

最後一列只計PyTorch已分配的GPU記憶體，不含CUDA執行環境、配置器保留空間或重設前的載入峰值。因此它不是整張顯示卡的總占用，更不是最低硬體需求。這次雲端工作請求4個CPU與32 GiB主記憶體；那是配置值，未量得CPU記憶體峰值。

完整訓練核對摘要保留量測範圍與完成檢查。換電腦、照片預算或對話長度，時間與占用都可能不同；推論另有自己的量測，不能直接拿訓練數字替代。

2. 取得固定資料，確認真正的資料目錄¶

先依資料說明取得這一版的照片、OCR 與語音快照，並完成逐檔核對。它們解包到 data/natural-v4，題目由 docs/natural-assistant/v4/manifest.json 配對；來源、授權與修改另保留在包內及來源清單。

.venv-natural/bin/python scripts/fetch_natural_data.py --manifest docs/natural-assistant/v4/manifest.json --revision 9a61ecf524c9518f33f1501c28aa72997d4a82d0 --manifest-sha256 0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60 --output data/natural-v4 --list
.venv-natural/bin/python scripts/fetch_natural_data.py --manifest docs/natural-assistant/v4/manifest.json --revision 9a61ecf524c9518f33f1501c28aa72997d4a82d0 --manifest-sha256 0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60 --output data/natural-v4
.venv-natural/bin/python scripts/fetch_natural_data.py --manifest docs/natural-assistant/v4/manifest.json --revision 9a61ecf524c9518f33f1501c28aa72997d4a82d0 --manifest-sha256 0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60 --output data/natural-v4 --verify


訓練讀 rows 中的 train 題目，驗證與最後測試分開保留。audio_rows 在這份微調中不進入更新器；FLEURS 檢查聽寫，AISHELL 問句再比較兩路聊天。錄音來源叫訓練集，也不會因此被這次訓練讀入。

下載核對說明「資料與清單相同」，不能代替「答案寫得正確」。你若自行修改資料，請另存清單、素材與資料版本，避免改動教材固定答案；後面的 --manifest 與 --data-root 必須指向同一份配對資料。

3. 準備底座，固定各個元件版本¶

先下載正式配方使用的圖文底座與 ASR，保存模型準備紀錄。LoRA 將從固定底座新建，ASR 留給之後的轉寫驗證。它們分別有版本，不用「抓最新」代替原配方。

.venv-natural/bin/python scripts/natural_assistant.py prepare \
  --output outputs/natural-my-v4/models \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --asr-variant turbo \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 \
  --max-tokens 2048 --max-new-tokens 384 --seed 42 \
  --cache-dir .cache/natural-v4-models


完成後保留 .cache/natural-v4-models。後續命令的 --local-files-only 只使用這份本機快取，缺檔時停止，不暗中換版本。模型下載與訓練計時不同；準備好權重，只表示拿到指定零件，尚未更新任何參數。

4. 保存配方，開始自己的新訓練¶

使用 outputs/natural-my-v4 保存自己的工作。若這個名稱已用過，換成新的實驗名稱，並同步改後面的輸出與 adapter 路徑。先把完整命令保存為腳本，再執行，之後才知道用了哪種設定。

mkdir -p outputs/natural-my-v4
cat > outputs/natural-my-v4/train.sh <<'BASH'
.venv-natural/bin/python scripts/natural_assistant.py train \
  --output outputs/natural-my-v4/train \
  --manifest docs/natural-assistant/v4/manifest.json \
  --data-root data/natural-v4 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --asr-variant turbo \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 \
  --max-tokens 2048 --max-new-tokens 384 --seed 42 \
  --cache-dir .cache/natural-v4-models --local-files-only \
  --steps 2077 --learning-rate 0.00003 \
  --lora-rank 8 --gradient-accumulation 2 \
  --checkpoint-every 25 --checkpoint-steps 1039,2077 \
  --max-seconds 3300
BASH
bash outputs/natural-my-v4/train.sh


cat把開頭<<'BASH'與結尾BASH之間的內容保存為train.sh；最後的bash才執行這份腳本，不是再訓練第二次。

學習率控制每次更新的步幅，這份完整配方固定為0.00003；它與概念小例子追蹤一步更新時使用的步幅分開。這份配方已完成2,077次更新，每次累積兩題的答案梯度再更新；實際讀取4,154筆練習，2,077道訓練題各讀兩次。第1,039與2,077次更新後另保存不可覆寫的候選檢查點；它們用同一份驗證題比較，最後測試不參與挑選。種子42固定本次資料排序與隨機起點；訓練長度上限是--max-tokens 2048。你自己的執行是否完成，仍按下一節的紀錄核對。

LoRA 加在語言注意力的 q、v 投影，rank 為 8，alpha 為 16；底座仍參與前向計算，只有修正參數進入更新器。視覺底座與 ASR 不因這個步驟重新訓練。這種做法保留原權重，把可更新範圍縮小；能否維持舊用途並完成新用途，仍要在後面驗收。

這次實際更新1,605,632個LoRA參數，分在112個修正張量中；112個張量的前後指紋都改變。加入LoRA後的圖文模型共有2,129,137,664個參數，語音辨識器另計。每份候選的adapter_model.safetensors為6,438,952 bytes，約6.44 MB；這只是修正權重檔，完整續訓檢查點還包含下一節所列的更新器與進度。

每道題都讀取問題、歷史與照片，直接計分的是最後助手回答及配套模板的結束部分。程式先核對提問編碼是完整對話的精確前綴，再遮住問題與歷史的直接代價；這就是「可以讀，但不要求抄成答案」的差別。輸入超出宣告長度時報錯，不偷偷切掉答案繼續訓練。

步數指更新器真正改權重的次數。若每次累積多道題再更新，題目讀取次數會更多；讀到相同題目也不代表有更多獨立資料。訓練紀錄保存實際題目 ID、有效答案位置與已完成更新，讓這些數字可以回查。

圖片預算與編碼長度限制每題的運算範圍，不是直接聲明「讀得懂多少中文字」。把圖片壓小可能省運算，也可能抹掉筆畫；需要改設定時，另存配方並重新驗證。時間上限會在更新前檢查，到時保存目前進度；模型載入、單次更新和存檔仍可能使整個命令超過這個軟上限。

5. 檢查完成狀態；必要時從自己的檢查點續訓¶

訓練結束後，先讀 train/result.json 的 status、requested_steps 與 completed_steps。只有完成所求更新，才稱完成這份訓練。代價下降是訓練訊號，還不是照片、讀字與聊天都通過的證據。

輸出內容各有用途：

路徑（省略 outputs/natural-my-v4/）	用來做什麼
train/adapter/adapter_model.safetensors、train/adapter/adapter_config.json	載入修正權重及配對設定，進行推論。
train/adapter/training_state.pt、train/adapter/training.json	保存更新器、隨機狀態與進度，搭配完整 adapter 目錄續訓。
train/result.json、train/training.json	核對完成狀態、更新步數、有效答案位置與配方。
train/provenance.json	核對底座、資料指紋及軟體配套。
train/checkpoints/step-001039/、train/checkpoints/step-002077/	各自保存完成該步的修正、更新器狀態與進度，供驗證選版。

optimizer_only_lora 表示更新器範圍符合設定；修正張量的前後指紋檢查是否真的改變。凍結檢查比對底座幾個固定抽查位置，不能把抽查說成二十億個數全部逐位比較。這些檢查證明執行方式，回答品質仍由驗收決定。

遇到時間上限或中斷，先讀最近完整檢查點的 completed_steps，未保存的更新不能恢復。只在未到目標總步數時續訓，保持相同底座、資料、rank、累積方式、種子與學習率，並另開 resumed 輸出目錄。

下面從目前最近保存的train/adapter恢復。第一段讀取真實進度，只要求保存尚未經過的1,039／2,077步候選；已保存的舊候選不會覆寫：

PENDING_CHECKPOINTS=$(
  .venv-natural/bin/python -c 'import json; from pathlib import Path; n = json.loads(Path("outputs/natural-my-v4/train/adapter/training.json").read_text())["completed_steps"]; assert 0 <= n < 2077, "先確認尚未完成2,077步"; print(",".join(str(step) for step in (1039, 2077) if step > n))'
)
.venv-natural/bin/python scripts/natural_assistant.py train \
  --output outputs/natural-my-v4/resumed \
  --manifest docs/natural-assistant/v4/manifest.json \
  --data-root data/natural-v4 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --asr-variant turbo \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 \
  --max-tokens 2048 --max-new-tokens 384 --seed 42 \
  --cache-dir .cache/natural-v4-models --local-files-only \
  --adapter outputs/natural-my-v4/train/adapter \
  --steps 2077 --learning-rate 0.00003 \
  --lora-rank 8 --gradient-accumulation 2 \
  --checkpoint-every 25 --checkpoint-steps "$PENDING_CHECKPOINTS" \
  --max-seconds 3300


例如已完成1,000步，剩下的保存點是1039,2077；已完成1,100步，則只要求2077。這兩個數字指總共完成的更新，不是再加上這麼多次。

原候選保留在train/checkpoints/，續訓中新保存的候選位於resumed/checkpoints/。以下驗證命令的兩份路徑，應各自指向真正保存那一份候選的位置；例如1,039步在原目錄、2,077步在續訓目錄，就只替換第二份路徑。若又到時間上限，先讀resumed/adapter/training.json，下次把讀取與載入路徑都改成那個完整adapter，再另開新的輸出目錄。

公開試用配套用來推論，沒有恢復這張訓練工作桌所需的更新器和進度。

6. 先驗證用途，再選定自己的版本¶

用 validation 比較同一底座關閉與開啟 LoRA 的結果。圖片、題目、歷史、處理預算與生成設定相同，才能知道差別來自哪個候選。這一刻還不讀最後 test。

.venv-natural/bin/python scripts/natural_assistant.py validation \
  --output outputs/natural-my-v4/validation \
  --manifest docs/natural-assistant/v4/manifest.json \
  --data-root data/natural-v4 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --asr-variant turbo \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 \
  --max-tokens 2048 --max-new-tokens 384 --seed 42 \
  --cache-dir .cache/natural-v4-models --local-files-only \
  --split validation \
  --adapter outputs/natural-my-v4/train/checkpoints/step-001039 \
  --comparison-adapter adapter-step-001039=outputs/natural-my-v4/train/checkpoints/step-001039 \
  --comparison-adapter adapter-step-002077=outputs/natural-my-v4/train/checkpoints/step-002077 \
  --learning-rate 0.00003 \
  --max-seconds 3300


這一步只讀取權重並生成回答，不執行更新。生成不抽樣：每一步取當下機率最高的下一個token，最多新增384個token。這些是固定驗證設定，不是每題一定會寫滿384個。

這條命令一次產生三份候選：原底座、1,039步修正、2,077步修正。輸出保存generations-base.json、generations-adapter-step-001039.json、generations-adapter-step-002077.json與result.json；語音逐字稿另存transcripts.json，三個聊天候選共用同一份真正ASR結果。先看是否完成全部所求題目，再按用途閱讀回答。僅保存部分結果，就不是整份驗證已完成。

程式會產生原始回答與部分自動統計，還不會替你完成照片與聊天的語義評分或選版。請按manifest.json中的references與本次配方的約定判準，逐題閱讀完整回答與來源素材；若希望減少先看版本名稱的偏見，可以請另一人先把三份生成隱去候選名稱、保留題號，再讓你判讀。照片關鍵字出現或聊天逐字相同，都不能代替用途是否完成。OCR的原樣與指定正規化匹配也要照各題規則，不自行增加簡繁轉換。

照片看物件、動作、關係和無根據新增；OCR 看逐字、正規化規則、換行與順序；聊天看請求和歷史條件。語音先比較逐字稿，再讀正確文字與真正 ASR 輸入的兩份回答。分開的判尺見20.9–20.12，不拿關鍵字碰巧出現當成整段正確。

選版前先約定哪些能力必須保留，以及可以接受哪些取捨。只看一個混合總分，可能讓讀字進步掩蓋聊天退步。候選按驗證判準決定，選定後保存權重、資料、生成設定與這次決定；教材作者的選版紀錄不能替代你自己跑出的結果。

原底座也列在候選中；只有LoRA的回答都完整結束、符合各項保留門檻，且約定的綜合分數嚴格高於底座，才採用修正。本課兩份候選在照片題進步，卻未通過語音聊天保留與回答完整性要求，因此驗證決定保留底座。具體對照集中在20.8，這裡不重複完整成績。ASR在各個圖文候選中保持相同，使用同一份實際逐字稿；這次選版沒有把聽寫改善當成微調的效果。

7. 固定選定版本，再做最後測試¶

確定不再改本版設定後，先保存指紋。教材本版已選底座，你自己重做則依自己的驗證決定。下面假設你的驗證選中了1,039步候選，示範如何固定那份修正；若選中2,077步，後續的SELECTED_ADAPTER都替換成那份實際路徑。若像教材一樣選中底座，保存底座版本、資料與驗證配方，最後命令省略--adapter。

SELECTED_ADAPTER=outputs/natural-my-v4/train/checkpoints/step-001039
sha256sum \
  "$SELECTED_ADAPTER/adapter_model.safetensors" \
  "$SELECTED_ADAPTER/adapter_config.json" \
  docs/natural-assistant/v4/manifest.json \
  outputs/natural-my-v4/validation/result.json \
  > outputs/natural-my-v4/chosen-before-test.sha256


上例以選用LoRA的情況保存指紋。若選的是自己的中途檢查點或續訓版，換成那份實際路徑；若保留底座，則保存底座版本、資料與驗證配置並在最後命令省略--adapter。同時保存已約定的判準與完整生成設定。指紋確定的是同一組檔案，回答能力仍須讀取最後結果。

.venv-natural/bin/python scripts/natural_assistant.py evaluate \
  --output outputs/natural-my-v4/final \
  --manifest docs/natural-assistant/v4/manifest.json \
  --data-root data/natural-v4 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --asr-variant turbo \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 \
  --max-tokens 2048 --max-new-tokens 384 --seed 42 \
  --cache-dir .cache/natural-v4-models --local-files-only \
  --split test --selected-only \
  --adapter "$SELECTED_ADAPTER" \
  --learning-rate 0.00003 \
  --max-seconds 3300


--selected-only只評估已固定的配置；載入修正時只用那份LoRA，省略--adapter則只用底座。最後測試不重新比較兩份修正，也不替你選版；不要據此換權重、改答案或換正規化規則。

最後報告保留完整回答、停止原因與評分範圍。正式摘要使用相同配套的完整結果；照片題數、OCR 區域數、錄音句數及助手回答數各有分母，不合成一個通用正確率。

最後測試發現的不足，可以成為下一版的練習目標。若你已根據這些錯題更改資料或設定，它們就成了已知回歸題；下一版需要另一份尚未參與決策的最後考卷。這樣才能區分「修好了已知題」與「學會了新情境」。

8. 使用自己的模型，或換一台 GPU 執行¶

驗收後，可以用自己選定的配置開啟相同本機介面。採用修正時載入自己的 adapter，保留底座時省略它；兩種情況都配對自己固定的底座與 ASR，不讓教材的公開下載清單替新版本背書。

沿用上一節已經指定的SELECTED_ADAPTER：

.venv-natural/bin/python scripts/natural_assistant.py serve \
  --output outputs/natural-my-v4/ui \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --asr-variant turbo \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 \
  --max-tokens 2048 --max-new-tokens 384 --seed 42 \
  --cache-dir .cache/natural-v4-models --local-files-only \
  --adapter "$SELECTED_ADAPTER" \
  --host 127.0.0.1 --port 8766


若你選的是底座，省略--adapter "$SELECTED_ADAPTER"；其他模型與生成設定仍相同。等服務開始，用相同電腦的瀏覽器開http://127.0.0.1:8766/。這條路直接載入自己的模型配置，不透過教材的公開成品下載清單。

沒有自己的 GPU，可以按GPU 環境指南準備 Modal，再使用自然助理工作流程。底座、資料與驗收概念相同；遠端執行另需設定自己的運算服務與權重儲存。遠端工作流程也要明寫batch_id=natural-v4、manifest=docs/natural-assistant/v4/manifest.json、steps=2077、learning_rate=0.00003、asr_variant=turbo、checkpoint_steps=1039,2077、max_pixels=524288、seed=42與GPU階段max_seconds=3300。使用自己的運算帳號與新的工作名稱，不能把本機outputs/路徑當成遠端檔案。

若卡在資料核對，先分清清單與 --data-root；若快取缺檔，先完成模型準備；若是記憶體不足，檢查設備、其他程序與本題大小。非有限代價、梯度或凍結樣本改變時，保留錯誤和輸出，這次不算完成訓練。驗證若只跑到部分題目，另存完整驗證後才選版。

最後把模型版本、配方、資料來源、實際驗收與資源範圍寫進自己的能力卡，格式可參考20.13。別人拿到你的 LoRA 時，才能知道它對應哪本原書、學過哪份練習，以及已確認哪些用途。

回到頂部