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

--checkpoint-every 25保存的是供續訓使用的最近進度：每完成25的倍數步，更新train/adapter/；若距離上次這種存檔已滿60秒，也會在完成一次更新後保存。這個目錄會換成較新的完整版本，正常結束時還會再保存一次。因此它不代表每25步都留下獨立候選。--checkpoint-steps 1039,2077才另外保存上表兩份不可覆寫的候選，後續更新不會改動它們；最近進度用來恢復工作，兩份候選用來固定比較時刻。

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

本次約定先檢查下列門檻。表中的「底座」指同一份驗證題上，未加修正的原模型；題數比較的是各項答對幾題，不能以其他用途的進步抵銷。

檢查用途	修正候選必須滿足什麼？
回答停止情況	全部132份回答都實際以EOS結束，沒有截斷或停止原因不明。EOS是模型的結束符號；單看「已生成132份」還不夠。
照片主描述、指定順序讀字	每項答對題數都至少與底座相同。
照片可見事實、有無中文字、單區讀字、一般文字聊天	每項最多比底座少答對1題。
正確文字問句聊天、真正ASR文字問句聊天	兩路各自的答對題數都至少與底座相同。

通過門檻後，才比較判準中的綜合分數primary。正確率是答對題數除以該用途題數；先算五組能力，每組在綜合分數中占五分之一：

能力組	怎麼算這組分數？
照片	28題主描述正確率與56題可見事實正確率的平均。
有無中文字	18題的正確率。
單區讀字	10題的正確率。
指定順序讀字	3題完整逐字與順序都符合判準的正確率。
聊天	9題一般文字聊天、4題正確文字問句聊天、4題真正ASR文字問句聊天，三個正確率的平均。

再把五組分數相加除以5。這樣照片題數較多，也不會因為題多就占掉其他用途的比重。聽寫的字元錯誤率CER另行報告，不計入這個LoRA綜合分數；聊天組計的是助理能否回答問句。

用一個假設算例練習：底座的五組分數若為1/2、1、4/5、1/3、1，綜合分數就是它們的平均，109/150，約72.67%。若某個候選把照片提高到3/4，卻把正確文字問句聊天從4/4降為3/4，另兩路聊天仍滿分，聊天組便是(1＋3/4＋1) / 3 = 11/12。其餘不變時，候選綜合分數是19/25 = 76%；分數雖高，仍因那一路聊天退步而不能選。這些是假設數字，並非本版成績。

實際選版使用未四捨五入的分數比較：候選必須嚴格高於底座；符合門檻的候選中選分數最高者。兩個修正同分就選較早保存的候選；與底座同分，或沒有修正通過全部要求，都保留底座。小份驗證中一題就可能改變取捨，而且同照片的題目彼此相關，這個分數不能當成一般使用的成功率保證。

本課兩份候選在照片題進步，卻未通過語音聊天保留與回答完整性要求，因此驗證決定保留底座。具體對照集中在20.8，這裡不重複完整成績。ASR在各個圖文候選中保持相同，使用同一份實際逐字稿；這次選版沒有把聽寫改善當成微調的效果。

7. 固定選定版本，再做最後測試¶

確定不再改本版設定後，先保存指紋。教材本版已選底座，你自己重做則依自己的驗證決定。若也保留底座，下面這段只用Python整理已生成的紀錄，不載入模型或使用GPU：

.venv-natural/bin/python - <<'PY'
import json
import shutil
from hashlib import sha256
from pathlib import Path

manifest = Path("docs/natural-assistant/v4/manifest.json")
validation = Path("outputs/natural-my-v4/validation/result.json")
protocol = Path("docs/natural-assistant/v4/validation-protocol-lower-lr.json")
record = json.loads(validation.read_text())
assert record["status"] == "completed" and record["split"] == "validation"
assert record["manifest_sha256"] == sha256(manifest.read_bytes()).hexdigest()
configuration = {
    key: record[key]
    for key in (
        "model", "model_revision", "asr_model", "asr_revision",
        "device", "dtype", "min_pixels", "max_pixels", "max_tokens", "seed",
        "manifest_sha256",
    )
}
configuration.update(
    selected_variant="base", adapter=None, max_new_tokens=384, do_sample=False,
)
snapshot = Path("outputs/natural-my-v4/chosen-base-before-test")
snapshot.mkdir()
for source in (manifest, validation, protocol):
    shutil.copyfile(source, snapshot / source.name)
(snapshot / "configuration.json").write_text(
    json.dumps(configuration, ensure_ascii=False, indent=2) + "\n"
)
(snapshot / "files.sha256").write_text(
    "".join(
        f"{sha256(file.read_bytes()).hexdigest()}  {file.name}\n"
        for file in sorted(snapshot.iterdir())
    )
)
print(snapshot)
PY


這會保存底座與ASR的實際revision、資料清單、驗證報告、判準及生成配方。max_new_tokens=384與do_sample=False對應第6步的固定生成方式；若你另改過生成設定，要保存自己的實際配方。目錄已存在時程式停止，避免覆寫先前決定。保存後，最後測試命令省略--adapter，只評估底座。

若驗證採用修正，則保存那份權重的指紋。下面假設你的驗證選中了1,039步候選；若選中2,077步，後續的SELECTED_ADAPTER都替換成那份實際路徑：

SELECTED_ADAPTER=outputs/natural-my-v4/train/checkpoints/step-001039
sha256sum \
  "$SELECTED_ADAPTER/adapter_model.safetensors" \
  "$SELECTED_ADAPTER/adapter_config.json" \
  docs/natural-assistant/v4/manifest.json \
  outputs/natural-my-v4/validation/result.json \
  > outputs/natural-my-v4/chosen-before-test.sha256


上例以選用LoRA的情況保存指紋。若選的是自己的中途檢查點或續訓版，換成那份實際路徑，同時保存已約定的判準與完整生成設定。指紋確定的是同一組檔案，回答能力仍須讀取最後結果。下面命令沿用修正候選的示例；保留底座時刪去--adapter "$SELECTED_ADAPTER"這一行：

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