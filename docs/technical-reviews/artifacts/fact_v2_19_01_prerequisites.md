## 7.1 對話怎麼表示？

模型會續寫短句，還不代表它懂用戶正在提問。要讓它學習問答，資料先把「誰說話」與「說了什麼」分開保存，再串成有邊界的一條序列。模型一次處理的片段或結構標記叫token，實際切多大由切詞方法決定，不必是一個完整中文字。這裡模型仍按[4.7](04.md#4.7)猜下一token，只是前文現在含問題，期望後續是回答。

一筆對話用消息清單表示，每條消息是一個字典，`role`記角色、`content`記內容。user是使用者，assistant是回答方；這些英文值是資料工具的約定，不是要模型靠普通文字猜身份。清單與字典見[W.2](../first-steps.md#W.2)，結構專用ID見[6.6](06.md#6.6)。

本節用位元組切詞：byte（位元組）能保存0到255的數，用八個只有0或1的位元記錄。UTF-8把文字轉成一串這樣的數；英文字元常用一個byte，中文字可能用三個，詳見[6.1](06.md#6.1)。本例的普通內容每個byte占一個token，開始、結束與角色標記則各占一個專用token。後面的ByteTokenizer就是做這種編碼的工具。

```python
from tiny_perceptron.data import ByteTokenizer, render_chat

messages = [
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "2"},
]
tok = ByteTokenizer()
x, y = render_chat(messages, tok)
print("模型輸入X", x.tolist())
print("答案標籤Y", y.tolist())
print("有效目標", y[y != -100].tolist())
```

`render_chat` 將消息轉成一條序列並建立下一token答案。原始順序為BOS、user、問題bytes、EOS、assistant、答案bytes、EOS。BOS表示開始，EOS表示一段結束；轉換後x取除末項，y則取右移後的監督標記，長度都是10。輸出X為 `[1,3,57,51,57,69,71,2,4,58]`，其中4是assistant，58是字元2的byte50加8。

Y前八項都是-100，最後為58與2；-100是忽略這個位置的直接答案代價，不是輸入字表ID。`y[y!=-100]` 用真/假條件選出有效項，應輸出 `[58,2]`，要求模型學會回答2以及回答結束。user問題仍在X裡供後面的注意力讀，不需要它們每格都作為loss答案。下一節會看推論開頭，之後再拆監督位置的對齊。

這種用正確示範回答繼續訓練的方法叫SFT（supervised fine-tuning，監督式微調）。fine-tuning表示在已有模型上繼續調參數；本文先建立資料接口，實際接續與評估會在 [7.11](#7.11) 展開。資料格式不等於能力，只有按正確目標更新、再用留出題檢查，才知道是否學會回應。

每條消息的role與content都應來自明確欄位，別靠掃描內容裡的 `<assistant>` 判斷換角色；用戶可能只是討論這串符號。多輪對話則按清單順序串接，保留每次邊界；不應把每句各編碼後丟掉是誰說話。

練習只把問題改成 `"2+1=?"`，先預測角色邊界、長度與有效答案仍58、2，只有問題中的某個byteID改變，再執行核對。答案還固定2，是為了隔離格式變化；若作為正確算術訓練例，應另外將回答改成3，不能讓資料默默帶着舊錯答案。


## 10.1 圖片是什麼 tensor？

手機畫面放大後會看到一格格像素。每個像素的紅、綠、藍亮度組合，形成我們看到的顏色；紅色像素可以用三個數字[1,0,0]表示。當圖片要進程式，這些數字需要有明確的排列。本節只回答「圖片的各條軸代表什麼」，不要求模型已能認出物件。

前置是[W.3的tensor與軸](../first-steps.md#W.3)。回顧一下，tensor是一個有形狀的數字盒子，每個軸可以用索引選位置。RGB指紅Red、綠Green、藍Blue三個通道；單張圖常用[C,H,W]，C是通道數、H是高、W是寬。因此[3,16,16]是有三種顏色通道的一張圖，不是三張圖。

```python
from tiny_perceptron.multimodal import scene

image = scene("red", "square")
print("形狀", tuple(image.shape))
print("中央RGB", image[:, 8, 8].tolist())
print("左上RGB", image[:, 0, 0].tolist())
print("加入批次軸", tuple(image.unsqueeze(0).shape))
```

輸入`scene`是專案的合成圖片工具：red要求紅色，square要求方塊，預設大小16×16，背景黑色。`tuple(image.shape)`把軸長度轉成括號形式供列印；`.tolist()`把取出的tensor數值轉成Python清單，兩者都不會修改原圖片。`image[:,8,8]`保留全部顏色通道，取第8列第8行的同一個像素；索引從0開始。輸出形狀`(3,16,16)`、中央`[1.0,0.0,0.0]`、左上`[0.0,0.0,0.0]`。0表示該色沒有亮度，1表示這個例子設定的最大亮度，所以方塊紅而背景黑。

`unsqueeze(0)`在最前面增加長度1的軸，得到`(1,3,16,16)`。第一個1叫批次大小，表示一次送一張圖；這個操作沒有新增像素或改變顏色。若一次放兩張圖，批次軸會是2。不同讀檔工具也可能給[H,W,C]，這時需要換軸成模型要求的順序，不能把數字表直接誤讀為另一種形狀。

實際照片常以整數0到255保存亮度，轉成浮點數並除255後可落在0到1。此處`scene`已直接提供這個範圍，不要再除一次255而讓數值縮小。圖的數值尺度、通道順序與訓練設定必須一致，否則同一顏色送進模型也可能被解讀錯。

若把三條通道當成三張灰階圖，也不是完全無法顯示，但會改變你正在描述的物件：紅通道裡方塊是亮的，綠與藍通道裡全黑，三者合起來才是原紅色圖。中央索引8、8選的是同一座標，不是分別在不同位置找紅綠藍。批次軸則用來分不同圖片，例如image_batch[0]取第一張整圖，image_batch[0,:,8,8]才取它的中央RGB。清楚說出每個索引選哪一軸，比記住一個形狀字串更能避免後面切塊時排錯。

練習只把red改成blue，先預測中央RGB為[0,0,1]、形狀保持不變，再跑程式核對。你會看到改變的是第三個通道的亮度，不是第三張圖片。能這樣讀懂軸，下一節才能把圖片切成小塊而不打亂像素。


## 12.1 音訊是什麼 tensor？

一根琴弦振動，空氣壓力隨時間上下起伏。數位音訊把這種起伏按固定時間間隔記成一串數字，叫波形waveform。一次起伏有多大是振幅，起伏重複有多快是頻率。音比較高不代表振幅一定更大；本節先認清模型最前面收到哪一串數字。

前置是[W.3的一維tensor](../first-steps.md#W.3)。單聲道有N個時間樣本，形狀[N]；一批B段等長單聲道可為[B,N]。左右雙聲道則需要另外一條通道軸，不能把左右兩串接成一段兩倍長的錄音，否則時間意義會變。

```python
from tiny_perceptron.multimodal import tone

wave = tone(440.0)
print("樣本形狀", tuple(wave.shape))
print("前三個值", [round(v, 3) for v in wave[:3].tolist()])
print("最小最大", round(wave.min().item(), 2), round(wave.max().item(), 2))
```

輸入`tone(440.0)`生成440Hz的正弦單音，Hz表示每秒重複幾次。正弦是一種特定的平滑曲線；在這個例子裡，它規律地從0升到最高、回到0、降到最低、再回到0，這完整一輪叫一個週期。440Hz就是每秒走完440輪。工具預設長0.1秒、每秒記錄16000個樣本，振幅0.5。它不讀取檔案或播放聲音，直接算出浮點tensor。輸出形狀`(1600,)`，前三值約[0.0,0.086,0.169]，最小最大約-0.5與0.5。波形先從0向上，之後會繼續上下往復；負數不是錯誤，而是相對基準的另一方向起伏。

每個數字本身沒有附時間，必須另外知道取樣率16000，才能說第1個索引在1/16000秒。1600是樣本數量，不是1600Hz，也不是1600個語言token。進入音訊編碼器之前還會切成短時間框，每框形成一條特徵，這個轉換會在後面逐步解釋。

振幅是這份數字記錄的尺度，常與聲音強弱有關，但真實聽感還受設備與頻率影響；不要把0.5直接叫固定分貝。頻率440控制起伏速度，而工具裡的0.5控制上下幅度，兩者在這個生成器裡能獨立改變。

若把波形畫成圖，橫軸是時間或樣本索引，縱軸是振幅；曲線越密集通常表示起伏越快，而不是樣本編號越大代表聲音越高。下圖只放大一個週期，用「這一輪走完多少」作橫軸：走到1/4輪時振幅是0.5，1/2輪時回到0，3/4輪時是-0.5，走完一輪又回到0，接著重複。

![振幅0.5的正弦波，一週期從0到最高、0、最低，再回到0](../figures/multimodal_wave_cycle.svg)

相位指起伏目前走到週期中的哪個位置，可以先當作「這一輪的進度」。相位不是振幅：同樣振幅0，可能是在起點向上，也可能是在半輪時向下。工具從圖左的0開始，索引0是第一點；索引1是1/16000秒後的第二點，這時440Hz的波走了440/16000輪，還沒到最高點，生成器算出的振幅約0.086。要解釋的是取樣時刻在曲線上的哪裡，不需要先學正弦公式。

這條曲線是確定生成的測試輸入，方便我們知道答案，不是從音樂中成功抽取了音高。真實錄音還可能有噪聲、靜音或多個聲源，單看前三個值無法辨識全部內容；後面才會在短框內匯整這些變化。

練習只把440改880，先預測形狀仍1600、極值仍約±0.5，而前幾個值變得更快上升，再執行核對。它在同樣時間內起伏約兩倍次數，不會因此多出兩倍樣本。下一節將解釋樣本數量如何與真實時間聯繫。


## B.5 同樣有數字，何時需要計算器？

學生知道1加2等於3，老師仍可能要求計算題用計算器，留下可核對的結果；但問「加法是什麼」，拿出計算器反而沒有解釋概念。同樣出現數字或「加」字，眼前要完成的工作卻不同。本節先訂一個明確的工作習慣：需要精確數值時使用可用的計算器，需要說明或照抄已給的資訊時直接回答，缺少必要資訊時先詢問。

前置是[B.3工具真的執行與回填](0B.md#B.3)和[B.4完成信號](0B.md#B.4)。直接回答也仍然是在產生文字；使用工具時，模型先產生的是請求文字。「預測下一個token」描述輸出的方式，token是[第6章的文字單位](06.md#6.1)，並不代表模型內部不能學到計算步驟。這裡改變的是下一段文字要做什麼，而不是替模型加上人類的自我意識。

先約定三張動作卡。DIRECT表示直接回答；TOOL表示接下來需要工具；ASK表示先向使用者求助。在資料不足時，求助是問缺少的數字；若任務要求可核對的精確計算，計算器卻停用，求助可以是請使用者啟用工具或提供其他核對方式。三個英文名稱只是程式方便辨認的固定標記，不是給使用者看的回答。

| 使用者問題 | 本例的條件 | 適合的動作 |
| --- | --- | --- |
| 1加2等於多少？ | 計算器可用，要求精確數值 | TOOL |
| 請解釋加法的意思。 | 問概念 | DIRECT |
| 請原樣回答數字3。 | 答案已經提供 | DIRECT |
| 請算總價。 | 尚缺單價與數量 | ASK |
| 1加2等於多少？ | 計算器停用，仍要求可核對計算 | ASK |

這是我們選定的教學策略，不是數學題永遠只能用工具的規定。對某個已可靠掌握的小任務，直接回答也可能更省時間；[B.8](0B.md#B.8)才會比較這個取捨。現在先用固定策略，避免同時要求初學者判斷能力、等待成本和資料是否齊全。

下圖的三條路都從同一份問題與工具狀態出發。綠色是直接回應，藍色是送往[B.1–B.4](0B.md#B.1)的工具流程，橘色是先補齊資訊或核對方式。箭頭表示可選的路，不代表目前已有一個模型能選對；接下來的程式也只是人工標註。

![同一個問題可能直接回答、使用工具或先求助，依任務需求與工具狀態選擇](../figures/tool_choice.svg)

```python
examples = [
    {"question": "1加2等於多少？", "calculator": True, "action": "TOOL"},
    {"question": "請解釋加法的意思。", "calculator": True, "action": "DIRECT"},
    {"question": "請原樣回答數字3。", "calculator": True, "action": "DIRECT"},
    {"question": "請算總價。", "calculator": True, "action": "ASK"},
    {"question": "1加2等於多少？", "calculator": False, "action": "ASK"},
]
for row in examples:
    print(row["question"], "計算器可用", row["calculator"], "→", row["action"])
```

輸入是五筆由我們寫好的標註，沒有載入或更新模型。每筆字典的question保存問題，action是期望動作。calculator只有True或False兩種值：True表示計算器可用，False表示停用；這種只表示「是／否」的值叫布林值。字典、清單與迴圈可回看[W.2](../first-steps.md#W.2)。輸出依序為TOOL、DIRECT、DIRECT、ASK、ASK。第一筆與最後一筆問題相同，只換了工具狀態，合適動作便改變；第二筆有「加」字，仍不需要算兩個數字。因此不能只靠找到關鍵字就宣布完成任務判斷。

練習只把第一筆calculator改成False，先預測原本action=TOOL會違反這份策略，再執行觀察。程式仍會印TOOL，因為它只讀出我們填的標註，沒有自動重標。將action改成ASK後再核對，並用自己的話說明求助的原因：這次不是缺數字，而是缺少策略要求的核對方式。


## 19.11 下載一份權重，就能從中斷處繼續訓練嗎？

把助理交給另一位學生時，有兩種需求：一位想直接試用，另一位想接著完成訓練。第一位需要模型權重與正確的結構、tokenizer；第二位還需要更新器狀態、隨機狀態、抽樣位置與訓練階段。只帶走期末作業，和帶走能繼續作業的完整工作桌，資料量與用途不同。

前置是[checkpoint與resume](05.md#5.7)、[tokenizer與模型的配套](06.md#6.5)、[本章父階段](19.md#19.4)與[量化、Dense學生兩條壓縮路線](19.md#19.10)。FP32成品使用`capstone-v1`格式，量化儲存版使用`capstone-ptq-v1`；檔案另外標明資料版本與階段。公開推論檔可以去掉更新器等狀態，方便下載；去掉之後就不能稱能完全恢復原訓練軌跡。

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from tiny_perceptron.capstone import CapstoneModel, load_capstone, save_capstone

model = CapstoneModel()
with TemporaryDirectory() as directory:
    path = Path(directory) / "example.pt"
    save_capstone(path, model, stage="sft", step=0, inference_only=True)
    loaded, metadata = load_capstone(path)
    print("格式", metadata["format_version"], "訓練步數", metadata["step"])
    print("只供推論", metadata["inference_only"], "參數相同", model.description() == loaded.description())
```

這段把隨機模型存到暫存資料夾，再真的讀回，結束後暫存夾自動清除。step=0明確表示沒有完成SFT；`stage="sft"`是格式標籤，不能拿來冒充已訓成果。輸出應為capstone-v1、0、True、True；最後一個True只確認架構與參數數量描述一致，完整數值與逐題輸出相等還要另檢。

正式交付把小型合成資料與來源規格保留在本專案，權重放[Hugging Face公開模型庫](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed/course/course-integration-v2)。本輪已逐檔匿名下載、核對指紋並讀回全部11份模型，正式[公開清單](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-public.json)固定在revision `33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed`。revision像這一版包裹的封條編號；SHA-256則是每個檔案的內容指紋，不只檢查檔名相同。下載不需要登入或提供token。

先照[19.1](19.md#19.1)準備Python環境，在專案資料夾執行：

```bash
python scripts/fetch_capstone.py --list
python scripts/fetch_capstone.py --stage joint
```

`--list`只印出11份清單，不下載權重，也不訓練。每一行的`stage`是下一行`--stage`可用的名字；`joint`是驗證後選出的推薦成品。其他名稱可這樣讀：

| 可以下載的名字 | 用途 |
| --- | --- |
| `pretrain`、`sft`、`joint`、`dpo` | 四站FP32推論檔，用來比較加入一項訓練後的變化 |
| `joint-int4`、`joint-int8` | 推薦joint的4／8-bit儲存版 |
| `dpo-int4`、`dpo-int8` | 偏好分支的4／8-bit儲存版 |
| `student-ce`、`student-kd`、`student-kd-int4` | [19.10](19.md#19.10)較小Dense學生的答案訓練、蒸餾與蒸餾後量化比較 |

FP32指權重以32-bit浮點數儲存；`int4`與`int8`指這裡的壓縮儲存格式，載入後仍還原成FP32計算，並不保證較少執行記憶體或更快。Dense學生的能力也不能直接當成joint能力。每個下載資料夾都包含`model.pt`、模型卡`README.md`、`LICENSE`、`THIRD_PARTY_NOTICES.md`與這次合成資料`data.json`；設定與tokenizer另存在模型檔內。各份用途、父檔來源與失敗案例要連同模型卡閱讀。

第二行會安裝到`checkpoints/capstone/joint/`。其中公開的`model.pt`是1,327,950 bytes，SHA-256為`d85cca83cdb4952f4653ef6b58d94562b41403db3a9db2ea31ff57e31246ca16`。這是移除訓練內部狀態與整理交付資訊後的正式檔案，不能直接拿[19.10](19.md#19.10)實驗當時的檔案bytes當它的大小；權重數值保留相同，封裝資訊有變。下載程式會先核對全部五個檔案的大小、SHA與模型身分，成功後才把完整資料夾放到目的地。若joint已在，就直接使用它；想另留一份，可以改用`python scripts/fetch_capstone.py --stage joint --output checkpoints/capstone-second`，後續模型路徑也改成`checkpoints/capstone-second/joint/model.pt`。

量化版可以另外下載，並用相同的CPU命令讀回：

```bash
python scripts/fetch_capstone.py --stage joint-int4
python scripts/capstone.py infer --checkpoint checkpoints/capstone/joint-int4/model.pt --prompt "1+2等於多少？" --device cpu
```

這份公開joint-int4檔是270,855 bytes；下載清單保留它自己的SHA。命令會選正確載入方式，再跑同一工具迴圈。若換成`student-kd-int4`，要把下載名稱與路徑一起換；成功下載、成功載入，仍不代表答案正確。例如本輪這位學生把1+2請求錯寫成2+9，工具回11後自己答12。我們保留這類失敗，不把「程式能跑」算成「模型會答」。本機網頁的`serve`目前只接受FP32格式，所以仍使用19.1的`joint/model.pt`；不要把`joint-int4/model.pt`交給它。

中斷續訓使用另外的`model-training.pt`與原有資料版本。DPO還需儲存reference狀態；只重新建立一份會隨當前policy改變的reference，並不等於接著原實驗跑。是否能逐步重現，也受裝置與運算實現影響，因此報告應區分同環境恢復測試與跨環境重新訓練。

如果要自己重做四站，下面才是會更新權重的命令；它們不是上方試用的必要步驟，也不會在本節短程式中自動執行。`--batch-size 24`表示每次更新取24筆訓練資料一起計算誤差；`--seed 42`用同一個種子數字設定新實驗的隨機初值與取樣起點，方便在相同條件下比較實驗。先用自己的資料版本與程式產生pretrain，再把每站真正完成的原始`model.pt`交給下一站。請一行一行執行，先讀該站的`train-report.json`，確認`schedule_completed`為`true`，才往下走：

```bash
python scripts/capstone.py train --stage pretrain --output checkpoints/my-capstone/pretrain --steps 300 --seed 42 --batch-size 24 --device cpu
python scripts/capstone.py train --stage sft --input-checkpoint checkpoints/my-capstone/pretrain/model.pt --output checkpoints/my-capstone/sft --steps 1400 --seed 42 --batch-size 24 --device cpu
python scripts/capstone.py train --stage joint --input-checkpoint checkpoints/my-capstone/sft/model.pt --output checkpoints/my-capstone/joint --steps 600 --seed 42 --batch-size 24 --device cpu
python scripts/capstone.py train --stage dpo --input-checkpoint checkpoints/my-capstone/joint/model.pt --output checkpoints/my-capstone/dpo --steps 100 --seed 42 --batch-size 24 --device cpu
```

這四行示範本課原配方的階段關係，並沒有另跑一次CPU訓練來宣稱重現正式GPU分數；各站實報在[19.4](19.md#19.4)。DPO是要觀察的比較分支，不是成品必須接受的最後更新：本輪推薦的是joint。命令每次預設最多訓練540秒，電腦較慢時可能只完成部分步數。到了時間上限仍會留下工作檔；例如自己那站joint原本排定600步、中途停下，可以用：

```bash
python scripts/capstone.py train --stage joint --resume checkpoints/my-capstone/joint/model-training.pt --output checkpoints/my-capstone/joint --steps 600 --seed 42 --batch-size 24 --device cpu
```

這裡的600是原本總步數，不是追加600步。`--resume`要求同一站、原定步數、batch大小、資料指紋與程式指紋一致，並讀回更新器、隨機狀態和進度；取樣接著工作檔的位置走，不會因為再次寫seed 42就從第一筆重來。DPO還要原reference。若尚未完成，先繼續同一站，不要以不完整父檔開始下一站。跨站的`--input-checkpoint`則以已完成父模型開始一份新的更新器與抽樣流程，兩者用途不同。實際檢查條件可見[訓練器原始碼](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/scripts/course_experiments/capstone.py)。

公開推論檔沒有上述完整工作狀態，也沒有本課串階段命令所需的原始`data_manifest`與完成排程證明，因此不能把公開`checkpoints/capstone/joint/model.pt`代入這裡的`--resume`或正式`--input-checkpoint`。推論權重本身仍可作為另寫訓練程式的初始數值；那是建立一個新實驗，並不是恢復本課已驗證的訓練。公開交付的是可試用、可比較的11份推論版本，不包含作者的完整續訓工作檔。

練習把step改成1，觀察metadata確實顯示1。但模型仍沒有更新過，這說明格式欄位可以寫入，卻不能自行證明訓練發生；正式報告還需有效目標、權重變更與執行記錄。

