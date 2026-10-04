## T.1 先決定要拿哪些例子教模型

訓練資料是交給模型練習的題目，不是模型已學好的數字。最簡單的文字資料是一篇篇短文；對話資料則保存問題與希望它回答的內容。先看[1.2的背誦與新題](chapters/01.md#1.2)：教過的題目與用來檢查的新題需要分開，同一句話重複出現不能當成新的能力證據。

第一次動手可先用本頁後面的規則資料。它們由程式生成，不用網路，也容易知道標準答案。如果想用公開短故事、對話、圖片或聲音，可以到[固定資料包說明](../assets/training/README.md)挑選；這批是小型起步樣本，不是完整的大規模語言訓練集。

```bash
python scripts/fetch_training_assets.py --list
python scripts/fetch_training_assets.py --asset tinystories
```

第一行列出可選的包與大小，不訓練模型；第二行只取得 TinyStories 這個短故事樣本包，核對檔案後解到 `data/training/`。資料包的來源、版本、授權和 SHA-256 都有記錄。SHA-256 是由檔案內容算出的指紋，用來確認取得的內容與記錄一致；它不保證資料本身沒有偏差或錯誤。

現在不只備好了資料，也已完成[T.4的兩個小型文字實驗](#T.4)：TinyStories包有512篇完整英文故事，中文詩包有365首完整古典詩。前者選自上游training材料，後者是未切分的詩集；我們在訓練前自行留出一部分來檢查，而不是把包內每篇都交給模型練習。這份自己的最後檢查不能冒稱官方test，唐詩的結果也不能代表現代中文對話能力。完整來源、固定版本與資料授權可由[實跑報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/real_text.json)追到各資料包的記錄。

把資料讀進訓練工具前，先打開幾筆看實際欄位，再安排未用來教模型的題目。例如選[本頁T.4的屬性問答](#T.4)，把單一能力限定為「從列出的屬性回答形狀」，答案必須與shape欄位完全相同。這裡先用可直接閱讀的示意小表練習分組，不需要執行指令，也不代表產生器預設會採用這份切分。

| 題號 | 教學輸入 | 標準答案 | 題目家族 |
| --- | --- | --- | --- |
| A | `color=red;shape=square;pitch=low;shape?` | `square` | red:square:low |
| B | `color=red;shape=square;pitch=low;describe` | `square` | red:square:low |
| C | `color=blue;shape=circle;pitch=high;shape?` | `circle` | blue:circle:high |

此例的describe也要求回答形狀。A與B只是同一組屬性換個問法，要放在同一側；可以用A、B教學，把C整個家族留作最後檢查。實際T.4產生的對話紀錄，問題在messages中role為user的content，標準答案在role為assistant的content，family保存題目家族。先查看這三處，就能分清原始資料、教學輸入與希望回答。文字接續的TinyStories則使用text欄位保存完整故事，沒有這種問題／標準回答配對，不應硬把兩種格式當成同一種。

圖片與音訊尤其要核對檔案路徑、尺寸與取樣率。一段錄音是一筆訓練例子；取樣率的「樣本」則是聲音測量點，例如每秒8,000個點，兩者不同。這批FSDD原錄音為8,000點／秒，[T.6的音訊訓練入口](#T.6)要求16,000點／秒且不自動轉換；取樣與時間軸見[12.1](chapters/12.md#12.1)。本節先選資料，使用前需明確重採樣，不能只改檔名。

現在也有[真實圖片與錄音的完整實跑](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/real_modal.json)。Fashion-MNIST的50張原training圖，十類各按3／1／1張分開，成為我們自己的30／10／10；FSDD的60段則固定jackson教學20段、nicolas驗證20段、theo最後檢查20段，每人0至9各兩次，不把同一說話者拆到三側。服飾測試3/10、真人數字3/20，皆只是一個很小的配方結果，不是官方benchmark或自然圖片問答。錄音從8kHz實際插值成16kHz、點數加倍而時長不變，完整方法見[12.2](chapters/12.md#12.2)，辨識失敗見[12.8](chapters/12.md#12.8)。

現在先預測：用A教學，再拿B答對，能否證明模型會做新的屬性組合？按[1.2的完整題目分組](chapters/01.md#1.2)核對，答案是不能，因為A、B屬於同一家族。把它們一起留在教學側，C留在最後檢查側；C的合格答案是circle，答square就不合格。再選一包，仿照小表寫出一筆教學題、一筆未教過家族的新題、各自標準答案與分組理由。若還不能定義答案對錯，先縮小任務，再增加資料。

