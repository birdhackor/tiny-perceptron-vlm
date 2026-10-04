# 照片、中文讀字與語音：資料怎麼教到用途？

<!-- 作者發布前核對並刪除此區。
本稿預期正式位置 docs/natural-assistant/v4/DATA.md，相對連結按正式位置書寫。
1. 本文只介紹新版素材及用途，不把 v3 素材/成績/舊字卡/CC-OCR混成新資料。
2. 以最終 manifest 作唯一計數：目前 root提供 train照片439；NVIDIA710單行/117多行裁切（153保留原頁），Commons仍最後過濾；chat122/9/13rows；audio38heldout=FLEURS30+AISHELL8，不訓ASR。這些暫時數量只在此註記，不預填正文正式表。
3. 在下方 AUTHOR_DATA_COUNTS 填素材、家族、衍生題數各split與bundle bytes，音訊分兩任務；聊天source及自製分開。
4. fetch_natural_data.py 已支援 --manifest/--revision/--manifest-sha256，但閱讀時 ARCHIVES 只有v3三包；需新增v4 vision/ocr/voice白名單、固定公開Git pin和新manifest SHA，真匿名list/fetch/verify後再插入 AUTHOR_DATA_DOWNLOAD_COMMAND。不只改--manifest稱已可用。
5. 三包源資料目錄 vision/ocr/voice，chat文字在manifest及repo來源JSON；公開學生快照不得含unused錄音或授權尚不確定的素材。
6. reconstruction命令源碼欄位已讀，正文命令仍請真正重建到新目錄並比對final manifest；不能把source上游完整包SHA說成親自全部下載。
-->

模型要學會「桌上有幾隻貓」，需要圖片、這個問題與合適答案。要學抄寫招牌，則需要看得見的字與原樣轉寫；只告訴它「這是商店」，沒有教到每個字。準備資料的第一件事，是讓每道練習對上用途，接著才談題目夠不夠多。

本章接手已有圖文能力的 Qwen3-VL，再用照片問答、中文讀字和聊天示範更新一份 LoRA。聲音由現成 ASR 先轉成文字；真人錄音用於驗收兩站流程，不拿來更新這次圖文 LoRA 或重新訓練 Whisper。各項資料只是有限的教學樣本，模型原本的能力仍來自上游訓練。

概念入口是[20.4 的題型](../../../course/chapters/20.md#20.4)及[20.5 的資料家族](../../../course/chapters/20.md#20.5)。以下說明這一版真正需要的來源、標註與取得方式。

## 1. 先分素材，再製作題目

同一張貓照片可以問數量，也可以問姿勢；同一頁文件可以切出十行字。題目變多了，來源卻沒有變成十張新照片或十頁新文件。如果其中一題用來訓練，另一題稱為新素材測試，就可能提前讓模型熟悉考卷。

因此，我們先按來源家族分 `train`、`validation`、`test`：訓練用來更新權重，驗證用來選設定，最後測試留到版本固定以後。同張照片、同頁文件的裁切與改問法留在同一側。相近照片還依 DOCCI 的場景群組處理；聊天以對話樹分家族；真人問句則把來源錄音與說話者分組一起考慮。

已用來選過版本的題目，之後仍可檢查舊能力是否保留，但不能重新稱為未見過的最後測試。公開底座的上游訓練資料也可能與公共素材重疊；這份切分保證的是本課資料的隔離，不保證上游從沒見過任何相似內容。

執行程式讀取[整合清單](manifest.json)。其中 `rows` 收錄聊天與圖片問答，`audio_rows` 另列錄音及逐字稿；後者不等於帶有助手訓練答案。數素材時看檔案與來源，數練習時看題目，兩個數字應各自保留。

<!-- AUTHOR_DATA_COUNTS
插入一張正式表：用途／獨立素材或對話樹／來源家族／train題數／validation題數／test題數。
至少分DOCCI主場景與可見事實、NVIDIA單行/多行、Commons裁切/整圖、presence、聊天含歷史、FLEURS僅轉寫、AISHELL兩路聊天。
可能同照片供scene與presence，表中合計不得把共享素材重複算成獨立照片。Commons照片及其裁切要區分。
另列三包compressed bytes/unpacked bytes/檔案數，不能從聲音來源原先train誤計新訓練筆數。
-->

## 2. 自然照片：答案只寫畫面支持的事

照片來自 [DOCCI](https://google.github.io/docci/)（[原始論文](https://arxiv.org/abs/2404.19753)）。原資料附有人工英文詳細描述，記錄物件、姿勢和位置關係。本課作者實際查看所選照片、閱讀完整原描述，再寫一至兩句的繁中主場景，以及簡短的可見事實問答。

例如[20.4 的貓照片](../../../course/chapters/20.md#20.4)，兩隻貓站或坐在書桌上，「有幾隻貓」可以直接回答「兩隻」；「左邊的貓在做什麼」則要回答前腳等可見姿勢。畫面沒有提供牠的想法、主人身分或拍照前發生的事，不把這些猜測寫成示範。

繁中題目與回答是 AI 助理依原圖及來源改寫的標註，不是 DOCCI 官方的人工中文答案。每筆保存完整原始英文描述、支持句、圖檔指紋與實際看圖記錄，讓後續使用者能回查。保留來源，也讓另一位讀者有機會發現標註不夠清楚；檔案指紋本身不能證明答案無誤。

使用的圖片是發布者提供的固定縮圖，並非下載整個高解析度原圖庫。大場景可以直接觀察，小筆畫或遠處物件則可能不足以看清。照片裁切或放大只能重新分配可見資訊，不能找回來源圖片原本沒有的細節。完整答案的判尺見[20.9](../../../course/chapters/20.md#20.9)。

來源與配對資料見[照片來源清單](data/vision-sources.json)，活動補充各有[訓練來源](data/vision-activity-sources.json)和[留出來源](data/vision-activity-heldout-sources.json)。照片與原描述由 Google LLC 以 CC BY 4.0 提供，照片署名 Jason Baldridge and family；本課保留署名、來源連結與繁中改寫說明。

## 3. 中文讀字：合成文件與真實招牌各練一件事

合成文件來自 [NVIDIA OCR-Synthetic-Multilingual-v1](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1) 的繁體中文部分。它們是合成文件，不是自然街拍。這份教學包取出有來源轉寫的文字區域，保留單行與多行兩種題型，讓模型練習原樣抄寫。

裁切要同時看文字和範圍。如果只指定抄寫一行，裁切裡卻留了隔壁行的一角，讀者和模型都可能不知道該不該抄它。本包記錄原頁、裁切位置、目標文字與範圍判準；同原頁的全部裁切留在同一個家族。多行題另外指定閱讀順序，不能只因字都出現便算完成。

真實文字照片另取自 [Wikimedia Commons](https://commons.wikimedia.org/) 個別署名與授權的作品，提供招牌、路牌或背景中的中文字。訓練可以用清楚區域，留出題則按題目指定整圖或目標區域，保留文字與周圍情境。斜拍、反光、背景和字距，都可能與合成文件不同，結果要分開看。

Commons 的繁中轉寫由本課作者實際查看圖像後編寫，並非上游另附的人工中文標準答案。逐張來源、作者、授權與修改記錄在[OCR 來源清單](data/ocr-sources.json)，題目及原樣文字在[OCR 標註](data/ocr-labels.jsonl)。NVIDIA 資料採 CC BY 4.0；Commons 每張按自己的 CC BY、CC0 或公有領域條件處理，不把全部照片統一視為 MIT。

是否有中文字另有題型。看見字但讀不清，和完全沒有中文字，是不同情況；英文圖也不能自動被當成中文字圖。對參考空字串，CER 沒有可除的字數，因此有字／無字另列判準。逐字、正規化與行序的差別見[20.10](../../../course/chapters/20.md#20.10)和[20.11](../../../course/chapters/20.md#20.11)。

## 4. 中文聊天：保留完整請求與對話條件

聊天示範包含 [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) 的中文對話，以及本課另寫的短對話。資料樹上的每個助手回合可以形成一筆練習，前面的回合則成為歷史。即使同一棵樹產生多筆題目，仍一起分到同一側，不能用另一個分支偷看測試的前文。

選用的路線保留完整請求和可用回答，再按繁中需求改寫；原始文字與更動原因另外保存。社群回答的來源排名不保證事實正確，簡繁轉換也不能代替語義閱讀。要求「兩句」時應真的給兩句，要求記住飲食條件時應使用那段歷史，不靠資料夾名稱教出指令遵循。

本課另外編寫的條件對話主要用來訓練，例如前一輪說「我不吃辣」，下一輪問晚餐。這些是教材作者寫的文字，不是新增真人對話錄音。後續抽到同一題多次，也只增加練習比重，沒有創造更多獨立情境。

來源、原始對話和題目清單見[聊天來源](data/chat-sources.json)與[聊天標註](data/chat-labels.jsonl)。上游資料採 Apache-2.0，保留原來源及 NOTICE；本課另寫文字的來源身分也逐筆記錄。這份小包適合教資料整理與對話微調，完整中文聊天能力仍需更廣的任務、情境和評估。

## 5. 真人錄音：分開檢查聽寫與聊天

本包有兩類真人朗讀，不是用文字轉語音產生的檔案。第一類是 [FLEURS](https://huggingface.co/datasets/google/fleurs) 的華語朗讀句子，附來源逐字稿，用來檢查 ASR 轉寫。第二類是 [AISHELL-1](https://www.openslr.org/33) 的問句形狀語料，例如「如何加強與員工的溝通」，用來比較同問句的兩條聊天路線。

FLEURS 一句旅遊敘述，可以有正確逐字稿，卻沒有「助手接下來該回答什麼」的來源答案。因此它在這份清單中只測轉寫，不把臨時編出的回覆當成官方聊天真值。AISHELL 問句則先訂好合理回應的語義判準，分別輸入來源正確文字與真正 ASR 辨識文字，再看助理是否完成要求。

這些錄音都只供本章驗證與測試。來源語料原先叫 `train`，也不表示本課曾拿那段音訊重新訓練 Whisper。真人讀出問句仍不等於即興聊天；停頓、地方口音、背景聲、打斷與多人重疊都需要另外驗收。

來源簡體字形與原始錄音位元組保留；AISHELL 輸入文字移除來源分詞空格，這種整理明寫在記錄中，不更正成模型可能輸出的字。評估保存原樣及指定正規化後的 CER，聊天則另讀完整回答。使用者在介面更正逐字稿是人工操作，自動測試不能暗中換入來源真值，差別見[20.12](../../../course/chapters/20.md#20.12)。

逐檔資訊見[FLEURS 來源](data/voice-sources.json)及[AISHELL 問句來源](data/voice-question-sources.json)。FLEURS 採 CC BY 4.0；AISHELL-1 語料採 Apache-2.0。快照核對的是所選錄音，來源大型壓縮包的完整指紋則另標明取得範圍，不能把只讀到的部分串流稱為親自驗過整包。

## 6. 取得固定快照，核對後再使用

只想試用成品，依[學生操作指引](STUDENT.md)即可；準備重新訓練或查看資料時，才需要這些快照。照片、OCR 與語音三包使用 Git LFS 保存，大型二進位檔不直接塞進一般 Git 歷史。聊天文字、標註和來源清單可直接讀 repo。

<!-- AUTHOR_DATA_DOWNLOAD_COMMAND
此處插入真匿名測過的三個完整命令：list、fetch、verify。
應明寫 --manifest docs/natural-assistant/v4/manifest.json --output data/natural-v4，以及真正新 --revision 40hex / --manifest-sha256 64hex。
不可使用 fetch_natural_data.py 的舊v3預設pin/shas；白名單同步更新後才發布。
-->

下載程式先核對固定版本清單，再核對壓縮包，最後逐檔核對解包後的大小與 SHA-256。Git LFS 指標檔只是一張告訴 Git「真正檔案在哪」的小紙條，不是圖片或錄音包本身；完整資料必須通過指紋檢查才能使用。

成功後的 `data/natural-v4` 內，`vision/` 放照片，`ocr/` 放文字圖片與裁切，`voice/` 放錄音，各包保留 ATTRIBUTION、NOTICE 及相應授權。訓練程式以 repo 的 `manifest.json` 配對題目和這些相對路徑；清單所在位置不等於 `--data-root` 的資料目錄。

已有相同資料時核對後重用；不同內容時停止並保留原目錄，改用新的下載位置即可。自己的檔案另放新目錄，不修改正式來源真值來配合模型答案。指紋只能證明拿到哪份檔案，回答和標註是否符合畫面仍要閱讀內容。

## 7. 從來源重建，或設計自己的下一份資料

固定快照是最簡單的學生入口；想研究材料如何形成，可以從來源重建。這條路讀公開固定來源、重做指定裁切並保留同一標註，可能需要比直接下載小包更多的讀取量。它還原相同資料，不會替你重新挑題、改答案或訓練模型。

<!-- AUTHOR_DATA_RECONSTRUCTION_COMMAND
源碼支援以下命令，發布前請root於新目錄重建並和最終清單比對：
.venv-natural/bin/python scripts/build_natural_v4_assets.py --reconstruct --data-root data/natural-v4-rebuilt --assets outputs/natural-v4-rebuilt/archives --manifest outputs/natural-v4-rebuilt/manifest.json --verify-manifest docs/natural-assistant/v4/manifest.json
此命令不得直接寫canonical assets或manifest。記錄實際網路讀取範圍與全部來源/所選素材核對，不宣稱親自下載上游完整大包。
-->

想加入新情境時，先另建資料版本，按來源家族重新安排切分，再寫與用途相符的題目。訓練資料變了，原來的最後考卷可能已被你參考過；評估時要分清已知回歸題與新的留出題。保留來源、修改與判準，才能讓下一位讀者理解你究竟教了什麼。

資料準備好後，按[訓練指引](TRAINING.md)更新自己的 LoRA，再依[20.9–20.12 的判尺](../../../course/chapters/20.md#20.9)核對不同用途；最後把能力與範圍寫進自己的能力卡。
