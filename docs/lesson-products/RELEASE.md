# B／C v1 發布與執行紀錄

學生操作見 [README](README.md)；本頁保留維護者回查來源的方法。

正式模型庫是 [固定 HF commit](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016)，共32檔。`tiny_perceptron/lesson_products/release.json`固定B／C推論清單，公開包另含兩個機制對照的完整最後checkpoint、最後報告、設定凍結檔與壓縮逐題分數。原始分數的大小與SHA在包內`raw/`清單；普通Git沒有加入原始大型分數或權重。

[測試前commit](https://github.com/birdhackor/tiny-perceptron-vlm/commit/821a2c717f05ad7b57e08e88cd965e92ff0c1200)先固定設定，再首次執行最後測試。正式結果沒有用來重新訓練或挑設定。後續教材回填的commit不同，不會改寫那次權重、考卷或成績。

[正式發布工作](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/38871109465)使用來源commit `b137418e7e4fda757ab295fafa8515bc98a53f56`。[發布收據](results/publication-v1.json)、[規格副本](results/publication-spec-v1.json)及[包內檔案清單](results/publication-package-v1.json)保留實際來源；收據中的`source_hashes`路徑要在該次執行commit解析，不能直接假設是目前main的檔案。

傳輸使用隔離分支`lesson-products-transport-20261016`的LFS包，再送到既有draft。[真正傳輸工作](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/38869556549)完成上傳、依asset ID取回及SHA／大小比對，[原收據](results/transport-roundtrip-v1.json)保持不變。大型包與一次性傳輸工具留在該分支；普通main沒有合併它們。手動transport workflow是為選定該來源分支執行而登記，不能以main代替原來源；原draft已有asset，再次執行也會拒絕覆寫。學生下載直接使用公開HF固定版。

首次publisher工作因GitHub draft讀取權限而在預約前回403；後續只調整該workflow的GitHub權限，沒有提高Modal額度。完成的publisher只用有上限CPU，沿用既有gross政策；`reserved_usd`是預約上限，不是實際花費。官方workspace gross觀測也不能當成這一個工作的獨立費用。

作者在新CPU環境與空產品目錄完成[實際CLI／HTTP檢查](results/public-cpu-acceptance-v1.json)。獨立審核者另匿名取回32個固定HF檔案逐一核對；訓練、CPU示範、逐題重算與獨立審核各自保留自己的範圍。三數加法、措辭變化與其他失敗仍見[最後報告](results/final-report-v1.json)。
