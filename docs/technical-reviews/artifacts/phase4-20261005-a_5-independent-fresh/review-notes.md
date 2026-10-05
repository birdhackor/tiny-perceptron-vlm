# A.5 本人實際查核紀錄

Reviewer task: `/root/phase4_factual_coordinator/factual_a_5`。本輪新的單節正確性審阅，非第一次順序盲讀。

全文讀取 A.5（當次來源行 143–186）；必要上下文僅 A.2，用以確認 K017/C2/D1 與回覆格式。原始 UTF-8 小節 bytes 保存為 section.md，完整 0A.md 是当次 frozen input，不能冒充日後整章版本。沒有讀舊技術報告、reader 報告、作者修正摘要或停止 task。

JSON 先列上層 key/type，再列必要下一層形狀。實際解釋的是 24 個 correct_context/retrieved_context 原始樣本及具名測量、判準、provenance leaves；完整 pointer 清單見 inspected-pointers.json。全檔的 opaque 保存與 SHA-256 沒有用來讀取多餘作者評語。原實作先 AST 列函式位置，讀 applications.py 的 prompt、資料生成、split、_grounding、run_rag 與 _sample 必要區塊，及 common.py 的 split/hash/write_json，retrieval.py 的 lexical_terms/retrieve，data.py 的 ByteTokenizer。一般資料規約、API/docstring 與方法限制可讀，沒有接觸既有結果解說或教材審阅結論。

三笔人工 flags 的原 fence 是加總示範。原程式 stdout 與預期相等；只把乙的 correct 改 True，hit 仍是 2/3，correct 變成 2/3。沒有把人工旗標當成真正檢索器或模型的輸出。

既有結果的两条件都涵蓋同一 12 個唯一 test family。本人獨立由 documents 的 key/address/source 檢查所問公告實際存在，由 raw text/token 與固定 fact 檢查地址及引用，由公告存在與否決定協議 expected；再對照原評分程式與保存的 booleans。correct_context 中沿用的 retrieval_hit 指另一次 lexical retrieval attempt，不能當成該條件上下文的公告存在；本次沒有這樣使用。

直接公告 12/12，地址與引用正確 5/12，協議成功 5/12。檢索公告 9/12，地址與引用正確 3/12，協議成功 6/12；三個 miss 都輸出 UNKNOWN，這三次屬協議成功而非地址正確。七個直接公告失敗都為 UNKNOWN。兩條的同 key、有相同公告的九對輸入其實相同；保存的題目值、原始提示、byte token、來源 ID 映射與分母均一致。本節没有聲稱獨立重訓、普遍因果效果或一般長文件能力。

回覆正確、引用在上下文有效、引用包含所報地址是分離判準。_grounding 的支持檢查只比對一個 address 欄，不能當一般語意蘊涵或來源可信性驗證。評分 fixture 改字串/文件只用來核對這些判準，沒有產生新模型答案。教材的表格使用真值地址與指定 ID 的 exact-match，原 sample 也記錄 valid/support；本次逐題全部核對，兩表中成功的地址/ID 都由所問公告支持。

正式原生成由已發布結果的 samples 按原 write_json 格式還原為 original-generations.json，208121 bytes，SHA-256 精確等於原 artifacts/generations.json 紀錄。其他 60 列只為完整 hash opaque 還原，不解釋其模型表現。資料切分、12 個新事實與 72 條語料以雜湊吻合的原方法在 CPU 重建；dataset 與 corpus 的保存 bytes/hash 也與原 artifacts 精確相等。命名 reconstructed-* 明示是確定性重建，不是假稱下載原模型/資料。不保存或載入權重，不新跑生成或訓練。

本人直接 HTTPS 取回 RAG arXiv:2005.11401v4（首頁 12 Apr 2021）及 KILT arXiv:2009.02252v2（首頁 9 Apr 2021），保存 PDF 與 pdftotext；讀 RAG §2 的兩组件/上下文與 §4.4 的 gold evidence overlap、KILT §5 的下游分数、檢索分數、完整來源集合規則。來源庫只用來找候選，正式支持是实际檢查的原文。固定權重的 gold-context 測試可以提示檢索是否限制該次表現；改變提示長度或其他資料會改變比較，因此正文保留「很可能」與非絕對因果證明的界線。

本節沒有圖像引用、SVG 或需判讀的圖片；三個人工 flags 的來源/答案對應由程式直接列出，12 題對應由表格與文字明列。無空間、影像素材、箭頭或形狀需要視覺查核，所以 figure_consistency 是具體 NA，沒有把渲染成功冒充檢視。

首次 reviewer harness 有意外 unary + 造成 TypeError，在實質 dataset/sample 查核開始前退出。保存 verify-failed-attempt.py 及實際 stderr，修正後以 capture.py 執行一次，全部斷言完成、exit 0；這是自製工具錯誤，不是教材缺陷，也沒有來源污染事件。

結論：本版 A.5 的重要主張均有相稱支持，沒有未解實質問題。數字只支持本次 12 個合成公告題與固定協議；「用好」在這裡指提供真值地址和指定來源，沒有把形式正確當成一般閱讀或自然語言證據能力。
