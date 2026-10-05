# R.3 本人獨立核查記錄

Reviewer: `/root/phase4_factual_coordinator/factual_r_3`。日期：2026-10-05。

本輪親讀現稿 `course/README.md#R.3` 全節（原檔第36–67行），原始UTF-8
section SHA-256 為 `709a303ea1c58c1b81d57a7407d45055b69fb83a5e18e2d523648afd330f7723`。
`section.md` 保存原始bytes；`frozen-inputs/course/README.md` 是本次讀取的完整凍結輸入，
不是目前整章版本的持續斷言。各個完整原檔與副本的SHA在 `frozen-input-manifest.json`；
audit實際逐檔比較原件／副本，並在檢查結束再次確認輸入未改變。

## 真實閱讀與操作範圍

已完整讀取本輪 `factual-reviewer-instructions.md`、checker schema、section_facts helper、
clear-tutorial SKILL.md、review-protocol.md，以及原始 `outline.md`、`rewrite-contract.md`。
讀取全部23個章原稿的章名與所有小節標題，親自將表中的學習問題與原標題對應；
具體字串與行號保存在 `chapter-heading-inventory.json`。
必要的正文讀取是第6章6.3第86–104行（byte-level BPE材料）、6.7第237–251行
（byte切分主題）；第19、20章第1–10行的主線／延伸導言。
這些前置正文只用來確認導航所指範圍，沒有將其中的既有實測當成R.3的能力證據。
本節不是第一小節，因此沒有另讀閱讀指南導言或R.1/R.2/R.4正文。

讀取 `course/lessons.md` 的生成表（初次工具輸出有截斷，沒有聲稱該次已完整看完）；
隨後以原生成器 `--check` 實際逐位元組核對完整表、index與全部Notebook。
`lesson-contract.json` 先只列上層key與型別，再讀 `/schema_version`、`/purpose`、
`/lesson_ids`；`lesson-index.json` 先列list形狀與第一項key/type，再檢查每項
`id/title/source/notebook`。Notebook只另外讀 `/metadata/lesson_id`，未印出模型輸出。
其他前置頁只讀 numbered headings以確認導航輔助頁與章內lesson的分類。

以AST定位 `scripts/build_course.py` 的函式，再讀其第64–236行：
`cell`、`lesson_anchor_aliases`、`notebook_reading_links`、`notebook`、`build`。
這些是生成／連結契約；没有讀作者額外評分常數、舊技術審閱、舊讀者判定、
TRAINING/DATA/worklog或修正摘要。初次 `rg --files` 檔名定位範圍過大，輸出中出現
歷史artifact路徑名稱，但未開啟歷史檔或接觸判定文字；後續均限定確切原檔。
外部source locator索引僅列top-level key/types，未使用索引作為技術答案。

實際原文擷取命令：
`.venv/bin/python docs/review-tools/section_facts.py 'course/README.md#R.3' --output outputs/course-revision-20261005/phase4-r_3-factual-own-extract`。
它返回exit0、0個Python fence、0個SVG引用。永久section副本由本人有界audit另從raw bytes擷取。

## 逐列對照的判斷

| R.3章列 | 親自核對的原小節定位 | 結論的範圍 |
| --- | --- | --- |
| 1 | 1.1、1.12 | 編號及更新問題確實位於該章 |
| 2 | 2.1–2.3 | 多位置前文及特徵組合位於該章 |
| 3 | 3.1、3.2 | 混合位置、改變讀取比例位於該章 |
| 4 | 4.1、4.5 | 順序及注意力／FFN組合位於該章 |
| 5 | 5.1、5.7、5.10 | 訓練、續存與評估位於該章 |
| 6 | 6.1–6.3、6.7 | 字元、byte與片段切分主題均存在 |
| 7 | 7.9、7.11、7.19 | 對話學習、EOS及輸入／生成長度位於該章 |
| 8 | 8.4、8.5、8.14、8.16 | 文風、內容範圍及格式要求位於該章 |
| 9 | 9.2、9.4、9.6 | 不知道時回答與拒絕取捨位於該章 |
| 10 | 10.1、10.3、10.5 | 圖片表示、特徵及梯度入口位於該章 |
| 11 | 11.8、11.14、11.15、11.18 | 是否使用圖片、位置、筆畫與指定區域位於該章 |
| 12 | 12.1、12.9、12.16 | 聲音表示、接入與共享對話位於該章 |
| 13 | 13.5、13.10、13.13 | DPO、reward model及PPO位於該章 |
| 14 | 14.1、14.7、14.10 | 位置及現代零件主題位於該章 |
| 15 | 15.1、15.4、15.10 | Dense與expert選擇／成本位於該章 |
| 16 | 16.3、16.4、16.12、16.13 | 重用計算及限制連結位於該章 |
| 17 | 17.2、17.3、17.15 | 數字刻度及誤差問題位於該章 |
| 18 | 18.1、18.2 | 教師訊號及較小學生位於該章 |
| 19 | 導言；19.6、19.7 | 同一自訓練核心與多模態／工具是整合設計主題 |
| 20 | 導言；20.1、20.4、20.8 | 成熟模型延伸、資料與交付驗收是該章主題 |
| A | A.3、A.4、A.10 | 找資料、回到來源及更新條件位於該章 |
| B | B.1–B.3 | 請求、執行及結果返回位於該章 |
| C | C.1、C.3、C.4 | 中間步驟、重試及候選驗證位於該章 |

開場「前七章」對應1–7章從接字至對話的排列，也符合原outline的基礎先行約定。
表格列的是學習問題，沒有推導PPO/DPO/MoE等方法、數值比較或模型成績。
因此不存在需要代入計算或查原論文的實質技術主張；沒有為schema新增虛構concept。

原outline的「主線成品的有限任務」明說第五階段設計目標尚不是新成品能力；
rewrite-contract「保留內容與連結」將19章設為自行訓練主線，20章設為Qwen/Whisper延伸。
现19章導言也明說新成品尚未完成訓練驗收。R.3使用「整合專題」及「如何共同工作」的問題式
導航，没有新增已驗收能力宣稱。现20章導言提及LoRA候選既有結果，這不是R.3主張，
本輪沒有查核或重新產生成熟模型分數。

## 覆蓋與執行

`audit_navigation.py` 本次實際exit0；保存命令、stdout、stderr、environment與JSON結果。
它核對23章順序、26個R.3本地連結、283個章內lesson ID與283個Notebook ID。
原 `build_course.py --check` 本次實際exit0：`283 個小節：同步檢查通過`。
原build從 `course/chapters/*.md` 生成；它不為W/R/T/G暖身、導覽、操作與名詞頁生成Notebook。
R.3標題與表格上下文是「各章」；「所有小節與Notebook」據此解讀為前述各章的課程小節，
其章內覆蓋是完整的。輔助頁不是該Notebook清單的範圍；本輪不聲稱它包含所有前置頁小節。

沒有原fence可執行，沒有shell recipe、數值算例、GPU工作、模型／trainingdata下載或新分數。
没有Markdown image、HTML img、inline SVG或SVG引用。導航表以章名與問題直接提供對應，
不需要讀者想像圖片素材、位置、箭頭或資料形狀；figure render/view具體不適用。
本次不評閱讀時間、前後銜接或模型工程。沒有修改教材、圖、其他報告，也沒有spawn或commit。

最終判定：導航核對通過，沒有未解決實質問題。checker只核格式、指紋及身份，不代替此判斷。
