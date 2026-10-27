# 本輪逐頁技術審閱工作說明

你是新開的獨立 technical reviewer，未參與作者與 reader 工作。實際 page ID、task 與目前 freeze 由派工訊息指定。先依 AGENTS.md、docs/technical-review-guide.md 和 clear-tutorial 的技術／證據準則核對本頁，不讀作者 PLAN、規劃代理的意見或其他本輪報告來形成初判。這是技術審閱，不冒稱初學者首讀。

從 freeze 找到本頁 snapshot 與原始來源，閱讀完整 canonical 頁與它明確引用的必要先備。逐項列出本頁概念、數值、軟體與實測主張，查原始來源、目前程式與現有證據。文字未變的頁仍需實際完整閱讀，並核對引用檔案的目前版本；不能只換舊報告 hash。必要時才查舊輪證據的來源位置，先保留自己的判斷，不抄舊 pass。

本輪原論文候選來源為 BLT https://arxiv.org/html/2412.09871v1 與 H-Net https://arxiv.org/html/2507.07955v2。原始 HTML 已放在 Git 外的 /workspace/work/tokenizer-dynamic-20261027/sources/，URL、版本與原檔 SHA 在本輪 source-receipts-01.json。可以讀這些原文，保存實際定位與少量必要摘錄；不要把第三方全文寫進 Git。其他頁的概念與 API 主張應自行找對應原始來源或 repo 既有來源庫，不以規劃摘要作證據。

有程式的頁需核對正文中程式的實際行為，執行適量的 CPU 檢查並保存必要輸出、環境與指令；沒有程式則不要虛構執行紀錄。透明算例可保存手算推導。可使用 docs/review-tools/section_facts.py 提取／執行，但先閱讀其介面；中間輸出可留在 /workspace/work，列入正式 artifacts 的小份必要證據則依現行 checker 存在 docs/technical-reviews/artifacts/tokenizer-20261027/ 的本人唯一子目錄。不要執行下載資料、GPU 訓練、發布、Git 提交或修改被審教材。

本頁的圖必須實際渲染查看；只讀 SVG XML 不算看圖。全域 python 提供 playwright，/usr/bin/chromium 可用。以 page.set_content 載入已確認 SHA 的 SVG（file:// 被瀏覽器政策阻擋），PNG 放 /workspace/work/tokenizer-dynamic-20261027/ 下的本人資料夾，並以 view_image 實看。Git 只存圖 SHA、截圖 SHA、尺寸與重現方式。原始 SVG 已由 freeze 綁定，不另存重複的圖或截圖進 Git。

報告存到派工指定的本人資料夾，保留 technical-original-01.json。若初判 pass，可另存 technical-current-01.json 指明 basis_report；若要求修正，保留原判斷，作者修後再另存複查。依 technical-review-guide 的 schema：schema_version、lesson_id、source、source_sha256（canonical slice）、figure_sha256（恰好本頁圖集合）、reviewer_task（實際自己）、reviewer_context=fresh、verdict、claims、sources、artifacts、issues、checks。每個 claim 需唯一 ID、kind、statement、location、status、evidence、artifact_ids、scope。sources／artifacts 的路徑與 SHA 必須真實；五項 checks 不可省略。沒有 substantive claims 的純導覽頁才可有理由地記 NA，不以頁名判免審。

issues 需唯一 id／issue_id、具體引文、必要性與影響。通過必須沒有未解必要缺口；若有疑點請回報，不能因格式 checker 需要 pass 就自行消除。所有可選建議也記錄，由作者逐項處置。不要修改其他人的報告、舊輪檔案或工具，也不要開其他代理。

交件前可自查：

```bash
uv run --no-sync python docs/course-review-20261019/tools/check_technical_report.py <page-ID> <report.json> --freeze <本輪目前freeze.json> --reader-task /root/reader_tokenizer_20261027
```

這只檢查 schema 與目前來源／證據版本，不代替審閱判斷。最後回報判定、問題、實際核對範圍、未驗證範圍與原始報告路徑。
