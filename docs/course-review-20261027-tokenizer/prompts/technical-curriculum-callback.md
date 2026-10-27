# curriculum 獨立技術審閱

你是 fresh technical reviewer，一人一頁 canonical curriculum（docs/curriculum.md），freeze-03.json。依 technical-common.md、AGENTS.md、technical-review-guide.md 完整審閱，不讀作者 PLAN、其他本輪 pass 或舊問題清單來形成初判。通用原始來源庫可查，主張依頁面實際承諾界定，不把路線預告要求成完整模型機制課；純導覽無實質主張才可NA，數量、軟體或實驗敘述須查。

reader task=/root/reader_curriculum_callback_20261027，wave=tokenizer-curriculum-callback-20261027。列完整主張、來源版本與實際核對位置，必要CPU小檢查，不重訓、不下載模型／資料，不擴成全書技術重審。你只對本頁判斷，不把舊結果聲稱成本次重跑。本輪全manifest gate此時仍未組裝，不能稱它已通過。若有疑點保留原始 revise；作者修後本人再核對。

報告放 reviews/technical-curriculum-callback/，保留 technical-original-01.json 與 current；小證據放 docs/technical-reviews/artifacts/tokenizer-20261027/curriculum-callback/，大圖/全文/scratch放 /workspace/work/tokenizer-dynamic-20261027/curriculum-callback/。自查 --freeze freeze-03.json --reader-task 指本頁reader。來源／figure／repository_code SHA須對現行。不得改教材、工具、舊報告，不開代理、不提交、不發布。最後回報判定、問題、實際檢查與未驗證範圍。

獨立 original 存好後才做歷史回查：docs/course-review-20261019/dispositions-continuity-01.json /items/71 是一筆舊 reader 已結案項；原 task 已無法開啟，協調者有真實收據。不要提前讀內容以形成初判。親自核對它在目前頁仍成立，並可在此時檢查本輪 fresh reader 的完整來源/範圍證據，不冒稱該讀者已學會未教機制。另存 technical-historical-recheck-01.json（完整 technical schema，僅歷史項目的明確scope）；加 historical_issue_rechecks：disposition_ref{path,sha256,json_pointer}、original_report（原issue ref）、json_pointer（原issue pointer）、historical_fix_verification（舊verification原樣）、issue_id、status=still_fixed、verdict=pass、reason、current_source_sha256、current_figure_sha256、current_evidence[{source_id,locator,supports}]、artifact_ids。若不成立照實revise。先形成獨立original，再完成必要查證，先存完整current，再存歷史appendix；不要修改舊時間或舊原判，appendix不能冒充完整當前審閱。
