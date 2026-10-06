## 3. 發布到GitHub Pages

專案的Settings → Pages → Build and deployment → Source選擇GitHub Actions。`.github/workflows/pages.yml`在`main`更新或手動觸發時，取得必要資料、核對審閱、逐份啟動獨立CPU kernel、嚴格建置網站並檢查連結，再由Pages部署工作發布。

Notebook數量由當前小節目錄決定，新增小節時也要同步來源、索引與導覽。網站產物與已執行Notebook放在被Git忽略的`outputs/`，透過Pages artifact傳送；正文與核對證據留在Git中。

發布後查看`build-info.json`，核對實際commit、頁面與小節數、CPU輸出狀態及Zensical版本，再實際點開新增頁面與下載入口。工作顯示成功和讀者拿到正確版本都需要確認。資料包、模型配套與網站是三條不同的發布路線，前兩者的規則見[教材資產存放](asset-storage.md)。

