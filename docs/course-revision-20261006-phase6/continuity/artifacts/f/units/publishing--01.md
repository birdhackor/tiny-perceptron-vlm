## 1. 讀者怎麼使用？

1. 在網頁讀正文、看SVG圖解與已核對的CPU輸出。
2. 想修改短程式時，按本節的Colab入口，或下載`.ipynb`在本機Jupyter開啟。
3. 有程式的Notebook在第一個程式格準備工具，再按順序執行後面的程式格。沒有程式的小節則以閱讀為主。
4. 想重做完整訓練時，再走對應訓練指引，取得那項任務的資料與權重。

網站的Colab連結指向發布commit中的Notebook。其準備程式在Colab首次使用時會clone專案的預設分支；這與Notebook連結固定版本是兩回事。要重做舊版實驗，應先在Colab取得對應commit的專案再繼續準備工具，或在本機checkout那個版本。Colab使用其Python與pip環境；本機網站建置使用專案的`uv.lock`。本課沒有把本機kernel成功寫成Colab實機驗證。

