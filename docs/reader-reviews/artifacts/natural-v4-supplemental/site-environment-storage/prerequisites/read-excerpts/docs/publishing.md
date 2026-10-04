# 教材發布

本課的正文是一份來源，網頁與Notebook是兩種使用方式。網頁適合順著說明讀圖與結果；Notebook讓你改一個數字、再看結果怎樣變。兩者由同一份正文產生，讀者不必在兩本內容不同的教材之間來回找。

網站以[Zensical](https://zensical.org/)建置，發布到[GitHub Pages](https://birdhackor.github.io/tiny-perceptron-vlm/)。如果你只想學習，直接從[閱讀指南](../course/README.md)選路線即可；下面的建置步驟給想修改或維護教材的人使用。

## 1. 讀者怎麼使用？

1. 在網頁讀正文、看SVG圖解與已核對的CPU輸出。
2. 想修改短程式時，按本節的Colab入口，或下載`.ipynb`在本機Jupyter開啟。
3. 有程式的Notebook在第一個程式格準備工具，再按順序執行後面的程式格。沒有程式的小節則以閱讀為主。
4. 想重做完整訓練時，再走對應訓練指引，取得那項任務的資料與權重。

網站的Colab連結指向發布commit中的Notebook。其準備程式在Colab首次使用時會clone專案的預設分支；這與Notebook連結固定版本是兩回事。要重做舊版實驗，應先在Colab取得對應commit的專案再繼續準備工具，或在本機checkout那個版本。Colab使用其Python與pip環境；本機網站建置使用專案的`uv.lock`。本課沒有把本機kernel成功寫成Colab實機驗證。

