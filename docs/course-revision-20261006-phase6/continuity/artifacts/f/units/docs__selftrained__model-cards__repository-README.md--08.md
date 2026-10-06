## 第20章：成熟模型應用延伸

[學生操作](https://birdhackor.github.io/tiny-perceptron-vlm/natural-v4-student.html)使用另建的Python3.12／PyTorch2.8.0環境，載入固定[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203)與[Whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo/tree/41f01f3fe87f28c78e2fbf8b568835947dd65ed9)。照片與文字交給圖文底座；Whisper先把語音轉成文字，再放入同一聊天紀錄，回覆為文字。

本版依validation保留原底座，不加本課LoRA。圖文模型2,127,532,032參數、語音辨識器808,878,080參數；兩站能力來自上游訓練。其[固定公開配套](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/d3954d6900b3cf81e593d99d9b8b1a91e6f9741d/natural-v3/assistant-2b-v4)保留版本與說明，沒有重新散佈上游權重。Qwen採Apache-2.0，Whisper採MIT，來源資料另遵守各自授權。

第20章有限最後測試中，自然照片完整短描述25/42、同組可見事實58/84、自然中文字完整轉寫8/10、多區塊文字順序1/3、文字聊天2/13，4段真人問句真正ASR後聊天2/4。完整條件與不足見[20.13](https://birdhackor.github.io/tiny-perceptron-vlm/20.13.html)、[資料](https://birdhackor.github.io/tiny-perceptron-vlm/natural-v4-data.html)與[LoRA訓練指引](https://birdhackor.github.io/tiny-perceptron-vlm/natural-v4-training.html)。這些題目、規模與起點不同，不能拿來代替第19章從零模型的驗收。

