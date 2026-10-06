## 原各章權重與舊合成整合索引

原30組正式局部實驗共120份模型存檔仍在本庫。下載由[public-models.json](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/course-experiments/public-models.json)固定各組revision、配對檔案、大小與SHA-256；使用 `scripts/fetch_course_models.py` 和對應實驗入口。完整recipe、原始評估及proof保留在[歷史實驗紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/course-experiments/README.md)。局部機制與歷史結果各有用途，不是v2成品的能力證明。

舊合成整合另保留[course-integration-v2權重](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed/course/course-integration-v2)：8份階段／量化檔及3份Dense學生檔，配對[capstone-public.json](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/course-experiments/capstone-public.json)。舊328k文字底座是合成小世界的局部機制與歷史輔助材料，不是上面的5,447,107參數v2 MoE。舊joint與DPO各78/90、舊Dense示範／蒸餾各62/90與61/90的結果仍保留，不換成新版最後測試分數。

下列入口各連到原固定版本的模型卡與檔案：

| 原實驗ID | 固定模型卡與檔案 |
| --- | --- |
| `simple_models` | [course/course-v1/simple_models](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/fbbff36990db0d95a6e0af5ecdbc593f920938d8/course/course-v1/simple_models) |
| `text_foundation` | [course/course-v1/text_foundation](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/2ba1278993a68d4ab30091534376a955b188df4c/course/course-v1/text_foundation) |
| `real_text` | [course/course-v1/real_text](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/c8416bcf4d54cf40fbd270d3f636655a522310a0/course/course-v1/real_text) |
| `tokenizer` | [course/course-v1/tokenizer](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/58eb946c0eb9a37b8abb8d1930900685b42790bf/course/course-v1/tokenizer) |
| `sft` | [course/course-v1/sft](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/14293af762e79c5a65da5bdc5afed1bea68175a6/course/course-v1/sft) |
| `sft_ablation` | [course/course-v1/sft_ablation](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/e30b15712b798cecacf47b0787a92590d2a7d876/course/course-v1/sft_ablation) |
| `style` | [course/course-v1/style](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/23b58c077a2da2ec6b94f4902e88510c0364e3fc/course/course-v1/style) |
| `lora` | [course/course-v1/lora](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/1bfb0ed028d3a980711a29cb38a993f58d127d02/course/course-v1/lora) |
| `safety` | [course/course-v1/safety](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/308aa207d6917d5dc8f9c819f2d7257d5cad7f8e/course/course-v1/safety) |
| `dpo` | [course/course-v1/dpo](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/9c3601de9cd47293cfe3734f8ddb4317e4ea7ae4/course/course-v1/dpo) |
| `encoders` | [course/course-v1/encoders](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/93e45c77a7f6f21047bcd9f6e7ac3b5a7ad6c294/course/course-v1/encoders) |
| `contrastive` | [course/course-v1/contrastive](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/03cddf7bf0a2cdd69c1df876d3cff4b5e88cd621/course/course-v1/contrastive) |
| `projector` | [course/course-v1/projector](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/6d115860fc0af4669fccaa44bcdf2fad7c97e91e/course/course-v1/projector) |
| `vqa` | [course/course-v1/vqa](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/e17db92f550449d1fc195655dfda1a4a49af9a43/course/course-v1/vqa) |
| `vision_ablation` | [course/course-v1/vision_ablation](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/71b6a22c6934bb6aec8ef92f9a0ce75b25f2aacf/course/course-v1/vision_ablation) |
| `ocr` | [course/course-v1/ocr](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/ce803a4c1eb75c428397fc4a49352354d9c82e8e/course/course-v1/ocr) |
| `audio` | [course/course-v1/audio](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/1c946fcb7f1d75948711962f22559b18d879e8b9/course/course-v1/audio) |
| `joint` | [course/course-v1/joint](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/b7908e15691515c15ab93caa9e71327b7ffda87b/course/course-v1/joint) |
| `real_modal` | [course/course-v1/real_modal](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/0c284d926ee729291fa023976fc03d9260db4068/course/course-v1/real_modal) |
| `modern` | [course/course-v1/modern](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/59d993473b5155842cea7b40d37be91a7b0bc033/course/course-v1/modern) |
| `moe` | [course/course-v1/moe](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/17fc5f20686fb4f207bb06b1e501b09a29b1cbeb/course/course-v1/moe) |
| `efficiency` | [course/course-v1/efficiency](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/0e1628d51964fdbd6f6dbf6373ce1324927addba/course/course-v1/efficiency) |
| `precision` | [course/course-v1/precision](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/fb1d740aa140204efd41329a2c28e2a87daae6a4/course/course-v1/precision) |
| `quantization` | [course/course-v1/quantization](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/b41d097e4c10b29f6e6218728cbe1bafcc9db93c/course/course-v1/quantization) |
| `qat` | [course/course-v1/qat](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/30ea5592953d0f0db4a0827fe6b7f30bc53805f0/course/course-v1/qat) |
| `distillation` | [course/course-v1/distillation](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/fa69713c7dece0177b5088e14f49581d8ae97b7f/course/course-v1/distillation) |
| `multimodal_distillation` | [course/course-v1/multimodal_distillation](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/ae3c4263c4f9e73f0a5fb25b969f5f3d7fbd31d6/course/course-v1/multimodal_distillation) |
| `rag` | [course/course-v1/rag](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/660fa1d1fa67770b651c8397b1c4f44b11d200e5/course/course-v1/rag) |
| `tools` | [course/course-v1/tools](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/973d02736f4ebbfd7188e7f032fff1dfeddbc66c/course/course-v1/tools) |
| `reasoning` | [course/course-v1/reasoning](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/ac5ac599faabcb026a159af09433ec0490a397f7/course/course-v1/reasoning) |

