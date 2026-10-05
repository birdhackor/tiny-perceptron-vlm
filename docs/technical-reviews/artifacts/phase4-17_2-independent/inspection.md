# 17.2 本人獨立核對記錄

審閱 task：`/root/phase4_factual_coordinator/factual_17_2`。日期：2026-10-05。

實際閱讀本輪 factual-reviewer-instructions.md、checker schema、section_facts.py 介面與 CPU 執行契約、clear-tutorial 的 SKILL.md 與 review-protocol.md；另讀 cloud-environment-onboarding:setup 及 onboarding reference，以既有 .venv 進行有界 CPU 驗證，未需設定或安裝變動。

正文讀取範圍：目前 17.2 全節（course/chapters/17.md 第 50 行起，至下一個 ## 之前），以及必要前文 17.1 全節。章節整份原始 bytes 保存於 inputs/chapter17-frozen.md，為本次 frozen input；並未聲稱讀過其餘章節內容。17.1 的完整模型實報只是前文閱讀，未當作本節能力證據，未開啟其 JSON 或重評其成績。17.2 無既有實測 JSON、helper 模型、訓練 recipe。

17.2 的唯一 Python fence 原始 bytes 保存於 inputs/fence-1.py，直接以同一 .venv Python 在 CPU 執行，stdout/stderr 與 command、code SHA 保存。它只建立四個元素、捨入、截斷、型別轉换和乘刻度；没有 backward、optimizer、更新、訓練或模型推論成績。独立变体执行了 scale=0.25、六个正负半格 tie、六个超界／端点值、25,401 个範圍內格點、三个同碼输入及元素儲存检查。這些是程式／算例驗證，不是重訓。

親讀外部原始來源均由 PyTorch 官方仓库 v2.9.0 tag 的 raw HTTPS URL 获取。manifest.json 记全檔 SHA、exact source locator 和原始摘錄 SHA；不是來源庫摘要。

本人實際 inspection 定位：

- torch/ao/quantization/fake_quantize.py L131–164：完整 FakeQuantize docstring，核對 round/clamp/scale/zero_point 公式；zero_point=0 即本節公式。該文档是模拟 q/dq 的定义，只用于确认数学映射，不把 fake-quant 浮点输出当作本例 int8 存法。
- torch/_torch_docs.py L2800–2841：clamp 範圍與 min/max 式；L9538–9592：round 最近整數、half to even、相同 dtype、decimals 参数与有限精度規約；L13325–13359：quantize_per_tensor 的 scale、zero_point 與 quantized dtype 契約。
- torch/_tensor_docs.py L1754–1769、L5192–5281、L5342–5352、L5513–5533：element_size bytes、to dtype conversion、float=to(float32)、tolist 返回 Python 數字／list。
- docs/source/tensor_attributes.rst 只讀 L1–74：float32 是32-bit、int8是signed8-bit；其他行仅保存原文，未用于判定。
- docs/source/notes/numerical_accuracy.rst 只讀 L1–35：IEEE754有限精度与版本／设备差异；其他行仅保存原文，未用于判定。

官方原來源版本為 v2.9.0；安裝版為 torch 2.14.1+cpu，git version 5c4886908584029761b579af026dcfb627c84070，Python 3.13.5。兩者版本不同，报告不声称所存官方文档是安装版源码；本节普通 API 的实际行为由原fence及本轮CPU变体核实。无GPU build，无CUDA可用。

图：親讀原 SVG bytes，使用 Inkscape 1.4 的实际版本见 render.execution.json 渲染，再通过 view_image 亲看 grid-render.png。640×760 图中的 0.7→1.4→码1→0.5→差0.2、1.2→2.4→码2→1.0→差0.2 完整可見；末行明确限定未截断时的半格误差。两箱按行列计算步骤，无独立数据轴或连线箭头需推断。数字、符号、标签与正文相符。此為技术图检，不冒称本轮进行了整页桌面／手机易读性验收。

手算和单位／范围见 derivation.txt。图与正文的差值均定义原值减还原值；练习0.05是误差绝对大小，不是四项都为正。scale 的单位是原值每整数码步，q 无量纲；固定正格127个的范围确实由63.5降至31.75。

未读旧技术／读者报告、作者修订记录、相邻报告、旧 canonical 17.2.json；初始 rg 只显示了文件名，不包含判定正文。未发生摘要曝光事件。没有修改正文、图、模型或训练资料，没有下载模型／资料，没有 commit、push 或 spawn。

判定：pass。所有本节影响理解的实质主张已核对，未发现需要修订的实质问题。有限格点检查不当成数学证明，半格界由本人推导支持。玩具映射不支持完整模型压缩比例、推理速度或精度提升。
