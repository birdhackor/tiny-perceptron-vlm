# 本人實際獨立16.8查核記錄

Reviewer: `/root/phase4_factual_coordinator/factual_16_8`，fresh factual reviewer，2026-10-05。

本人的主要輸入是完整 `course/chapters/16.md#16.8` 原稿與SVG，不是第一次分段易讀性審閱；沒有讀舊technical/reader報告正文或結論、作者結果修正摘要，也沒有換身分。初讀全章原bytes保存為 `inputs/course__chapters__16.md`，其SHA是當次frozen input；只依賴16.8原bytes作正式source identity。協調者通知後來16.5選讀局部文字變更，我沒有依賴那段，因此未擴大重跑。另親讀3.4、5.17、16.4、16.7的原小節，只作QKV/eval/GQA/dtype背景，未重評其他節的GPU或品質成績。

先讀 factual-reviewer-instructions、checker schema、section_facts介面、clear-tutorial skill及review-protocol；環境setup skill也已讀，沿用現有CPU.venv，未安裝套件或改設定。section_facts只作原bytes擷取，沒有用其會建立symlink的execute模式；原fence由本人有界CPU腳本實執行。

原seed0 forward差為5.960464477539063e-08；教材seed1變體為1.1920928955078125e-07。反转SDPA mask差2.940368890762329并触发预期AssertionError。Q/K/V、scores、weights、输出轴、tril逐行1/2/3/4个allowed、future权重零、权重和约1均亲核。额外短CPU变体显示q重复缩放改变结果，eval下非零functional dropout仍生效。

本人以AST定位必要原函数；从每份result的revision亲自git show取得原architecture、attention、model、common等文件，逐檔对照code_sha256。JSON先列root/必要分支keys/types，只沿报告列的具名测量、配置、方法及provenance pointers核对；没有读取heldout成绩或总判定/作者解读。CPU重算9笔原forward samples的median，转换秒到毫秒；原40更新、0跳步、各2328目标计数、weight max及warm-step时单位与舍入均匹配。

为独立重算有效目标，我只取得协调者给出的精确原dataset bytes locator，不读其相邻报告。本人核其整档SHA匹配efficiency /artifacts/5，再保存19,822 bytes到自己的inputs/dataset-original.json，正式source只引用自己的副本。原45条train集合SHA也吻合。按原render_chat/text_examples、IGNORE=-100及seed42 random.choices，每批8题抽40次，每步有效目标相加2328；原前四题长度53/51/50/48，padding[4,53]。没有加载weights或运行optimizer/model评测。

Flash原profile的forward与forward_backward，FP16/BF16均亲核指定aten::flash算子和实际CUDA flash_fwd/flash_bwd kernel名；没有以API名字或原summary判定取代trace核对。原8份output/q/k/v comparison的finite、max_abs与atol/rtol逐份核：所有max_abs低于atol本身，独立充分满足allclose不等式。梯度表格确是q/k/v三份max的再max。原GPU tensors未保留，不能冒称重新逐格求值；原fixture.pt缺失，其声明SHA保留但未冒称恢复原bytes。另CPU重建同seed/shape recipe不保存新.pt；只以T=8强制MATH执行原manual/SDPA/error方法验证dtype/upstream契约，不作为GPU Flash或原成绩再现。

PyTorch docs/2.14的三个网页请求实际403；使用官方固定commit5c4886908584029761b579af026dcfb627c84070原functional.py的SDPA6367–6536，以及_torch_docs.py allclose833–863/matmul7907–7975。官方autograd原grad434–480核上游vector-Jacobian；此commit同时吻合本CPU和原L4测量。额外dtype文档候选路径404，位数由CPUtorch.finfo验证32/16/16。原Attention论文v7 §3.2.1 Eq(1)、§3.2.3及FlashAttention论文v2 Abstract/Introduction/Fig1说明亲读，支持机制而非本例性能保证。

本节SVG以Inkscape1.4真正渲染为1320×1316 PNG，本人调用view_image查看。Query为行Q0–Q3、Key为列0–3，下三角和对角10个绿1、未来6个灰0，Q0只读0、Q3读0–3，图例/颜色和正文及tril一致，没有箭头。Inkscape非致命字体包装警告没有阻断渲染；后次渲染SHA与已查看图完全一致。没有冒称手机/桌面网页易读性验收。

判定pass的范围是现稿的数值与计算契约、原既有测量的可追踪性以及成熟方法机制。没有训练、GPU工作、模型/资料下载、能力重评、正文/图修改、commit/push或subagent。未发现实质错误。原fixture缺失、GPU原tensors未存、37笔更新step latencies未存、官方网页403均保留为证据范围限制；现稿没有要求复原这些原bytes或宣称CPU重跑确认Flash。
