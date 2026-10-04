# 本人必要前置复核 2

同一 FACTUAL owner：`/root/v4_review_coordinator/factual_v4_13_17`。保全的原 pass 报告 SHA256 为 `5385c09b343ed2b3ab640ef9d91643a815e44548a775234ff8e7dd57132e6920`，其 22 项判断、原完整 main raw、CPU/原始记录复算、权威阅读笔记及五图实际 render/view 证据均保留。原已读 13.16 raw SHA 为 `12f128d0ef1e038399e74b95e57fd7d7655b98e264c301a62af2c9f2308d7978`。

本人实际完整读取新版 13.16 UTF8（包括全部空行），SHA 为 `99b78665cf25940b79388a79c9881fd1f7455a65c75d90079aa06fba4821a1ae`；与本人保存的原必要前置全文比对，唯一变化在第一个 Python 块后的第一段，增加 `range(len(answers))` 提供编号 0、1，以及 `key=lambda index: scores[index]` 按编号取分数的解释。该代码块、练习、正式 18 题表格、逐题例子与限制条件均未改变。

新语法核对使用本人实际下载并读取的官方 CPython v3.13.5 文档，完整第三方文件只在 ignored `outputs/natural-v4/factual-research/13.17/prerequisite-recheck-2/`，durable receipts 只记 URL/version/date/retrieval SHA。读取日期 2026-10-04。

- `Doc/library/functions.rst`：1127–1131 的 `len(s)` 返回容器/序列的项目数；1211–1232 的 `max` 返回 iterable 中的原项目，`key` 是单参数排序函数，最大值相同时取先出现者。本例没有并列，返回候选编号 1，不是分数 14。
- `Doc/library/stdtypes.rst`：1385–1411 的 Ranges 说明 `range` 是不可变数值序列；省略 start/step 时为 0/1，正步长内容小于 stop。因此 `range(len(answers))` 为 `range(2)`，可枚举编号 0、1。正文“列出”未承诺它是 eager `list`，本次执行也保留了真实类型 `range`。
- `Doc/reference/expressions.rst`：1881–1907 的 Lambdas 给出匿名函数语法与返回表达式的等价函数。因此当前 lambda 参数是编号，返回 `scores[index]`；新解释准确，未声称 lambda 可容纳一般语句。

本人实际在 Python 3.13.5 执行完整 current13.16 块及其正确性练习：长度分数 `[1,14]`，lambda 映射也是 `[1,14]`，选编号 1，输出长错误回答与 False；练习分数 `[1,0]`，选编号 0，输出 `3` 与 True。原/新 Python 块逐字相同。commands、预期/观察、stdout/stderr、环境与真正 assertions 见 `probe.py`、`probe-result.json`。

13.17 main raw 与本人原全文逐字节相同，SHA 仍为 `7c54f21313229f90df2b5ac4f37815d0cfe4063ec5df07d03b20517c7c6113fb`。注册的全部程序、原正式记录及原 artifacts 哈希未变，只有章 13 的整文件 SHA 因必要前置改变而变。五个 SVG 原字节均与本人旧图 map 相同；本次沿用本人原实际 render/view 的观察记录，没有声称重新看图或重新渲染。原 native CPU 训练/导数/分母核对继续适用，本次没有模型训练。

受影响范围裁决：新增语法说明只解释原人工长度评分反例如何选择候选。它不改变 13.17 的 DPO 公式、两卡数值、共同起点、训练预算、12/18 分项、回馈来源或成品分支。closest dependent c16 的任务结果、c18 的比较预算限制与 c21 的发布证据条件仍成立；未新增矛盾或未解决事实。保留 22 项原判断并加入真实 current 必要前置 closure，不能把本次称为新的整页 full-review 或 CPU training replication。
