# R.1 本輪独立检查记录

Reviewer: `/root/phase4_factual_coordinator/factual_r_1`。

亲读本轮 factual-reviewer-instructions、checker schema、section_facts、clear-tutorial skill 和 review-protocol；亲读现稿 course/README.md，包括章首导言与完整 R.1。整个 README 保留为明确标示的 frozen input，正式 source_sha256 只对应 R.1 原始 UTF-8 bytes。intro.md 是第一个 ## 之前的原始 bytes，不转换换行。

必要上下文实际检查：course/chapters/01.md 的章首与 1.1（原材料、字表、唯一 Python fence、还原解释）；course/first-steps.md 的 W.1（两种操作路径、CPU、本机 kernel、第一节文件）、W.2（清单与字典）、W.3（轴与形状）。一次读取的 chapter 01 上下文包含后续 1.2–1.4 开头，未将其结果当作本节证据。README 其他小节同属现稿；未使用它们证明 R.1。暖身 W.3 的图不作为 R.1 导览主张证据，本轮没有判定其视觉或数值正确性。

第一笔 filename locator 返回了大量 artifacts 文件名；没有打开旧 reader/technical report、作者结果评语、TRAINING、DATA 或工作记录的正文。此处是路径列举，不是读到旧判定；随后所有内容读取限于确切原文件。没有发生结果摘要污染事件。

程序先 AST 定位生成器与 exporter 的方法，再读生成契约。实际独立执行原 1.1 fence、改变排序方向与改变材料的有界版本，并调用 notebook() 比对现有 1.1.ipynb 原始序列化 bytes。选取 exporter 的原 AST 单节循环（原源码 272–297 行）进行一页输出契约检查；没有执行完整 build()、读时函数、发布、网络下载、bootstrap、图的 code cell、训练或模型推理。

JSON 先检查上层 key/types，再检查 measurements.json /inspected_json_pointers 指定的原始 inventory、Notebook 结构与当前正文 source。lesson-index 的其他选定 id/title 仅用来确认导言所述题材确实存在：/0、/6、/7、/12、/15、/71、/106、/135、/268 的 id/title/source/notebook 及文件存在性；没有递归列值或读取作者解释字段。

Notebook 原件 /cells/5/outputs 是空清单。bounded-render/1.1.md 中的执行结果来自本轮真实执行的字符编号 fence，只用于核对输出放在说明与程式同页的 exporter 契约，不是新模型成绩、正式网站建置或已发布结果。源码 checked_notebook() 96–108 行会检查既有执行副本的 cell 数量、原 source 与 execution_count，实际 build() 238–252 行会先核对再输出；本轮阅读此契约，未宣称执行全站副本核对。

R.1 自身没有 Python、shell fence、图片引用或 SVG；它是文字导览，不要求通过空间位置、箭头或视觉标签理解机制，因此 figure check 具体不适用。1.1 的图和 W.3 的图是被链接课程内容，不纳入 R.1 的 figure_sha256，也未宣称完成它们的图审查。

官方来源亲读：Jupyter nbformat `v5.10.4` 的 docs/format_description.rst，HTTPS 原件保留在 original-sources。读取定位：原文件 21–59 行 Top-level structure，62–80 行 Cell Types，82–101 行 Markdown cells，104–141 行 Code cells。支持 Notebook 保存正文、可执行 code 和关联 outputs 的结构；不支持保证 Colab 可用或教程已在云端安装。实际验证环境 nbformat 5.11.1，Notebook 本身是格式 4.4；官方 v5.10.4 文档说明的是共同的 v4 文件结构，不冒称两个发行版相同。

原始来源取得过程：未带 v 的 5.10.4 路径分别在 docs/ 与 docs/source/ 返回 HTTP 404（curl 错误 22）；确认 tag 前缀后 v5.10.4/docs/ 路径成功。成功 URL 又以 curl --fail --location --max-time 30 --silent --show-error 实际取得一次，exit 0，stderr 保存；未关闭 TLS 或任何校验。

判断范围：本轮支持第一节材料与原程序、四个原链接及生成目标、同节 Notebook、正文/程式/实际 stdout 的单页输出契约、Colab 与下载目标、W.1–W.3 的内容入口。R/W 指南是阅读页，生成器只给 course/chapters 的编号教学小节产生 Notebook；R.1「同一节」按前文第一节教学页理解，不声称导览自身有 R.1.ipynb。无需安装的是静态阅读；动手执行仍按 W.1 建立执行环境。本轮未验证远端服务、在线站点、完整训练或任何模型能力。

R.1 没有独立展开的机器学习机制、数学公式或成熟方法因果主张。关于固定 ID 与字义的最后两句是阅读后的练习问题，未把它们改造成不存在的经验或机制结论；编号来回的具体材料按实际软件行为核对。Notebook 定义已按原官方格式核对，不以 concept NA 删去这项主张。
