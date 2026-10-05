# 本人独立查证记录：12.1

任务身分：`/root/phase4_factual_coordinator/factual_12_1`。实际读当前章首导言、12.1 全节、前置 W.3；没有读后文实验、旧技术／reader 报告、作者修正摘要或其他人的结果解释。初次宽 filename 查询仅列档名，没有读取这些档案内容。正式小节与导言均保存原始 UTF-8 bytes，未正规化换行。

导言摘要：本章说明声音先按时间保存振幅，再经过固定频谱处理送到自行训练的入口。教学路线先以合成单音观察资料流与控制，后续区分听写和需求回答，并安排真人短句有限意图与打字共用对话。本次摘要记录导言的教学安排，不把后文成品能力当成本节已验收成果。

## 原始代码与资料契约

先 AST 列 `tiny_perceptron/multimodal.py` 的顶层 API，实际读 `tone` 第 44–46 行。契约是 `tone(frequency=440.0, seconds=0.1, sample_rate=16000)`，计算 `N=round(seconds*sample_rate)` 与 `times=arange(N)/sample_rate`，返回 `0.5*sin(2*pi*frequency*times)`。本节没有训练、参数更新或既有模型实测主张，也没有需核对的原实验 JSON。

先读 `section_facts.py` 的擷取、bootstrap 与执行契约，以原 fence 执行 CPU helper。源代码、bootstrap、环境、完整 stdout/stderr 和实际 worker argv 都在 `original-execution/`。实际 exit=0，guard_events=[]，torch=2.14.1+cpu，CUDA build=None、available=False。原 stdout：1600 样本；前三数 `[0.0, 0.086, 0.169]`；极值 −0.5 和 0.5。

`check_variants.py` 是本人的有界 CPU 检查，不写权重。0.2 秒实际产生 3200 样本，原 1600 点逐点完全相同，FFT 主峰 bin=88，间距=5 Hz，换算=440 Hz。原 0.1 秒 FFT 主峰 bin=44，间距=10 Hz，也为440 Hz。延长时长增加的名义周期数从44变88，频率参数与每点间距不变。没有播放声音或做真人音高测试；音高与频率的关系查官方教材。

振幅变体 `2*wave` 极值 ±1，主峰440 Hz；频率变体 `tone(880)` 样本数仍1600、极值仍±0.5，主峰880 Hz。两段1600点沿新增轴 stack 得到 `[2,1600]`。1600点与800点 padding后得到 `[2,1600]`，第二段末800个零是程式填入，valid mask 只计800个原样本；所以这些零不构成真人沉默的观察。

独立双精度代入 `x[n]=0.5*sin(2*pi*440*n/16000)`，前三值与float32原 helper最大误差7.292e−9（阈值1e−7）；极值阈值1e−6。样本间隔1/16000秒，周期1/440秒，约36.3636样本／周期，不能以第440样本作为频率定义。半开时间区间最后样本位于0.0999375或0.1999375秒；名义时长依N/fs为0.1或0.2秒，不以最后时间点代替时长。

## 权威原文的真实读取范围

Adobe 官方 Audition `sound.html`，Last updated Apr27,2021：亲读开场空气压力、零线／正负、Waveform measurements 中 Cycle、Frequency、Phase，以及 How sound waves interact。支持周期、Hz=每秒周期数、周期内高低位相、频率与单音音高、复杂声波由多波叠加。没有沿用该页把振幅描述为峰到谷的措辞去定义本例系数；本例0.5作为相对零线的峰值直接由正弦公式、SciPy正弦范例及实算支持。

Adobe 官方 `digitizing-audio.html` 同版本：亲读 Analog audio: positive and negative voltage、Digital audio: zeroes and ones、Understanding sample rate、Understanding bit depth 与 How Adobe Audition digitizes audio。支持每个sample记当时幅值、sample rate为每秒数、存储按时间顺序。未把本节扩展成量化精度或Nyquist教学。

SciPy 官方 `scipy.io.wavfile.write`：亲读 Parameters 的rate（samples/sec）、1-D/2-D data契约、Notes 的多声道形状、Examples 的 `amplitude*sin(2*pi*fs*t)` 单音。本节 `[2,1600]` 明确是batch轴；它不是该SciPy WAV API的多声道 `[Nsamples,Nchannels]` 契约。本节没有调用此API写档。

PyTorch 官方2.14 Tensor页面亲读多维tensor介绍；2.14 pad_sequence页面亲读整段API、batch_first的B×T×*输出、padding_value与长度定义。PyTorch原始API docstrings还直接核对安装版2.14.1+cpu（upstream git 5c4886908584029761b579af026dcfb627c84070）：先AST定位后读 `_torch_docs.py` stack1918–1976、max6919–6945、min7553–7573、arange9679–9727、sin10335–10361；`_tensor_docs.py` item2788–2805、shape4760–4778、tolist5445–5465。相关API合组核对tensor显示、采样和batch契约，没有逐语法列claim。

原-source-locators仅用于定位官方raw档：读取顶层keys/types、列表长度与item keys/types、`/locators/*/url`，再具名读取43、44、53的url/version_label/original_path/sha256。保存43、44的原始文件后与目前安装版逐bytes一致；亲核实际AST范围，未读其他人的inspection或判定。53原rnn raw仅保存未用作证据。官方网上部分请求503/403及一次错误Adobe路径404均保存在downloads清单；成功直接下载的原文和安装版原source足以支持本节相关范围，未将失败请求当作验证。

## 图与页面

`multimodal_wave_cycle.svg` 用Inkscape实际渲染，并view `figure.png`；Chromium以1280×800与390×844打开HTTP8765的12.1页面并实际view两份截图。当前练习、代码和正文匹配冻结现稿。横轴写周期进度／相位，0、1/4、1/2、3/4、1对应幅值0、0.5、0、−0.5、0。蓝点是周期内示意取样位置，未标成默认16000Hz生成的前五个点；连续曲线是示意，不能当成完整1秒录音或语音素材。基本标示在桌面和手机可见，手机代码横向滚动。没有其他实测数值图。

## 本人支持范围

本节是合成mono单音的表示、显示、时长和受控参数变动，另说明batch与padding来源约定。没有语音辨识、泛化、训练或真人沉默检测能力宣称。单峰值不能保留复杂声音内容的充分资讯：不同频率的两个正弦已有相同最大值，最高点本身也不包含时间演变；这只是信息不足的限制，不是用玩具实验评估语音系统。
