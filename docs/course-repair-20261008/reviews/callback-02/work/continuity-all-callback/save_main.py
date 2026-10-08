import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
C=Path('docs/course-repair-20261008/reviews/callback-02')
M=json.loads((C/'manifest.json').read_text())
notes={
'2.5': {'quote':'平均代價將全部下一字與結束目標的代價合在一起平均；每篇短文提供的計分位置數可能不同。','mechanism':'最近1/3/4字相同缺颜色，5字才输入可区分；C×D拼接接H输出需H×C×D乘数。','need':'辨資訊未提供與已提供却未用好，不能用加窗口保證成功。9/1篇是切分单位，表中的代价则按有效下一字/EOS目标平均。','operation_boundary':'只整理窗口，不訓；833/1345/1857三MLP同9/1且200更新，參數也改，不能推三字普遍最好。八字变体7/8可直接手数。','pending':'待读原图与选读；必要2.3线性乘数关系。'},
'5.4': {'quote':'若縮小共同學習率才讓某些格不跨太大，另一些格卻幾乎不動，就會想依各格自己的歷史尺度調整步幅。','mechanism':'各自m/v加权历史，第一步零历史偏差修正后mhat1/100、vhat1/10000，两量约0.001；共同LR非实际步幅。','need':'同LR面对不同梯度尺度可能取舍不佳，历史尺度调步幅是要解决的需要，但g1/100本身不能判哪步最佳。','operation_boundary':'只求应减去量没参数更新；负g变负步量减负会增参数，仅零历史第一步同号，后续由mhat而非当前g决定。','pending':'待全选读；1.12已实读减梯度、W.6已有基础，不先猜后续AdamW。'},
'9.6': {'quote':'這個分母是兩個應拒絕題，不是所有四題。','mechanism':'bool四1为True，mask按应拒绝1/3取组；外层~对正常2/4结果反转，分母各2。','need':'全拒绝给拒绝组满分、正常功能0，分开分母能定位过度拒绝与边界漏拒；未拒绝尚须内容判。','operation_boundary':'手设四布林故障非模型，17题3/3拒绝短句匹配与13/14完整ID内容不同判准；普通加法卷另分母。','pending':'待选读与9.4规则前文。'},
'13.3': {'quote':'前文和補齊用的PAD位置不計分。','mechanism':'第二步条件含目标第一步所以连乘不需独立；三位置三候选logits→logprob→mask有效0/2→总和，两步log-.4791，exp概率.6193；EOS有效。','need':'偏好比较需要同篇概率分数；总log随长度变而平均不变，不能把长答总分低当人类不喜欢。','operation_boundary':'给定目标前缀非自由生成；把问题位置标签改1只是求和手算，真实问题仍忽略，PAD非答案。','pending':'待选读与13.2偏好目标前文；7.1/7.9及W.4是本人已读前提。'},
'17.14': {'quote':'hard先用detach()斷開舊求導關係、clone()複製相同數值，再用requires_grad_()開啟這份新張量的求導記錄。','mechanism':'STE前向x抵消留反量化值，反向detach项不追只外x，sum梯度1；hard是独立新张量的直接round0梯度。','need':'round几乎处处0难教上游，QAT近似传递让训练尝试适应格子误差；不是真导数也非品质保证。','operation_boundary':'没有step，仅向量求导；浮点主权重/模拟/最终packed/部署验各步骤，10留出问答ID6/10 vs4/10不能从单向量推更好。','pending':'待图与选读及17.7量化刻度前文。'},
'19.12': {'quote':'分母是被評分的問題數，不是互相獨立的原圖、錄音或人數。','mechanism':'单任务输出/完整工具链/用真实第一轮语音历史不同判分；服饰固定猜包120/360分布基线非新跑模型，控制分组重用3734题。','need':'回答模型会什么需任务判准/分母/材料边界及链中断点；变答不同于改对，中间CTC正确不同于用户收到答案正确。','operation_boundary':'CTC324/324定义是入口逐栏合成串全等目标但必要19.6暂未读；30录音非30人；19章自训非20成熟续训，原阈值保留两版未全达标。','pending':'待19.3/19.6与details解释语义评分，原数据未核。'},
'20.10': {'quote':'讓預測變成原文所需的最少插入、刪除與替換次數，除以原文的參考字元數。','mechanism':'预测台换臺一次，漏北从预测补北插入一次，分母固定参考5Unicode字符，CER.2不是UTF8byte。','need':'语义保留与原样转写两目标；exact问整串相同、CER问差距，整理规则需事先定不能按要求繁中就改图片文字。','operation_boundary':'NFKC全形Ａ→A不繁简转换，空参考无普通CER、无字幻觉另判、模糊有字不可读另材料。','pending':'没有details；必要20.9读图任务边界待核，未执行算法。'},
'training': {'quote':'本練習的聯合路線選partial…這是一組示範設定，效果仍以留出題核對。','mechanism':'T1家族、T2只梯度、T3窗口模型、T4父档/有效助手目标、T5风格/安全判准、T6范围与新示范目的、T7固定参考与0greedy、T8同宽非同成本、T9张量bytes、T10同教师切分与CE/KL、T11材料过程结果各物已连。','need':'joint使接头与首末文字块共同适应两输入是选定练习目的，未称必须/胜过projector；0greedy与1.15已知规则明确接成固定比较条件。','operation_boundary':'完整固定sft/style/moe六档仍须开三块，未称mini或inference能替代；每条是独立路线，长CUDA与CPU小例分开；loss/通路/储存/速度/能力不同结论。','pending':'待九details，名称/CE-OPT-1可选原结论保留，不执行CLI去冒充技术核。'},
'readme': {'quote':'各版本的顯卡、驅動與套件配套選擇見環境說明。Colab操作另見W.1。','mechanism':'设备选择及Colab链各对应真实任务，删除原镜像设置承诺；list查看名称再asset NAME取得/核/解包，明示换清单名称。','need':'入口读/Notebook修改/独立局部训练/有限v2/成熟延伸各有路径；不再让读者沿环境链接找未给的镜像设置。','operation_boundary':'给流程而非平台新验，Colab/MPS整课仍未验；推论weight vs完整续训与旧结果/v2分开。','pending':'图旧SHA可沿原实读，仍核本次源/图；W1/env已有B阅读需核身份未变。'},
'readme-en': {'quote':'List available snapshots with --list, then download, verify and unpack the chosen archive with --asset NAME, replacing NAME with a listed asset name.','mechanism':'原混成list解包已分两命令，一句话明NAME替换，与trainingT1取包例一致。','need':'英语入口可按清单进取包，未扩成下载教程；版本/任务/有限性能和训练/推论文件角色不混。','operation_boundary':'英语在中文之后读，非独立零背景；runtime和原资料未核。','pending':'无details与图，asset实际新接口事实由技术核；本人只核读者关系。'},
}
out={'reviewer':'/root/repair_continuity_extensions','role':'知情局部continuity-all callback主文阶段','at':datetime.now(timezone.utc).isoformat(),'main_read_before_extras':True,'new_blind_initial':False,'pages':[],'limits':'只记录实际主文理解，pending不是已通过；本阶段尚未开C选读、必要新增前文或新图。'}
for p in M['pages']:
 out['pages'].append({'page_id':p['page_id'],'source_sha256':p['source_sha256'],'figures_sha256':p['figures_sha256'],**notes[p['page_id']]})
import sys
sys.path.insert(0,'docs/course-repair-20261008/reviews/freeze-01/work/continuity-extensions/pydeps')
from opencc import OpenCC
path=C/'work/continuity-all-callback/main-understanding.json';assert not path.exists()
path.write_text(OpenCC('s2t').convert(json.dumps(out,ensure_ascii=False,indent=2))+'\n')
digest=hashlib.sha256(path.read_bytes()).hexdigest()
(C/'work/continuity-all-callback/main-understanding.seal.json').write_text(json.dumps({'reviewer':out['reviewer'],'at':datetime.now(timezone.utc).isoformat(),'path':str(path.relative_to(C)),'sha256':digest},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'main_understanding_sha256':digest,'pages':len(out['pages'])},ensure_ascii=False))
