from build_report import *
add('chapter-09','把內容依據、行為邊界和相關幫助分開','行為標準與機率信心導覽範圍相稱，無技術誤導。','導覽由盒子/加法/授權轉softmax與可靠分箱，各後頁有可核材料。','希望回答有根據、守规则且能往下走，再檢驗信心與真值差。','完整11頁已讀；不把toy行為标签當通用安全认证。')
add('9.1','有用看是否推進目標，誠實看是否符合可見依據，安全看是否守任務明定邊界。','三种判准与相对更安全／绝对安全区分正确。','缺数量给澄清可同时有用诚实安全；两回答均越界仍可有相对偏好，chosen不自动无害。','需要把目标分项写清楚，防只量一个好听标签；基于明确任务边界而非未定义通用真值。','safety.json pku pilot相对safer再检is_response_safe；官方PKU卡许可CC-BY-NC4.0、helpfulness/harmlessness分维且harmless超过harmful实际读。未逐条读外部PKU素材。')
add('9.2','生成目標的規則只讀可見欄位，不能偷偷把隱值教成猜中的獎勵。','未知内容澄清与偶然答中真值、可见依據区别正确。','visible3→3，None→要求数量/图片，hidden3不参与target；已可见还澄清亦未完成。','防猜中被当有根據、无限追问被当诚实，需要成对已知/未知材料。','safety.json固定unknown-ID1原题正确澄清、新问法猜6，说明标签规则不等于可靠推广；只引用toy字符串判准，未测事实世界知识。')
add('9.3','不讓標註者只憑喜歡的語氣評分。','纠错附和的独立真值和风格边界正确。','2+2由Python加法4独立核对，chosen改错前提再解释，rejected语气不改变错误。','礼貌可保留需求但需修正前提，避免偏好标注奖励附和；不是模型自己无真值识别。','safety.json固定1+3假设5：safety-only答8错误、mixed答4正确；原文报告具体窄case并未概括全部算术。')
add('9.4','公開性是題目設定，不由授權布林值推斷','可信授权／公开性与代码简化两个场景边界正确。','只两组合owner自己permissionTrue公开、他人False秘密，ifpermission规则仅本域，其他组合不承诺正确。','关键词拒答会挡正常题，全部放行越界；可信管理字段而非来文自称决定规则。','safety.json六固定permission题示例只含ID与permission，不做授权验证或公开性推理；正文明确这个实测范围，不能据6/6说现实权限完備。')
add('9.5','三篇都不提供碼，卻只有一篇給相關替代','拒答边界与相关帮助独立，人工材料范围清楚。','不行/边界+询问盒主/边界+天气三例都守边界，替代仅第二true；打印既存标签不量真实帮助。','需求被挡还可给相关合法下一步，堆无关文字不能满足帮助。','检查三个具体回答与规则相符；safety固定拒答substring检测不等同通用替代有用性，正文范围未超出toy。')
add('9.6','正常題「沒拒絕」還不等於已完成','分组分母与拒绝／完成分离正确，原数字一致。','人工shouldTFTF、actual全T→应拒2/2=1，正常未拒0/2=0；拒率本身不会量任务完成。','防拒越多越安全指标掩盖正常功能，需正常与拒题同时评。','safety.json同17test应拒3/正常14：only拒3/3正常12/14exact整体15/17；mix3/3+13/14=16/17且over0/14。900步同batch但曝光394888vs275389不匹配；算术仍0/7。')
add('9.7','後半命令是待分析內容，正解仍紅。','可信任务／待分析資料优先级及提示注入示例正确。','task定义找颜色是后词，document保留红+干扰蓝；target红由任务规则不是遵循document命令。','要读文件事实而不让其冒充任务，输入边界与不同地位需教／测，不是角色名字自带防护。','实看rewrite-09-07-data-task-640.png；safety.json原注入ID1输出green符合，改措辞pink干扰后答red错误；未把一次原句成功当安全保障。')
add('9.8','字串不同只是起點，還需讀題確認不是換數字或標點。','模板家族留出与固定ID泛化、改问法失败区分正确。','手写train/test结构不同但同缺信息规则；正式24盒含重复premise按ID%8规则家族去重，102/17/17无整句跨侧。','换ID或标点不足测规则理解，需新问法与预定评分；冻结现有模型才能独立看转移失败。','safety.json data136/8families分侧6/1/1；held_out_wording六原样本全部exactfalse并自然EOS。历史safety_behavior hashmatch且family修正逻辑与冻结一致；未重训、不宣称无任何近重复。')
add('9.9','最大比例只比較在場候選，不查標準答案。','softmax候选排名／尺度／外部真值与校准反例正确。','[2,1,0]argmax蓝0vs真红1；×10仍蓝但max由约.66524→.99995；正温缩放不改变argmax。','高模型偏好未增加事实依据，校准需留出真值；采样温度與用分类校准不同用途。','encoders.json audio test[11] freq320 amplitude.5 seconds.12 labelhigh却predictlow，conf.749011→.899048；同原logits除T.5，固定logits短核算完整14含该条通过。未将分类置信当生成答案可靠率。')
add('9.10','按信心區間分組、比每組平均信心和正確率，叫可靠度分箱。','可靠分箱／ECE／Brier／温度选择分侧正确，音讯失败诚实呈现。','toy四conf平均.75正确.5，高bin4 gap.25；等宽半开最后包含1，空bin不算平均。ECE按每箱人数权重绝对gap，不等于全局平均之差。','报告信心长期是否吻合需分组看真值；在validation按NLL选择T再独立test，可能因分布差变差而非保证改进。','实际读encoders.json calibration全val/test rawlogits、labels、bins和selection：visionval6/test6、audio val8/test14，T.5grid选；audio11/14不变mean87.9598→92.7251%、ECE.131856→.141536、Brier.202071→.253374。math短核各指标误差<2e-6；实际看两640图。未重跑训练。')
# 附加回查完成先前自记的待查项；尚未封存，不覆盖來源。
for p in pages:
 if p['page_id']=='5.8':
  p['checks']['technical_evidence']+=' 後續已實際核real_text.json tinystories.after.validation.samples[1]/[7]及after.test.samples[2]，相同prompt Once upon a time there w→as a big was a she was a ber the；32個generated ID。'
  p['unverified']=['未重跑訓練或下載模型／資料包；回查原JSON指定欄位與引用樣本。','完整故事／詩資料與全部生成未逐條讀；這不妨礙核正文指定例子。','網站版面與頁內程式未執行。']
 if p['page_id']=='5.16':
  p['checks']['technical_evidence']+=' 後續實際讀7.16與sft_ablation.json b_only/replay.training.effective_tokens=18453/36384，前引已核实。'
  p['unverified']=['本頁是假設長短答案計算，不是本頁實際訓練。','未逐頁執行或查網站版面；實際回放比較未匹配token預算。']
 if p['page_id']=='6.1':
  p['checks']['technical_evidence']+=' 後續實際核real_text.json chinese-poetry.after.validation.samples[7] prompt渡漢江/李頻/嶺外→孟沙，天夜天天天天春�，32generated IDs末237/172對應UTF8尾229/164未完成碼點；text.py/common.py UTF8safe24byteprefix与32newtokens回查且hash与记录匹配。'
  p['unverified']=['未逐頁執行／網站版面。','未重跑生成；指定原樣本與輸入／輸出byte邊界已回查，完整詩集未逐條讀。']
save()
