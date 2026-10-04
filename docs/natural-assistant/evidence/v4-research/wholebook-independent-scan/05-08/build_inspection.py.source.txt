from pathlib import Path
import datetime
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
manifest = json.loads((OUT/'source-manifest.json').read_text())
notes = {
'5.1': '固定三個下一字目標用40步檢查更新；embedding梯度與loss下降可診斷接口，不能證明新題泛化。',
'5.2': '長短兩筆合計五個有效答案，按答案而非筆數平均；忽略label不表示輸入看不見。',
'5.3': '動量把梯度歷史累進v，第三步新梯度反向但v仍0.71；方向記憶不能保證逐步下降。',
'5.4': 'Adam每格保存m/v并做偏差修正；第一步1和100都給約0.001位移，不外推成每步固定符號步長。',
'5.5': 'AdamW獨立衰減使2變1.96；零梯度與None區別有實現依据，任務loss可能暫時上升。',
'5.6': '暖身與餘弦是按步索引的計画；五步暖身后索引5仍峰值，尾端接近一成而非零。',
'5.7': '保存權重、optimizer歷史与RNG后可檢查同一步接續；例子不含隨機dropout，正式抽樣與排程需另保留。',
'5.8': '正確前文、自由生成、整串匹配衡量不同事件；左右續寫沒給方位線索，不能当事实问答错误。',
'5.9': 'FP32純參數6104×4与訓練記憶體/CPU前向時間分開；GPU同步与计时范围均解释清楚。',
'5.10': '驗證參與設定選擇，test在定案后評；上游train檔与本次自行留出的完整故事不是同一切分名稱。',
'5.11': '合併空白后的SHA指紋用來判完全內容重複；保留原文，不能当作語義去重。',
'5.12': '兩字Jaccard4/8與三字3/7計相近程度，左右标签不同仍可同家族切分；實跑未做近重复聚类。',
'5.13': '模型寬度×資料數四格設計用共同留出題比較；同150步/4筆未匹配token或FLOPs，四格不建立scaling law。',
'5.14': '平方函數候選都從同起點走一步，過大步幅跨谷底；换起点未证明最佳步幅改变。',
'5.15': '同seed重建可重現字表初值，不是品質排名；配對seed与相同資料順序仍需分开控制亂數消耗。',
'5.16': '90短筆/10長筆得到90/200答案token；抽樣比例、loss權重和实際累計監督量是不同記錄。',
'5.17': 'eval、no_grad与requires_grad各控制模組行为、計算記錄与参数求導；固定权重仍可回傳到輸入。',
'6.1': '碼點、byte、token三种長度分开；byte覆蓋原文不保证生成UTF-8合法，中文/英文32格不可当同字数。',
'6.2': '在文件内数相鄰對并非重疊合并ab；圖與Counter一致，訂BPE規則与訓練回答模型分开。',
'6.3': '完整256byte起始字母表确保未見字可表示，280预算只留少量合併空間；新句roundtrip不等於懂句意。',
'6.4': 'V×D的輸入/輸出表与T²序列成本形成取舍；共同原文上byte3894格/BPE2112格但参数增加。',
'6.5': '同原文總NLL/(bytes×ln2)比較BPB，明确EOS分子/原文bytes分母与窗口不重複計目標。',
'6.6': '結構0–7与內容byte+8分區，普通assistant字面拼寫不能變角色；BPE也要保存內容編碼规则。',
'6.7': 'UTF-8字可能跨token，单片decode的替代符不证明原bytes坏掉；延遲或增量解碼需保留尾段。',
'6.8': '窗口需取C+1格再shift一次，資料先按文件分側；窗口长度不改变tokenizer或词表。',
'6.9': 'I/O把hello切hel/lo再編碼會失去跨段合併；Whitespace边界成功不代表所有byte-level规则成功。',
'7.1': 'role/content→BOS-user-内容-EOS-assistant-内容-EOS；Y只計回答2与EOS，数据格式还不是回答能力。',
'7.2': 'prompt尾端assistant格負責首答案，EOS结问题与assistant起回答有不同功能；未知答案不能放入prompt。',
'7.3': 'label mask跟下一项目标shift，问题仍在X供读取；中文答案byte數决定有效目标数。',
'7.4': 'Q/A六输入格中first=4，assistant4→A73，而A73→EOS2；圖与代码明确分位置索引/ID/label。',
'7.5': 'BOS logits直接梯度0但Q embedding非零，因为答案讀Q；-100不等于detach或禁读。',
'7.6': '追加忽略label后平均保持0.8635；这里只检查loss统计，实际PAD输入等价留给下一节。',
'7.7': '因果许可AND有效key，加上保持真token位置0/1/2，才使左PAD等价；空key行有数值处理。',
'7.8': '每笔选最大True物理索引，左PAD不能用有效数-1；全False需拒绝，简易generate并未声称支持batch。',
'7.9': 'EOS作有效答案学停止，生成max_new_tokens截停是另一來源；cire+EOS证明正常结束不等于答对。',
'7.10': '截断后逐笔拒絕空監督，长度19只保留首byte目标，22才全答+EOS；UltraChat片段EOS并非原回答完成。',
'7.11': 'CPU代码只有forward/backward并未step或加载预训练；两条实际接续起点与额外250步预算明确区分。',
'7.12': '按整组合家族留出与未见属性是不同挑战；45训练/5验证/10末检，表格平均目标和生成匹配分開。',
'7.13': 'JSON结构与独立内容真值各打分，基模保留实际5/10而非假定满分；回归基准被选择后不是新test。',
'7.14': '交叉熵遵从提供label，错标可使自身loss更低；实跑8.9%非均匀污染不外推普遍损失。',
'7.15': 'A/B前后四格辨別旧能力退步与新能力提高；实际只练B使A5/10→0，但B新组合也未成功。',
'7.16': '回放重新用当前参数算旧题，并非存梯度；保住旧题的实跑监督量近翻倍，未声称纯同token回放效应。',
'7.17': '预训练从原文取下一项目标，后训练集中示范/反馈；成本是透明假设，小世界接续不证明产业两阶段必勝。',
'7.18': '示范、偏好、作答后reward→SFT/DPO/RL；RLHF与PPO、人工与程序验算分清，图中间流程可更明确。',
'8.1': '正確与比喻对应拆成rubric；固定積木句通过只能量狭窄写法，不代表灵性或洞察。',
'8.2': '换prompt影响这次上下文、不永久改权重；同模型/题/greedy的比较只支持教过的style切换。',
'8.3': '配对事实的短/长答案监督2/38位置；无style前缀再训练展示权重写法变化，但新加法仍全错。',
'8.4': '同题配各风格条件避免题材混杂；真实3种格式均合规而算术0/7，不能称通用指令或比喻能力。',
'8.5': '整回答能解析且恰好answer字段的格式规则与值4的内容规则分开；前缀散文与多字段都拒绝。',
'8.6': '必要date缺少才问，完整才确认；日期按家族留出，实际固定澄清成功但新日期复制失败。',
'8.7': '评分风格权重改排序不改正確标签；长固定句的平均loss分母不同，不支持算术或洞察提升。',
'8.8': '冻结W并加A16→2/B2→12共56参数，B零初始化令A首步梯度0；全模型LoRA6.7%不是总内存6.7%。',
'8.9': 'clone备份并copy两矩阵、记录alpha/rank与基模，ABA恢复与合并容差各有明确范围。',
'8.10': '0.75是与人工3/4一致而非3好答案；需回读分歧及校准混淆例，人工也不是绝对真值。',
'8.11': '交换显示A/B后映射回回答身份，永选首位变correct→wrong；手写裁判不是外部模型偏差实测。',
'8.12': '重复一句给5→50字符但只1种字符串，长度裁判故意故障；不同句子数不当成普遍信息分数。',
'8.13': '固定BA只改alpha验证修正倍数；换rank会改形状，不可说ΔW必减半，公式与基模指纹均需保存。',
}

issues = [
 {'id':'WB05-08-P2-01','priority':'P2','kind':'optional_pedagogical_clarity','section_id':'7.18',
  'source':'course/chapters/07.md','line':597,'figure':'course/figures/posttrain_signals.svg',
  'problem':'图的偏好卡右支直接抵达PPO框，仅以「也可先教獎勵模型」文字表示中间阶段。初读者虽然可由正文补回流程，单追箭头容易把离线chosen/rejected对误当成PPO作答后的reward。',
 'action':'在此支线加一个小节点「獎勵模型→替新回答評分」，或把箭头明确接到右上「作答後得到分數」卡，再进入PPO；保留三条可选路线，无须增加算法推导。',
  'evidence':['rendered/posttrain_signals.png','primary-sources/instructgpt-v1.txt:295-319'],
  'blocking':False,'status':'closed_after_parent_figure_revision',
  'closure':'Root将偏好支线接回「作答後得到分數」卡、标「先教評分員／再評新回答」，保留score→PPO箭头。本人完整重读7.18行578–605并读新版SVG、render/view；中间版caption末字接近下箭头，又建议x684→660。Root落实后本人再次读完整SVG并真render/view，18px两行清楚无重叠，流程与正文及InstructGPT §3.1一致。',
  'closure_evidence':['sources/figures/posttrain_signals.revised.svg','rendered/posttrain_signals.revised.png']},
 {'id':'WB05-08-P2-02','priority':'P2','kind':'optional_term_order','section_id':'8.8',
  'source':'course/chapters/08.md','line':278,
  'problem':'「低秩」首次出现在节首，并先称中间宽度为rank；窄通道为何限制自由方向的直觉直到11处模型改装与实测表之后才解释。数学基础够但初接LLM的读者可把2×16+12×2算对，却要读完较多工程细节才知道低秩究竟指什么。',
  'action':'将「所謂低秩，是修正先經過…窄通道」及rank增加的取舍提前到首次ΔW公式或图前；偏置12个数的说明可仍留在末尾。只调整既有段落顺序。',
  'evidence':['sources/chapters/8.8.md','primary-sources/lora-v2.txt:205-225'],
  'blocking':False,'status':'optional_not_required',
  'author_disposition':'Root倾向保留现有顺序；本人认可这不是理解阻塞，不要求修改。'},
]

executed=json.loads((OUT/'numeric-checks.json').read_text())
by_section={row['section_id']:row for row in executed['checks']}
issue_ids={sid:[issue['id'] for issue in issues if issue['section_id']==sid] for sid in notes}
reading=[]
for entry in manifest['files']:
    for section in entry['sections']:
        row=dict(section)
        row.update(read_order=len(reading)+1, reading_method='complete_sequential_text_read',
                   own_understanding=notes[section['section_id']], issue_ids=issue_ids[section['section_id']],
                   execution=by_section.get(section['section_id'], {'status':'not_executed_by_this_reviewer'}))
        reading.append(row)

support=[]
for rel in ['tiny_perceptron/training.py','tiny_perceptron/data.py','tiny_perceptron/model.py',
            'tiny_perceptron/alignment.py','tiny_perceptron/adapters.py','scripts/course_experiments/behavior.py']:
    path=ROOT/rel; dst=OUT/'sources/support'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
    support.append({'source':rel,'snapshot':str(dst.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'read_scope':'relevant helper definitions / adapter training and loading portions, not claimed full-file reading'})

raw=[]
for name in ['text_foundation','real_text','tokenizer','sft','sft_ablation','style','lora']:
    rel=f'docs/course-experiments/results/{name}.json';path=ROOT/rel;dst=OUT/'sources/results'/f'{name}.json'
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
    raw.append({'source':rel,'snapshot':str(dst.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'read_scope':'chapter-explicitly-linked raw results; inspected schema, aggregate fields and selected claims; no review verdict used; no long-run reproduction'})

figure_notes={
'6.2':'两文件、ab3/ba2/ac1及合并后2/3项完整可读；未跨文件连对。',
'6.5':'同4193bytes，左每token2.77947/3.98741与右BPB4.02906/3.43401相反；文字和bars一致。',
'7.4':'六格位置和ID各列分清；assistant4→A73及A73→EOS2，前四label忽略但X保留。',
'7.17':'一般文章→共用底座→三任务分叉，均写有后续资料/评测，箭头未承诺能力。',
'7.18':'三类信号与SFT/DPO/PPO都可读；偏好经reward模型打分支线建议补明确中间节点。',
'8.7':'1/1与3/0分项保持，两权重总分1.2/0.6与3/6，布局无重叠。',
'8.8':'W上路与A→B下路并行相加，alpha/rank=1与56参数说明完整可读。',
}
for fig in manifest['figures']:
    if fig['section_id']=='7.18':
        fig['initial_inspection']={key:fig[key] for key in ['source','snapshot','sha256','rendered']}
        fig['initial_inspection']['rendered_sha256']=hashlib.sha256((ROOT/fig['rendered']).read_bytes()).hexdigest()
        fig['intermediate_inspection']={'snapshot':'outputs/natural-v4/wholebook-review/05-08/sources/figures/posttrain_signals.intermediate.svg',
          'rendered':'outputs/natural-v4/wholebook-review/05-08/rendered/posttrain_signals.intermediate.png',
          'observed':'流程已修，但x684两行caption右缘与PPO箭头尖头近贴；建议中心左移24px。'}
        for field,key in [('snapshot','sha256'),('rendered','rendered_sha256')]:
            fig['intermediate_inspection'][key]=hashlib.sha256((ROOT/fig['intermediate_inspection'][field]).read_bytes()).hexdigest()
        fig['snapshot']='outputs/natural-v4/wholebook-review/05-08/sources/figures/posttrain_signals.revised.svg'
        fig['sha256']=hashlib.sha256((ROOT/fig['snapshot']).read_bytes()).hexdigest()
        fig['rendered']='outputs/natural-v4/wholebook-review/05-08/rendered/posttrain_signals.revised.png'
        figure_notes['7.18']='新版支线先教评分员再评新回答并回到作答后分数卡，既有score→PPO保留；完整正文重新读过，修前问题与修后图分别保留。'
    fig['source_xml_read']=True
    fig['visual_inspection']='rendered_png_viewed_directly'
    fig['renderer']='Inkscape command line, existing installed executable'
    fig['own_inspection']=figure_notes[fig['section_id']]
    fig['rendered_sha256']=hashlib.sha256((ROOT/fig['rendered']).read_bytes()).hexdigest()

primary=[
 {'name':'instructgpt-v1','sections_supported':['7.17','7.18'],'read_text_lines':[[38,85],[295,324]],
  'observed':'§1下一token目标不等于helpful instruction；§3.1明确pretrained→示范SFT→偏好RM→PPO scalar reward。'},
 {'name':'deepseek-r1-v1','sections_supported':['7.18'],'read_text_lines':[[220,297],[431,466]],
  'observed':'§2.2 R1-Zero基座直接RL且准确/格式规则reward；§2.3.1 R1用数千cold-start CoT先微调。原文方法名GRPO，教材未误称为PPO。'},
 {'name':'dpo-v1','sections_supported':['7.18'],'read_text_lines':[[23,39],[205,270]],
  'observed':'摘要与§4 Eq7直接偏好目标使用reference policy，绕过独立reward模型拟合与RL优化。'},
 {'name':'lora-v1','sections_supported':['8.8','8.9'],'read_text_lines':[[144,173]],
  'observed':'早期v1已明确冻结W、A随机/B零、并行相加、可合并；v1缩放写1/r，未将它当成alpha/r公式证据。'},
 {'name':'lora-v2','sections_supported':['8.8','8.9','8.13'],'read_text_lines':[[195,240]],
  'observed':'§4.1明确BA低秩更新、冻结原W、Gaussian A/zero B与alpha/r缩放。'},
 {'name':'rslora-v1','sections_supported':['8.13'],'read_text_lines':[[73,92],[150,177],[177,205]],
  'observed':'Eq4明确alpha/sqrt(r)，说明同名rank/alpha在不同缩放约定下不是同一倍率。'},
]
receipts=json.loads((OUT/'primary-sources/fetch-receipt.json').read_text())
for item in primary:
    receipt=next(row for row in receipts if row['name']==item['name'])
    item.update(url=receipt['url'],pdf=f"primary-sources/{item['name']}.pdf",text=f"primary-sources/{item['name']}.txt",
                pdf_sha256=receipt['pdf_sha256'],text_sha256=receipt['text_sha256'],download_exit_code=receipt['exit_code'])

checks=[]
for item in manifest['files']+manifest['figures']+support+raw:
    now=hashlib.sha256((ROOT/item['source']).read_bytes()).hexdigest()
    checks.append({'source':item['source'],'recorded_sha256':item['sha256'],'current_sha256':now,'unchanged':now==item['sha256']})

inspection={
 'schema':'fresh-independent-wholebook-precheck/v1',
 'identity':{'agent_task_name':'/root/v4_wholebook_scan_05_08','parent_task_name':'/root',
             'identity_basis':'actual delegated task identity in NEW_TASK message; no invented reviewer credential',
             'role':'fresh independent assigned-chapter precheck','audience':'数学较好的高中生/有基本数学的大学生，首次接触LLM'},
 'scope':{'full_files':['course/chapters/05.md','course/chapters/06.md','course/chapters/07.md','course/chapters/08.md'],
          'chapter_count':4,'sections_read':57,'lines_read':1922,'figures_directly_viewed':7,
          'reading_sequence_chunks':[{'source':f'course/chapters/{name}.md','ranges':ranges} for name,ranges in
           [('05',[[1,180],[181,360],[361,575]]),('06',[[1,155],[156,292]]),('07',[[1,170],[171,340],[341,505],[506,605]]),('08',[[1,160],[161,310],[311,450]])]],
          'snapshot_timing':'sources were copied after the first complete sequential read; chapters remain unchanged; root changed only the 7.18 figure during review and its original, intermediate and final inspected revisions are preserved separately',
          'rereading_events':[{'section_id':'7.18','source':'course/chapters/07.md','lines':[578,605],
             'method':'complete section reread following root figure mutation; complete revised SVG source and direct rendered PNG inspection',
             'reason':'original P2 diagram path finding corrected by root during review'}],
          'not_a_final_wholebook_pass':True,'chapter_20_read':False,'prior_review_verdicts_read':False,
          'canonical_edits':False,'baseline_checker_or_review_metadata_edits':False},
 'files':manifest['files'],'sections':reading,'figures':manifest['figures'],
 'supporting_code':support,'chapter_linked_raw_result_sources':raw,
 'issues':issues,'priority_counts':{'P0':0,'P1':0,'P2_closed':1,'P2_optional_not_required':1},
 'historical_failures_assessment':{'removal_required':[],
   'reason':'未见无教学价值的环境安装、补丁追逐或尝试流水账。保留的训练失败分别检验loss与生成、截断含义、组合泛化、遗忘、格式与内容、LoRA容量/成本，不因失败就应删除。'},
 'actual_execution':{'numeric_log':'numeric-checks.log','numeric_json':'numeric-checks.json','executed_fences':17,'completed':17,'failed':0,
   'numeric_claims_observed':{'5.1':'loss 2.272893667→0.000443551，首embedding梯度0.332364','5.7':'恢复后预测相等与各更新一步后预测相等两断言通过',
     '7.4':'first4、X=assistant4、Y=A73','7.7':'PAD有效位置最大差0','8.3':'位置10/46，目标2/38','8.8':'可训练56、初始相同、A梯度全0、B非0',
     '8.13':'alpha翻倍误差2.9802322387695312e-8'},
   'raw_gpu_results':'仅查阅链接原始JSON与摘录中的数值，没有复跑GPU实验',
   'raw_result_aggregate_consistency':'raw-result-checks.json: 28 printed aggregate values consistent at printed precision; two independently recomputed BPB values have difference 0',
   'primary_source_checks':primary},
 'not_executed':['全部notebook从头到尾运行','本章所有练习变体','文中600/400/900/1000/450步GPU长训练或其续训/adapter文件检验','真正外部LLM裁判调用','网站/手机/浏览器最终渲染'],
 'limits':['完整阅读范围只覆盖05–08；不授权把本报告写成全书最终通过','数值检查17段使用CPU/PyTorch2.14.1+cpu/tokenizers0.23.2，精确速度与GPU结果不互相替代',
           '图以Inkscape渲染后直接检视；未验证最终网站缩放和浏览器字体','已阅读章内明确链接的原始实验汇总字段，未独立证明其数据来源或复训统计可靠性',
           '链接前置章节以本章自身说明判读；未宣称已完整重读01–04、13、19或附录'],
 'source_final_check':{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_recorded_sources_unchanged':all(row['unchanged'] for row in checks),'files':checks},
 'environment_observation':{'setup_skill':'cloud-environment-onboarding:setup read; preserve checkout and distinguish observed/unrun used; no config draft change needed for read-only review',
   'system_python_torch_probe':'failed: torch missing','resolved_by':'reuse existing .venv/bin/python; no install or repository dependency changes'},
}
assert len(notes)==len(reading)==57
assert all(row['unchanged'] for row in checks)
(OUT/'inspection.json').write_text(json.dumps(inspection,ensure_ascii=False,indent=2)+'\n')
lines = [
 '# 05–08 章 fresh independent 全章預檢', '',
 '實際 task：`/root/v4_wholebook_scan_05_08`；parent：`/root`。對象是數學較好的高中生或有基本數學的大學生，首次接觸 LLM。完整依序閱讀 05、06、07、08 共 1,922 行、57 節與全部 7 張引用圖，沒有讀取既有 review verdict；未讀第 20 章。本報告只屬分章預檢，不能當成全書最終通過。', '',
 '沒有發現 P0/P1 技術錯誤或必須刪除的無教學價值失敗歷史。原先提出 7.18 圖流程的 P2 易讀問題，主代理已修正，本人已完整重讀 7.18 並檢視最終新版圖，問題關閉。另有 8.8 段落順序的可選 P2，現有說明與圖足以理解，不要求修改。本人未更動 canonical 教材、程式、baseline checker、review 或 reading metadata。', '',
 '## 實際問題與處置', '',
 '**WB05-08-P2-01，7.18 圖：已關閉。** 原圖偏好卡的側線直接進入 PPO 框，只有「也可先教獎勵模型」旁註。初讀者追箭頭可能略過訓練評分員、替新回答評分的中間階段。InstructGPT 原論文 §3.1 明確分示範 SFT、比較資料訓練 RM、再以 RM scalar reward 進行 PPO。主代理把支線改接「作答後得到分數」卡，加兩行「先教評分員／再評新回答」。中間版兩行標籤接近下箭頭，我又建議 x684→660；最終版已落實，18px 字與箭頭有間距、無重疊。三條路線與原有 SFT／DPO／score→PPO 箭頭仍可清楚追蹤。', '',
 '修前、第一次修正、最終修正 SVG 與 PNG 都保留。`inspection.json` 中的 figure 記錄包含三版 SHA 與 source，`rendered/posttrain_signals.revised.png` 是本次直接檢視的最終圖。7.18 正文沒有變，本人仍完整重讀行 578–605；修正流程與其示範、偏好、reward、RLHF/PPO 的區別一致。', '',
 '**WB05-08-P2-02，8.8：可選，不構成阻塞。** 「低秩」的窄通道與自由方向限制直覺在行 278，位於 11 處全模型改裝與實測表之後。可把這段既有直覺提前到首次 ΔW 公式旁，讓數學初學者更早把 rank 與限制連起來；不需要增加矩陣秩推導。主代理傾向保留既有順序，本人接受，因前段 16→2→12、56 參數的圖與算式已能支援閱讀。', '',
 '## 已執行查核', '',
 '- 執行 17 段原文 Python fence，全部無例外完成。只 5.1 做 40 次參數更新，其餘為短算例或接口查核。使用既有 `.venv` 的 CPU、PyTorch 2.14.1+cpu、tokenizers 0.23.2，沒有安裝或改依賴。原碼、stdout、stderr、版本與 fence SHA 保留在 `numeric-fences/`、`numeric-checks.json`、`numeric-checks.log`。',
 '- 5.1 loss 2.272893667→0.000443551；5.7 重載及再更新一步兩次 equality 斷言通過；7.4 first=4，assistant4→A73；7.7 左 PAD 等價差 0；8.3 位置 10／46、有效目標 2／38；8.8 可訓練 56、初始相同、A 首步梯度全 0／B 非 0；8.13 alpha 翻倍誤差 2.98e-8。',
 '- 閱讀章內明確連結的原始實驗 JSON，28 個印出精度的彙總數值一致；以總目標×平均 NLL／原文 bytes／ln2 獨立重算 6.5 兩個 BPB，與原始值差均為 0。這是引用數值與算術核對，沒有重跑長訓練。詳 `raw-result-checks.json`。',
 '- 完整讀取 7 張 SVG XML，使用已安裝 Inkscape 渲染，再直接檢視 PNG。7.18 的最終修改另外實際重新渲染、檢視，並保存 renderer stderr。',
 '- 取得並閱讀 InstructGPT v1 §1/3.1、DPO v1 §4/Eq7、DeepSeek-R1 v1 §2.2/2.3.1、LoRA v1/v2 相應方法段、rsLoRA v1 Eq4；PDF、抽取文字、TLS 預設 curl 命令、exit code、SHA 與實際閱讀行數均保留在 `primary-sources/` 與 `inspection.json`。LoRA v1 的縮放寫 1/r，沒有誤拿它作 alpha/r 證據；alpha/r 由 v2 與本專案程式核對，alpha/√rank 由 rsLoRA 原文核對。', '',
 '## 失敗結果的教學價值', '',
 '05–08 的失敗結果有具體概念用途：5.8 分開留出 loss 與自由生成；6.1／6.7 解釋 byte 尾段與替代符；7.10 檢查截斷後答案與語義是否完整；7.12／7.15／7.16 分開组合泛化、新任務尚未學會、舊任務退步與回放預算；8.3–8.8 分開寫法、內容、有效目標分母與 LoRA 訓練參數。沒有發現環境修補、反覆重試或過往版本成敗流水帳。這些段落不宜僅因成績差就刪掉；它們也沒有把 toy 的狹窄成功升格成一般 LLM 能力。', '',
 '## 每節獨立理解記錄', '',
 '以下每節皆為本人完整順讀後的理解。每節的精確起訖行、來源、SHA、保存文字、Python fence 狀態及圖關聯見 `inspection.json`；SHA 用來綁定實際閱讀版本，不取代閱讀。', '',
 '| Section | 本人理解／問題 |', '| --- | --- |',
]
for row in reading:
    suffix='；'+','.join(row['issue_ids']) if row['issue_ids'] else '；未見需修問題'
    lines.append(f"| {row['section_id']} | {row['own_understanding'].rstrip('。')}{suffix}。 |")
lines.extend(['','## 未執行與限制','',
 '未執行整份 notebook、所有練習變體、文中 400–1,000 步 GPU 訓練、真實 adapter 載入／合併檔案檢驗、外部 LLM 裁判或網站／手機最終渲染。長實驗只查引用原始結果，不獨立證明資料來源或統計可靠性。前置章節以本章已有說明判讀，未宣稱完整重讀 01–04、13、19 或附錄。這些限制不應寫成已通過。','',
 '## 來源回核','',
 '四章全文及 57 節保存於 `sources/chapters/`；原版與最終版圖保存於 `sources/figures/`。初讀之後主代理只修改 7.18 圖，本人保存原版、再次完整讀節與重新檢視新版。`inspection.json` 的最終回核以最後實際檢視版本為準；全部記錄的現行 chapter／figure／support／raw-result source SHA 一致。', '',
 '| 全文來源 | SHA-256 |','| --- | --- |'])
for entry in manifest['files']:
    lines.append(f"| {entry['source']} | `{entry['sha256']}` |")
lines.extend(['','實際順讀分段：05 行 1–180／181–360／361–575；06 行 1–155／156–292；07 行 1–170／171–340／341–505／506–605；08 行 1–160／161–310／311–450。圖修改後再完整讀 7.18 行 578–605。',''])
(OUT/'report.md').write_text('\n'.join(lines))
print(json.dumps({'sections':len(reading),'numeric_fences':17,'figures':len(manifest['figures']),'sources_unchanged':all(row['unchanged'] for row in checks)},ensure_ascii=False))
