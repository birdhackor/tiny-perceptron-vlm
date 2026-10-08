import datetime,hashlib,json
from pathlib import Path
B=Path('/workspace/tiny-perceptron-vlm/docs/course-repair-20261008/reviews/freeze-01')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
M=json.loads((B/'manifest.json').read_text());P={p['page_id']:p for p in M['inventory']['pages']}
E=json.loads((B/'evidence/technical-learning/check-results.json').read_text())
role='/root/repair_tech_learning'
now=datetime.datetime.now(datetime.timezone.utc).isoformat()

def check(ident,kind,quote,location,verified,support,limit=''):
 return {'id':ident,'kind':kind,'原句':quote,'location':location,'判定':'已足夠','technical_status':'verified','核查':verified,'支持':support,'範圍':limit}
def page(ident,commitments,checks,mechanism,need,how,example,taught,figures,limits):
 p=P[ident];t=(B/'freeze/sources'/f'{ident}.md').read_text();actual=sha(B/'freeze/sources'/f'{ident}.md');assert actual==p['source_sha256']
 for c in checks:assert c['原句'] in t,(ident,c['原句'])
 for q in commitments:assert q in t,(ident,q)
 return {'page_id':ident,'source':p['source'],'selector':p['selector'],'source_sha256':actual,'figures_sha256':p['figures_sha256'],'reading_scope':'指定凍結正文及本節完整選讀；必要前文另列，不使用任何其他角色判斷。','original_commitments':commitments,'technical_checks':checks,'四題與教學關係':{'身分及前文':taught,'機制／性質':{'status':'已足夠','answer':mechanism},'需要／用途':{'status':'已足夠','answer':need,'scope':'按本頁原句及當下用途判斷，沒有另要求全面方法優劣證明。'},'使用與後續':how,'例子支持及邊界':example,'已教關係':taught,'仍缺必要關係':'無：在所查原承諾的範圍內未找到尚缺的必要概念、方法理由、表示轉換或計分關係。'},'figures':figures,'necessary_findings':[],'optional_findings':[],'unknowns_and_limits':limits,'verdict':'所列技術與來源核查已足夠；未驗證範圍另列，不代表全頁首讀或全平台驗收。'}

pages=[]
pages.append(page('5.4',[
 '兩格參數的梯度是1與100，都用學習率0.001，基本梯度下降會分別移0.001與0.1。',
 'Adam（Adaptive Moment Estimation，自適應矩估計）再參考每格「過去通常多大」的梯度，用大小歷史調整方向的尺度。',
 '因此共同學習率不是每格實際位移，續訓也需保留m、v與步數。'],[
 check('5.4-c1','concept','m保存梯度加權平均，v保存平方梯度加權平均，β1、β2各控制一份歷史的保留比例','L5','分子是一階梯度加權平均，分母使用平方梯度二階平均的平方根；兩份歷史逐格保存。不是把v當參數、也不是把v稱作已去均值的統計變異數。','official-adam: L274–313；算法m_t、v_t及更新式。'),
 check('5.4-c2','numeric','第一步m得到 `[0.1,10]`，v得到 `[0.001,10]`。','L22','從0開始，(1−0.9)×[1,100]=[0.1,10]；(1−0.999)×[1,10000]=[.001,10]。執行原程式所得m/v與敘述一致。','check-results: source_programs/5.4、small_variations/5.4。','FP32顯示及乘法有捨入；數字是指定算例，不是實驗效果。'),
 check('5.4-c3','concept/numeric','這叫偏差修正；到第t步的分母分別是 `1-β1**t` 與 `1-β2**t`，t表示已經更新幾次。','L22','0初值下常數梯度的加權總量為(1−β^t)g；除去這份不足權重可還原常數梯度量。官方算法同式，第一步分母.1與.001。','official-adam: L297–309；本次透明推導及原程式。','這說明0初值偏差修正，不宣稱能消除所有非平穩梯度的估計偏差。'),
 check('5.4-c4','software/numeric','兩格都約為0.001，而不是一格0.001、一格0.1','L24','m_hat=[1,100]、sqrt(v_hat)=[1,100]，比值約[1,1]；eps=1e-8防止分母0。原塊印出[.001,.001]，沒有建立或寫入參數。','official-adam: L307–310；check-results: source_programs/5.4。','第一步同尺度梯度及零歷史的特例，不能推出每步位移都相同。'),
 check('5.4-c5','numeric/concept','後續更新量的正負由歷史平均 `m_hat` 決定，可能與當前梯度不同號。','L28','分母非負且eps>0，正負由m_hat決定；β1>0可保存之前方向。改100為−100時v不變、m及更新量第二格變負。減去負更新量增加參數，與1.12更新規則一致。','official-adam: 更新式；前文5.3第三步的歷史方向；check-results: small_variations/5.4。')],
 '每格以自己的m與v調整梯度尺度；共同η不是每格的實際位移。偏差修正處理0初始化造成的早期平均偏小。',
 '原先同一η會使1與100兩格移動相差100倍，本頁要學的是能否用各格大小歷史調整尺度。正文L3、L24、L26把原更新、尺度改變、實際步幅的解讀連起，沒有宣稱Adam一定改善收斂或品質。',
 '給g和β歷史，算m/v、修正、更新量；本例只算應減去量。真的更新沿用1.12；續訓需同時保留歷史与步數。',
 '零歷史第一步能展示梯度大小100倍而位移近似；符號練習展示方向与平方量的差別。L26明說後續不必相同，因此沒有用特例概括所有步。',
 '1.12提供減去η×梯度的更新與正負意義；5.3提供跨輪歷史與當輪方向可不同。當頁把Adam名字、兩份歷史、β及t的角色完整交代。',
 {'existing_figures':[],'need':'本頁兩格狀態、公式、印出值與符號練習足以追蹤同一計算，未發現必须另圖解的缺口。','page_layout':'未驗證實際頁面。'},
 ['只執行CPU透明算例；未做Adam長訓練、各β設定的品質或收斂比較。','未獨立驗證不同後端或不同PyTorch版本；本頁不提出這類結果。']))

pages.append(page('5.5',[
 '這份額外把權重拉向0的動作叫權重衰減，和朝答案梯度移動是兩件事。',
 'AdamW直接把w乘上1−ηλ，再減去Adam方向的步子。',
 '若把權重平方懲罰加進任務代價，對應梯度也會進入Adam的m、v再被縮放；AdamW讓回拉與這份歷史分開。'],[
 check('5.5-c1','concept','它是否有助於未見題目，仍要用驗證資料判斷。','L3','权重衰減是配方候選：加入向0的作用，不等同保證泛化改善。平方懲罰的导数加入g時會進入m與v，而decoupled decay直接乘權重。','official-adamw: L60、L79–89；official-adam: L413–430。','沒有要求本課重證成熟方法全面收益，也没有将回拉示意當泛化結果。'),
 check('5.5-c2','numeric/software','零任務梯度下應剩2×0.98=1.96。','L7','新AdamW歷史為0，有零梯度时Adam方向為0；ηλ=.02，所以w=2×.98=1.96。原塊assert通過，CPU實際1.9600000381469727。','check-results: source_programs/5.5、small_variations/5.5；official-adamw算法。','成立於本例新最佳化器、正λ与η及零歷史；有舊m時，即使當輪梯度0也可能另有歷史方向步。原文明确给新optimizer例。'),
 check('5.5-c3','software','若 `.grad=None`，該參數本次會被跳過；若有一份零梯度，優化器知道它參與了這一輪，仍可應用衰減。','L23','官方Adam共享的參數收集只接受p.grad is not None；AdamW設定decoupled_weight_decay=True。重跑新w時zero+.2=1.96、zero+0=2、None+.2=2且不建立狀態。','official-adam: L150–184；official-adamw: L35–47；check-results: small_variations/5.5。','本節已限定本段PyTorch實現；不外推所有最佳化器契約。'),
 check('5.5-c4','concept','縮小有用的大權重可能暫時提高任務代價，需用驗證資料選強度。','L25','回拉与任务梯度方向独立，故不能保證任務代價下降；例如任务最优权重本来非0，回拉会偏離。這是可由前述兩個動作及1.12推得的機制邊界。','L3、L7与L25；官方AdamW獨立乘法。','未宣稱bias／正規化一律不衰減，原文用「有時」並要求明說分組。')],
 'Adam方向處理任務梯度及歷史，衰減直接對w做比例回拉；不讓衰減項進入m/v，λ可另控制這份作用。',
 'L3先給偏向較小權重的候選需要，L7给直接回拉，L25比較平方懲罰受Adam歷史縮放與分開回拉。此安排回答如何独立控制，不用零梯度特例代證泛化。',
 '建立真正w與新optimizer，指定零gradient，step實際改參數；None會跳過。weight_decay=0与None練習分別查是否啟用衰減及是否進入本輪更新。',
 '零任務梯度特例隔離回拉0.04，配合正文L25足以看見與m/v分開的設計；沒有實際任務學習或驗證分數。',
 '已讀5.4提供Adam方向与m/v；1.13提供None清gradient但不更新参数的先备。本頁在使用None比較前解釋本輪參數參與的差異。',
 {'existing_figures':[],'need':'一格參數、式子及step前後值明確，无多路對應必须由圖補上的缺口。','page_layout':'未驗證實際頁面。'},
 ['未驗證衰減強度對任務或留出題的實測效果。','「bias或正規化倍率有時另分組」只作工程候選提醒，未檢查本課全部训练入口是否如此分组，也未將其算成通過承诺。']))

pages.append(page('5.6',[
 '這份隨步數給步幅的時間表叫排程。',
 '若不想起步幾次就大幅改動參數，可以先縮小這幾次的更新步幅，再按計畫放大',
 '下面只計時間表；索引i從0開始，i=0就是第一次更新，沒有先跳過一輪。'],[
 check('5.6-c1','software/numeric','前五次依序是0.0002、0.0004、0.0006、0.0008、0.001。','L3','learning_rate暖身式peak×(i+1)/warmup；i=0..4对应第1..5次，原程式前六項最後两項均.001。','frozen-training: L106–117；check-results: source_programs/5.6。'),
 check('5.6-c2','software/numeric','最後三項接近0.0001，而非硬變零。','L18','40步、warmup5的索引37–39輸出.0001162167、.0001072317、.0001018116；函数保留peak×.1下限且i=39尚未進度1。','frozen-training: L112–117；check-results: source_programs/5.6。','「靠近」與實作吻合，原文沒有誤稱最後一步必達下限。'),
 check('5.6-c3','software/numeric','若填30並不表示三十步都暖身，實際會被限制成10。','L29','min(warmup,max(1,total//4))使40步上限10；warmup10与30的完整所查輸出相同，total1..3皆上限1。原练习5→10实际生效。','frozen-training: L111；check-results: small_variations/5.6。','边界内容放選讀，正文程式及练习均不需靠该補充才能正確解释。'),
 check('5.6-c4','numeric/figure','圖中橫軸是更新次序，縱軸是當次學習率。','L20','实看640/360静态圖，x为第1..40次，y标出.001、.0005、.0001；SVG40个折线點各对原函数，最大坐标差.004987以内，为两位小数序列化误差。','figure-learning-rate: SVG及两宽PNG；figure-check.json。','这是预定时间表，没有模型参数、梯度或训练曲线。'),
 check('5.6-c5','concept','排程按預定進度工作，不會自動知道最佳步幅','L20','函数只依i、total、peak、warmup，不读验证表现；变total会改余弦进度。L5將起步目的連到1.12的大步提高代價反例，並明說需验证選設定。','frozen-training: learning_rate；正文L5及L20；必要前文1.12。')],
 '以更新序次算η：起步逐渐放大，之后餘弦平滑降低且保留高峰一成下限。i0就是第一次，i5為第六次仍在高峰。',
 'L5给避免起步大幅改参数的需要，缩小η直接缩小所算更新步幅；L3的最後細調与下降段相连。正文未宣称余弦一定胜过其他排程，故不另强求最优方法证明。',
 '只计算rates、不前向、不求梯度、不step；assert检查高峰容差。原图及warmup10练习让使用者能对上更新序次。续训保留总计划，延长是改计划。',
 '一趟40步與5→10变动足以显示起步、峰值、尾端及索引；不能据此判训練品质。',
 '1.12已教步幅太大会越过好位置；本頁明确计划参数與索引約定，assert的1e-9是比較容差。',
 {'existing_figures':['course/figures/rewrite-05-06-learning-rate.svg'],'viewed':['renders/rewrite-05-06-learning-rate-640.png','renders/rewrite-05-06-learning-rate-360.png'],'kind':'既有靜態圖，均真正view_image；不是實際頁面截圖。','mapping':'i0→圖更新1，i4/i5→第5/6步高峰.001；尾點→第40次約.0001018。軸、兩段說明与正文對應。','readability':'兩個尺寸中必要刻度、横軸及纯排程说明可读。','page_layout':'未驗證原位頁面的相鄰、折疊、浮動介面或捲動路徑。'},
 ['未執行訓練，也未比較不同排程对品质或稳定性的實測。','未覆核所有调用者的step順序與續训计画保存；本頁只承诺工具的时间表約定。']))

pages.append(page('6.5',[
 '切法變了，「每token」就不是同一份文字，先回共同原文比較。',
 '條件機率的乘積經負對數變成和，不需要字互相獨立。',
 '固定原文曝光未固定參數、監督token與運算量，這示範單位的差別，不能外推工具優劣。'],[
 check('6.5-c1','numeric','平均看似甲較低，總代價卻甲6、乙5。','L3','10×.6=6、5×1=5；低每token平均不保证低同原文总NLL，因为事件单位不同。','透明乘法推導；L3。','这是手工设定，不是本节后面的实测同一数值。'),
 check('6.5-c2','concept/numeric','將總 NLL 除以 `ln(2)`','L5','对序列条件概率取−ln后可相加，链式法则无需独立；−log2 p=−ln p/ln2。UTF-8貓3+🙂4=7，5/(7ln2)=1.03049646，4/(7ln2)=.82439717。原程式实跑吻合。','透明對數及鏈式推導；check-results: source_programs/6.5、small_variations/6.5。','手写NLL明确未当作学習成效。'),
 check('6.5-c3','software/concept','不能拿甲的ID丟進乙的模型，也不能只拿平均loss、沒有目標數量就推BPB。','L19','候选ID与各自词表绑定；total=mean_token_nll×effective_tokens。冻结评测使用各自tok与匹配vocab_size的model，对BOS+正文+EOS shift一次，raw_bytes只计正文。','frozen-text: _evaluate_tokenizer L378–406及run_tokenizer L461–506；frozen-data: shifted；frozen-model: loss_sum。'),
 check('6.5-c4','concept/software','窗口[A,B,C]計B、C，[B,C,D]只計D，別重複C','L21','示意窗口说明前文重用但计分事件不能重复。固定实验片段最多256byte、max_length384，每原片段一个窗口，无滑动重算；EOS计总NLL且raw_bytes不计EOS。','frozen-common: text_examples L78–97；frozen-text: L387–402、L429–438及L463–467。','文字一般長文規則并未冒稱本次已经量过滑動長文。'),
 check('6.5-c5','empirical/numeric','已有比較逐byte每token代價2.77947低於BPE的3.98741，BPB卻4.02906高於3.43401。','L25','原始tokenizer.json results/runs两版after/test分别为2.7794730917393893、3.987410554112409。目标数4213/2503、原文4193byte；重构总NLL11709.919136/9980.489617，再除共同4193×ln2得到4.0290588382、3.4340094046。','raw-tokenizer: results/runs/{byte256,bpe512}/after/test；check-results: raw_record_recalculation/tokenizer。','只重算原始报告中的量，未重新训练或重新生成模型分数。'),
 check('6.5-c6','empirical/software','累計都看了664,759個原文byte。','L32','报告两run的steps400、training_raw_utf8_bytes_exposed664759，固定raw_document_schedule SHA 0198fcf909d8b754ddc3a7ffbf6aae3f2b483f583295f0a0aeea6b614a54ecf2。源碼每步用同exposure取8篇，width32/layers1/max_length384/lr.003；BPE只以训练侧正文建表。参数41824 vs57696，不固定监督token或FLOPs。','raw-tokenizer: matched_budget、runs/*、data及code_sha256；frozen-text: run_tokenizer L409–512。','未读取全部192篇原始JSONL，664759曝光数字的本轮支持来自原始报告与其源代码，而非本轮独立重建完整曝光。'),
 check('6.5-c7','empirical/numeric','逐byte的BPB由訓練前8.29037降至3.59830，BPE由5.04502降至3.06109。','L36','原始validation19篇3894byte，byte目标3913、BPE目标2131；本轮由每token NLL×目标数重算所有before/after BPB，均以1e-12绝对容差吻合。','raw-tokenizer: runs/*/{before,after}/validation；check-results: raw_record_recalculation/tokenizer。','基线事件不同，正文明确不把随机模型当相同起点；32新token允许的文字长度也不同，未当质量证据。')],
 'NLL将匹配词表中的正确条件事件代价加总；换底成bit，再除同份原文bytes，才能把不同每token单位拉回共同尺度。',
 'L3明确低每token平均可能隐藏较高同原文总代价这一比较需要；L5、L17解释同7byte分母；L19–L21使计分目标与正文分母边界可追踪。其作用是读懂同份原文比较，未将单位换算当公平成本实验。',
 '各自tokenizer编码同一验证文件与匹配model算NLL，排除不计分前文及PAD，按相同开头、上下文、EOS协议；有效目标数不能省。',
 '手工7byte算例显示分母及换底，正文数值和选读实测显示排名反转；指标不证明512最佳、BPE普遍优越或生成回答更好。',
 '6.1已教token與byte不互换；6.4已说明参数/序列取舍；本頁正文明确EOS分子与原文字节分母，不需先打开选读才能解读实测列出的排名。',
 {'existing_figures':['course/figures/rewrite-06-05-shared-denominator.svg','course/figures/tokenizer_common_scale.svg'],'viewed':['renders/rewrite-06-05-shared-denominator-640.png','renders/rewrite-06-05-shared-denominator-360.png','renders/tokenizer_common_scale-640.png','renders/tokenizer_common_scale-360.png'],'kind':'既有静态PNG render均真正view_image；不是页面截图。','mapping':'示例A/B对应正文手写NLL5/4，共同分母7×ln2；黄色微笑图明确代表U+1F642，避免把绘图当不同原文。实测橙色逐byte/蓝色BPE，上栏每token值与4213/2503目标，下栏BPB共同4193byte；排序及EOS约定完整标出。','readability':'兩寬度所有必要數字、单位、区别与局限可读；条长方向随各值，且均标低为佳。','page_layout':'未驗證原位相鄰、选读折叠或实际页面捲动负担。'},
 ['未重新訓練byte或BPE模型、未生成新文本，未重新前向取得历史模型NLL。','未取得并逐篇读取tokenizer原始train/validation/test JSONL，所以原文共享与曝光支持到公开原始report+原实现，不冒称本輪重現完整語料。','报告记录GPU torch2.14.1+cu126与历史revision a253d1262bf5f361f9ac4e19232ae752f0ecc7a3；本次CPU小算例不是同平台实验复现。']))

pages.append(page('7.5',[
 '因此問題格的候選分數梯度可以是0，Q的輸入特徵卻從後面答案收到梯度。',
 '兩種梯度針對不同變量，不能混看。',
 '這段只求梯度，沒有step；它核對資料依賴，尚未調整出回答能力。'],[
 check('7.5-c1','software','Q位於X索引2、ID89。embedding.weight第89列是Q的字向量','L5','ByteTokenizer ASCII81+8=89；render_chat给X=[1,3,89,2,4,73]，Y=[-100,-100,-100,-100,73,2]。索引2的输入Q和索引4预测A、索引5预测EOS与前文7.4一致。','frozen-data: ByteTokenizer L14–27、render_chat L54–69；check-results: source_programs/7.5/x、y。'),
 check('7.5-c2','software','PyTorch一般不保留中間節點的 `.grad`，所以 `retain_grad()` 明確要求留下它。','L7','logits是output后非叶节点；官方retain_grad允许backward填入其grad，is_leaf说明只叶节点默认保留。原块调用retain_grad后可读取候选分数梯度。','official-tensor: L6595–6602及L6631–6636；check-results: source_programs/7.5。'),
 check('7.5-c3','numeric/software','問題位置的標籤是-100，輸出的logits梯度大小應0；Q字表列的梯度通常大於0','L24','loss_sum仅对有效label求和。官方ignore_index不贡献cross_entropy输入梯度，所以候选分数张量索引2的grad norm=0；本次Q输入表列norm=.00571822514757514。候选分数并不是后面位置输入，后面attention读取Q的中间特征，所以可有另一條導數路。','official-functional: cross_entropy L3487–3505；frozen-model L43–50、L68–89、L91–106；frozen-attention L49–82；check-results: source_programs/7.5。','norm是不同变量的导数量，不做绝对大小比较；一次非0只说明这个配置的依赖。'),
 check('7.5-c4','software/numeric','先預測問題位置logits梯度仍0、被讀取的Z字表列通常有梯度','L30','同seed、A不动，Z的ID98；问题logits norm仍0，Z输入表梯度非0。Q与Z两次backward前后全部state_dict参数逐字equal，没有step或写权重。','check-results: small_variations/7.5/Q、Z。','没有将随机未训练的梯度通路宣称回答能力；模型默认tied=False，输入列导数不是共享output表引入的替代原因。'),
 check('7.5-c5','concept','忽略user loss沒有凍結它。','L28','Y忽略部分位置并没有删掉X，causal attention在后面答案位置允许读Q；loss经这些中间特征回传到输入表。detach或遮断读Q属另一个操作，说明不同机制不能代换。','正文L3、L26、L28；frozen-data、model、attention。','没有实际执行detach或全体user隔离实验，不把这一界限当本轮已测所有模型。')],
 '−100控制直接输出计分；X仍参与输入查表與後续注意力。logits位置导数和embedding字表列导数分别回答输出计分與输入影响。',
 '模型需保留问题线索才能回答，而希望只学助手答案。L3给需要，L5给输入特征通路，L26给两种变量的判断，能防止把零logits梯度读成user完全冻结。',
 'render_chat提供X/Y，随机TinyLM生成logits，retain_grad保存非叶节点grad，masked_loss.backward求导，分别读位置2和ID89行。只做backward且无step，核对依赖不验收能力。',
 '最短Q/A與Q→Z变动隔开位置、ID、计分和输入表四种对象。大小只看零/非零；不能从一个例子推出全部user内容、所有网络结构或绝对影响大小。',
 '前文7.3−100是忽略目标而非输入token；7.4已给Q/A位置与移位对应。當頁在retain_grad前说明中间节点梯度保存，未把程序当作介绍替代品。',
 {'existing_figures':[],'need':'7.4的已教位置对应与当页X索引2/ID89定位足以追踪兩種梯度；本页没有必须新增图才能区分变量的缺口。','page_layout':'本轮未看7.4前文图的render，也未验实际页面；位置技术核查用前文文字和实际X/Y。'},
 ['只查CPU、width8、seed42、Q/A及Z/A。','未训练回答能力；未执行detach分支或禁止全体user attention；这些作为概念边界保留，未算已运行结果。']))

pages.append(page('7.13',[
 '後續改風格、拒答或壓縮時，需先保存固定題和分項規則，才能知道改善的是交付方式，還是回答內容。',
 '這是兩個可以同時一過一錯的檢查。',
 '它在最後檢查只答對5／10，因此後續改動要對照這個實際起點，不能先假定舊能力已經滿分。'],[
 check('7.13-c1','software/numeric','輸出格式True、內容False','L20','原reply={answer:3}严格通过只有answer且value类型int的格式，内容3!=2。执行原塊assert通过；reply2得到两True，改名result得到两False。','check-results: source_programs/7.13与small_variations/7.13。','本例使用有效JSON字典；并非声称一般无效JSON或非dict也能稳定评分的生产评估器。'),
 check('7.13-c2','software','若改用 `isinstance(value, int)`，它們卻也會通過，因此這裡需要更嚴格的檢查。','L20','Python bool是int子类，True/False的type是bool；JSON true/false decode后type不是int。重跑variants两者格式False；2.0内容数值等于2而格式False，分项没有混成同一判准。','原程序严格type契约；check-results: small_variations/7.13。','Python类型行为透明验证，未要求模型生成JSON或网络服务测试。'),
 check('7.13-c3','concept','格式提升可能抵消事實退步；保存指標向量，也就是幾個分項數字，能保留這種行為差異。','L22','格式True+内容False这个直接例子足以说明两个责任分开；仅总分不能辨别同和向量，例如(1,0)与(0,1)。L22明说开放答案须适当判准、不一律字符串完全相同。','L3、L5、L20–L24；透明分项推理。'),
 check('7.13-c4','empirical/numeric','從零練900次的模型，不是較好的`pretrain-sft.pt`','L31','原sft.json results/training.steps=900、checkpoint=model.pt；after/test有10个samples。逐条生成ID去EOS后对tok.encode(expected)重算得5匹配，10条都有EOS；两阶段另在pretrain_then_sft字段。','raw-sft: results/training、checkpoint、after/test；check-results: raw_record_recalculation/sft。','没有重新运行权重推论，也未检验全部后续实验真的沿用了同一个checkpoint。'),
 check('7.13-c5','software/source','本課後續實驗保留的屬性基準是[T.4實跑](../training.md#T.4)的`sft/model.pt`','L31','冻结training.md L201明确输出outputs/course-experiments/course-v1/<组>/、sft/model.pt直接对话版；run_sft以name=model训练随机TinyLM、pretrain-sft另外命名。与raw report的model.pt及Sha b4184ed88631596b40b9d8c9b2db79f80e11404da308d3d760ed0505fae9801f一致。','frozen-training-guide: L195–201；frozen-text: run_sft L556–590；raw-sft artifacts/model.pt。','artifact摘要证明基准命名及所报版本，未取权重文件核验其bytes。'),
 check('7.13-c6','concept','反覆用基準選設定後它就是開發資料，最終仍需未參與選擇的新題。','L24','设置与报告选择会依赖已看的题，后续同题不再是独立最终证据；正文先固定规则再比较且提醒保留新题，没有由JSON解析代判安全或开放建议。','正文L22–L24；前文7.12的家族留出角色。')],
 '固定基准与分项规则使格式、内容、停止等责任可分别重测；JSON是文字格式契约，内容另按题意判。',
 'L3先展示漂亮格式仍答错的目标问题，L5拆开格式与内容，L22解释总分抵消的风险；这把保存指標向量直接连到辨别后续改变改了什么。',
 '解析固定字串检查欄位及整数类型、另检查1+1=2；保存完整题目、权重/tokenizer/生成设定，未来同题重测。原例不生成模型结果。',
 '3→2與answer→result变化分别显示内容改善及格式/字段契约失效；原實测5/10告诉读者实际旧起点，不代表所有指令或安全题能力。',
 '7.12已教材料与留出；当前正文解释JSON、dict、type、bool、键集合/get的作用。完整外部实测只用于基准版本与起点，未拿它补主文评分关系。',
 {'existing_figures':[],'need':'格式/内容两键True/False与逐项解释足够完成当前辨别任务，未发现必须新增图的缺口。','page_layout':'未驗證實際頁面。'},
 ['历史sft结果来自固定报告；本轮只重算已保存生成ID与答案，不重新加载model.pt生成。','没有审查所有后续章节的实验依赖和基准管理；本页主要提供比较要求。','无效JSON、顶层list等输入的错误处理不在本页固定字典算例承诺内，未扩写生产评估器。']))

pages.append(page('9.6',[
 '助手全部拒絕，在應拒絕組是滿分，在正常組卻零功能；單一拒絕率看不見這兩個責任。',
 '正常題「沒拒絕」還不等於已完成',
 '這個拒絕判讀只搜固定「無法提供」，不是真實安全裁判。'],[
 check('9.6-c1','numeric/software','這個分母是兩個應拒絕題，不是所有四題。','L18','should_refuse bool选择1/3，取全True并float mean得1.0。正常子组2/4再反转actually_refuse得0.0；两分母都2。原程序与匹配/第一项失败练习分别得(1,0)、(1,1)、(.5,1)。','check-results: source_programs/9.6、small_variations/9.6。','两组均非空；没有扩展到空组mean的生产报表。'),
 check('9.6-c2','concept','一句好呀沒拒絕，卻未必完成；正常組還須檢查內容。','L22','non_refusal不等同task exact completion；本节示例先演示组分母，原表另外用完整答案匹配。没拒绝但乱答与拒绝边界失误分属不同责任。','正文L5、L22及表L33–39；raw-safety before/test与run分项。'),
 check('9.6-c3','empirical/numeric','其中三題未授權，按規則應拒絕；其餘14題應正常完成','L29','逐读17题的原始samples，permission False有3题；normal为授权True3、known3、unknown3、true_premise1、false_premise1、injection3，共14。正常责任包含适当澄清。','raw-safety: results/before/test/samples及runs/*/safety/test/samples；check-results: raw_record_recalculation/safety/test_audits。'),
 check('9.6-c4','empirical/numeric','盒子行為混入加法 | 3／3 | 13／14 | 0／14','L37','从每题generated_ids去EOS、按ByteTokenizer完整ID与expected比对，并独立在decode后的串搜索無法提供，重算三行: before=(0/3,0/14,0/14)，safety-only=(3/3,12/14,0/14)，mixed=(3/3,13/14,0/14)，各版EOS17/17。','raw-safety SHA d98ee412c2c19029d731406e629c6dbd86a99a16f215ef26c3c89cdf4f85d0ed；check-results: raw_record_recalculation/safety。','以固定拒绝模板与ID exact衡量这个玩具任务，非真实安全分类器或普遍助手能力。'),
 check('9.6-c5','empirical/software','都是沒有用來教模型的留出題','L29','本轮在记忆体调用frozen/live一致的_safety_records与split_records(seed42)，重建train102、validation17、test17；序列化三侧JSONL的SHA分别fcbb3ae3...、1b708936...、cd927de5...与报告完整值吻合，家族为6/1/1且交集空。test是safety-rule-1、validation是safety-rule-0。','frozen-behavior: _safety_records L385–433、run_safety L503–559；frozen-common split_records L50–71；raw-safety data；check-results: split_reconstruction。','这个核查验证数据切分与manifest一致，不重演历史训练过程；未保存或提交原始JSONL。'),
 check('9.6-c6','software/concept','EOS是回答結束的特殊標記，用來表示停止輸出，與答案內容另行檢查','L39','ByteTokenizer EOS ID2；evaluate_lm在首EOS前取raw ID作exact，eos单列。拒绝判读是固定字符串，原文已区分。普通加法另一份7题，不能混进17分母或把答错当拒绝。','frozen-data specials；frozen-common evaluate_lm L243–281；raw-safety samples与arithmetical字段；正文L41。')],
 'should_refuse给题目责任，actually_refuse给观察；分别选两个子组统计，正常未拒绝还须另验内容与停止。',
 'L3把全拒绝的反例给出，显示高拒绝率可能伴随零正常功能；L5、L22再限制非拒绝率不能代表完成。这直接连到避免一个总比率掩盖过度拒绝或无功能的评估需要。',
 '布林选择题类，反转拒绝状态，按各组自己的题数mean；实测另把拒绝模板、答案ID exact、EOS三项分开，并保留普通加法的独立卷。',
 '4题手工示例足以显示两责任不可互代；17题实测显示正常题没拒绝但仍有内容失败。mixed唯一正常失败是盒子1 count=1生成2，不是拒绝；不能据混合版多对一题宣称通用减过拒绝方法。',
 '题类、dtype bool、筛选两次~及每项分母都在正文先给。选读进一步说正常责任包括澄清，EOS与答案ID分开。9.8仅按原文链接查训练/留出设定，未扩大其他章节审阅。',
 {'existing_figures':[],'need':'四个明确排列的bool结果、索引位置说明及两组表格充分呈现分母与结果；未发现必须新增图的缺口。','page_layout':'未驗證實際頁面。'},
 ['不做真实安全或开放式助手评估，固定無法提供模板无法覆盖所有拒绝措辞。','历史报告revision a864a60bbf72583afc9bbaf45e052bd4fe076c62的behavior.py摘要SHA94ab92aa...与本轮frozen SHA2ccdd39e...不同；本輪未把二者当逐byte同一实现。数据重建SHA与已保存逐题ID核查支持本页数字，未宣称独立复现历史训练。','未重新加载或训练behavior模型；本輪未读取原始data/test.jsonl实体，已在记忆体逐条重建并核对它的bytes SHA。']))

for p in pages:
 assert not p['necessary_findings'] and not p['optional_findings']

sources=[]
for n in ['adam.py','adamw.py','functional.py','tensor_docs.py']:
 meta=next(x for x in json.loads((B/'evidence/technical-learning/official/sources.json').read_text()) if x['name']==n)
 ident={'adam.py':'official-adam','adamw.py':'official-adamw','functional.py':'official-functional','tensor_docs.py':'official-tensor'}[n]
 sources.append({'id':ident,'kind':'official_source','path':'evidence/technical-learning/official/'+n,**meta,'inspection_note':{'adam.py':'實讀L150–184、L274–331、L413–430與偏差修正更新行。','adamw.py':'實讀L19–122：继承Adam、decoupled=True、算法与eps约定。','functional.py':'實讀cross_entropy L3487–3509：输入/目标与ignore_index。','tensor_docs.py':'實讀retain_grad及is_leaf L6595–6636。'}[n],'supports':'只用于对应公式及API机制；未将官方source所在性当作教學说明充分的替代。'})
for ident,path in [('frozen-training','tiny_perceptron/training.py'),('frozen-data','tiny_perceptron/data.py'),('frozen-model','tiny_perceptron/model.py'),('frozen-attention','tiny_perceptron/attention.py'),('frozen-text','scripts/course_experiments/text.py'),('frozen-behavior','scripts/course_experiments/behavior.py'),('frozen-common','scripts/course_experiments/common.py')]:
 sources.append({'id':ident,'kind':'repository_code','path':'freeze/implementation/'+path,'sha256':sha(B/'freeze/implementation'/path),'expected_sha256':M['implementation_sha256'][path],'live_equals_frozen':sha(B/'freeze/implementation'/path)==sha(Path('/workspace/tiny-perceptron-vlm')/path),'version':'freeze-01','checked_original':True,'inspection_note':'实际读取报告引用的函数正文；check-results/integrity保存执行前frozen/live/expected核对。'})
for name in ['tokenizer','sft','safety']:
 path='docs/course-experiments/results/'+name+'.json';d=json.loads((B/'freeze/technical-data'/path).read_text())
 sources.append({'id':'raw-'+name,'kind':'raw_experiment_record','path':'freeze/technical-data/'+path,'sha256':sha(B/'freeze/technical-data'/path),'expected_sha256':json.loads((B/'checks/technical-inputs-manifest.json').read_text())['files_sha256'][path],'version':d['revision'],'historical_environment':{'device':d['device'],'torch':d['torch_version'],'python':d['python_version'],'seed':d['seed']},'fixed_hf_revision':d.get('hf',{}).get('revision'),'fixed_hf_result_revision':d.get('hf',{}).get('result_revision'),'checked_original':True,'inspection_note':'实际读取对应结果字段与逐题记录；只有tokenizer代價彙總沒有逐目標log機率，本輪重算其尺度，不冒称重新模型评测。'})

prior_ids=['1.12','1.13','5.3','6.1','6.4','6.6','7.3','7.4','7.12','9.8']
priors=[{'page_id':i,'path':'freeze/sources/'+i+'.md','sha256':sha(B/'freeze/sources'/f'{i}.md'),'role':'必要前文或正文明确链接的设定；只作为技术依赖核对，不扩大为这些页的完整新审阅。'} for i in prior_ids]
priors.append({'path':'freeze/original/course/training.md','sha256':sha(B/'freeze/original/course/training.md'),'locations':'只取T.4相关基准路径说明，尤其L195–201；rg另回传的其他章节相同路径不作该章节审阅结论。','role':'核对7.13链接的sft/model.pt命名。'})
artifacts=[]
for path,description in [('evidence/technical-learning/check_learning.py','离线执行脚本：七段冻结代码、小变化、原始ID重算及安全分侧记忆体重建。'),('evidence/technical-learning/check-results.json','本次完整CPU执行输出与核对结果；无训练、无模型生成重跑。'),('evidence/technical-learning/figure-check.json','40个SVG坐标按learning_rate数值核对，含映射及容差。')]:
 artifacts.append({'path':path,'sha256':sha(B/path),'description':description})
for stem in ['rewrite-05-06-learning-rate','rewrite-06-05-shared-denominator','tokenizer_common_scale']:
 for size in [640,360]:
  path=f'renders/{stem}-{size}.png';artifacts.append({'path':path,'sha256':sha(B/path),'description':'本轮真正view_image的既有静态render；未冒称原位网页。'})

report={'schema_version':1,'review_stage':'technical-source-initial','group':'learning','reviewer_task':role,'reviewer_context':'來源獨立初判；未讀任何讀者、作者、同行或舊問題判斷。','at':now,'scope':{'pages':M['groups']['learning']['pages'],'kind':M['kind'],'repository':M['repository'],'freeze':'freeze-01','baseline_commit':M['baseline_commit'],'manifest_sha256':sha(B/'manifest.json'),'reading':'先核对实际reader-source全组seal gate，随后完整指定正文/选读、必要前文、冻结实施与原始数据；没有假称技术专家全文阅读为逐段盲读。'},'criteria':{'path':'freeze/criteria/SKILL.md','sha256':M['criteria_sha256']['SKILL.md'],'read':['SKILL.md','references/review-protocol.md','references/calibration.md','references/project-context.md'],'read_sha256':{x:sha(B/'freeze/criteria'/x) for x in ['SKILL.md','references/review-protocol.md','references/calibration.md','references/project-context.md']},'generic_guides':['CLAUDE.md','docs/editorial-guide.md','docs/technical-review-guide.md'],'precedence':'按本轮七页group授权与冻结规约；不因旧逐页task规约改派工或扩大范围。'},'gate':{'path':'checks/reader-source-all-sealed.json','sha256':sha(B/'checks/reader-source-all-sealed.json'),'inspected_before_source_judgments':True,'role':'实际report身份seal gate，内容判断报告未打开。'},'technical_inputs_manifest':{'path':'checks/technical-inputs-manifest.json','sha256':sha(B/'checks/technical-inputs-manifest.json'),'kind':'只含原始实验参考快照指纹，不含判断。'},'integrity':{'all_assigned_sources_match_manifest':True,'all_assigned_svgs_match_manifest':True,'execution_imports_verified_before_run':True,'pyproject_expected_difference':'已实际diff：live只增加docs/course-repair-20261008/reviews的Ruff immutable archive排除及注释；freeze SHA725fbfc7...，live SHA4a909002...；uv.lock SHA完全一致，依赖与运算未变。'},'sources':sources,'necessary_prior_material':priors,'pages':pages,'execution':{'command':'/workspace/tiny-perceptron-vlm/.venv/bin/python '+str(B/'evidence/technical-learning/check_learning.py'),'exit_code':0,'result':'七段冻结Python块、数值小变化、逐题生成ID与分项重算、分侧SHA断言全部通过。','environment':E['environment'],'artifact_path':'evidence/technical-learning/check-results.json','meaning':'技术算例执行与原始结果重新计分，不是再训练或历史模型独立再推论。'},'artifacts':artifacts,'findings':{'necessary':[],'optional':[],'statement':'所审七页未发现需要修正的技术矛盾或当下必要关系缺项；不以本轮未执行的未知补成缺陷，也不以它们算通過。'},'unknowns_and_limits':['沒有读取B/reports、traces、notes、checks/root-*、authors、diagnosis.json、旧audit/reader/technical/continuity结论或本轮作者修补记录。','三张图都实看640/360静态render；实际网页平台、桌面/手机原位图文、DOM、折叠、遮挡与阅读路径均未验证。','没有重训、安装依赖、调用成熟模型、覆写历史实验或提交原始数据；只在自己的evidence目录保存小检查和官方源快照。','技术判断不是实际高中生首读或真人学生测试；其独立性只支持本轮未看同行判断的来源初判。','报告支持数字来源、公式与已保存结果读法；未重新获取历史权重和tokenizer全量JSONL。安全数据在记忆体重建SHA，tokenizer曝光/完整语料只按原始report和冻结源代码查核。'],'cross_review':'尚未打开；报告封存后停止，由root在全来源初判封存后安排。'}
for f,h in report['criteria']['read_sha256'].items():assert h==M['criteria_sha256'][f]
for p in pages:
 for f,h in p['figures_sha256'].items():assert sha(B/'freeze/original'/f)==h
out=B/'reports/technical-learning-initial.json';out.parent.mkdir(exist_ok=True);assert not out.exists(),'不要覆盖已封存初判'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print({'report':str(out),'bytes':out.stat().st_size,'sha256':sha(out),'pages':len(report['pages']),'claims':sum(len(p['technical_checks']) for p in pages),'necessary_findings':0,'optional_findings':0})
