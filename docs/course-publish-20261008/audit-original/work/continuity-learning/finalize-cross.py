import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path('/workspace/work/tutorial-audit-20261008')
R = B / 'reports'

def read(name):
    return json.loads((R / name).read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

own = read('continuity-learning-initial.json')
reader = read('reader-learning-main.json')
final = read('reader-learning.json')
tech = read('technical-learning-initial.json')
pages = {p['page_id']: p for p in own['pages']}
prereqs = {p['page_id']: p for p in own['actual_prerequisites']}
manifest = json.loads((B / 'manifest.json').read_text())
inventory = {p['page_id']: p for p in manifest['inventory']['pages']}

inputs = [
    'reports/reader-learning-main.json', 'reports/reader-learning.json',
    'reports/technical-learning-initial.json', 'reports/continuity-learning-initial.json',
    'reports/continuity-learning-main-notes.json', 'notes/learning-main-seal.json',
    'reports/technical-learning-initial.seal.json',
    'reports/continuity-learning-initial-seal.json',
    'work/continuity-learning/main-notes-seal.json', 'cross-review-instructions.md',
]
input_hashes = {p: sha(B / p) for p in inputs}
assert input_hashes['reports/continuity-learning-initial.json'] == '321fd54c17816c91d99e43c5db75cde7cc3659eb99ebb2b22dcc42d9961af8dd'
assert input_hashes['reports/continuity-learning-main-notes.json'] == '96af382d232005866776dde0a095ed39f37ee969c867cb72bf21fd04770a7eb9'
assert json.loads((B / 'work/continuity-learning/main-notes-seal.json').read_text())['sha256'] == input_hashes['reports/continuity-learning-main-notes.json']
assert json.loads((R / 'continuity-learning-initial-seal.json').read_text())['sha256'] == input_hashes['reports/continuity-learning-initial.json']
assert json.loads((B / 'notes/learning-main-seal.json').read_text())['sha256'] == input_hashes['reports/reader-learning-main.json']
assert json.loads((R / 'technical-learning-initial.seal.json').read_text())['report_sha256'] == input_hashes['reports/technical-learning-initial.json']
assert reader['issues'] == final['issues']
assert all({k: v for k, v in p.items() if k != 'optional_read'} == reader['pages'][i] for i, p in enumerate(final['pages']))

origin_by_id = {}
for report_name, data in [('reader-learning-main.json', reader), ('technical-learning-initial.json', tech), ('continuity-learning-initial.json', own)]:
    for issue in data['issues']:
        iid = issue.get('issue_id', issue.get('id'))
        origin_by_id[iid] = {'report': 'reports/' + report_name, 'sha256': input_hashes['reports/' + report_name], 'original_issue': issue}

def dependency(pid, quote, purpose, essential=True):
    return {'page_id': pid, 'source_sha256': inventory[pid]['source_sha256'], 'scope': 'main; actually read in independent continuity stage and/or cross reread', 'quote': quote, 'supports': purpose, 'essential_to_current_check': essential}

def item(cid, pid, origins, quotes, promise, completed, mechanism, need, gap, payoff, repair, burden, severity='optional'):
    return {
        'cross_id': cid, 'page_id': pid, 'source_sha256': inventory[pid]['source_sha256'],
        'origin_ids': origins, 'origins_preserved': [origin_by_id[o] for o in origins],
        'source_quotes': quotes, 'original_promise': promise, 'already_completed': completed,
        'checks': {'mechanism_property': mechanism, 'necessity_use': need},
        'remaining_minimal_gap_or_rebuttal': gap, 'understanding_or_operation_payoff': payoff,
        'minimum_repair': repair, 'new_burden_and_handling': burden,
        'severity': severity, 'necessary': severity == 'burden', 'disposition': '採用',
        'rewrite_scale': 'paragraph',
        'provenance': '互見後交叉覆核；保留各來源初判，非新的獨立發現',
    }

issues = []
issues.append(item(
    'cross-learning-5.5-motive', '5.5', ['L-5.5-motive'],
    ['這輪任務梯度恰好0，我們還想把它稍微縮小。',
     '若把權重平方懲罰加進任務代價，對應梯度也會進入Adam的m、v再被縮放；AdamW讓回拉與這份歷史分開。',
     '縮小有用的大權重可能暫時提高任務代價，需用驗證資料選強度。'],
    '章首由固定題改善進入訓練／評估流程，以便選模型、比較配方；正文首次把額外回拉引入任務梯度更新之外，同時教獨立控制。不是承諾AdamW必勝，也不是只憑頁名把新訓練偏好的採用理由排除。',
    '乘法衰減、與m/v分離、零梯度與None、驗證選強度均已完成。2→1.96及λ=0/grad=None兩個變化都足以檢查操作。',
    {'judgment': '已足夠', 'reason': '新optimizer零歷史且實際零梯度使示例隔離乘法回拉；平方懲罰進m/v與直接衰減分開的理由明說。沒有把有舊動量時的零梯度泛化為只有衰減。'},
    {'judgment': '需補說明', 'reason': '獨立控制已選定回拉的需要成立；更前一層為何在希望降低答案代價的訓練中加入小權重偏好，仍只有「想縮小」。1.12教降低任務代價，1.2教新題檢查，兩者都不能推出縮小權重能回應哪個訓練問題。驗證選λ是選擇程序，也不代答加入這份偏好的目的。'},
    '只缺把額外小權重偏好接回訓練中的採用條件；不否定獨立控制理由，不要求正則化完整理論或新增實驗。',
    '讀者能分清「按任務梯度改答案」與「另加一份可關閉的偏好」，知道何時值得列為候選設定，而非從w變小反推它一定對新題有用。',
    '開頭或平方懲罰段補一小段，明說作者為何想在本训练流程加入小權重偏好、它打算約束什麼，以及仍須用驗證選是否採用／強度。可舉希望避免少數權重無限制放大的設計偏好，但這個需要須由教材明教；不可由reviewer視為原已教，也不可寫成必然泛化改善。保留暫時提高任務代價的限制。',
    '新增一個「任務目標之外的偏好」关系，限一小段、普通語言；不加入完整L2推導、先備課或長訓練。', 'burden'))
issues[-1]['actual_dependencies'] = [
    dependency('chapter-05', '分清它們，才能選模型、比較配方和解讀結果。', '上層訓練／配方承諾'),
    dependency('1.12', '希望下降的是代價，參數本身可能增加。', '原任務梯度的目的'),
    dependency('5.4', '兩格各保存自己的歷史', 'm/v逐參數歷史，支撐解耦机制'),
    dependency('1.13', '若每輪只想用當前資料更新，應先清舊梯度，再算新代價、求導與更新。', 'None與零不是同一狀態的前文位置'),
    dependency('1.2', '驗證資料**用來選設定', '驗證只能選設定，不能自行產生衰減动机', False),
]
issues[-1]['strongest_counterargument_and_adjudication'] = 'technical與自己的initial均以「為獨立控制回拉」判need足夠；這個子理由保留。交叉重新核對章首訓練與配方承諾、額外偏好及可能提高任務代價後，認為它未包辦採用額外回拉的上層理由。小節末沒有指定後續承接動機的位置，折疊只有安裝通知。故採用reader的局部理解負擔，自己的5.5初判仍不修改。'
issues[-1]['initial_continuity_judgment_preserved'] = pages['5.5']['checks']

issues.append(item(
    'cross-learning-5.6-warmup', '5.6', ['L-5.6-warmup'],
    ['一趟40步訓練，可以先小步起步，再放大，最後細調。', '排程按預定進度工作，不會自動知道最佳步幅'],
    '教按時間給不同步幅、讀懂暖身／餘弦時間表與續訓契約；正文稱常見選擇，未承諾起步一定不穩或排程必勝。',
    'i0第一次、五步暖身、第5/6次同峰、尾端非零、修改總計畫與保存計畫完整；折疊補暖身夾值。',
    {'judgment': '已足夠', 'reason': '公式(i+1)/warmup与数列／圖讓读者可预测改warmup10的轨迹，不需执行长训练。'},
    {'judgment': '已足夠；可選說明可更清楚', 'reason': '預先指定較小／较大位移及後期細調是可理解的設定用途，1.12已教步子過大可能跳過低代價位置。正文沒有宣稱某階段必然需要最佳暖身或自動識別風險。可补小步起步的有条件用途，不能把这个可选项升成效果理由缺陷。'},
    '無當下必要操作／目的缺口；可補一句起步時想限制前幾次位移的條件，連到已教大步风险。',
    '讀者理解暖身是在指定起步階段的步幅偏好，能把它與即時代價反馈分開，避免仅凭「常見」選用。',
    '用一兩句連到1.12：若不想起步幾次就走大步，可先限制這幾次的位移，再按計畫放大；這不會偵測是否真的過大，仍須驗證選設定。無需新增「早期梯度一定不穩」之類未教結論。',
    '復用1.12已教大步反例，增加最多兩句；不增加新的梯度統計理論或訓練效果比較。'))
issues[-1]['actual_dependencies'] = [dependency('1.12', '正確方向配太大步子仍可能變差', '步幅選擇而非方向選擇'), dependency('5.6', '這工具在暖身區使用 `(i+1)/warmup` 乘高峯。', '指定時間表本身')]

issues.append(item(
    'cross-learning-6.5-nll-name', '6.5', ['L-6.5-nll-name', 'technical-learning-name-6.5'],
    ['正確token的負對數機率加總叫 NLL。', '這裡用自然對數 ln，Python 的 `math.log` 也是這個底數'],
    '把不同切法的代價換到共同原文byte尺度，教NLL→bit→BPB與比較契約。',
    '中文角色、自然log、加總、ln2換bit、共同原文及EOS計分均明說；名称缺失不妨碍计算。',
    {'judgment': '已足夠', 'reason': 'NLL总和除ln2再除原文bytes，7byte例与实测总分母是同关系；不需独立事件假设。'},
    {'judgment': '已足夠', 'reason': '每token分母因切法而变，共同原文尺度回应比较需要；完整英文名不是该推论必要前提。'},
    '同一名稱關係缺口，兩origin合併一處括注。',
    '中文計算角色能與常見英文NLL名稱對上，便於辨認引用／工具。',
    '首個NLL後加「Negative Log-Likelihood，負對數似然」。保留前面的具体中文解释。',
    '僅一个括注，不新增似然理論先備，不重复两位reviewer的两条修补。'))
issues[-1]['actual_dependencies'] = [dependency('1.8', '正確答案得到的機率越少，代價越大；代價不是猜錯題數。', '負log代價的角色'), dependency('6.1', 'token是模型一次處理的片段', '不同切法改变token单位')]

issues.append(item(
    'cross-learning-7.13-json-name', '7.13', ['L-7.13-json-name'],
    ['JSON把欄位名稱與值保存為文字。', '下面要求只有answer且值真的是整數，內容則另問是否等於2'],
    '建立後續前後比較的固定題與分項指標，区分交付格式与答案内容。',
    '示例、json.loads、唯一answer鍵、type為int與expected2都具體；正文还排除bool並拒用解析代判內容。',
    {'judgment': '已足夠', 'reason': '{"answer":3}结构合格而内容错误，不靠英文展开也可预测answer→result与3→2的变化。'},
    {'judgment': '已足夠', 'reason': '後續改寫法需看改善与退步分别发生在哪项，answer3在同一题体现差异。'},
    '仅格式名稱展开；不是schema或用途介绍缺失。',
    '把可读的字段／值文字格式对上完整标准名称，方便辨认外部资料。',
    '首次JSON加「JavaScript Object Notation，JavaScript物件表示法」括注。',
    '不要引入JavaScript课程，也不要把名称误写成需要用JavaScript实现；最多一个括注。'))
issues[-1]['actual_dependencies'] = [dependency('5.8', '代價與實際回答應一起看', '分開平均代價與實際交付结果'), dependency('7.13', '這是兩個可以同時一過一錯的檢查。', '局部已明说分项目的')]

issues.append(item(
    'cross-learning-5.4-adam-name', '5.4', ['technical-learning-name-5.4'],
    ['Adam再參考每格「過去通常多大」的梯度，用大小歷史調整方向的尺度。', '後續m、v記憶長度不同，不必再相同'],
    '解释共同学习率与各参数实际更新量的区别、历史与偏差修正；没有承诺Adam一定更优。',
    'm/v公式、首步修正与方向、后续不等位移及存状态都具体，名稱是剩余的小补。',
    {'judgment': '已足夠', 'reason': '[1,100]在零历史首步约同.001，换负梯度平方不变方向变；后续不同历史不得泛化首步。'},
    {'judgment': '已足夠', 'reason': '实际位移不能由共用lr单独读出，历史调尺度回应这个训练参数解读需要；无须新编收敛或优势承诺。'},
    '名称未展开不挡机制；补完整名可减少检索负担。',
    '将本文的两种历史与外部Adam完整名对齐，避免只记工具名。',
    '5.4正式介绍处加「Adam（Adaptive Moment Estimation，自適應矩估計）」。',
    '一个括注；5.1的优化器工具入口仍可合理指向后续，不加全算法第二次说明。'))
issues[-1]['actual_dependencies'] = [dependency('1.12', '更新規則是 `新參數=舊參數−學習率×梯度`。', '共同学习率下原始位移'), dependency('5.3', '本節的 v 則在更新後仍保留，讓過去方向繼續影響下一輪。', '历史更新的已教背景')]

issues.append(item(
    'cross-learning-7.5-duplicate', '7.5', ['CL-L-01'],
    ['`logits` 是中間結果，PyTorch一般不保留中間節點的 `.grad`，所以 `retain_grad()` 明確要求留下它，相關葉參數區別見 [1.11](01.md#1.11)。'],
    '用兩種变量梯度分清直接不计问题答案代价与后续答案仍读取问题。',
    'retain_grad用途在代码前完整给出，代码后同句原样重现；核心梯度路径在随后段清楚。',
    {'judgment': '已足夠', 'reason': '中间logits保grad的本地规则充分；问题输出分数梯度0与Q输入表梯度非0针对不同变量。'},
    {'judgment': '已足夠', 'reason': '读取问题以回答但不直接训练问题文本，变量区分回应监督选择需求。重复不产生必要知识缺口。'},
    '编辑重复，不是需补新概念；原冻结来源同一句确出现两次。',
    '代码后直接进入两种norm对象及0/非0的解释，减少重复回读与核心关系间距。',
    '删代码后的第二个retain_grad介绍句，保留前面的首次说明与后面的变量区别。',
    '净减一段／一句，保留首次必要说明；与下一项在同一段处理但不混为一个理解关系。'))
issues[-1]['actual_dependencies'] = [dependency('1.10', '每段的敏感度相乘，就是**鏈式法則**', '后续答案的梯度依赖链'), dependency('3.6', '位置 i 只准讀位置 j≤i，即自己與更早的位置。', '后答案能读早问题的输入')]

issues.append(item(
    'cross-learning-7.5-link-scope', '7.5', ['CL-L-02'],
    ['相關葉參數區別見 [1.11](01.md#1.11)', '它與參數形狀相同，每格敏感度都有自己的參數可對應。'],
    '辅助链接声称1.11已教叶参数区别，本地则教中间logits的grad保存。',
    '1.11确实教各参数偏导、梯度同形与沿各路回算；7.5本地已明说retain_grad当下用途。',
    {'judgment': '本地已足夠；链接范围不符', 'reason': '实际重读1.11正文并无叶／中间节点保存区别，不能由只出现参数.grad推成已教该区别。'},
    {'judgment': '已足夠', 'reason': '当前可执行和解释来自7.5本地，辅助链接失配只影响选择回看者，不升成主文必须补一门叶节点理论。'},
    '链接承诺多于所链前文；保留角色明确的本地规则便足够。',
    '读者回看能找到实际对应的内容，不会寻找不存在的叶节点区分或误认漏读。',
    '删「相關葉參數區別見1.11」；若保留链接，改称「每格参数梯度可回看1.11」。与前项去重后保留的首次说明一起编辑。',
    '无新先备；不要为了救一个可选链接重写1.11或在7.5补完整autograd分类。'))
issues[-1]['actual_dependencies'] = [dependency('1.11', '它與參數形狀相同，每格敏感度都有自己的參數可對應。', '实际链接内容核对', False), dependency('7.5', '`retain_grad()` 明確要求留下它', '当下保grad规则在本页完成')]

pass_reasons = {
    '5.3': '三步算例只证明方向历史延迟转向；正文独立给「穿过小抖动」条件及过头代价，不需由位置更负推质量。与.grad本轮累积的两个状态明确分开。',
    '5.17': '固定权重仍有dy/dx，训练上游输入产生器的需要独立于冻结产物；no_grad会切此次记录，eval只改前向模式。公式已足以连到分工，不要求更多运行样本。',
    '7.4': '首答目标73从原位置5配到输入assistant索引4，EOS配到A索引5；问题长度和答案内容两种变换分别验位置与ID。有效个数相同不能掩盖错位。原位图先前已亲看桌面／手机。',
    '7.7': '保持真实题预测是独立需要；禁读PAD与保真实位置编号各影响前向，labels=-100只管计分。因果遮罩一项不能代答另两项。',
    '7.19': '同一书店请求需要历史店名和角色，12前文+最多4新增在16容量内；G2截停、缺小小店、正常EOS反例分别不完成同一交付，不能只数token或EOS判成功。',
    '7.22': '远目标沿真实中间token条件化，标准停只进入后一路预测了，不倒进主雨核心；顺序目的和数据流同时成立。图两模块示意／实际V3一个额外模块、推论无标准未来都限定清楚，不借此放行加速效果。',
    '8.8': '保留原模型路线并减少更新状态的需求在正文独立给出，A/B窄修正通道与参数数连接；表达受限、原权重仍存/算、A首步grad0及层sum不是回答训练均明确。不是从56小于192自编通用效果需要。',
    '8.14': '当前只要两行中文的请求是独立需要；全抄事实真却超范围、少一行、同一行三个失败关系已有，不能只认字或事实正确放行指令完成。',
    '8.15': '同告示不同要求的训练目标实际不同，错配将教违令；同家族新措辞与新告示检验不同迁移，已有对照足以教方法，无须把完整配方移入正文。',
    '8.16': '联合可交付要求是同题内容、范围、格式、自然结束四项合取；五例各轴失败与全过例使分项均值不能冒充联合成功，失败保原分母。IFEval不包办本题更窄schema/EOS合同。',
    '9.4': '安全需要在明示可信管理场景内才成立；只有两有效场景的bool简化分支可通，扩到用户自称等场景要可信字段与规则。实际实验只有permission输入不证明owner/public事实，正文明确限制。',
    '9.10': '新增低信心答对同时accuracy升／ECE升是欠信与加权差的定义应用；温度保持argmax故不增正确率。音频同14题11对而信心升、ECE/Brier变坏是实际反例，整体均值差不代binECE。',
}
passes = []
for p in own['strong_pass_samples']:
    pid = p['page_id']
    passes.append({'page_id': pid, 'source_sha256': inventory[pid]['source_sha256'],
                   'original_independent_pass_preserved': p,
                   'cross_verdict': '保留通過', 'source_supported_reason': pass_reasons[pid],
                   'actual_dependencies': pages[pid]['dependency_support'],
                   'cross_limits': pages[pid]['unverified'],
                   'provenance': '互見後核對已有原文／自己的實讀與已封存技術檢查；非新獨立發現'})

assert set(origin_by_id) == {oid for i in issues for oid in i['origin_ids']}
quote_checks = []
for i in issues:
    raw = Path(inventory[i['page_id']]['snapshot']).read_text()
    for quote in i['source_quotes']:
        # Some cross quotations explicitly come from the referenced supporting prior page.
        found = quote in raw or any(quote in Path(inventory[d['page_id']]['snapshot']).read_text() for d in i.get('actual_dependencies', []))
        quote_checks.append({'cross_id': i['cross_id'], 'quote': quote, 'matched': found})
    for d in i['actual_dependencies']:
        quote_checks.append({'cross_id': i['cross_id'], 'dependency_page': d['page_id'], 'quote': d['quote'], 'matched': d['quote'] in Path(inventory[d['page_id']]['snapshot']).read_text()})

out = {
    'schema_version': 1, 'reviewer': '/root/continuity_learning', 'group': 'learning',
    'stage': 'sealed-initials-authorized-cross-review', 'created_at': datetime.now(timezone.utc).isoformat(),
    'criteria_sha256': own['criteria_sha256'], 'input_reports_and_seals_sha256': input_hashes,
    'independence': '此輪已看同組各來源初判；所有裁定是交叉覆核，不算第二次獨立發現。來源initial/main-notes均保留，未讀歷史review，未改教材，未訓練，未派agent。',
    'actual_report_reading': {
        'reader_main': '全部issues、79頁小節末核心判斷；5.4/5.5/5.6/6.5/7.5/7.13相關逐段checkpoint另讀。未重演244個盲增量，不聲稱全244checkpoint逐字重新審閱。',
        'reader_final': '核對main內容與issues逐項和reader-main相同；讀全部補充觀察的核心理解／影響、圖與未驗範圍，另讀所有有issue頁的optional_read。',
        'technical': '全部issues與79頁機制／用途核心技術判斷及證據範圍；讀execution/visual/coverage metadata。引用其已封存數值核查為該輪證據，不聲稱本人重跑或重核rawJSON。',
        'continuity': '自己的79頁全文初判、strong passes與原始seals；此輪保留原判，只增加cross裁定。',
        'independent_addenda': '搜尋報告／notes檔名未見本組獨立補件；只見上述seal。',
    },
    'actual_source_reading': {
        'independent_stage': '全部79頁main＋全部79頁extras範圍（71折疊）、20實際前文；以已封存initial/main-notes與helper紀錄為準。',
        'cross_reread_main': ['chapter-05', '5.4', '5.5', '5.6', '7.5', '7.13', '1.2', '1.12', '5.10', '1.11'],
        'cross_targeted_quote_checks': ['5.3', '6.1', '6.5'],
        'cross_new_external_or_code_reading': [],
        'actual_prior_scope_preserved': own['actual_prerequisites'],
        'supplemental_scope_preserved': own['supplemental_scope'],
    },
    'issues': issues, 'origin_coverage': {'unique_origin_ids': sorted(origin_by_id), 'all_8_processed': True, 'deduplicated_relations': 7, 'merged_relation': '兩個NLL名稱origin合為一項，兩項7.5雖同段編輯仍保留各自理解關係'},
    'passes_rechecked': passes,
    'notable_counterexamples': [
        {'page_id': '5.5', 'reason': '2→1.96、λ0與None變化只答得出控制規則，不能單憑能预测數字放行額外偏好的採用理由；独立控制子理由仍有效。'},
        {'page_id': '5.4', 'reason': '首步兩格約同位移不足证明一般同位移或优越；正文後續历史不等與無效果承諾使当前教學可通，不要求自編收益目標。'},
        {'page_id': '8.14–8.16', 'reason': '所有店名真且可解析都不包辦指定范围、格式及自然结束；原文已同题体现多个相互独立的判准。'},
        {'page_id': '9.4', 'reason': '可信场景给出的规则可通，但6/6固定bool输出不扩大成现实授权判断；不能一面接受正文约定、一面用它替正式实验补缺字段。'},
    ],
    'unknown': own['unknown'],
    'unknown_cross_handling': '保留2类未验范围。technical的原JSON/固定logits核查提供跨轮支持，未转写成本reviewer本人已执行；外部引用、完整实验复现、网站77页与runtime/annotations仍不由本轮放行。未知不自动算source缺陷。',
    'visual_scope': {'cross_new_image_views': [], 'basis': '沿用initial实际亲看19图640/360与7.4/7.22原位captures；当前问题没有新增图/版面争议，不以DOM或source代看图。', 'initial_scope_preserved': own['visual_scope'], 'runtime_outputs_and_reading_annotations': '新preview省略；本reviewer仍未验。', 'route_deviation_preserved': own['visual_scope']['route_deviation']},
    'execution_scope': '本轮未执行教材／训练／新增数字重算。仅做hash、来源引文匹配、origin覆盖和JSON结构校验。',
    'counts': {'origin_necessary': 1, 'origin_optional': 7, 'deduplicated_necessary': 1, 'deduplicated_optional': 6, 'unknown_scopes': 2, '採用': 7, '延後': 0, '保留原文': 0, '待查': 0},
    'repair_assessment': {'paragraph': 7, 'page': 0, 'chapter': 0, 'order': 0, 'page_chapter_order_rewrite_needed': False, 'reason': '必要补写只在5.5額外小權重偏好的採用關係；其餘括注、一兩句或刪改同段。主要表示、监督、数据／评估和后训练路线已有必要已教关系，不支持章重写或顺序重排。'},
    'source_quote_validation': quote_checks,
}
assert all(q['matched'] for q in quote_checks), [q for q in quote_checks if not q['matched']]
destination = R / 'cross-learning.json'
assert not destination.exists(), 'Do not overwrite a sealed cross report'
destination.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
seal = {'path': str(destination), 'sha256': sha(destination), 'sealed_at': datetime.now(timezone.utc).isoformat(), 'reviewer': '/root/continuity_learning', 'stage': 'cross-review', 'input_reports_and_seals_sha256': input_hashes, 'counts': out['counts'], 'original_seals_rechecked': True, 'meaning': 'SHA封存互見後交叉裁定；不是獨立來源初判、不是教材已修改或已驗收'}
(R / 'cross-learning-seal.json').write_text(json.dumps(seal, ensure_ascii=False, indent=2) + '\n')
assert all(sha(B / p) == h for p, h in input_hashes.items())
print(json.dumps({'path': str(destination), 'sha256': seal['sha256'], 'counts': out['counts'], 'repair_assessment': out['repair_assessment']}, ensure_ascii=False, indent=2))
