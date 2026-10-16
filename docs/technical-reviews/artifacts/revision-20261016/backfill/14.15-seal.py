"""Seal only this reviewer's independently checked 14.15 report and small receipts."""

import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parent
WORK = Path('/workspace/work/curriculum-20261016/backfill-14.15')
FULL = WORK.parent / 'fulltexts'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, body):
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + '\n')


probe = json.loads((BASE / '14.15-probe-result.json').read_bytes())
assert probe['status'] == 'all assertions passed'
frozen = json.loads((ROOT / 'docs/course-revision-20261016/freeze-14.json').read_bytes())
page = next(x for x in frozen['pages'] if x['page_id'] == '14.15')
paper_raw = FULL / 'huginn-v2.source'
paper = BeautifulSoup(paper_raw.read_text(), 'html.parser')
paper31 = paper.find(id='S3.SS1').get_text(' ', strip=True)
paper33 = paper.find(id='S3.SS3').get_text(' ', strip=True)
autograd = (FULL / '14.15-autograd.md').read_text()
loss = (FULL / '14.15-loss.source').read_text()
external = {
    'accessed_on': '2026-10-16',
    'scope': 'Original paper sections and upstream PyTorch versioned sources were actually read. Full third-party texts stay in /workspace/work; only short supporting excerpts and fingerprints are retained here.',
    'sources': [
        {'id': 's-paper', 'url': 'https://arxiv.org/html/2502.05171v2', 'version': 'arXiv:2502.05171v2, 2025-02-17', 'local_fulltext': str(paper_raw), 'sha256': sha(paper_raw), 'locators_read': ['3.1 Macroscopic Design: recurrence and final-state readout', '3.3 Training Objective: sampled iteration counts and endpoint loss; truncated-backpropagation paragraph'], 'excerpts': [paper31[:1100], paper33[:1050]], 'inspection': 'Repeated R(e,s) and final C(s_r) support variable-depth recurrence. Random r changes which terminal output receives loss. Their main run uses log-normal Poisson sampling and truncated BPTT k=8; the course odd1/3/5 full-unroll mechanism is its own design, not a replication of this paper.'},
        {'id': 's-autograd', 'url': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/docs/source/notes/autograd.md', 'version': 'PyTorch v2.14.1 tag', 'local_fulltext': str(FULL / '14.15-autograd.md'), 'sha256': sha(FULL / '14.15-autograd.md'), 'locator': 'How autograd encodes the history, lines 10–36', 'excerpt': autograd[autograd.index('## How autograd'):autograd.index('(saved-tensors-doc)=')], 'inspection': 'A differentiable unrolled computation graph propagates terminal loss through intermediate hidden states by the chain rule; indirect participation does not equal adding intermediate readout losses. Nonzero gradient magnitude for every input is not promised.'},
        {'id': 's-ce', 'url': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/loss.py', 'version': 'PyTorch v2.14.1 tag', 'local_fulltext': str(FULL / '14.15-loss.source'), 'sha256': sha(FULL / '14.15-loss.source'), 'locator': 'CrossEntropyLoss, lines 1200 onward; class-index formula and reduction mean', 'excerpt': loss[loss.index('class CrossEntropyLoss'):loss.index('class CrossEntropyLoss')+2680], 'inspection': 'For 19-class integer targets without weights, ignore labels or smoothing, mean CE is mean(-log softmax(logits)[gold]). Exact-match accuracy uses argmax and can stay constant when CE changes.'},
    ],
    'access_notes': ['Versioned docs.pytorch.org /2.14/ pages returned HTTP 403; the corresponding version-tagged upstream sources above were successfully read.', 'autograd.rst at v2.14.1 returned 404; the actual version uses autograd.md, which was successfully read.']
}
write(BASE / '14.15-original-sources.json', external)
figure_receipt = {
    'figure': 'course/figures/looped-mechanism-measured.svg',
    'frozen_path': frozen['figure_snapshots']['course/figures/looped-mechanism-measured.svg']['path'],
    'source_sha256': page['figures_sha256']['course/figures/looped-mechanism-measured.svg'],
    'render': str(WORK / 'figure.png'),
    'render_sha256': sha(WORK / 'figure.png'),
    'command': ".venv/bin/python - <<'PY'\nimport cairosvg\ncairosvg.svg2png(url='docs/course-revision-20261016/frozen-figures/36232631aa6614854ea3deb88511c0996934eea950713204c64f56b6717c45fa.svg', write_to='/workspace/work/curriculum-20261016/backfill-14.15/figure.png', output_width=1000)\nPY",
    'viewed_with': 'tools.view_image at /workspace/work/curriculum-20261016/backfill-14.15/figure.png',
    'observations': ['實際查看1000 px寬PNG。標題是相同初始化與1500次更新；藍色固定3輪、橘色抽1/3/5，與正文對應。', '上圖CE的10^0與10^-1刻度為對數軸，低者較好；下圖0–100%線性軸，高者較好；橫軸完整1–6輪。', '兩曲線第一輪較差；固定3在第三輪CE最低且accuracy較高，其後退化；抽奇數2/4可交卷，6輪保持本次有限表現。沒有畫成單調改善或漏掉不利點。', '原圖無箭頭／空間方向主張；曲線點、顏色、圖例、尺度與正文一致。小幅accuracy差異由相鄰表格提供精確數字。', '此任務只驗證圖本身；未聲稱完成網站桌面／手機閱讀路徑驗收。'],
    'coordinate_verification': probe['figure_coordinate_check'],
    'large_render_policy': 'Self-generated PNG remains outside Git; this small receipt preserves source/render hashes, acquisition command, and actual viewing observations.'
}
write(BASE / '14.15-figure-view.json', figure_receipt)
derivation = {
    'terminal_supervision': '令 h_r=F_theta(h_(r-1),x)，z_r=G_theta(h_r)，L_r=CE(z_r,y)。抽 D∈{1,3,5} 並只最小化L_D，不等於最小化L_1+...+L_D。D=5時 ∂L_5/∂h_4=(∂h_5/∂h_4)^T ∂L_5/∂h_5；∂L_5/∂h_2再經第3、4、5輪Jacobian。沒有detach時存在反傳路徑，但梯度可能因輸入或參數為零；所以「間接梯度」指圖上的參與，不保證每樣本非零。加入第2輪loss後，2屬直接監督；4若無對應loss，仍只間接參與。',
    'depth_locations': '終止集合S={1,3,5}；min S=1、max S=5；1<2<5且1<4<5，2、4不屬S，故稱未直接監督的範圍內測試／深度插值。6>5，故稱深度外推；這只是整數位置分類，不是聲稱函數連續、平滑或準確率有下界。',
    'counterexample': 'h_0=-1；h_r=-h_(r-1)=(-1)^(r+1)。固定readout P(correct|h)=0.9 if h>0 else0.1，故r=1,3,5均0.9，而2,4,6均0.1。單一反例已否定「共享更新且奇數輪好⇒偶數輪必好」；它沒有訓練，也不估計Transformer的實際頻率。',
    'work': '固定3：1500*128*3=576000。抽奇數的可重現hash選深度：1輪491步、3輪502步、5輪507步，所以128*(491+3*502+5*507)=580096；(580096-576000)/576000*100=0.711111...%，四捨五入0.71%。區塊序列數不是同FLOPs或同牆鐘時間的保證。',
    'metrics': 'CE=(1/N)Σ(logsumexp(z_i)-z_i,y)，accuracy=(1/N)Σ1[argmax z_i=y_i]。例如兩模型都argmax正確，但正解機率0.6與0.9導致CE分別-log0.6與-log0.9。故兩者衡量不同關係，低CE、高accuracy為各自方向。',
    'scope': '透明推導加上本次CPU核對；不宣稱這些式子證明有限任務以外的品質。'
}
write(BASE / '14.15-derivations.json', derivation)

def artifact(identifier, name, kind, description, **extra):
    path = BASE / name
    return {'id': identifier, 'kind': kind, 'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'description': description, **extra}


artifacts = [
    artifact('a-probe-code', '14.15-probe.py', 'code', '本審閱者的有界CPU查核原始碼：重算已存在的逐題logits、讀checkpoint及GPU原始收據、做單一自製輸入的forward/backward，不更新權重。'),
    artifact('a-execution', '14.15-probe-result.json', 'execution', '實際執行本審閱者查核的完整小型結果、輸入SHA、版本、表格重算、配對權重、工作量與圖座標。GPU訓練是既有primary receipts，非本審閱者親跑。', command='uv run --frozen --extra cpu python docs/technical-reviews/artifacts/revision-20261016/backfill/14.15-probe.py', result='exit 0; all assertions passed；12組CE/accuracy與凍結正文一致；兩份實際last.pt均第1500步且SHA吻合freeze；24個圖點吻合；未執行optimizer step或新的formal test。', environment=probe['environment']),
    artifact('a-original', '14.15-original-sources.json', 'source_snapshot', '實際讀取的原論文及PyTorch版本化原始碼之定位、SHA、短摘錄與支持範圍；全文留在work。'),
    artifact('a-figure', '14.15-figure-view.json', 'figure_render', '實際render/view紀錄；大型PNG留work，保存SHA、重現命令、目視判斷及24點座標核對。'),
    artifact('a-derivation', '14.15-derivations.json', 'derivation', '終止監督與間接梯度、深度位置、週期反例、工作量及兩評分量的透明推導。'),
    artifact('a-seal-code', '14.15-seal.py', 'code', '只封存本節新報告與本審閱者小型收據的程式；不改教材、active manifest或舊review。'),
]

def repo_source(identifier, path, title, note):
    return {'id': identifier, 'kind': 'repository_code', 'title': title, 'path': path, 'sha256': sha(ROOT / path), 'version': '2026-10-16 checked working-tree bytes pinned by SHA-256', 'verified': True, 'inspection_note': note}


sources = [
    {'id': 's-paper', 'kind': 'paper', 'title': 'Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach', 'url': 'https://arxiv.org/html/2502.05171v2', 'version': 'arXiv 2502.05171v2, 2025-02-17', 'verified': True, 'checked_original': True, 'accessed_on': '2026-10-16', 'authority_reason': '架構與訓練方法作者的原論文；只支持recurrence與抽取終止深度的角色。', 'inspection_note': '已實讀HTML原文3.1與3.3，包括R(e,s)遞迴、C(s_r)末端readout、E_r L目標和truncated BPTT k=8。未把其大模型效果當成本課奇數輪實驗的證明。原檔SHA與短摘錄見a-original。'},
    {'id': 's-autograd', 'kind': 'official_source', 'title': 'PyTorch Autograd mechanics (v2.14.1)', 'url': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/docs/source/notes/autograd.md', 'version': 'PyTorch v2.14.1 tag; docs/source/notes/autograd.md', 'verified': True, 'checked_original': True, 'accessed_on': '2026-10-16', 'authority_reason': '目前實測版本PyTorch的上游官方文件原始碼。', 'inspection_note': '實讀How autograd encodes the history，lines10–36：forward記錄圖並以chain rule回傳，任意Python control flow依實際執行建立圖。用來核對不detach的間接梯度路徑，不保證所有梯度非零。SHA與短摘錄見a-original。'},
    {'id': 's-ce', 'kind': 'official_source', 'title': 'PyTorch CrossEntropyLoss (v2.14.1)', 'url': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/loss.py', 'version': 'PyTorch v2.14.1 tag; torch/nn/modules/loss.py', 'verified': True, 'checked_original': True, 'accessed_on': '2026-10-16', 'authority_reason': '上游官方CrossEntropyLoss契約與公式；與本專案的F.cross_entropy計算相符。', 'inspection_note': '實讀class CrossEntropyLoss lines1200起，integer-label的-log-softmax公式、無權重mean reduction及targets範圍。實際每題19 logits，gold long且無ignore/smoothing/weight；accuracy另由argmax exact-match計算。SHA及短摘錄見a-original。'},
    repo_source('s-frozen', page['snapshot'], '14.15 frozen original slice', '完整讀取唯一派定節；與scripts.reading_time.lesson_slices的14.15原始UTF-8逐位元比較一致，不宣稱整章已查證。'),
    repo_source('s-model', 'tiny_perceptron/lesson_products/model.py', 'LoopedSolver model', '實讀Tokenizer、Config、Block與LoopedSolver.forward，重點lines114–145：同一完整tokens、固定original每輪注入、共享block、loop範圍1–6、all_depths/末端readout，hidden沒有detach。其餘檔案內容不作本節未需主張的證據。'),
    repo_source('s-training', 'tiny_perceptron/lesson_products/training.py', 'Terminal loss, sampler and metrics', '實讀sample_batch/logits_for/step/predict/metrics及state_digest；lines100–133的odd以獨立hash決定末端1/3/5且只在該末端算CE；multi才有1–5加權loss；backward未切斷hidden。metrics為integer argmax accuracy及CE。'),
    repo_source('s-train', 'tiny_perceptron/lesson_products/train.py', 'Training initialization and actual last checkpoint', '實讀train()初始化seed、獨立minibatch generator、train/policy分工、迴圈step_number、last.pt與best.pt保存；配對模式不按development_target提前停。相同batch資料抽樣序列由獨立depth hash保持；不宣稱相同實際FLOPs。'),
    repo_source('s-data', 'tiny_perceptron/lesson_products/data.py', 'Finite box-world task generator', '實讀answer、render、family/split_of、box_worlds及core/stress分界；swap交換、設值覆寫並順序執行；query A/B共用世界family。又從實際b.jsonl重算全部2956個test boxes gold與family/train不重疊。'),
    repo_source('s-eval', 'tiny_perceptron/lesson_products/evaluation.py', 'Frozen paired mechanism evaluation', '實讀select_settings配對驗證、b_report、frozen_guard及evaluate_frozen的mechanisms段，特別lines468–491：讀last.pt而非best.pt，在相同rows測1–6並寫逐題logits；本次沒有重新呼叫formal-test入口。'),
    repo_source('s-report', 'docs/lesson-products/results/final-report-v1.json', '2026-10-16 final primary result receipt', '核對mechanisms.fixed3/odd全部六深度boxes結果、actual_last_step、原始predictions檔名/SHA/bytes與freeze/data SHA；本節以boxes為評分範圍，未把sum或C結果混入。實際逐題分數獨立重算見a-execution。'),
    repo_source('s-settings', 'docs/lesson-products/results/settings-freeze-v1.json', 'Pretest frozen mechanism checkpoint choices', '核對兩個mechanisms的last.pt、training-receipt、初始digest、step、seed、config、recipe和tokenizer；實際本機檔案SHA與其一致。protocol要求matched actual last-step，test_outcomes_observed=false。'),
    repo_source('s-pretest', 'docs/lesson-products/results/pretest-freeze-record-v1.json', 'Pretest primary choice record', '實讀選擇在test之前、paired actual1500、initial SHA及test命令；所引settings SHA吻合，選擇明定mechanism用actual last-step。這是既有原始收據，非審閱者親見當時操作。'),
    repo_source('s-gpu-fixed', 'docs/lesson-products/results/mechanism-fixed3-runner-result.json', 'Actual fixed3 GPU runner primary receipt', '核對reservation/execution job、GPU NVIDIA L4、completed/confirmed、training seed8159/batch128/last1500、initialdigest/data/sourcehash/work；policy best1200與表格所用last1500分開。GPU訓練未重跑。'),
    repo_source('s-gpu-odd', 'docs/lesson-products/results/mechanism-odd135-runner-result.json', 'Actual odd135 GPU runner primary receipt', '同樣核對實際GPU execution與job、seed8159/batch128/last1500、initialdigest/data/sourcehash/work；policy best1400與actual last1500分開；不是本審閱者重新訓練。'),
    repo_source('s-plot', 'scripts/lesson_products/plot_results.py', 'Plotting the complete primary six-depth curves', '完整讀plot()：fixed3藍#175a97、odd橘#b34813，範圍1–6，上CE對數，下accuracy百分比0–102。實際SVG的24點另核對primary logits及view，未只依程式碼判圖通過。'),
    {'id': 's-derivation', 'kind': 'derivation', 'title': 'Transparent recurrence, depth-range and work derivations', 'verified': True, 'details': '\n'.join(f'{k}: {v}' for k, v in derivation.items())},
    {'id': 's-execution', 'kind': 'execution', 'title': 'Independent CPU evidence recomputation and bounded model probe', 'verified': True, 'artifact_id': 'a-execution'},
]

def evidence(source, locator, supports):
    return {'source_id': source, 'locator': locator, 'supports': supports}


def claim(identifier, kind, statement, location, evidence_rows, scope, verification=None, artifact_ids=None):
    value = {'id': identifier, 'kind': kind, 'statement': statement, 'location': location, 'status': 'verified', 'evidence': evidence_rows, 'artifact_ids': artifact_ids or [], 'scope': scope}
    if verification is not None:
        value['verification'] = verification
    return value


def executed(expected, observed, details, denominators=None, tolerance=None):
    value = {'method': 'executed', 'expected': expected, 'observed': observed, 'details': details}
    if denominators:
        value['denominators'] = denominators
    if tolerance:
        value['tolerance'] = tolerance
    return value


claims = [
    claim('c1', 'concept', '共享遞迴可選不同推論終止輪數；能執行某輪數不構成該輪數品質或奇偶泛化的保證。', '第1段及反例後「共享更新可以產生這種週期」', [evidence('s-paper', '§3.1 R(e,s_(i-1)), final C(s_r); §3.3 sampled r', '支持可重用core而改變終止深度；不借用論文實測當本課odd1/3/5保證。'), evidence('s-derivation', 'counterexample', '共享F(s)=-s的反例在奇數0.9、偶數0.1，邏輯上否定無條件保證。')], '只確認機制可設定與不存在此種邏輯保證，不宣稱週期是實際Transformer常態。', artifact_ids=['a-original', 'a-derivation', 'a-execution']),
    claim('c2', 'concept', '只對抽到的1/3/5末端算CE時，第2/4輪仍在5輪反傳路徑中，應稱未直接監督終止深度；若另加2輪loss，2就屬直接監督。', '第2段及末段練習', [evidence('s-autograd', 'How autograd encodes the history, lines10–36', 'chain rule沿實際unroll圖傳到中間hidden，與在每個中間readout加loss不同。'), evidence('s-training', 'step lines100–133', 'odd只取logits_for(...loops=terminal)後單一CE，multi另行對1–5輪計loss。'), evidence('s-model', 'LoopedSolver.forward lines127–145', 'hidden遞迴未detach。'), evidence('s-derivation', 'terminal_supervision', '明列L_D與Jacobian路徑及加入2輪loss對監督集合的影響。')], '全unroll、grad enabled且無detach的本課機制；「間接梯度」不保證每一參數／輸入的梯度非零。CPU自製樣本第2/4輪grad norm實測非零，只是局部查核。', artifact_ids=['a-original', 'a-derivation', 'a-execution']),
    claim('c3', 'numeric', '對終止集合{1,3,5}，2與4屬範圍內未直接監督位置，6超出最大5；插值／外推名稱沒有平滑承諾。', '第3段', [evidence('s-derivation', 'depth_locations', '直接比較2/4/6與min/max終止集合。')], '這是深度座標分類；不等於狀態從未算過，也不推出品質。', {'method': 'hand_calculation', 'expected': '2、4位於1–5且不屬{1,3,5}；6>5。', 'observed': '1<2<5、1<4<5、6>5；集合判斷成立。', 'tolerance': '整數與集合精確相等。', 'details': derivation['depth_locations']}, ['a-derivation']),
    claim('c4', 'numeric', '變號反例的1/3/5輪正解機率0.9，2/4/6輪0.1。', '手工state程式碼與緊接解釋', [evidence('s-derivation', 'counterexample', 'h_r=(-1)^(r+1)，readout依符號固定。'), evidence('s-execution', 'counterexample.executed_frozen_code_stdout', '實際執行凍結正文內唯一Python區塊並逐字比對輸出。')], '手工無學習反例，不是神經網路實測。', executed('1 0.9;2 0.1;3 0.9;4 0.1;5 0.9;6 0.1', probe['counterexample']['executed_frozen_code_stdout'], '抽取frozen slice的Python區塊執行，並與推導比對。', tolerance='輸出字串精確相等。'), ['a-execution', 'a-derivation']),
    claim('c5', 'numeric', 'A=2、B=6交換後查B得2、查A得6；核心測試按序執行交換／設值，事件長度1–4，只計整數完全正確。', '「兩盒追蹤」任務段', [evidence('s-data', 'answer/render/box_worlds and core/stress split', 'swap同時交換，set覆寫指定盒，query最後取值；世界家族先分再衍生A/B題。'), evidence('s-execution', 'task_label_audit及recomputed_table', '獨立順序解題核對全部2956題gold及1–4長度；指標用argmax整數精確比對。')], '有限語法與有限整數範圍；本節只報boxes子任務。訓練資料另有sum次要任務，不把表格解釋為所有任務。', executed('交換後A6/B2；2956盒題gold符合順序規則且長度只有1–4。', json.dumps(probe['task_label_audit'], ensure_ascii=False), '不呼叫訓練或正式test；從現有b.jsonl逐題重算盒值並比對gold。', tolerance='整數標籤、題數及事件長度精確一致。'), ['a-execution']),
    claim('c6', 'software', '本課LoopedSolver可設定1–6輪；每輪處理同一完整題目與原始輸入注入，沒有每輪再收一個新事件。', '首段「設成二輪或四輪」與任務段末', [evidence('s-model', 'LoopedSolver.forward lines127–145', 'tokens與original在迴圈外建立，每輪共享block讀hidden+0.5*original；loop整數1–6。'), evidence('s-execution', 'bounded_cpu_probe', '實際執行自製問題1–6輪，與all_depths的對應輸出完全相等。')], '只驗證已查版本CPU與已限定loop介面；單一輸入的跑通不代表測試品質。', executed('1–6均可執行；單輪指定depth輸出與相同depth的all_depths一致。', '六個max_abs_difference皆0.0；loops2/4已實際執行。', '載入odd實際last.pt，使用教材格式自製A2/B6/swap/B，不重開formal test。'), ['a-execution']),
    claim('c7', 'empirical', '兩個機制模型同seed8159、batch128、初始化與資料，均完成1500更新；比較last1500而非policy最佳步。', '「本課實際從同一組隨機權重」段', [evidence('s-gpu-fixed', 'execution.job/training, source_hashes', '原始L4收據：fixed3、1500、seed8159、batch128及初始digest。'), evidence('s-gpu-odd', 'execution.job/training, source_hashes', '原始L4收據：odd、1500及相同起點/資料；best1400另列。'), evidence('s-settings', 'mechanisms.fixed3/odd', '凍結last.pt與recipe/seed/config。'), evidence('s-pretest', 'choice_before_test/paired_actual_steps', '事前明定使用actual last-step。'), evidence('s-train', 'torch.manual_seed; generator; training loop', 'depth hash不消耗minibatch RNG，recipe同而終止深度不同。'), evidence('s-eval', 'evaluate_frozen lines468–479', '正式分數載入last.pt。'), evidence('s-execution', 'paired_training/recreated_seed8159_initial_state_sha256', '親自讀實際兩份checkpoint核SHA與metadata；CPU重建8159初始權重digest也相同。')], '既有GPU訓練由primary receipts確認；審閱者未親跑GPU。這是相同更新數配方比較，非相同算力；一seed不能外推普遍優劣。', executed('同初始SHA，兩last步皆1500，seed8159/batch128。', '初始SHA均4ad5df4b…feccde；last步均1500；best步fixed3=1200、odd=1400未混用；last檔SHA吻合freeze。', '讀原始GPU execution、freeze、actual last.pt及receipt；核對相關訓練原始碼SHA仍吻合runner。', {'seeds': '1 (8159)', 'updates_per_model': '1500', 'batch_per_update': '128', 'sampled_rows_per_model': '192000 with replacement', 'comparison': 'actual last1500, not best policy checkpoint'}), ['a-execution']),
    claim('c8', 'numeric', '固定3的區塊序列工作量576000，抽奇數580096，多0.71%。', '訓練對照段工作量句', [evidence('s-training', 'step work return lines129–133', 'block_sequence_work=batch rows*terminal loops。'), evidence('s-derivation', 'work', '依1500次hash選深度實際頻數推算。'), evidence('s-execution', 'paired_training.*.depth_histogram/work_increase_percent', '重算全部1500個終止深度，與兩個GPU收據work相等。')], '僅區塊序列計數，沒有宣稱FLOPs、GPU秒或wall time完全相同。', executed('576000、580096、增加0.71%。', 'odd頻數1:491、3:502、5:507；增加0.7111111111111111%。', derivation['work'], tolerance='計數精確一致；比例以百分比小數二位四捨五入。'), ['a-execution', 'a-derivation']),
]
for mode, cid, label in [('fixed3', 'c9', '固定3輪'), ('odd', 'c10', '抽1/3/5輪')]:
    values = probe['recomputed_table'][mode]
    claims.append(claim(cid, 'empirical', f'{label}模型在同一2956題／1478世界家族盒追蹤最後測試上，1–6輪CE與答案正確率皆如表。', f'六列表格「{label}訓練」欄及對應曲線', [evidence('s-report', f'mechanisms.{mode}.evaluation.1..6.boxes', '原始正式報告的全部六深度，不只挑選最佳點。'), evidence('s-eval', 'b_report; evaluate_frozen lines468–491', '相同rows、actual last.pt、逐題19 logits與整數gold。'), evidence('s-execution', f'recomputed_table.{mode}; inspected_input_sha256', '對原始predictions.jsonl逐題穩定logsumexp重算CE，argmax exact-match重算accuracy，核對相同IDs、gold及family。')], '只支持seed8159、有限盒追蹤heldout family；train/test家族無重疊但同有限語法，非開放語言推理保證。原始GPU訓練與既有CPU正式test沒有重跑。', executed('六個CE四位小數／accuracy百分比二位小數均等於凍結表。', '; '.join(f'{d}輪:{values[str(d)]["display"]}' for d in range(1,7)), '原始檔56528235 bytes、SHA8b593421…cf4896；僅篩product=mechanism、相應mode、task=boxes、split=test。19個finite logits；CE=mean(logsumexp-logit_gold)，accuracy=correct/N；無忽略／權重／校準。表格12組全部吻合。', {'questions_per_depth': '2956', 'world_families_per_depth': '1478', 'depths': '1,2,3,4,5,6', 'training_seed': '8159', 'checkpoint_step': '1500', 'labels': 'one integer per question; 19 class logits', 'test_event_lengths': '1–4'}, 'CE與原始float32 receipt差<1e-6；accuracy差<1e-7；表格顯示精確相等。'), ['a-execution']))
claims += [
    claim('c11', 'concept', '較低CE與較高答案正確率回答不同問題，兩者不可互代。', '表格前及圖說「越低的CE與越高的正確率」', [evidence('s-ce', 'CrossEntropyLoss integer-index formula and mean reduction, lines1200 onward', 'CE受gold機率影響，並非只比較argmax。'), evidence('s-derivation', 'metrics', 'argmax一致但0.6/0.9機率的CE不同，兩指標沒有一對一對應。')], '本表未校準的19類integer-target mean CE與exact-match accuracy；不解釋為自我信心校準。', artifact_ids=['a-original', 'a-derivation']),
    claim('c12', 'numeric', '圖中藍／橘曲線完整呈現1–6輪：CE是對數縱軸，accuracy是百分比線性縱軸；24點與表及primary scores相同。', 'looped-mechanism-measured.svg及圖後解說', [evidence('s-plot', 'plot() complete function', '六深度、色彩、CE log axis及accuracy 0–102設定。'), evidence('s-execution', 'figure_coordinate_check', '獨立以實際SVG path座標核對全部4*6個點及軸轉換。'), evidence('s-derivation', 'metrics', '較低CE／較高accuracy的方向。')], 'PNG已實際render/view；不是網站版面驗收。小幅accuracy差異精確值依相鄰表格。', executed('24點在log10 CE/linear百分比映射下等於重算結果。', '四序列最大SVG座標誤差依序3.97e-6、7.70e-6、9.45e-6、4.51e-6；目視圖例與尺度一致。', '核對原始SVG等距x、顏色及四條6點path；對應每個mode的raw logits；另view_image查看1000 px PNG。', tolerance='SVG序列反推誤差<3e-5座標單位；原始圖SHA與freeze及live相同。'), ['a-execution', 'a-figure']),
    claim('c13', 'empirical', '本次odd模型2/4輪確實有高正確率，6輪是外推位置的有限觀察；固定3模型在第3輪更高；不推成普遍成功定理。', '圖後結果解釋與單seed限定', [evidence('s-execution', 'recomputed_table.fixed3/odd', 'odd@2=2877/2956，@4=2916/2956，@6=2909/2956；fixed3@3=2941/2956高於odd@3=2916/2956。'), evidence('s-report', 'mechanisms; scope', '原始結論限finite fixed grammar，更多輪可使accuracy/CE變差。')], '單seed、同有限盒追蹤留出題，不證明所有偶數輪、所有任務或無限深度；fixed3與odd比較有0.71%工作量差。', executed('odd2/4本次能正確作答多數留出題，fixed3@3高於odd@3。', 'odd2=97.33%，odd4=98.65%，odd6=98.41%；fixed3@3=99.49% > odd@3=98.65%。', '直接由完整逐題重算曲線比較，不擇最佳點、不重跑測試。', {'questions': '2956', 'families': '1478', 'seed_count': '1', 'seed': '8159', 'checkpoint_step': '1500', 'eval_depths': '1–6'}), ['a-execution']),
    claim('c14', 'software', '部署成品B另用seed7421與multi目標直接監督1–5輪，不能當作本表seed8159奇數末端機制對照。', '最後「這兩份是機制對照」段', [evidence('s-training', 'step mode=multi branch, lines106–114', 'multi對1–5輪各算CE且五個正權重；odd分支只算一個末端。'), evidence('s-execution', 'deployment_b_separate_recipe及inspected_input_sha256中B training-receipt', '實讀實際B training-receipt seed7421、mode multi、depth_weights全1。')], '只核對B配方與本節對照不同；不替19.15其他部署能力或速度背書。', executed('B seed7421，multi，直接監督1–5。', '實際receipt seed7421、mode=multi、depth_weights=[1,1,1,1,1]，reference_depth5。', '讀本機正式成品B既有GPU training-receipt；源檔SHA封存於查核輸出，並核對multi loss原始碼。'), ['a-execution']),
]

report = {
    'schema_version': 1, 'review_stage': 'technical', 'lesson_id': '14.15', 'source': 'course/chapters/14.md#14.15',
    'reviewer_task': '/root/backfill_tech_14_15', 'reviewer_context': 'fresh', 'author_tasks': ['/root'],
    'source_sha256': page['source_sha256'], 'figure_sha256': page['figures_sha256'], 'verdict': 'pass',
    'claims': claims, 'sources': sources, 'artifacts': artifacts, 'issues': [],
    'checks': {
        'factual_accuracy': {'status': 'pass', 'details': '已逐項核對直接／間接監督、可執行與泛化區別、有限任務答案規則、同起點actual last1500對照與部署B不同配方。原論文只作recurrence/sampling來源，沒有冒稱複現其大型結果。', 'claim_ids': [f'c{i}' for i in range(1,15)]},
        'numeric_verification': {'status': 'pass', 'details': '實跑凍結反例；2956題gold與1478family核對；由既有primary logits重算全部12組CE/accuracy，精確吻合正文顯示；1500步hash深度頻數重算工作量及0.71%；未重訓或重啟formal test。', 'claim_ids': ['c3','c4','c5','c7','c8','c9','c10','c12','c13']},
        'figure_consistency': {'status': 'pass', 'details': '親自以CairoSVG render frozen SVG並view_image查看；核對藍fixed3／橘odd、全部1–6輪、CE log尺度與accuracy百分比方向；獨立比對24個SVG點與重算分數。大型PNG在work，正式小收據含SHA、命令及目視紀錄。', 'claim_ids': ['c9','c10','c12','c13']},
        'source_verification': {'status': 'pass', 'details': '實讀huginn v2 §§3.1/3.3與PyTorch v2.14.1 autograd/CE上游原始碼；保存短摘錄、URL、版本、SHA、定位。另核對本節所需實作、pretest/settings freeze、原始GPU runner receipts、actual last.pt與原始逐題scores；不是沿用他人review。', 'claim_ids': ['c1','c2','c6','c7','c9','c10','c11','c14']},
        'limitations': {'status': 'pass', 'details': '正文將手工反例與實訓分開，將未直接監督與未經過狀態分開，將深度插值／外推位置與品質保證分開。實測限1seed8159、有限語法boxes heldout-family；非同算力；表用actual1500不混best；部署B種子與監督不同。此審閱未執行GPU訓練、formal test、調參、Modal或HF寫入，未驗網站閱讀路徑。', 'claim_ids': ['c1','c2','c3','c4','c5','c7','c8','c9','c10','c13','c14']},
    },
    'review_scope': '唯一派定14.15 frozen-14來源；只讀本節、對應圖與必要一手證據。未讀作者plan、其他review或舊review結論；未改教材、舊報告、active manifest，未提交Git。',
    'execution_scope': '親跑：有界CPU source/scores/checkpoint檢查、單一手工輸入forward/backward、反例及圖render。既有primary receipts：L4訓練與2026-10-16正式CPU heldout scores。兩者明確區分。',
}
destination = ROOT / 'docs/course-revision-20261016/reviews/technical-14.15-backfill-original-01.json'
assert not destination.exists(), 'Do not overwrite a sealed original'
write(destination, report)
print(json.dumps({'report': str(destination.relative_to(ROOT)), 'sha256': sha(destination), 'claims': len(claims), 'artifacts': len(artifacts)}, ensure_ascii=False))
