"""Seal this reviewer's independently checked 14.16 report and small receipts."""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.check_technical_reviews import _validate
from scripts.reading_time import lesson_slices

HERE = Path(__file__).resolve().parent
FULL = Path('/workspace/work/curriculum-20261016/fulltexts')
TASK = '/root/backfill_tech_14_16'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def relative(path):
    return path.relative_to(ROOT).as_posix()


loopus = BeautifulSoup((FULL / 'loopus.source').read_text(), 'html.parser')
cv = BeautifulSoup((FULL / '14.17-sklearn-cv.source').read_text(), 'html.parser')
cv_section = cv.find(id='cross-validation-evaluating-estimator-performance')
cv_paras = [x.get_text(' ', strip=True) for x in cv_section.find_all('p', recursive=False)]
loopus_extracts = {}
for locator in ['S2.SS0.SSS0.Px2', 'S3.SS1.SSS0.Px2', 'S3.SS1.SSS0.Px3']:
    loopus_extracts[locator] = loopus.find(id=locator).get_text(' ', strip=True)
loopus_extracts['S3.SS2.SSS0.Px2'] = loopus.find(id='S3.SS2.SSS0.Px2').get_text(' ', strip=True)
app = loopus.find(id='A2').get_text(' ', strip=True)
start = app.index('This is a diagonally preconditioned')
end = app.index('More directly, our training objective')
loopus_extracts['Appendix B, equations 21-24, fixed-point and extra potential assumptions'] = app[start:end]
receipt = {
    'reviewer_task': TASK,
    'inspected_on': '2026-10-16',
    'sources': [
        {'url': 'https://arxiv.org/html/2605.11011v1', 'version': 'arXiv v1, submitted 2026-05-10 11:05:20 UTC', 'original_sha256': digest(FULL / 'loopus.source'), 'original_local_path_outside_git': str(FULL / 'loopus.source'), 'version_confirmation_sha256': digest(FULL / '14.16-loopus-abs.html'), 'excerpts': loopus_extracts, 'inspection_note': 'Personally read background Looped LLMs; 3.1 random supervision/stop; 3.2 equations 10-13 and Figure 4 pseudocode; Appendix B equations 21-26 and Appendix E equations 27-41. Paper uses SiLU(delta), not the tutorial illustrative hinge. Appendix B adds a potential and smoothness/step-size assumptions for energy descent. The tutorial makes no claim to reproduce the paper or prove correctness.'},
        {'url': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/loss.py', 'version': 'PyTorch v2.14.1 tag', 'original_sha256': digest(FULL / '14.16-torch-loss.py'), 'original_local_path_outside_git': str(FULL / '14.16-torch-loss.py'), 'locator': 'CrossEntropyLoss lines 1200-1242', 'excerpt': '\n'.join((FULL / '14.16-torch-loss.py').read_text().splitlines()[1199:1242]), 'inspection_note': 'Personally fetched pinned official source and read logits/index-target formula, weighted mean and LogSoftmax+NLL equivalence. Used unweighted class-index definition with no ignored rows for score check; lower CE need not alter argmax.'},
        {'url': 'https://scikit-learn.org/1.7/modules/cross_validation.html#cross-validation-evaluating-estimator-performance', 'version': 'scikit-learn 1.7.2, DOCUMENTATION_OPTIONS.theme_switcher_version_match in cached original HTML', 'original_sha256': digest(FULL / '14.17-sklearn-cv.source'), 'original_local_path_outside_git': str(FULL / '14.17-sklearn-cv.source'), 'locator': '3.1 introductory paragraphs on unseen data, heldout validation and final test', 'excerpts': [p for p in cv_paras if any(t in p for t in ['methodological mistake', 'When evaluating different settings', 'To solve this problem'])], 'inspection_note': 'Personally read the original introductory explanation, including test leakage from adjusting settings and the validation-before-final-test workflow. Supports using validation to choose a fixed loop hyperparameter; does not promise out-of-domain quality.'},
    ],
}
receipt_path = HERE / '14.16-primary-source-receipts.json'
save(receipt_path, receipt)
execution_path = HERE / '14.16-execution.json'
execution = json.loads(execution_path.read_text())

def repo_source(identifier, title, path, note):
    return {'id': identifier, 'kind': 'repository_code', 'title': title, 'path': path, 'sha256': digest(ROOT / path), 'version': 'working-tree bytes inspected 2026-10-16, SHA-256 pinned', 'verified': True, 'inspection_note': note}


sources = [
    {'id': 'loopus', 'kind': 'paper', 'title': 'LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models', 'url': 'https://arxiv.org/html/2605.11011v1', 'version': 'arXiv v1, 2026-05-10; original HTML SHA-256 ' + digest(FULL / 'loopus.source'), 'verified': True, 'checked_original': True, 'accessed_on': '2026-10-16', 'authority_reason': 'Original paper proposing LoopUS; directly states its supervision, regularizer, inference stop rule, and conditional geometric interpretation.', 'inspection_note': receipt['sources'][0]['inspection_note']},
    {'id': 'ce', 'kind': 'official_source', 'title': 'PyTorch CrossEntropyLoss original source and contract', 'url': receipt['sources'][1]['url'], 'version': 'PyTorch v2.14.1, original SHA-256 ' + digest(FULL / '14.16-torch-loss.py'), 'verified': True, 'checked_original': True, 'accessed_on': '2026-10-16', 'authority_reason': 'PyTorch-maintained loss implementation and mathematical API contract.', 'inspection_note': receipt['sources'][1]['inspection_note']},
    {'id': 'validation', 'kind': 'official_docs', 'title': 'scikit-learn 3.1 Cross-validation: evaluating estimator performance', 'url': receipt['sources'][2]['url'], 'version': receipt['sources'][2]['version'] + '; original SHA-256 ' + digest(FULL / '14.17-sklearn-cv.source'), 'verified': True, 'checked_original': True, 'accessed_on': '2026-10-16', 'authority_reason': 'Official scikit-learn user guide describes the evaluation/selection distinction.', 'inspection_note': receipt['sources'][2]['inspection_note']},
    {'id': 'math', 'kind': 'derivation', 'title': 'Transparent loss and guarantee counterexamples', 'verified': True, 'details': 'For assumed losses [1.2,.8,.6,.7,.9], min=.6 at round3. Hinge max(0,after-before) gives .2 for (.2,.4), zero for (.5,.4): unchanged later quality and worse earlier quality can reduce penalty. With J=L1+L2+lambda*hinge and lambda=1, feasible candidate losses [.001,.002] violate monotonicity but J=.004, lower than nonviolating [1,1] with J=2; a finite soft term alone imposes no hard constraint. Increasing earlier CE by .3 adds .3 to its own unweighted CE term, so per-round CE makes that shortcut costly without proving universal optimality. CE=-log p_gold: binary gold0, [.2,.8] to [.4,.6] lowers CE 1.6094379 to .9162907 while argmax1 stays wrong. A mean drop [1,1] to [0,1.5] does not imply each item improves. Fixed-point counterexample: scalar h_(r+1)=h_r/2 converges to0; a decoder with constant probabilities [.1,.9] for gold0 remains wrong at every state. Thus state convergence does not prove correct answers. No finite training-set observation constrains an arbitrary unobserved input without further assumptions. If validation round3 minimizes loss and round4 maximizes exact-answer accuracy, the stated acceptance metric selects which objective to optimize; neither metric is an algebraic proxy for the other.'},
    {'id': 'cpu', 'kind': 'execution', 'title': 'Reviewer CPU check of frozen code, raw scores and local metadata', 'verified': True, 'artifact_id': 'a-execution'},
    repo_source('body', 'Frozen 14.16 body', 'docs/course-revision-20261016/freeze-10/sources/14.16.md', 'Read entire assigned section; matched exact UTF-8 lesson_slices body and freeze-14 page SHA. No other section is asserted reviewed.'),
    repo_source('primary', 'Original final-report-v1 mechanism scores', 'docs/lesson-products/results/final-report-v1.json', 'Inspected mechanisms.fixed3/odd at depths3,4,6; original prediction file SHA/bytes; scope and freeze SHA. Four relevant boxes cells independently recomputed from raw logits.'),
    repo_source('settings', 'Settings freeze original mechanism metadata', 'docs/lesson-products/results/settings-freeze-v1.json', 'Inspected mechanisms fixed3/odd checkpoint SHA, seed8159, last_step1500, matching initialization and tokenizer. Checked final-report settings SHA and actual local checkpoint/receipt hashes.'),
    repo_source('eval-code', 'Mechanism evaluation and raw serialization implementation', 'tiny_perceptron/lesson_products/evaluation.py', 'Read b_logits lines38-40 and mechanism final-evaluation lines468-492: reads actual last.pt, evaluates each depth1..6, serializes full 19-class logits, labels and IDs. Did not call evaluate_frozen or restart test.'),
    repo_source('metrics-code', 'Classification metrics implementation', 'tiny_perceptron/lesson_products/training.py', 'Read metrics lines147-158: CPU float32 logits, class-index targets, argmax exact accuracy, mean unweighted F.cross_entropy. Independent reaggregation uses stable logsumexp float64 over the same 2956 rows.'),
    repo_source('model', 'Shared loop update implementation', 'tiny_perceptron/lesson_products/model.py', 'Read Config, Block and LoopedSolver.forward lines87-148: shared block reused, original input injected, per-depth norm/readout, 19 classes and explicit loops1..6; no descent projection or correctness test is imposed.'),
    repo_source('runner-fixed3', 'Existing fixed3 actual GPU runner receipt', 'docs/lesson-products/results/mechanism-fixed3-runner-result.json', 'Inspected execution.status, gpu_name, job seed/batch/steps, execution.training and local receipt identity. Original NVIDIA L4 training confirmed by existing receipt; this reviewer did not execute GPU work.'),
    repo_source('runner-odd', 'Existing odd135 actual GPU runner receipt', 'docs/lesson-products/results/mechanism-odd135-runner-result.json', 'Inspected execution.status, gpu_name, job seed/batch/steps, execution.training and local receipt identity. Same distinction between existing primary receipt and personally executed CPU check.'),
]


def evidence(source, locator, supports):
    return {'source_id': source, 'locator': locator, 'supports': supports}


def claim(identifier, kind, statement, location, scope, refs, artifacts=(), verification=None):
    value = {'id': identifier, 'kind': kind, 'statement': statement, 'location': location, 'status': 'verified', 'evidence': refs, 'artifact_ids': list(artifacts), 'scope': scope}
    if verification is not None:
        value['verification'] = verification
    return value


claims = [
    claim('c1', 'numeric', '假設五輪 loss 為1.2、0.8、0.6、0.7、0.9，第三輪的 loss 最低。', '第1段人工假設', '僅按相同資料上的 loss 選擇；這組數字明示假設，不是成品實測，也不證明部署分布相同。', [evidence('math', 'hypothetical minimum', '逐一比較五個數值得唯一最小值。')], ['a-execution'], {'method': 'executed', 'expected': 'round3, loss0.6', 'observed': 'best_round=3, losses原樣；assert通過', 'tolerance': '輪數與輸入精確相等。', 'details': '在已保存CPU程式中以argmin求得。'}),
    claim('c2', 'concept', '共享循環區塊本身未被定義成每輪朝正解下降的最佳化器。', '第1段末句', '循環是結構安排；特定額外約束可帶來有條件性質，但權重共享本身沒有答案或loss單調契約。', [evidence('loopus', 'Background Looped LLMs; Appendix B equations21-26', '重複區塊定義與固定點/勢能的附加假設分開。'), evidence('model', 'LoopedSolver.forward', '本課共享區塊未執行loss下降投影。')], ['a-sources']),
    claim('c3', 'concept', '逐輪答案監督與後輪退步正則可以促進品質，但有限權重軟目標不能提供任意新題逐題單調保證。', '第2段', '教材hinge為通用示意；LoopUS用抽樣深度和SiLU差額，兩者沒有被混稱同一配方。未聲稱這些方法在本課模型上已訓練成功。', [evidence('loopus', '3.1 Random Deep Supervision; 3.2 equations10-13; Appendix B equations24-26', 'CE、monotonicity與confidence共同最佳化，幾何下降另有前提。'), evidence('math', 'finite soft penalty and unseen-input argument', '具體有限候選空間中違反單調的解仍能有較小總目標，故軟罰項不等於硬限制。'), evidence('validation', '3.1 introductory evaluation paragraphs', '訓練觀察與未見資料泛化需要區分。')], ['a-sources', 'a-execution']),
    claim('c4', 'numeric', 'max(0,L_after-L_before)只罰正差額；把早輪由0.2改成0.5而後輪固定0.4，懲罰由0.2變0。', '第2-4段公式、人工算例與解讀', '只證明懲罰可能被提高早輪loss來降低；不宣稱實際梯度一定採此退路。各輪CE會對早輪惡化加上相應代價。', [evidence('math', 'hinge calculation and added per-round CE cost', '差額與各輪品質是不同目標。')], ['a-execution'], {'method': 'executed', 'expected': 'penalty(.2,.4)=.2; penalty(.5,.4)=0', 'observed': 'stdout「懲罰 0.2」「懲罰 0.0」', 'tolerance': 'Python輸出與兩個精確十進位展示值一致；無大規模浮點推論。', 'details': '直接執行凍結正文唯一Python block，核對前後輸入、max公式與完整stdout。'}),
    claim('c5', 'software', '正文penalty程式使用Python內建max並印出兩組預期懲罰。', '唯一Python程式區塊', '只保證本次Python3.13.5 CPU程式行為；不是神經模型訓練或效能實驗。', [evidence('body', 'Python block', '固定待執行原始碼。'), evidence('cpu', 'frozen_code_stdout', '逐字核對實際輸出。')], ['a-execution', 'a-code'], {'method': 'executed', 'expected': '早輪0.2後輪0.4懲罰0.2；早輪0.5後輪0.4懲罰0.0', 'observed': execution['frozen_code_stdout'], 'details': '從lesson_slices擷取未修改block執行並assert兩行stdout完全一致。'}),
]

for identifier, mode, depths, statement in [
    ('c6', 'fixed3', [3,6], '固定3輪模型的第3/6輪CE為0.0238/0.1072，正確率為99.49%/97.67%。'),
    ('c7', 'odd', [3,4], '抽奇數模型第3/4輪正確率均98.65%，CE由0.0389略升到0.0395。'),
]:
    observed = execution['score_reaggregation'][mode]
    claims.append(claim(identifier, 'empirical', statement, '第5段「上一節的實測」', '同一2956題兩盒追蹤最後測試，1478世界家族、單seed8159、實際第1500步機制模型；不是部署成品B、開放域泛化或所有樣本的單調結論。個別題目答對集合不必相同。', [evidence('primary', f'mechanisms.{mode}.evaluation depths {depths}, boxes', '原始固定報告值、分母及正確題數。'), evidence('eval-code', '468-492', 'last checkpoint、逐深度logits與測試ID序列。'), evidence('metrics-code', '147-158', 'mean CE與argmax判準。'), evidence('cpu', f'score_reaggregation.{mode}; checkpoint_and_receipt_checks', '親自以原始19類logits重算，並核對實際last.pt/freeze/既有訓練收據。')], ['a-execution', 'a-code'], {'method': 'executed', 'expected': statement, 'observed': json.dumps(observed, ensure_ascii=False), 'tolerance': 'CE對原報告絕對差<2e-6，correct整數精確相同；本節4位CE及2位百分比全相同。', 'details': '只重新聚合已存在且SHA吻合的逐題JSONL；穩定logsumexp以float64計算每題負log p_gold後平均，argmax與gold比較。未推論、未重啟formal test、未重訓；GPU更新依既有primary runner receipts。', 'denominators': {'samples': 2956, 'world_families': 1478, 'classes': 19, 'split': 'heldout boxes final test', 'seed': 8159, 'actual_updates': 1500, 'batch_size': 128, 'inference_depths': depths, 'label_criterion': 'argmax integer exactly equals gold', 'ce_reduction': 'unweighted mean over all 2956 boxes rows, no ignored labels'}}))

claims.extend([
    claim('c8', 'concept', 'CE下降不一定改變argmax正確率，選輪數須按產品驗收目標；兩條曲線不必一致。', '第6段CE與argmax；末段練習', '不同目標可選出不同輪數；未把loss最小自動等同答案正確率最高。', [evidence('ce', 'CrossEntropyLoss lines1218-1242', 'CE取決於gold機率而非僅argmax對錯。'), evidence('math', 'binary CE counterexample', 'p_gold .2→.4時CE下降而argmax仍錯。')], ['a-sources', 'a-execution']),
    claim('c9', 'concept', '平均訓練loss、每個訓練樣本、新題與正確率是不同保證範圍。', '第6段首兩句', '只要求先定義保證對象；未提出任何未證明的新題單調定理。', [evidence('ce', 'CrossEntropyLoss lines1230-1239', 'mean reduction與逐題loss不同。'), evidence('math', 'mean [1,1]→[0,1.5] and CE/argmax counterexamples', '平均下降不能推出每項下降；loss不等同正確率。'), evidence('validation', '3.1 opening and validation paragraphs', '訓練與unseen資料的評估分開。')], ['a-sources', 'a-execution']),
    claim('c10', 'concept', '狀態收斂到固定點不等於固定點答對。', '第6段末句', '收斂條件、收斂速度與答案正確性分開；未主張本課狀態已收斂。', [evidence('loopus', 'Appendix B equations21-26', '原文把fixed-point、額外勢能假設與task-aligned CE surrogate分開。'), evidence('math', 'h_next=h/2 with constant wrong decoder', '具體反例同時滿足狀態收斂與答案永遠錯。')], ['a-sources']),
    claim('c11', 'concept', 'LoopUS共同研究訓練正則與自適應停止，逐題停止須使用推論時可取得的訊號。', '第7段研究方向及末句', '僅引用研究方向與機制；其confidence門檻不等於可靠正解保證。教材沒有把max hinge冒充LoopUS原式，也沒有宣稱本課成品採該停止頭。', [evidence('loopus', '3.1 equation9; 3.2 equations10-13; Appendix E equations27-41', 'joint CE/SiLU/confidence目標與推論confidence、state change、CDF訊號。')], ['a-sources']),
    claim('c12', 'concept', '一般部署可先在validation選固定輪數，再以保留測試驗收；不能用測試結果持續調設定來推論未見表現。', '第7段「一般部署可先在validation選定固定輪數」', '本節為部署建議而非已執行選型紀錄；第一段假設保留集不當作新的正式測試重調授權。', [evidence('validation', '3.1 introductory paragraphs on hyperparameters, validation and final test', '輪數屬可在驗證集選擇的設定，最後test須保留。')], ['a-sources']),
])

def artifact(identifier, kind, path, description, **extra):
    return {'id': identifier, 'kind': kind, 'path': relative(path), 'sha256': digest(path), 'description': description, **extra}


report = {
    'schema_version': 1, 'review_stage': 'technical', 'lesson_id': '14.16',
    'source': 'course/chapters/14.md#14.16', 'reviewer_task': TASK,
    'reviewer_context': 'fresh', 'author_tasks': ['/root'],
    'source_sha256': execution['source_sha256'], 'figure_sha256': {},
    'verdict': 'pass', 'claims': claims, 'sources': sources,
    'artifacts': [
        artifact('a-execution', 'execution', execution_path, '親跑CPU凍結程式、原始分數重算、checkpoint與既有receipt對照；含命令、環境、原始來源SHA及實際輸出。', command=execution['command'] + ' > ' + relative(execution_path), result=execution['result'], environment=execution['environment']),
        artifact('a-code', 'code', HERE / '14.16-check.py', '本審閱者相稱CPU查證程式；不呼叫訓練或正式test入口。'),
        artifact('a-sources', 'source_snapshot', receipt_path, '已讀LoopUS、PyTorch與scikit-learn原始來源的小段原文、版本、SHA、定位及本人inspection note；全文留work/。'),
    ],
    'issues': [],
    'checks': {
        'factual_accuracy': {'status': 'pass', 'details': '逐一核對12項主張；共享結構、軟目標、兩種品質指標、固定點與停止訊號均保留適用範圍。', 'claim_ids': [c['id'] for c in claims]},
        'numeric_verification': {'status': 'pass', 'details': '人工argmin/hinge程式與四個實測格均親跑核對；2956題/1478家族及correct整數一致，CE浮點差僅約1e-8，正文顯示精度吻合。', 'claim_ids': ['c1','c4','c5','c6','c7']},
        'figure_consistency': {'status': 'not_applicable', 'details': '凍結14.16本節無SVG或其他必要圖引用；上一節的實測摘句直接查原始分數，沒有聲稱審閱14.15的圖或頁面排版。', 'claim_ids': []},
        'source_verification': {'status': 'pass', 'details': '真正讀LoopUS v1 §3.1/3.2/Appendix B/E、官方PyTorch2.14.1原始碼與scikit-learn1.7.2原文；本課證據另核對原始final-report/settings/runner與有限實作定位，沒有用來源庫摘要充當原文。', 'claim_ids': [c['id'] for c in claims]},
        'limitations': {'status': 'pass', 'details': '區分人工示意、有限CPU查算、既有L4訓練收據與新題定理。單seed有限任務實測僅證反例；LoopUS原式SiLU不同於通用hinge，且勢能下降需額外假設。無GPU重訓或formal test重啟，未驗證網站版面。', 'claim_ids': [c['id'] for c in claims]},
    },
    'review_scope': 'Only frozen14.16 substantively reviewed; necessary14.15 read solely to locate the empirical primary source. No author plans, author dispositions or other/old review conclusions read. Source-receipts candidate location consulted but judgments not used as authority. No curriculum, old reports or active manifest edited.',
    'sealed_at': datetime.now(UTC).isoformat(),
}
body = lesson_slices((ROOT / 'course/chapters/14.md').read_text())['14.16']
errors = _validate(ROOT, ROOT / 'course/chapters/14.md', '14.16', body, report, {TASK: ['14.16']}, set())
assert not errors, errors
destination = ROOT / 'docs/course-revision-20261016/reviews/technical-14.16-backfill-original-01.json'
assert not destination.exists(), 'Do not overwrite a sealed original'
save(destination, report)
validation_path = HERE / '14.16-validation.json'
save(validation_path, {'reviewer_task': TASK, 'report': relative(destination), 'report_sha256': digest(destination), 'validator': 'scripts.check_technical_reviews._validate', 'source': 'course/chapters/14.md', 'lesson_id': '14.16', 'reviewer_owners': {TASK: ['14.16']}, 'reader_tasks': [], 'errors': errors, 'meaning': 'Format/hash/identity self-check only; substantive checks were performed before sealing.'})
print(json.dumps({'sealed_report': relative(destination), 'sha256': digest(destination), 'verdict': report['verdict'], 'claims': len(claims), 'validation_errors': errors}, ensure_ascii=False))
