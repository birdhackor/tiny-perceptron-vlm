import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DEST = Path(__file__).resolve().parent
REL = DEST.relative_to(ROOT).as_posix()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = json.loads((DEST / 'previous-report.json').read_bytes())
assert report['reviewer_task'] == '/root/p6_fact_19_2'
assert report['verdict'] == 'pass'
recheck = {
    'reviewer_task': '/root/p6_fact_19_2',
    'review_type': 'same original factual reviewer actual current-source callback',
    'checked_on': '2026-10-06',
    'previous_report_opaque_backup': {'path': f'{REL}/previous-report.json', 'sha256': sha(DEST/'previous-report.json')},
    'current_source_sha256': sha(DEST/'current-19.2.md'),
    'previous_source_sha256': report['source_sha256'],
    'actual_read_ranges': {
        'course/chapters/19.md': '完整目前19.2（74-124）；必要前文章導言與19.1（1-73）實際閱讀。前文僅為19.2語境，不新增19.1判定。',
        'course/chapters/15.md': '19.2明確引用15.13，與本人首次保存原文逐byte核相同；其方法限制適用範圍重新核對。',
        'tiny_perceptron/selftrained/model.py': 'description421-448實際重讀count/shared/expert/active算式；完整SHA與首次證據同。',
        'tiny_perceptron/model.py': 'TinyLM54-66實際重讀embedding/output/tied賦值；完整SHA同。',
        'tiny_perceptron/selftrained/dataset.py': 'train_tokenizer84-91實際重讀train-only篩选與ASCII固定規約；完整SHA同。',
        'tiny_perceptron/selftrained/tokenizer.py': '7-41實際重讀specials/OCR12/逐字编码与unknown fallback；完整SHA同。',
        'docs/review-tools/factual-reviewer-instructions.md': '本輪規約全文再次閱讀。'
    },
    'source_diff': '唯一差異是SelftrainedConfig九個既有參數改為逐行排版（增加七行）；fence外原文逐byte相同，AST忽略位置資訊精確相同。',
    'execution': '原樣執行目前完整fence；另有界CPU程序核当前section、AST、38份原證據、六份原碼及原始metadata输入完整SHA，獨立重算FFN262912與表格公式。当前fence六行输出與首次精確同。',
    'retained_evidence_scope': '初次Mixtralv1/PressWolfv3原來源、15段raw receipts pointer核對、untied/top-k/路由小變化與固定train-only字表重建證據原件均SHA相同；無新實質主張，逐claim仍適用。沒有重新訓練、heldout評分、生成或下載。',
    'claim_assessment': '重新閱讀目前每段及表格，16項實質claim均未改：4選2、shared/tied/count邊界、306883 perception、550字表、兩路最終目標/選定差異及不能作架構因果/同計算同記憶體結論。',
    'figures': '19.2無引用圖；沒有需要render/view的本節素材。前文19.1的圖不在此回查判定範圍。',
    'verdict': 'pass'
}
(DEST/'owner-recheck-record.json').write_text(json.dumps(recheck,ensure_ascii=False,indent=2)+'\n')
def add_artifact(i, f, kind, description, **extra):
    report['artifacts'].append(dict(id=i,path=f'{REL}/{f}',sha256=sha(DEST/f),kind=kind,description=description,**extra))
for i,f,kind,desc in [
    ('a_owner_current_source','current-19.2.md','source_snapshot','原owner本次完整讀取的目前19.2 UTF-8原始bytes。'),
    ('a_owner_current_frozen','current-frozen-19.md','source_snapshot','本次讀取時完整19.md frozen input；不是全章目前內容一致的宣告。'),
    ('a_owner_current_fence','current-fence.py','code','從目前19.2 fence原样擷取的實際執行程式。'),
    ('a_owner_recheck_code','recheck_current.py','code','本輪AST/文字邊界/原碼與證據SHA、原樣fence執行及獨立算式複查程式。'),
    ('a_owner_ast','fence-ast.txt','source_snapshot','實際目前fence AST，忽略行列屬性後與初次AST相同。'),
    ('a_owner_diff','section-diff.txt','source_snapshot','初次已審原始section與目前section真實diff，只改參數換行。'),
    ('a_owner_semantic','semantic-check.json','source_snapshot','真實AST、fence外byte、six source SHA及38份artifact核對結果與環境。'),
    ('a_owner_record','owner-recheck-record.json','source_snapshot','本人完整current-source callback範圍、方法、保留證據理由及判定。'),
    ('a_owner_output','current-fence-output.txt','source_snapshot','本輪目前fence真實六行stdout，與首次結果精確相同。')
]:
    add_artifact(i,f,kind,desc)
add_artifact('a_owner_execution','recheck-output.txt','execution','本次原owner实际CPU複查：原文/AST/證據scope、完整目前fence及獨立公式。',command=f'PYTHONPATH=. /workspace/tiny-perceptron-vlm/.venv/bin/python {REL}/recheck_current.py > {REL}/recheck-output.txt 2>&1',result='exit0；目前fence AST同、fence外bytes同、6原碼/38證據SHA同、目前原樣fence六行輸出同；FFN262912与表格推導精確吻合。',environment={'python':'3.13.5','torch':'2.14.1+cpu','device':'cpu','threads':'1'})
report['sources'].append(dict(id='s_owner_current',kind='execution',title='原owner本次目前原文與程式語義實際CPU複查',verified=True,artifact_id='a_owner_execution'))
locations = {
    'c1':'19.md:76','c2':'19.md:78','c3':'19.md:80-102',
    'c4':'19.md:109','c5':'19.md:110','c6':'19.md:80,111',
    'c7':'19.md:105','c8':'19.md:102,105','c9':'19.md:105',
    'c10':'19.md:112','c11':'19.md:114','c12':'19.md:114',
    'c13':'19.md:116','c14':'19.md:87,121','c15':'19.md:121','c16':'19.md:121'
}
for c in report['claims']:
    c['location'] = locations[c['id']]
    c['artifact_ids'].append('a_owner_record')
    c['evidence'].append({'source_id':'s_owner_current','locator':'owner-recheck-record.json#/claim_assessment；semantic-check.json#/ast_identical及/outside_fence_bytes_identical','supports':'本次完整閱讀此主張目前原文，確認沒有語義/數字/支持範圍變更；實作與原證據SHA均相同，原逐來源核對仍適用。'})
    if c['id'] in ['c3','c4','c5','c6','c8','c10','c12','c14']:
        c['artifact_ids'].append('a_owner_execution')
        c['verification']['details'] += ' 本次原owner已重讀目前原文和必要實作分支，執行排版後原樣fence並獨立重核算式；語義/輸出精確同，舊執行仍為原先有界驗證證據。'
for a in report['artifacts']:
    if a['id']=='a_input':
        a['description']='首次獨立審閱19.2原始UTF-8 bytes歷史快照；當前來源以a_owner_current_source為準。'
report['source_sha256'] = recheck['current_source_sha256']
report['owner_rechecks'] = [dict(record_artifact_id='a_owner_record',execution_artifact_id='a_owner_execution',source_artifact_id='a_owner_current_source',source_sha256=recheck['current_source_sha256'],previous_source_sha256=recheck['previous_source_sha256'],verdict='pass')]
report['current_recheck_frozen_input'] = dict(artifact_id='a_owner_current_frozen',sha256=sha(DEST/'current-frozen-19.md'),meaning='本次callback當時保存的整章frozen input；本節scope SHA獨立核對，不以其他小節格式變動推定內容變更。')
report['checks']['factual_accuracy']['details'] += ' 本次完整current-source owner callback確認16項claim語義不變。'
report['checks']['numeric_verification']['details'] += ' 本次重新執行目前原樣fence，六行輸出與初次精確同；重新核FFN與shared/active公式。'
report['checks']['source_verification']['details'] += ' 本次確認6來源原碼、38原證據及原始metadata輸入SHA同，保留首次原權威來源核讀紀錄。'
report['checks']['limitations']['details'] += ' 本次再次讀全部範圍說明及字表補充；fence外文字逐byte相同，無新增外推。'
(ROOT/'docs/technical-reviews/19.2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('OWNER_UPDATED',report['verdict'],report['source_sha256'])
