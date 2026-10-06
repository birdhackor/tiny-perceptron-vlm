"""Update only this original reviewer's canonical report after actual narrow inspection."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK = '/root/phase4_factual_coordinator/factual_18_3'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_bytes())

preservation = load(HERE/'prior-preservation.json')
prior_path = HERE/'prior-canonical.opaque.json'
assert sha(prior_path) == preservation['prior_canonical']['sha256']
report = load(prior_path)
receipt_path = HERE/'current-inspection.json'
receipt = load(receipt_path)
assert report['reviewer_task'] == receipt['reviewer_task'] == TASK
assert receipt['new_substantive_issues'] == [] and receipt['events'] == []
assert receipt['current_primary']['current_primary_sha256'] == report['source_sha256']
assert receipt['new_model_cpu_execution'] is False and receipt['network_fetches'] == 0
assert receipt['result'].startswith('Current primary unchanged')
environment = load(HERE/'environment.json')
command = 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-18_3-current-callback-20261006/inspect_current.py > docs/technical-reviews/artifacts/phase4-18_3-current-callback-20261006/inspection.stdout.txt 2> docs/technical-reviews/artifacts/phase4-18_3-current-callback-20261006/inspection.stderr.txt'
commands = {
    'cwd': str(ROOT), 'shell':'bash', 'login':False,
    'inspection':{'command':command,'exit_code':0,'code':'inspect_current.py','stdout':'inspection.stdout.txt','stderr':'inspection.stderr.txt','scope':'Pure Python files/AST/hashes and named saved raw leaves. No torch import or model execution, network, downloads, training or rendering.'},
    'canonical_update':{'command':'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-18_3-current-callback-20261006/update_canonical.py','code':'update_canonical.py','scope':'Only own report mutation after actual current inspection; prior canonical preserved opaque first.'},
    'prior_preservation':{'operation':'Read original canonical and known own prior artifact files as raw bytes; copy canonical/proof/history bytes, then write SHA/byte manifests before latest instructions and source inspection. Original old directory remains unchanged.','manifest':'prior-preservation.json'},
}
(HERE/'commands.json').write_text(json.dumps(commands,ensure_ascii=False,indent=2)+'\n')

current_artifact_id = 'current-inspection-20261006'
execution_artifact_id = 'current-inspection-execution-20261006'
for index, path in enumerate(sorted(HERE.rglob('*')),1):
    if not path.is_file() or '__pycache__' in path.parts:
        continue
    name=path.relative_to(HERE).as_posix()
    if name.startswith(('canonical-update','checker')):
        continue
    assert not path.is_symlink() and path.suffix != '.pt'
    identifier = (current_artifact_id if name=='current-inspection.json' else
                  execution_artifact_id if name=='inspection.stdout.txt' else
                  f'current-file-{index}-20261006')
    kind='code' if path.suffix=='.py' else 'source_snapshot'
    artifact={'id':identifier,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),
              'kind':kind,'description':'本人18.3 current callback實際保存的原稿、必要ctx、opaque history、source/hash inspection或方法證據：'+name}
    if name=='inspection.stdout.txt':
        artifact.update(kind='execution',command=command,result='exit0；親核current原稿與必要context，43份原artifact精確hash相符、必要方法/source scope重讀、原測量leaves回算；無torch/model/GPU/train/download/render。',environment=environment)
    report['artifacts'].append(artifact)

report['sources'].append({'id':'current-inspection-execution','kind':'execution','title':'同一原審閱者本人18.3 current narrow callback實際檢查','verified':True,'artifact_id':execution_artifact_id})
locations = {
    'c1':('CURRENT_PRIMARY_COMPLETE_READ;REUSED_MINILLM_ORIGINAL_INTRO;CURRENT_METHOD _fit_text','現版文字黑盒訊號仍一致；親重讀原MiniLLM分類和相同CE原方法，不從舊判定取得科學答案。'),
    'c2':('current-primary bytes == original section;REUSED_PRIOR_ACTUAL_CPU_STDOUT 保留筆數1/2','正文與原fence未變，原code/command/stdout/env精確hash已核；沿用先前本人原fence及answer變化的真執行，沒有再跑。'),
    'c3':('NECESSARY_CONTEXT7.11,6.1,7.3;CURRENT_METHOD ByteTokenizer/render_chat/ModelConfig/_example;REUSED_ORIGINAL_AUTHORITY tokenizers/pytorch','真正必要前文切片重讀且與原切片bytes相同，当前共享264-ID/specials/遮罩方法契約和原權威映射一致。'),
    'c4':('CURRENT_METHOD _hard_targets;REUSED_ORIGINAL_AUTHORITY pytorch;SAVED_RAW_LEAF_RECHECK','現版品質與來源/成本規約支持範圍相同；只核教師調用、保存方法與錯答案原葉，未用作者extra說明證明。'),
    'c5':('SAVED_RAW_LEAF_RECHECK;SAVED_RAW_TARGET_RECORD_LEAVES','只按保存原ID/gold/EOS/targets葉重算45/45、180/185、230EOS與5筆+10日；原資料與指紋未變，不是新生成或新score。'),
    'c6':('CURRENT_METHOD _example/_hard_targets/_distill_case/generate/generation_report;REUSED_PRIOR_ACTUAL_CPU_STDOUT CONTROLLED_GENERATION/TOKEN_CONTRACT','当前原方法精確相同；保留原受控返回值/無EOS/空目標/控制ID的既有本人CPU證據及限制，未重跑模型。'),
    'c7':('SAVED_MATCHED_TRAINING_AND_SAMPLES_LEAVES;CURRENT_METHOD _fit_text/_distill_case;prior_cpu_reuse','再核原init/batch/final SHA及保存samples/分母一致；權重等同仍以原parameter SHA與先前45筆x/y真檢查支持，未載.pt或重新訓練。'),
    'c8':('REUSED_GSM8K_ORIGINAL_PROVENANCE','原OpenAI immutable README指紋相同，親重讀人寫資料/contractor steps的必要段落；不下載資料、不把provenance名稱當生成操作。'),
}
for claim in report['claims']:
    locator,supports=locations[claim['id']]
    claim['evidence'].append({'source_id':'current-inspection-execution','locator':locator,'supports':supports})
    for identifier in [current_artifact_id,execution_artifact_id]:
        if identifier not in claim['artifact_ids']:
            claim['artifact_ids'].append(identifier)
    if 'verification' in claim:
        claim['verification']['current_callback']='2026-10-06 same original reviewer；unchanged code/primary與必要slice重查；原CPU內容指紋及scope核後沿用，callback只有saved raw leaves/hash/AST inspection，未重跑model CPU。'

current = {
    'artifact_id':current_artifact_id,'path':receipt_path.relative_to(ROOT).as_posix(),'sha256':sha(receipt_path),
    'reviewer_task':TASK,'reviewer_identity_scope':'Same original fresh reviewer callback; not a new fresh agent or new original experiment.',
    'checked_on':'2026-10-06','current_primary_sha256':report['source_sha256'],
    'current_primary_path':(HERE/'current-primary.md').relative_to(ROOT).as_posix(),
    'current_primary_read_scope':f"Complete current18.3, lines{receipt['current_primary']['current_primary_first_line']}-{receipt['current_primary']['current_primary_last_line']}; no chapter introduction or18.4 inspected in this callback.",
    'intro':None,'figure_sha256':{},
    'necessary_context':[{'source':c['source'],'read_scope':c['read_scope'],'first_line':c['first_line'],'last_line':c['last_line'],'path':c['slice_path'],'necessary_slice_sha256':c['necessary_slice_sha256'],'section_sha256_hashed_only':c['section_sha256_hashed_only'],'necessary_prefix_equal_to_original':c['necessary_prefix_equal_to_original']} for c in receipt['current_primary']['necessary_context']],
    'prior_canonical':copy.deepcopy(preservation['prior_canonical']),
    'reuse_scope':receipt['scientific_support_boundary'],'prior_cpu_reuse':receipt['prior_cpu_reuse'],
    'events':[],'unresolved_issues':[],
}
report['current_inspection']=current
report['read_scope']['current_callback']=copy.deepcopy(current)
report['verdict']='pass'
for issue in report['issues']:
    issue['current_callback_recheck']='2026-10-06：本人完整重讀未變18.3與必要context前綴，親核current方法/權威來源支持及原證據hash；原ID契約仍一致，無新未定主張。'
report['checks']['factual_accuracy']['details']+=' 2026-10-06本人narrow callback完整重讀current18.3及真正必要ctx；不機械重查整節ctx或整章。'
report['checks']['source_verification']['details']+=' Current callback親核43份正式原artifact精確hash，原權威支持段與方法界線重讀、只具名raw leaves；明示沿用而未重fetchpaper/modelCPU/render。'
report['checks']['limitations']['details']+=' Current callback：7.11/6.1完整ctx哈希有變，但實際依賴的開頭slice完全相同且已親重讀；原完整章frozen input保持其原SHA與真快照。'

canonical=ROOT/'docs/technical-reviews/18.3.json'
canonical.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
actual=load(canonical)
assert actual['reviewer_task'] == TASK
assert actual['current_inspection']['reviewer_task'] == TASK
assert actual['source_sha256'] == sha(HERE/'current-primary.md')
assert actual['current_inspection']['sha256'] == sha(receipt_path)
assert actual['verdict']=='pass'
assert actual['frozen_input']==load(prior_path)['frozen_input']
print(json.dumps({'canonical_report':canonical.relative_to(ROOT).as_posix(),'report_sha256':sha(canonical),'reviewer_task_asserted':TASK,'current_primary_sha256':actual['source_sha256'],'current_inspection_artifact':{'id':current_artifact_id,'path':receipt_path.relative_to(ROOT).as_posix(),'sha256':sha(receipt_path)},'prior_canonical_opaque':preservation['prior_canonical'],'necessary_context':current['necessary_context'],'verdict':'pass'},ensure_ascii=False))
