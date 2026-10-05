"""Original 1.14 reviewer narrows an ambiguous cross-section report scope.

This saves the reviewer's own initial report and current raw source bytes.
It does not modify the course, train or execute any model.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
REL = OUT.relative_to(ROOT).as_posix()
TARGET = ROOT / 'docs/technical-reviews/1.14.json'
def sha(raw):
    return hashlib.sha256(raw).hexdigest()

original_report = TARGET.read_bytes()
original_hash = sha(original_report)
history = ROOT / 'docs/technical-reviews/history' / ('phase4-1_14-own-initial-' + original_hash + '.json')
history.parent.mkdir(parents=True, exist_ok=True)
if history.exists():
    assert history.read_bytes() == original_report
else:
    history.write_bytes(original_report)

report = json.loads(original_report)
assert report['reviewer_task'] == '/root/phase4_factual_coordinator/factual_1_14'
assert report['reviewer_context'] == 'fresh'
claim = next(c for c in report['claims'] if c['id'] == 'C3')
old_scope = claim['scope']
assert old_scope == '本段唯一fence與小變化；和T.3 helper中另有EOS0分開。此处不計算梯度、不更新參數、不做評測。'

folder = OUT / 'scope-recheck'
folder.mkdir(exist_ok=True)
raw = (ROOT / 'course/training.md').read_bytes()
headers = list(re.finditer(rb'(?m)^## [^\r\n]+', raw))
i = next(i for i, h in enumerate(headers) if h[0].startswith(b'## T.3 '))
section = raw[headers[i].start():headers[i+1].start()]
(folder / 'T.3.current.raw.md').write_bytes(section)
sources = {'course/training.md#T.3': {'sha256': sha(section), 'actual_inspection': '目前T.3全節L62–122；主體是simple/字元表與MLP，補充L99–111另用text_foundation Transformer/infer.py與byte JSON例子，所以不能把EOS0泛稱整個T.3。'}}
for path, locator in [
    ('scripts/course_experiments/text.py', '親讀_simple_examples L102–110、_simple_sample L113–125、run_simple_models建vocabulary L131–138；各篇+0邊界，UNK1、char ID從2開始，_simple_sample在answer==0停止；這是具體simple/字元helper。'),
    ('scripts/infer_simple.py', '親讀load_simple_checkpoint L13–37、generate_simple L40–65；simple-v1字元vocabulary，邊界0/UNK1；這是C5實際公開模型推論入口之一。'),
    ('tiny_perceptron/data.py', '親讀ByteTokenizer L10–28；SPECIALS中<eos>位於index2且byte編碼+8；與simple-v1字符vocabulary是不同契約。'),
    ('scripts/infer.py', '親讀L1–66；load_tokenizer讀saved，generate傳eos_id=tok.eos_id；沒有固定套用simple-v1數字0。'),
    ('tiny_perceptron/tokenization.py', '親讀load_tokenizer L58–95；預設264詞表時使用ByteTokenizer，否則依checkpoint詞表驗證載入；不由T.3概括全部tokenizer。'),
    ('tiny_perceptron/model.py', '親讀generate L109–134；依傳入eos_id判定停止。這裡只讀契約，不推論text_foundation模型能力，不執行/下載。'),
]:
    original = (ROOT / path).read_bytes()
    snapshot = folder / path
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(original)
    sources[path] = {'sha256': sha(original), 'actual_inspection': locator}

historical = json.loads((OUT / 'simple_models.raw.json').read_text())
assert sources['scripts/course_experiments/text.py']['sha256'] == historical['code_sha256']['scripts/course_experiments/text.py']
prior_execution = json.loads((OUT / 'bounded.results.json').read_text())
assert prior_execution['public_export_inference']['public_checkpoint_sha256'] == '2a51d84afd66eb2a3d528d2447cfe978e8bbbadd2faf451525f50b900d8bddd6'
assert prior_execution['public_export_inference']['samples_via_generate_simple'][0]['generated_ids'] == [4,13,3,0]

new_scope = '本段唯一fence與小變化；停止行為只描述這個手動四字表。本段不計算梯度、不更新參數、不做評測。C5另依 scripts/course_experiments/text.py::_simple_sample 的字元 vocabulary 契約核對既有模型推論。'
claim['scope'] = new_scope
note = {
    'reviewer_task': report['reviewer_task'], 'reviewed_on': '2026-10-05',
    'trigger': '協調者指出C3.scope新增的「T.3 helper中另有EOS0」沒有精確實作/編碼範圍，要求原審閱者親核後限定。',
    'own_initial_report': history.relative_to(ROOT).as_posix(), 'own_initial_report_sha256': original_hash,
    'old_scope': old_scope, 'new_scope': new_scope,
    'finding': '初稿數字0只適用本輪C5已執行的_simple_sample/generate_simple及simple-v1字元vocabulary；當前T.3補充另有Transformer/byte編碼例子，故含糊的整節指向過寬。移除C3跨節數值說法、限定C3四字表停止行為，具體helper核對留於原有C5。',
    'version_and_encoding': {'character_helper': 'scripts/course_experiments/text.py::_simple_sample；SHA04b0a75d151b6920dd8c28b5f7b6cadf94d27a71776a7ca7952be39ae05a0b59，與revision26f34ebb5d1e237611567697d2b3ea4d64669331原結果code hash相同；普通字元由訓練vocabulary從2編號，0邊界/停止，1未知；不使用CharTokenizer物件。', 'execution_already_performed': 'bounded.results.json：固定公開revisionfbbff36990db0d95a6e0af5ecdbc593f920938d8的simple-v1公開bigram SHA2a51d8…；本輪原helper與generate_simple已執行，新增ID[4,13,3,0]。此次只重新讀契約與證據，不聲稱新推論或重訓。', 'scope_exclusion': '不將字元helper的數字0套至T.3全部內容，亦不為T.3作整節事實PASS。'},
    'current_sources_actually_read': sources,
    'substantive_claims_preserved': ['C3抽样/item/append/current回填/固定12次/句號不提前停止仍有原始CPU證據', 'C5具體字元helper的歷史分母/輸出仍完整保留；不刪claim避過schema', '本節教材/圖与source_sha256均不改'],
    'outcome': 'C3.scope修正後1.14維持pass；沒有未核實的實質教材主張。',
}
(folder / 'reinspection-note.json').write_text(json.dumps(note, ensure_ascii=False, indent=2)+'\n')
report['scope_reinspection'] = {'reviewer_task': report['reviewer_task'], 'date': '2026-10-05', 'note': REL+'/scope-recheck/reinspection-note.json', 'initial_report': history.relative_to(ROOT).as_posix(), 'outcome': note['outcome']}
for identifier, filename, description in [
    ('scope_recheck_note', 'scope-recheck/reinspection-note.json', '原審閱者親讀當前T.3及具體字符/byte helper，保存初稿問題、版本、定位、支持範圍與C3.scope修正理由；不作T.3整節PASS。'),
    ('scope_recheck_T3', 'scope-recheck/T.3.current.raw.md', '本輪複查親讀當前T.3完整raw UTF-8；用以限定報告跨節指向。'),
    ('scope_recheck_script', 'recheck_scope.py', '保存原審閱者自己的初稿歷史、當前source bytes、親讀紀錄并精確限定C3.scope；不修改教材/圖。'),
]:
    report['artifacts'].append({'id':identifier, 'path':REL+'/'+filename, 'sha256':sha((OUT/filename).read_bytes()), 'kind':'code' if filename.endswith('.py') else 'source_snapshot', 'description':description})
report['checks']['limitations']['details'] += ' 原審閱者複查C3.scope後移除含糊「T.3 helper中另有EOS0」，明確限定本段四字表；C5的字符helper依具體函式/vocabulary版本支持，不延伸至T.3的Transformer/byte補充。'
TARGET.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'old_report_sha256':original_hash,'new_report_sha256':sha(TARGET.read_bytes()),'source_sha256':report['source_sha256'],'verdict':report['verdict'],'changed':'C3.scope及真正複查證據/limitations；教材和圖未改'},ensure_ascii=False))
