"""Write only this reviewer's new canonical report; never read the prior report."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREFIX = HERE.relative_to(ROOT).as_posix()
TASK = '/root/phase4_factual_coordinator/factual_18_3'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rel(name):
    return PREFIX + '/' + name

spec = importlib.util.spec_from_file_location('section_facts', ROOT / 'docs/review-tools/section_facts.py')
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
body, whole, line = facts.original_section(ROOT / 'course/chapters/18.md', '18.3')
assert body == (HERE / 'section.md').read_bytes()
assert hashlib.sha256(body).hexdigest() == 'a69d6880083c580d589b55cab996392e9cf9ded6f03b4bd9c3f46d42b91b872d'
frozen = HERE / 'inputs/course/chapters/18.md'
assert sha(frozen) == '49a0ea53a85d5b1ca8c13bbf22e4b36010cf0bde90cfb1ec8ea14e61769f2b9d'

pointers = {
    'code/docs/course-experiments/results/distillation.json': [
        '/revision', '/seed', '/torch_version', '/python_version', '/device', '/code_sha256',
        '/artifacts (file paths, byte counts and SHA-256 only)',
        '/results/tasks/attributes/teacher_provenance (except metadata)',
        '/results/tasks/style_transfer/teacher_provenance (except metadata)',
        '/results/tasks/attributes/data (source,sha256,counts,families,family_intersections)',
        '/results/tasks/style_transfer/data (source,sha256,counts,families,family_intersections)',
        '/results/tasks/attributes/hard_target_generation (raw generation measurements and full audit)',
        '/results/tasks/style_transfer/hard_target_generation (raw generation measurements and full audit)',
        '/results/tasks/attributes/runs/w16_{ce,teacher_hard}/training/{initialization_sha256,final_sha256,batch_plan_sha256,steps,training_examples,effective_supervised_tokens}',
        '/results/tasks/attributes/runs/w32_{ce,teacher_hard}/training/{initialization_sha256,final_sha256,batch_plan_sha256,steps,training_examples,effective_supervised_tokens}',
        '/results/tasks/attributes/runs/w{16,32}_{ce,teacher_hard}/{validation,test} (raw metrics, criteria and generated_samples)',
        '/results/tasks/style_transfer/runs/w32_{ce,teacher_hard}/training/{initialization_sha256,final_sha256,batch_plan_sha256,steps,training_examples,effective_supervised_tokens}',
    ],
    'sources/sft-hard-targets.json': ['/records', '/audit'],
    'sources/sft-experiment.json': ['/revision', '/artifacts (model.pt locator/bytes/SHA only)', '/code_sha256'],
    'sources/style-experiment.json': ['/revision', '/artifacts (model.pt locator/bytes/SHA only)', '/code_sha256'],
}
inspection = {
    'reviewer_task': TASK,
    'independence': 'Fresh one-section reviewer; no old technical/reader report bodies, author repair summaries or review summaries read. Locator indexes used only for original-file paths, URLs and hashes.',
    'chapter_read_scope': {'source': 'course/chapters/18.md', 'read': 'Chapter introduction and complete 18.1, 18.2, 18.3; no later section text read.', 'last_line': line + body.count(b'\n') - 1},
    'necessary_pretext': ['course/chapters/07.md#7.3 (complete)', 'course/chapters/07.md#7.11 (complete)', 'course/chapters/06.md#6.1 (complete)'],
    'json_read_policy': 'Top-level keys/types first; then named raw/provenance pointers. No notes, review, results scope strings, matched_students explanation, limitations result summaries, or correction summaries read.',
    'json_pointers': pointers,
    'code_inspection': {
        'scripts/course_experiments/compression.py': 'AST function/statement/dictionary-key locators first; then source lines 1-28,31-44,47-52,55-146,216-284,337-417,574-627,720-802. No extra result interpretation dictionary strings read.',
        'code/compression-recorded.py': 'git show 5af615e5d7c9642afee800390fa072257f895d0c; whole-file SHA matches original result code_sha256. _prompt,_example,_hard_targets,_fit_text,_distill_case AST compared equal to current implementation. Personally read _example79-97,_hard_targets574-627,_json31-33,_sync36-40,_sha43-44; other equal methods checked against current method source. Selected original methods compiled unchanged into bounded CPU test.',
        'tiny_perceptron/data.py': 'lines1-86, including SPECIALS/ByteTokenizer/CharTokenizer/render_chat/pad_batch; SHA equals the result-recorded source SHA for distillation and both actual teachers.',
        'tiny_perceptron/model.py': 'ModelConfig15-28, loss_sum92-100,masked_loss103-105,generate109-131; SHA equals recorded experiment source.',
        'tiny_perceptron/tokenization.py': 'generation_report116-142; SHA equals recorded experiment source.',
        'tiny_perceptron/training.py': 'AST checkpoint function locators, load_checkpoint79-103.',
        'scripts/course_experiments/common.py': 'new_lm45-47,text_examples77-97,load_lm307-309; actual teacher original fit_lm text_examples call125 checked.',
        'scripts/course_experiments/text.py': 'run_sft556-573, plus AST calls in original sft revision; original run_sft AST matches current.',
        'scripts/course_experiments/behavior.py': 'run_style108-155, plus AST calls in original style revision; original run_style AST matches current.',
    },
    'external_original_inspection': {
        'MiniLLM': 'Original PDF arXiv2306.08543v6, first page version/title and page2 Introduction; first paragraph classifies text-only black-box KD and prompt-response finetuning. First ~10000 chars of pdftotext inspected, including original paper abstract and figure1 caption; paper measurement results not used to support this toy experiment.',
        'PyTorch': 'Official original torch/nn/functional.py at installed git5c4886908584029761b579af026dcfb627c84070, cross_entropy3478-3569.',
        'HuggingFace tokenizers': 'Official BPE model.rs v0.23.2; get_vocab453-454,593-598,token_to_id614-615,id_to_token618-619.',
        'CPython': 'Official v3.13.5 Doc/library/json.rst dumps248-255 and ensure_ascii497-499.',
        'GSM8K': 'Official OpenAI grade-school-math README fully read; Dataset Details and Socratic Dataset provenance specifically checked. Immutable commit3101c7d5072418e28b9008a6636bde82a006892c fetched and verified equal to initially fetched master snapshot.',
    },
    'raw_cache_copy': {'original_path': 'outputs/private-review-artifacts/original-distillation/sft-hard-targets.json', 'permanent_copy': rel('sources/sft-hard-targets.json'), 'sha256': sha(HERE / 'sources/sft-hard-targets.json'), 'verification': 'Original bytes and permanent copy match; SHA also equals distillation manifest hard_target_generation/sha256 and artifacts entry. Formal references use the permanent copy.'},
    'figure_check': 'No image or SVG reference appears in18.3. The learning question is answered by explicit messages/roles/answers and ID rules; no spatial layout or hidden visual material is needed. Rendering/viewing not applicable; no visual render claimed.',
    'stop_events': [],
    'harness_correction': 'First CPU harness failed because reviewer omitted _answer from selected original dependency nodes. Initial code/stdout/stderr retained, then unchanged original _answer added and all assertions passed. No textbook error inferred.',
}
(HERE / 'inspection.json').write_text(json.dumps(inspection, ensure_ascii=False, indent=2) + '\n')

env = json.loads((HERE / 'environment.json').read_text())
command = json.loads((HERE / 'commands.json').read_text())['command']
artifacts = []
artifact_ids = {}
for path in sorted(HERE.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts:
        continue
    assert not path.is_symlink() and path.suffix != '.pt'
    name = path.relative_to(HERE).as_posix()
    if name.startswith('canonical-write') or name.startswith('checker'):
        # These proofs are produced after the canonical report hash is fixed.
        continue
    identifier = 'a' + str(len(artifacts) + 1)
    artifact_ids[name] = identifier
    kind = 'code' if path.suffix == '.py' else 'source_snapshot'
    if name == 'stdout.txt':
        kind = 'execution'
    artifact = {'id': identifier, 'path': rel(name), 'sha256': sha(path), 'kind': kind,
                'description': '本輪本人保存的原始輸入／方法／來源／執行證據：' + name}
    if kind == 'execution':
        artifact.update(command=command, result='exit0；原fence及必要變化、230筆原測量、45筆原records、EOS/空目標/控制ID/回答遮罩斷言全部通過；沒有訓練或新score評測。', environment=env)
    artifacts.append(artifact)

sources = []
def repo_source(id0, title, name, version, note):
    sources.append({'id': id0, 'kind': 'repository_code', 'title': title, 'path': rel(name),
                    'sha256': sha(HERE / name), 'version': version, 'verified': True, 'inspection_note': note})

def authority(id0, kind, title, url, version, reason, note, snapshot):
    sources.append({'id': id0, 'kind': kind, 'title': title, 'url': url, 'version': version,
                    'authority_reason': reason, 'accessed_on': '2026-10-05', 'verified': True,
                    'checked_original': True, 'inspection_note': note,
                    'snapshot_path': rel(snapshot), 'snapshot_sha256': sha(HERE / snapshot)})

authority('minillm', 'paper', 'MiniLLM: On-Policy Distillation of Large Language Models', 'https://arxiv.org/pdf/2306.08543v6', 'arXiv2306.08543v6;31Jan2026', '論文作者的arXiv原論文，親核第一頁版本。', '第2頁Introduction第一段，親讀text-only black-box KD與prompt-response finetuning的原描述。', 'sources/fact_finish_g_4_minillm_original.pdf')
authority('pytorch', 'official_source', 'PyTorch cross_entropy原始API契約', 'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py', 'Installed PyTorch2.14.1+cpu;git5c4886908584029761b579af026dcfb627c84070', 'pytorch/pytorch官方原始庫且git版本和實跑wheel一致。', '親讀cross_entropy3478-3569：class-index目標、[0,C)範圍、ignore_index不計梯度及非忽略分母。', 'sources/pytorch-functional.py')
authority('tokenizers', 'official_source', 'HuggingFace BPE詞表與ID映射原碼', 'https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/tokenizers/src/models/bpe/model.rs', 'tokenizersv0.23.2;方法原碼查閱，未用該套件運行本課byte測試', 'huggingface/tokenizers官方標籤原碼。', '親讀get_vocab453-454/593-598、token_to_id614-615、id_to_token618-619；ID來自該model的詞表。', 'sources/tokenizers-bpe-model.rs')
authority('json', 'official_docs', 'CPython json.dumps及ensure_ascii文件', 'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/json.rst', 'CPythonv3.13.5，和實跑Python一致', 'python/cpython官方版本文件。', '親讀dumps248-255、ensure_ascii497-499；序列化str，False時非ASCII照原文輸出。', 'sources/python-json.rst')
authority('gsm8k', 'official_docs', 'OpenAI GSM8K原始資料說明與人寫解答來源', 'https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/README.md', 'Immutablegit3101c7d5072418e28b9008a6636bde82a006892c', '發布GSM8K的OpenAI原始庫。', '親讀全文README；Dataset Details的人寫grade school word problems、question/answer格式，以及Socratic段落contractor-provided解題步驟。未下載訓練資料。', 'sources/gsm8k-readme.md')
repo_source('data', '本課byte詞表、角色、回答遮罩', 'code/tiny_perceptron/data.py', 'SHA matches recorded distillation and both teacher runs', '親讀1-86，264=256+8，ID0..7依序pad,bos,eos,user,assistant,image,audio,system；byte+8；render_chat只監督assistant內容與EOS。')
repo_source('model', '本課模型詞表預設、CE及生成停止', 'code/tiny_perceptron/model.py', 'SHA matches recorded distillation source', '親讀ModelConfig15-28、loss_sum92-100、masked_loss103-105、generate109-131。')
repo_source('generation', '保留控制ID的生成審核', 'code/tiny_perceptron/tokenization.py', 'SHA matches recorded distillation source', '親讀generation_report116-142，控制ID與EOS分別標記。')
repo_source('compression', '指定版蒸餾原實作', 'code/compression-recorded.py', 'git5af615e5d7c9642afee800390fa072257f895d0c;SHA28d8ce258369672d71a24f7efbcbf68f3e5e8114e073620aba91c4d807043a44', inspection['code_inspection']['code/compression-recorded.py'])
repo_source('measurements', '指定GPU原始蒸餾結果及逐筆審核', 'code/docs/course-experiments/results/distillation.json', 'original experiment revision5af615e5d7c9642afee800390fa072257f895d0c;seed42;torch2.14.1+cu126;python3.13.3', '先列上層keys/types，再只讀inspection.json具名raw/provenance pointers。CPU僅重算原測量，未重訓。')
repo_source('sft_targets', '原始45筆SFT教師硬回答records及audit', 'sources/sft-hard-targets.json', 'Original artifactSHA8d09066e9d8390a08220219bb8696a254c1dc9b4ed4599f346d5c87fc3c9ab51', '先列records/audit keys/types，再親讀兩者；原件/永久副本/正式manifestSHA相同。')
repo_source('sft_teacher', '屬性教師原始訓練方法', 'code/sft-text.py-recorded.py', 'sft revisiona253d1262bf5f361f9ac4e19232ae752f0ecc7a3', '原版run_sft AST和現版相同，親核new_lm與mode=sft fit_lm呼叫。')
repo_source('style_teacher', '風格教師原始訓練方法', 'code/style-behavior.py-recorded.py', 'style revisionae7bbbf95537d228a44810041d2a9e978360d369', '原版run_style AST和現版相同；親核mode=sft fit_lm與共享new_lm呼叫。')
repo_source('teacher_common', '兩個原始教師的tokenizer與SFT準備方法', 'code/sft-common.py-recorded.py', 'SHA matches both original teacher results;df08fb4d378170e930bb5deb50b97749c33e5d97d0fbd3ce77ba916b12b07206', '親核new_lm45-47，text_examples77-97，fit_lm125呼叫；SFT tokenizer默認ByteTokenizer，render_chat做相同ID映射。')
sources.append({'id': 'cpu', 'kind': 'execution', 'title': '本人有界CPU程式與原測量核對', 'verified': True, 'artifact_id': artifact_ids['stdout.txt']})
sources.append({'id': 'reasoning', 'kind': 'derivation', 'title': '硬答案、文字轉ID及錯目標的支持範圍', 'verified': True,
                'details': '文字可被學生詞表重新encode，故沒有教師ID必須相同的要求；直接抄教師ID則要求同一ID代表同一token。CE=-log q(label)，把錯誤答案當label會鼓勵錯答案；模仿不是外部真值。生成調用教師、驗證及持久化各有操作成本，本文無確切成本數字。'})

def evidence(id0, locator, supports):
    return {'source_id': id0, 'locator': locator, 'supports': supports}

def executed(expected, observed, details, denominators=None):
    value = {'method': 'executed', 'expected': expected, 'observed': observed, 'details': details}
    if denominators is not None:
        value['denominators'] = denominators
    return value

claims = [
    {'id': 'c1', 'kind': 'concept', 'statement': '只有教師回答文字仍可用prompt-response SFT做黑盒蒸餾，文字不能恢復未保存的完整候選分布。', 'location': '18.3首段及回答/answer字段段落', 'scope': '方法與可取得的訊號；不保證教師真值或學生能力提升。', 'status': 'verified',
     'evidence': [evidence('minillm', 'p2 Introduction first paragraph', '黑盒只取得生成文字，並用API生成prompt-response pairs微調學生。'), evidence('pytorch', 'cross_entropy3493-3504,3520-3522', 'class-index硬目標與class-probability完整分布是不同輸入。'), evidence('compression', '_fit_text350-373', '無teacher_cache時使用普通CE，未從answer字段推造logits。')], 'artifact_ids': []},
    {'id': 'c2', 'kind': 'software', 'statement': '人工加法記錄以sum/str比較篩掉5，印出一筆assistant=4且provenance標為人工；改第二筆4後印出兩個重複record，沒有寫檔或訓練。', 'location': '18.3 Python fence與後續輸出/練習說明', 'scope': '普通相連Python資料整理API合組查核；只證明篩選、JSON輸出和重複資料，不是教師採集或訓練。', 'status': 'verified',
     'evidence': [evidence('cpu', 'stdout ORIGINAL_FENCE and FENCE_VARIATION_CHANGE_ONLY_SECOND_ANSWER;verify.py43-59', '本人執行原fence及只把第二answer5改4的變化。'), evidence('json', 'json.rst248-255,497-499', 'dumps回傳JSON str，ensure_ascii=False讓中文原樣输出。')], 'artifact_ids': [artifact_ids['stdout.txt'], artifact_ids['fence-1.py'], artifact_ids['fence-1-variation.py']],
     'verification': executed('2+2真值4；原版保留1，改第二answer後保留2，重複對話。', '原版1筆、變化2筆且兩record完全相同；teacher_revision=null。', '原fence編譯執行，未呼叫教師或模型；只print，沒有建立資料檔。精確整數與文字相等。')},
    {'id': 'c3', 'kind': 'concept', 'statement': '回答文字可由學生自己的tokenizer重切分而不要求教師詞表相同；學生只計assistant回答目標，user是可見前文。直接沿用原始ID的本次對照使用相同264-ID byte詞表與8個specials。', 'location': '18.3學生訓練段及選讀第二段的共享词表前提', 'scope': '外部文字重切分和本次原始ID支線分開；詞表尺寸相同本身不足以保證ID語義相同。', 'status': 'verified',
     'evidence': [evidence('minillm', 'p2 Introduction first paragraph', '黑盒可取得文字即可微調。'), evidence('tokenizers', 'get_vocab593-598,token_to_id614-615,id_to_token618-619', 'token與ID映射屬各模型詞表。'), evidence('pytorch', 'cross_entropy3500-3504', '-100不直接貢獻loss梯度。'), evidence('data', 'SPECIALS11;ByteTokenizer14-28;render_chat54-68', '同一byte+8與0..7specials以及回答遮罩。'), evidence('teacher_common', 'new_lm45-47,text_examples77-97,fit_lm125', '兩個實際教師採用同一ByteTokenizer/render_chat。'), evidence('compression', '_prompt131-142,_example79-97,_distill_case770-784', '學生默認264詞表，用同一角色與raw hard ID。'), evidence('cpu', 'TOKEN_CONTRACT,MASK_AND_LOSS', '4在byte tokenizer為60，char為1；回答label非忽略格有梯度、其餘logit梯度0。')], 'artifact_ids': [artifact_ids['stdout.txt']]},
    {'id': 'c4', 'kind': 'concept', 'statement': '教師答案需依任務規則查品質並保存來源，模仿程度不能當真值；教師生成、驗證、保存也屬於流程成本。', 'location': '18.3品質篩選、provenance與成本段', 'scope': '方法規約與操作成本構成；不提供金額、教師性能保證或因果提升估計。', 'status': 'verified',
     'evidence': [evidence('pytorch', 'cross_entropy3488-3504', 'CE直接依label要求模型提高該答案，沒有自動糾正錯label的外部真值機制。'), evidence('reasoning', 'CE=-log q(label);inspection.json支持範圍', '錯目標與可得真值不同；由公式直接得模仿不足以做正確性標準。'), evidence('compression', '_hard_targets582-606,611-625', '教師生成與審核/JSON持久化確實是獨立操作，原始ID和正確旗標逐筆保存。')], 'artifact_ids': []},
    {'id': 'c5', 'kind': 'empirical', 'statement': '原屬性硬回答45/45正確且EOS；原風格硬回答180/185正確，5筆已知日期回答日加10，包含2026-10-06→2026-10-16；合共230個目標均EOS、無非法控制ID或零長度。', 'location': '18.3選讀「實際硬回答、錯誤審核與正常結束」', 'scope': '指定原實驗保存的raw audit，非重新生成或重訓；完整style-hard-targets.json未在本輪單獨載入，核對的是正式結果中完整嵌入audit、同版寫檔方法與原artifact SHA/bytes。', 'status': 'verified',
     'evidence': [evidence('measurements', '/results/tasks/{attributes,style_transfer}/hard_target_generation/audit; /artifacts style-hard-targets.json locator', '逐筆teacher_ids、gold_answer、teacher_answer、EOS與測量，不用作者結果摘要。'), evidence('sft_targets', '/records;/audit', '45筆實際保存的原raw ID與旗標；永久副本hash和正式manifest匹配。'), evidence('generation', 'generation_report116-142', '核對控制token與EOS判準。'), evidence('cpu', 'RAW_HARD_TARGET_AUDIT', '本人按原ID重算內容正確、EOS、空目標與控制ID，並以date差重算五筆+10天。')], 'artifact_ids': [artifact_ids['stdout.txt']],
     'verification': executed('屬性45正確、風格180正確/5錯；總230EOS，零長度/控制ID0。', '45/45及180/185；EOS45+185=230；target tokens315+3783=4098；5筆date差均10天。', '每筆teacher_ids先按首EOS截出answer bytes，與ByteTokenizer.encode(gold)精確比；EOS均唯一且在尾，stored audit旗標與generation_report完全相符。', {'attribute_training_records':45,'style_training_records':185,'total_records':230,'attribute_target_tokens_including_eos':315,'style_target_tokens_including_eos':3783,'known_date_errors':5})},
    {'id': 'c6', 'kind': 'software', 'statement': '工具完整保存teacher_ids和實際EOS旗標，用原ID作回答標籤，不在未EOS時補假EOS；零目標會失敗並保留題數審核；普通文字decode可能隱藏控制ID。', 'location': '18.3選讀第二段', 'scope': '原實作合組契約；有界測試替換generate的返回值來查資料路徑，不宣稱重新呼叫訓練好的教師。非法控制case只核審核偵測與保留，原本230筆並無控制異常。', 'status': 'verified',
     'evidence': [evidence('compression', '_hard_targets579-627,_example80-93,_distill_case749-755', 'ID逐筆保存；尾labels直接cat/shift；空目標在_example及全流程失敗，保存全部審核不跳題。'), evidence('model', 'generate115-128', 'token/context上限和EOS停止各有真實條件，未EOS不造結束。'), evidence('data', 'ByteTokenizer.decode23-25', 'decode會忽略0..7控制ID，因此畫面文字不能當停止原因證据。'), evidence('generation', 'generation_report122-141', '原始控制ID與停止旗標分開保存。'), evidence('cpu', 'CONTROLLED_GENERATION,TOKEN_CONTRACT;verify.py', '原方法在EOS/無EOS/空目標/非法控制的四種受控輸出均保留真ID與審核，空目標label不被捏造。')], 'artifact_ids': [artifact_ids['stdout.txt']],
     'verification': executed('ID與EOS原樣保存；無EOS標籤只[60]，空目標及不一致EOS報錯；控制ID不靠普通decode推定。', '四個受控generate返回值[60,2]、[60]、[]、[60,3]全部斷言通過；無EOS未補2、零目標audit題數1/zero1；decoder的4和完整審核4<user>區別成立。', '精確執行指定版原AST函式；generate是明示stub，無模型能力評分或訓練。')},
    {'id': 'c7', 'kind': 'empirical', 'statement': '屬性硬目標原ID與真值相同，因此本次相同初始化/batch計畫下兩個寬度的CE與teacher_hard學生權重及成績相同；不能當成額外教師知識帶來的提升。', 'location': '18.3選讀首段', 'scope': '指定seed42兩個寬度16、32原實驗；權重相同由原parameter SHA支持，本輪沒有載入完整.pt比較或重訓；不外推其他seed/資料/教師。', 'status': 'verified',
     'evidence': [evidence('sft_targets', '/records,/audit', '45筆保存原record的hard IDs含EOS精確等於gold編碼。'), evidence('compression', '_example79-97,_parameter_hash47-52,_fit_text350-409,_distill_case770-790', '同一init、seed/batch plan；同label的CE目標相同；final_sha是名稱+tensor bytes hash。'), evidence('measurements', '/results/tasks/attributes/runs/w{16,32}_{ce,teacher_hard}/training and {validation,test}/generated_samples', '兩寬度init/batch/final參數hash、更新步數與原分數一致。'), evidence('cpu', 'SFT_ORIGINAL_RECORDS;CE_AND_HARD_MATCH original_measurements', '本人重建並逐筆比較45組x/y；逐樣本重算原validation/test正確、完成與EOS分母。')], 'artifact_ids': [artifact_ids['stdout.txt']],
     'verification': executed('每個width內CE/hard x/y、init/batch/final hash及留出分數精確一致。', '45/45 x/y完全相等；width16兩支val0/5,test1/10；width32兩支val2/5,test4/10；两宽度各400次更新、44985監督token，final SHA相同。', '只核原測量並重算已保存sample的raw-ID判準；模型文件容器SHA可不同，本文核的是_parameter_hash的參數内容，不是.pt容器bytes。', {'training_records':45,'matched_student_widths':2,'optimizer_steps_per_branch':400,'batch_size':16,'effective_supervised_tokens_per_branch':44985,'validation_examples_per_branch':5,'test_examples_per_branch':10})},
    {'id': 'c8', 'kind': 'concept', 'statement': 'GSM8K為公開小學程度文字算術題與人寫解答來源；改provenance名字不能把原資料答案變成教師生成答案。', 'location': '18.3GSM8K與資料來源段', 'scope': '原GSM8K資料來源；Socratic變體的模型生成subquestions和人提供steps已在原來源分開，本文沒有宣稱下載或實跑GSM8K。', 'status': 'verified',
     'evidence': [evidence('gsm8k', 'README Dataset Details;Socratic Dataset final paragraph', '人寫grade school problems，question/answer，contractor-provided解題步驟；模型生成的Socratic subquestions另作明確區分。'), evidence('reasoning', 'provenance refers to actual generation operation', '改metadata名稱不改答案生成事實。')], 'artifact_ids': []},
]

report = {
    'schema_version': 1, 'review_stage': 'technical', 'lesson_id': '18.3',
    'source': 'course/chapters/18.md#18.3', 'source_sha256': hashlib.sha256(body).hexdigest(),
    'verdict': 'pass', 'reviewer_task': TASK, 'reviewer_context': 'fresh', 'author_tasks': [],
    'figure_sha256': {},
    'frozen_input': {'path': rel('inputs/course/chapters/18.md'), 'sha256': sha(frozen), 'meaning': '當次最初保存的完整chapter18 frozen input；不是聲稱目前整章仍等於此快照。'},
    'read_scope': inspection,
    'artifacts': artifacts, 'sources': sources, 'claims': claims,
    'issues': [{'id': '18.3-Q1', 'status': 'resolved', 'question': '教師硬回答的原ID直接作學生標籤時，正文、實作、教師與學生詞表和specials是否一致？',
                'resolution': '本人核原文共用264-ID明示前提、同SHA byte tokenizer方法、兩教師指定版SFT準備、學生原例與raw ID訓練方法、230筆保存ID/EOS。均一致；外部文本重新encode路徑已另行清楚說明。無需改正文。'}],
    'checks': {
        'factual_accuracy': {'status':'pass','details':'逐項核文字黑盒蒸餾、共享ID前提、答位遮罩、品質/來源與EOS含義；待查疑點由原碼、原測量和短CPU查證解決。','claim_ids':['c1','c3','c4','c6','c8']},
        'numeric_verification': {'status':'pass','details':'2+2整數篩選1→2筆；原45/45與180/185、總230EOS、五date差10天逐筆重算；同label、init/batch/final hash與原val/test分母亦核。CPU logits軸[2,10,264]，有效目標3，loss為ln264±1e-6且ignore梯度精確0。','claim_ids':['c2','c5','c7']},
        'figure_consistency': {'status':'not_applicable','details':'18.3沒有圖或SVG引用；兩筆messages/答案、角色與raw ID前提直接由文字和程式給出，無未呈現的必要視覺素材。未宣稱渲染。','claim_ids':[]},
        'source_verification': {'status':'pass','details':'親核MiniLLMv6原PDF與官方PyTorch/tokenizers/CPython/OpenAI原來源；指定原實驗JSON先keys/types再具名raw/provenance pointers。原方法按Git指定版SHA保存，正式sft raw record副本和原件/manifest SHA一致。','claim_ids':['c1','c2','c3','c4','c5','c6','c7','c8']},
        'limitations': {'status':'pass','details':'人工示例只資料整理；原GPU測量和本輪CPU核對分開。受控generate只查原ID資料流；未訓練、無新score評測、未下載資料模型、未留.pt。權重相同是原parameter SHA與same labels證據；style full records未另載，使用正式嵌入audit+manifest+原寫檔方法。限定本次共享詞表，不推到外部不同詞表直接抄ID。','claim_ids':['c1','c2','c3','c5','c6','c7','c8']},
    },
}
output = ROOT / 'docs/technical-reviews/18.3.json'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
canonical = json.loads(output.read_text())
assert canonical['reviewer_task'] == TASK
assert canonical['source_sha256'] == hashlib.sha256(body).hexdigest()
assert canonical['verdict'] == 'pass'
print(json.dumps({'canonical_report': str(output), 'reviewer_task_asserted': TASK, 'source_sha256': canonical['source_sha256'], 'report_sha256': sha(output), 'claims':len(claims)}, ensure_ascii=False))
