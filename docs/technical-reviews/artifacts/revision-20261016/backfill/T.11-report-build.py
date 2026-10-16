"""Archive the completed independent T.11 technical review."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from scripts.reading_time import lesson_slices
from scripts.check_technical_reviews import _validate

HERE = Path(__file__).parent
assert not (HERE/'T.11-archive-receipt.json').exists(), 'Completed review archive is immutable'
WORK = Path('/workspace/work/curriculum-20261016/T.11')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
observed = json.loads((HERE/'T.11-check-output.json').read_text())
body = lesson_slices((ROOT/'docs/course-revision-20261016/freeze-11/sources/training.md').read_text())['T.11']
(HERE/'T.11-source-slice.txt').write_text(body)

external = [
 {'id':'s-neurips','title':'NeurIPS Paper Checklist Guidelines','file':'neurips-checklist.html',
  'url':'https://neurips.cc/public/guides/PaperChecklist','version':'2026 website, retrieved 2026-10-16; original HTML SHA-256 pinned in receipt',
  'locator':'items 4 Experimental Result Reproducibility, 5 Open Access to Data and Code, 6 Experimental Setting/Details, 8 Experiments Compute Resource, 12 Licenses',
  'quote':'The instructions should contain the exact command and environment needed to run to reproduce the results.',
  'note':'實讀4–8及12項；列出模型存取或可重做入口、程式與資料、完整命令和環境、切分、超參數選擇、硬體／記憶體／時間。12項要求引用來源與版本、URL、授權名稱及遵守條件。支持本節紀錄欄位的目的，不保證僅有紀錄即可重現位元相同訓練。'},
 {'id':'s-hash','title':'CPython hashlib documentation','file':'hashlib.rst',
  'url':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/hashlib.rst','version':'CPython v3.13.5',
  'locator':'Hash algorithms; Usage; sha256 and hexdigest',
  'quote':'At any point you can ask it for the digest of the concatenation of the data fed to it so far using the digest() or hexdigest() methods.',
  'note':'實讀Hash algorithms與Usage，SHA256對輸入bytes計算digest；雜湊檔案內容不評估標籤、授權或模型正確性。指紋用於核對已取得內容；不解讀為由hash單獨下載檔案或絕對無碰撞證明。'},
 {'id':'s-git','title':'Git core tutorial: committing Git state','file':'gitobjects.adoc',
  'url':'https://raw.githubusercontent.com/git/git/v2.51.0/Documentation/gitcore-tutorial.adoc','version':'Git v2.51.0',
  'locator':'lines 109–116 object names; Committing Git state, lines 285–330',
  'quote':"creating a 'tree' object, and committing that 'tree' object as a 'commit' object together with an explanation of what the tree was all about",
  'note':'實讀object name及Committing Git state；tree綁定檔名、內容與權限，commit指向tree及parent。只支持有保存之受版控來源版本，不能覆蓋未追蹤資料、環境或能力。'},
 {'id':'s-split','title':'scikit-learn Cross-validation: evaluating estimator performance','file':'sklearn-cv.rst',
  'url':'https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst','version':'scikit-learn 1.7.2',
  'locator':'opening paragraphs, especially lines 59–82: hyperparameter tuning, validation set and final test',
  'quote':'training proceeds on the training set, after which evaluation is done on the validation set, and when the experiment seems to be successful, final evaluation can be done on the test set.',
  'note':'實讀開頭至Computing cross-validated metrics；對test調參會讓資訊洩漏，validation可供選設定，另留final test。沒有把本節要求誤讀成特定K-fold算法或所有資料皆獨立。'},
 {'id':'s-tool-flow','title':'OpenAI function calling guide','file':'openai-function-calling.html',
  'url':'https://developers.openai.com/api/docs/guides/function-calling','version':'2026-10-16 live official guide; original HTML SHA-256 pinned in receipt',
  'locator':'The tool calling flow, five high-level steps; Handling function calls',
  'quote':'Execute code on the application side with input from the tool call; Make a second request to the model with the tool output; Receive a final response from the model (or more tool calls).',
  'note':'實讀五步flow與Handling function calls，區分模型call、程式執行、回傳結果與後續模型回答；支持操作完成與單生成請求的差別，不替本課工具正確率背書。'},
]
receipts=[]
sources=[]
for item in external:
    receipts.append({k:v for k,v in item.items() if k not in ('id','note')} | {'accessed_on':'2026-10-16','original_sha256':sha(WORK/item['file']),'inspection_note':item['note']})
    sources.append({'id':item['id'],'kind':'official_docs','title':item['title'],'url':item['url'],
                    'version':item['version'],'verified':True,'checked_original':True,'accessed_on':'2026-10-16',
                    'authority_reason':'由該軟體／會議／API維護者發行的原始官方文件；核對範圍限於inspection_note。',
                    'inspection_note':item['locator']+'；'+item['note']})
write(HERE/'T.11-original-source-receipts.json',receipts)

def repository(sid,path,title,note):
    sources.append({'id':sid,'kind':'repository_code','title':title,'path':path,'sha256':sha(ROOT/path),
                    'version':'2026-10-16 actual file bytes; SHA-256 pinned','verified':True,'inspection_note':note})

repository('s-final','docs/lesson-products/results/final-report-v1.json','B/C fixed final report v1',
           '讀b[5].boxes/sum、c各question after、shifts.c_wording、predictions、gates、scope；由自身腳本逐題原始分數重算相應分母。未親跑formal test。')
repository('s-settings','docs/lesson-products/results/settings-freeze-v1.json','Pretest product settings freeze',
           '讀protocol核心門檻、calibration溫度、test_outcomes_observed=false、recorded_at與固定資料SHA；B主要門檻是5輪兩盒accuracy>=.90，三數加法target .85。')
repository('s-b-run','docs/lesson-products/results/initial-b-execution.json','Actual B NVIDIA L4 runner receipt',
           '核status completed、gpu_name L4、training隨機起點與無resume、1000次更新與選定1000、weights_changed；和本機完整training-receipt逐欄相等。這是既有實際GPU一手收據，非本審閱者親跑GPU。')
repository('s-c-run','docs/lesson-products/results/initial-c-execution.json','Actual C NVIDIA L4 runner receipt',
           '核status completed、gpu_name L4、training隨機起點與無resume、800次更新與選定800、weights_changed；和本機完整training-receipt逐欄相等。這是既有實際GPU一手收據。')
repository('s-train','tiny_perceptron/lesson_products/train.py','B/C genuine training initialization',
           '讀train中fresh分支建LoopedSolver或CandidateScorer、AdamW及seeded sampler、初始／選定weights digest；未載入成熟模型。分開read_dataset train與policy。')
repository('s-model','tiny_perceptron/lesson_products/model.py','Local B/C neural implementation',
           '讀Tokenizer、Config、Block、LoopedSolver、CandidateScorer；普通nn.Embedding/Linear隨機模組，B循環同一shared block，C獨立候選評分器，無MoE／外部pretrained loader。未對本節以外架構性質背書。')
repository('s-release','tiny_perceptron/lesson_products/release.json','Immutable public B/C release',
           '完整核HF revision 2c6a1062f3e66a602cf83eda6c703456ec18bca0、b/c prefixes及manifest SHA；自己匿名下載每產品5個檔案並核對。')
repository('s-delivery','tiny_perceptron/lesson_products/delivery.py','Anonymous public product download implementation',
           '讀download_product: urllib只加User-Agent、不送Authorization；revision格式、manifest及各檔bytes/SHA；本次用新work目錄實際下載與CPU推論。')
repository('s-local-bc','scripts/lesson_products/train_local.py','B/C local training entry',
           '完整讀main：以已提交job與固定data manifest建立本機配方，呼叫本機train函式，device cpu/mps/cuda；不呼叫Modal或HF。只核入口，不執行訓練。')
repository('s-v2-final','docs/selftrained/results/v2-final-public-results.json','Existing v2 final public result primary capsule',
           '讀scope、manifest_sha、公開HF revision、architectures.moe capacity.initialization、safe_weights與final_test定位；只核v2既有訓練／推論釋出／最後測試存在，不重新審全部能力表。未讀其所連其他review。')
repository('s-v2-test','docs/selftrained/results/public-raw/moe/test/evaluation-receipt.json','v2 actual final-test journal receipt',
           '完整讀expected_count=completed_count=3734、split=test、status=complete、checkpoint與outputs SHA；checkpoint恰等公開safe export SHA。既有primary receipt，未重跑test。')
repository('s-v2-start','docs/selftrained/results/training-raw/moe-pretrain/raw/train-receipt.json','v2 actual pretrain receipt',
           '核origin.kind=all-neural-weights-random、seed20261006、completed_requested_steps=true；不以此單段替所有後續品質背書。')
repository('s-v2-terminal','docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json','v2 terminal native training receipt',
           '核terminal joint/native已完成4000更新、interrupted=false、selected checkpoint與inference export可用、origin random；和同目錄execution completed/returncode0及final capsule selected_source run gha-37440329212-1一致。')
repository('s-v2-cpu','docs/selftrained/results/public-cpu-raw/tool_call-1-result.json','v2 retained actual public CPU operation',
           '讀argv CPU、returncode0、public_source authentication disabled、同一固定HF revision、兩次actual_generations與工具執行416後最終回答。是既有觀察，不是本審閱者重跑v2。')
repository('s-v2-data','docs/selftrained/v2-manifest.json','v2 fixed Git LFS manifest',
           '讀schema、initialization=random、package git-lfs commit08761d…與archive bytes/SHA；records12與assets8950數量；只核固定資料與來源分工，不重新逐一下載8962項。')
repository('s-v2-local','scripts/selftrained/train_local_stage.py','v2 local stage orchestration',
           '讀local_source、source_identity、main前半部：own best.pt/latest.pt與matching本機execution/receipt、manifest資料驗證及subprocess本機trainer；不必作者私有雲端路徑。')
repository('s-v2-guide','docs/selftrained/TRAINING.md','Linked v2 local operation guide',
           '完整讀本機固定Git LFS資料取得、pretrain至native逐段--init-checkpoint own best.pt、exact resume與收據範圍；本次不執行長訓練。')
repository('s-mature','docs/natural-assistant/v4/public-release.json','Mature model release configuration',
           '完整讀selected_variant=base、Qwen3-VL-2B-Instruct固定revision、Whisper-large-v3-turbo固定revision及adapter_parameters=0；僅核延伸沿用上游模型，未宣稱從零能力。')
repository('s-tools','docs/course-experiments/results/tools.json','Original immutable tool experiment result',
           '實讀results.split/test/protocol/samples及one_step_cap_samples逐步trace；56個generation的input_ids由自身腳本全數重建，36個messages metadata差異；36個有限工具結果另重算，19/20得分不變。')
repository('s-rag','docs/course-experiments/results/rag.json','Original RAG experiment result',
           '核results.split、全部84個samples的expected、generation.input_ids與生成sample；每種mode12題。只核紀錄完整性，不重跑模型或復審所有RAG能力。')
repository('s-reasoning','docs/course-experiments/results/reasoning.json','Original reasoning experiment result',
           '核comparison.direct/steps每個budget：各24題，input_ids、truth、candidate outputs，samples長度等於majority_accuracy denominator；共2×4組。只核紀錄及分母。')
sources.append({'id':'s-execution','kind':'execution','title':'Own independent CPU evidence and original-score recalculation','verified':True,'artifact_id':'a-execution'})
sources.append({'id':'s-math','kind':'derivation','title':'Hand quantization criterion and denominator counterexamples','verified':True,
 'details':'60000<80000，少20000 bytes；原4/5=.8、新3/5=.6，3<4，因此「較小且答對數不減」的合取判準為false。兩組正確ID={1,2,3,4}及{1,2,3,5}都4/5但錯題不同。若全5題只有1、2、3題對，先丟掉4、5題會報3/3而非原3/5，材料及分母不同不能直接當相同試題比較。這是透明手算及反例，不是量化模型實測。'})

def evidence(sid,locator,supports): return {'source_id':sid,'locator':locator,'supports':supports}
claims=[]
def claim(cid,kind,statement,location,scope,refs,expected=None,got=None,den=None,manual=False):
    row={'id':cid,'kind':kind,'statement':statement,'location':location,'status':'verified',
         'evidence':refs,'artifact_ids':[] if kind=='concept' else ['a-execution'],'scope':scope}
    if kind in ('numeric','software','empirical'):
        row['verification']={'method':'hand_calculation' if manual else 'executed','expected':expected,'observed':got,
                             'details':'本次執行T.11-check.py；既有訓練／final test收據為primary evidence，未重跑。' if not manual else '透明算式與集合反例見s-math；數字明示手寫，非模型實測。'}
        if kind=='numeric':row['verification']['tolerance']='整數與集合判斷精確相等；比例由相同分母直接相除。'
        if den:row['verification']['denominators']=den
    claims.append(row)

claim('c1','concept','保存原始模型／程式、資料版本與授權切分、轉換及兩版評估命令、生成與裝置精度、逐題輸出停止原因、量測邊界，可讓材料、做法與結果受核對。',
 '開頭兩段與六列紀錄表、有限比較結語',
 '是一次有限比較的必要紀錄建議；不宣稱這些欄位足以保證跨平台逐位元重現，也不由檔案大小推算時間或記憶體。純文字／圖排版修訂未改模型、資料與能力命題時，沒有新的訓練實驗條件；若有改動則需相應驗收。',
 [evidence('s-neurips','Checklist 4–6、8、12','官方準則明列完整命令、環境、資料切分、超參數、模型存取和compute邊界；並要求來源／版本及授權條件。')])
claim('c2','concept','SHA-256是檔案bytes的指紋，指紋相符不自動證明資料正確、授權或模型能力。',
 'SHA-256段','以已取得檔案bytes核對身份；雜湊不是內容語義評估，也不把碰撞機率描述為零。',
 [evidence('s-hash','Hash algorithms與Usage: sha256/digest/hexdigest','API計算輸入bytes的摘要；未計算標籤或能力，故只能核對版本。')])
claim('c3','concept','Git commit可標記受版控程式的固定版本。','SHA-256段',
 'commit所指tree內受追蹤程式；環境、未追蹤資料與權重另需固定。',
 [evidence('s-git','Committing Git state','commit連結檔案名稱、內容及權限的tree與parent；支持以commit辨別程式版本。')])
claim('c4','numeric','手寫例子80000→60000 bytes雖變小，但4/5→3/5不符答對數不減。','量化手寫例子段',
 '只驗手寫合取判準，不是量化成效或真實4-bit壓縮率。',[evidence('s-math','第一組算式','60000<80000且3<4。')],
 '較小為真，答對數不減為假，合取為假','60000<80000；3<4；.6<.8',manual=True)
claim('c5','numeric','兩版同為4/5仍可能錯不同題；刪掉難題後的比例不能直接當全檔相同條件的比較。','逐題結果與分母段',
 '同分母與逐題配對的透明反例，不把單題抽樣差異解讀成普遍模型優劣。',[evidence('s-math','正確ID集合及刪題反例','同樣4/5的錯題可不同，3/3與3/5材料不同。')],
 '能構成同分數異錯題以及刪題改分母的反例','{1,2,3,4}與{1,2,3,5}各4/5；丟掉兩道錯題後3/3而非3/5',manual=True)
claim('c6','concept','驗證題可供選設定；最後題留到設定選定後才評估。','逐題結果段末',
 '合理保留test防止選設定洩漏；不能僅由宣告推斷任一實驗實際遵守，B/C另核凍結與score receipts。',
 [evidence('s-split','opening hyperparameter/validation/test paragraphs','明述對test調參會洩漏，validation供評估選設定，final test最後使用。')])
claim('c7','empirical','既有v2已完成訓練並有固定最後測試與公開推論權重。','details第1段v2句',
 '只核既有MoE v2的完成與發佈身份；原有能力未全達標仍保留，未親跑production或完整複審v2逐題分數。',
 [evidence('s-v2-start','origin、completed_requested_steps','實際pretrain由隨機神經權重開始且完成要求步數。'),
  evidence('s-v2-terminal','steps/completed_requested_steps/interrupted/selected_checkpoint_available/inference_exported','terminal native stage實際完成4000更新，有選定checkpoint和推論匯出；同run出現在最終成品selected_source。'),
  evidence('s-v2-final','hf_revision、architectures.moe.safe_weights/final_test','固定公開inference export與final來源。'),
  evidence('s-v2-test','completed_count/status/checkpoint_sha256','3734題固定test已完成且使用同一公開權重SHA。'),
  evidence('s-execution','v2_existing_primary_receipts','本次讀取並交叉核對其identity。')],
 '已完成訓練收據、固定test完整、公開safe export身份相同','pretrain completed=true；terminal native完成4000更新、原execution completed/returncode0、selected source為同run；3734/3734 test complete；checkpoint SHA=26102d…等公開safetensors SHA',
 {'v2_moe_final_examples':3734,'v2_pretrain_origin_seed':20261006,'v2_terminal_actual_updates':4000,'evaluation_run':'gha-37448172517-1'})
claim('c8','software','v2公開CPU操作可供試用；本機stage指引以固定Git LFS資料接自己的完整訓練checkpoint，不依賴作者私有雲端空間。','details第1段CPU與本機訓練句',
 'CPU操作此處引用既有primary receipt；本次只查本機stage腳本與資料身份，未下載全8962檔或重跑stage。公開safe export本身不作續訓checkpoint。',
 [evidence('s-v2-cpu','argv/public_source/actual_generations','原始CPU匿名推論成功，明留實際輸入與最終結果。'),
  evidence('s-v2-data','package、records、assets','git-lfs固定commit、archive SHA及資料清單。'),
  evidence('s-v2-local','local_source與main','從本機資料與own best/latest載入，不須私人遠端路徑。'),
  evidence('s-v2-guide','安裝與固定資料、手動stage、新stage與exact resume','明確逐段接自己的checkpoint。'),
  evidence('s-execution','v2_existing_primary_receipts','本次驗過資料固定類型、CPU原收據與身分。')],
 '資料git-lfs固定版本；own checkpoint chain；原CPU命令成功','manifest含12 records+8950 assets，package git-lfs；原CPU returncode0/auth disabled；local_source要求own matching checkpoint sidecars')
claim('c9','software','第20章操作延伸沿用成熟Qwen與Whisper能力，與從零主線分開。','details第1段第20章句',
 '只核模型來源身份；沒有重跑成熟模型，也不把上游能力歸為本課從零訓練。',
 [evidence('s-mature','selected_variant/base_model/asr_model','runtime選base，Qwen3-VL與Whisper各固定revision；adapter_parameters=0。'),
  evidence('s-execution','mature_extension','實際讀取固定配置。')],
 '延伸配置載入Qwen與Whisper','Qwen/Qwen3-VL-2B-Instruct@89644892…；openai/whisper-large-v3-turbo@41f01f3…；selected_variant=base')
claim('c10','empirical','新增B/C與既有MoE分開，由隨機權重起步，並在NVIDIA L4完成正式訓練。','details第2段前半',
 '既有一手GPUrunner收據與本機receipt交叉核對；本審閱者沒有親跑GPU。B/C是有限世界獨立模組，不宣稱一般助理能力或多seed穩定性。',
 [evidence('s-b-run','status/gpu_name/training','B random init、no resume、1000/1000 steps、權重已變。'),
  evidence('s-c-run','status/gpu_name/training','C random init、no resume、800/800 steps、權重已變。'),
  evidence('s-train','fresh initialization branch','普通本機模組建構，不載入MoE或成熟神經模型。'),
  evidence('s-model','LoopedSolver/CandidateScorer','兩個獨立類別，無MoE loader。'),
  evidence('s-execution','training_receipts','本次逐欄確認local原始receipt等於runner.training。')],
 '兩產品隨機起點、無resume、L4完成更新','B seed7421 last/selected1000；C seed9041 last/selected800；兩者weights_changed=true與CUDA L4',
 {'b_actual_updates':1000,'c_actual_updates':800,'b_seed':7421,'c_seed':9041,'b_and_c_runs':2})
claim('c11','software','學生可在CPU匿名下載固定B/C權重推論，產品操作另提供本機重訓入口，可先只做短機制練習。','details第2段CPU與產品操作句',
 '自己在Python3.13.5/PyTorch2.14.1+cpu匿名下載各5個檔案並各推論一個指定例子；這只驗通路，不證明整體能力。不執行長訓練或機制訓練。',
 [evidence('s-release','products及40字revision','固定發佈版本與manifest SHA。'),
  evidence('s-delivery','download_product','匿名urllib下載與逐檔hash驗證。'),
  evidence('s-local-bc','main','本機訓練入口存在且只呼叫本機train。'),
  evidence('s-execution','own_public_cpu','本審閱者直接下載與CPU推論觀察。')],
 '兩產品匿名下載所有固定檔案並成功本機推論','每產品5檔SHA核對；B交換盒查詢答案2、loops3；C缺申請編號info選no；固定HF2c6a1062…')
claim('c12','empirical','B兩盒追蹤通過核心門檻，三數加法未通過。','details第2段成績句',
 '固定test、5輪、seed7421產品：兩盒有限事件語法；加法是次要未達標項，不把所有輸入能力混成一個成功率。',
 [evidence('s-settings','protocol.b_primary/b_secondary','主要兩盒5輪accuracy>=.90；加法target .85。'),
  evidence('s-final','b[5].boxes/sum與gates','原報告與本次重算一致。'),
  evidence('s-execution','raw_scores counts/correct/b_family_counts','原始逐題答案與gold重算。')],
 '兩盒>=.90、加法<.85','兩盒2928/2956=.990527740…；三數加法2/42=.0476190476…，各自通過／未通過',
 {'b_boxes_examples':2956,'b_boxes_families':1478,'b_sum_examples':42,'b_sum_families':10,'loops':5,'seed':7421})
claim('c13','empirical','C核心題全對；換說法仍有高集中度錯誤。','details第2段C成績句',
 '有限候選與固定核心測試；集中度使用calibration溫度.25及(n*pmax-1)/(n-1)，不是答對概率。wording shift與核心分開，不能稱開放域可靠。',
 [evidence('s-final','c.*.after、shifts.c_wording','核心各question全對，換說法各有錯誤。'),
  evidence('s-settings','calibration.temperatures','三類問題固定溫度均.25。'),
  evidence('s-execution','raw_scores','自己由raw_scores argmax與gold重算，錯誤再依溫度求softmax與集中度。')],
 '核心所有題答對；shift存在concentration>=.9錯誤','核心Choice11797/11797、Info10890/10890、Rubric10890/10890；shift分別304/117/117題錯，其中283/117/117題集中度>=.9',
 {'core_choice':11797,'core_info':10890,'core_rubric':10890,'wording_choice':11797,'wording_info':10890,'wording_rubric':10890,'seed':9041})
claim('c14','software','RAG、工具及推理原始報告保存實際輸入、生成輸出與分母。','details第3段歷史結果句',
 '只核三份實際連結結果的紀錄欄位；工具messages有下段所揭露的metadata缺陷，不能拿它單獨當逐步輸入。沒有重跑模型或把全部歷史實驗宣稱完整。',
 [evidence('s-rag','results.split/samples','84筆實際輸入IDs、生成樣本、expected，7 modes各12。'),
  evidence('s-reasoning','comparison.direct/steps.budgets','每mode每budget24筆，保存truth、input IDs與candidates、分母。'),
  evidence('s-tools','samples/one_step_cap_samples.trace','原始generation IDs、tool result與final。'),
  evidence('s-execution','historical_records/historical_tools','程式逐項查必要欄位與分母。')],
 '三報告有原始輸入、輸出及計分樣本分母','RAG84=7×12；reasoning direct/steps×4 budgets各24；tools20正常+18cap、56次generation')
claim('c15','concept','只生成工具請求還不等於完成操作；要留下實際工具結果及將結果交回模型後的回答。','details第3段工具句',
 '是本課所教帶回覆的工具循環完成定義；不把有final文字當正確，也不要求所有可能外部工具都必須返回自然語言。',
 [evidence('s-tool-flow','The tool calling flow五步','模型tool call之後另有application execute、交回tool output及final response。')])
claim('c16','software','舊工具generation.messages曾引用後續會變動的list，而實際input_ids正確；原始JSON、hash與得分保留。','details末段勘誤句',
 '只限tools.json指定歷史revision；本次從原始開始兩訊息、每次實際請求與有限結果重建，不是新模型觀察。不能把舊messages未經補件當當時輸入。',
 [evidence('s-tools','56個generation.input_ids、20正常samples','所有ID及token count重建逐個相等，19/20得分保留。'),
  evidence('s-execution','historical_tools','讀原run revision910aeb… applications.py，SHA8fbb7…與report code_sha相同；_sample回傳messages本體，_tool_episode +=回填，56次中36筆metadata差異。')],
 '56次所有input IDs與長度相符、36次messages不同、原JSONhash及得分不變','56/56重建相等，6630個input tokens；36 metadata差異；原JSON SHA3befd7fc…不變；正常19/20；36個有限工具執行結果另算吻合')

def artifact(aid,kind,name,description,**kwargs):
    return {'id':aid,'kind':kind,'path':(HERE/name).relative_to(ROOT).as_posix(),
            'sha256':sha(HERE/name),'description':description,**kwargs}
artifacts=[
 artifact('a-execution','execution','T.11-check-output.json','自身CPU下載／推論、原始分數重算、歷史receipt identity核對、56次工具輸入逐ID重建與分母檢查。',
          command='.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/backfill/T.11-check.py',
          result='全部assert通過；B2928/2956、2/42；C核心33577/33577；shift有517個集中度>=.9的錯誤；56次歷史輸入全符、36次metadata差異；兩產品匿名下載與CPU各一例通過。',environment=observed['environment']),
 artifact('a-code','code','T.11-check.py','可重查的有界CPU檢查程式；不重訓、不調參、不重啟formal test。'),
 artifact('a-source-receipts','source_snapshot','T.11-original-source-receipts.json','實讀五份官方來源的URL、版本、日期、原檔SHA、定位與短摘錄；外部全文留work。'),
 artifact('a-body','source_snapshot','T.11-source-slice.txt','freeze-14 training頁所指freeze-11來源的T.11原UTF8切片，使用lesson_slices，不正規化。'),
 artifact('a-v2-terminal','execution','T.11-v2-terminal-check-output.json','交叉核對v2最後native stage實際完成與final capsule所選run身份；不執行模型。',
          command='.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/backfill/T.11-v2-terminal-check.py',
          result='原terminal train-receipt完成4000步、未中斷、selected checkpoint與inference export存在，原execution completed/returncode0；run gha-37440329212-1等final selected_source。',
          environment={'python':'3.13.5','device':'cpu'}),
 artifact('a-v2-terminal-code','code','T.11-v2-terminal-check.py','最後stage一手收據的有界核對程式。')]
next(x for x in claims if x['id']=='c7')['artifact_ids'].append('a-v2-terminal')
ids=[x['id'] for x in claims]
report={'schema_version':1,'review_stage':'technical','lesson_id':'T.11','source':'course/training.md#T.11',
        'reviewer_task':'/root/backfill_tech_t_11','reviewer_context':'fresh','author_tasks':['/root'],
        'source_sha256':hashlib.sha256(body.encode()).hexdigest(),'figure_sha256':{},'verdict':'pass',
        'reviewed_on':'2026-10-16','source_freeze':'docs/course-revision-20261016/freeze-14.json',
        'claims':claims,'sources':sources,'artifacts':artifacts,'issues':[],
        'checks':{
         'factual_accuracy':{'status':'pass','details':'16個實質主張分開核對；紀錄方法、雜湊版本、固定資料與本機入口、既有v2、B/C訓練與能力、歷史trace均有對應原始來源。','claim_ids':ids},
         'numeric_verification':{'status':'pass','details':'手寫80000/60000與4/5、3/5精確核對；自身讀原始逐題分數重算B/C核心及wording錯誤，保留題數與家族數；工具56次6630 tokens逐ID核對，沒有正式評測重跑。','claim_ids':['c4','c5','c7','c10','c12','c13','c16']},
         'figure_consistency':{'status':'not_applicable','details':'T.11凍結切片沒有圖片、SVG或需渲染之圖；本次不宣稱linked第19章圖表版面已驗。','claim_ids':[]},
         'source_verification':{'status':'pass','details':'直接讀五份官方原文並記版本／定位／日期／SHA；專案行為讀實作，一手GPU與CPU收據對應其實際範圍，原始分數自行重算，未把來源庫摘要或其他review當證據。','claim_ids':ids},
         'limitations':{'status':'pass','details':'手寫例子非實測；hash與命令成功不等於能力；B加法失敗、C換說法高集中錯誤均明示；公開推論權重與續訓checkpoint分開，有限語法／單seed分數不可外推。自身只做CPU短查與重算，GPU訓練及舊production是既有primary receipts。','claim_ids':ids}}}
target=ROOT/'docs/course-revision-20261016/reviews/technical-T.11-backfill-original-01.json'
if target.exists():
    assert json.loads(target.read_text())['reviewer_task']=='/root/backfill_tech_t_11','Never replace another reviewer record'
write(target,report)
errors=_validate(ROOT,ROOT/'course/training.md','T.11',body,report,{'/root/backfill_tech_t_11':['T.11']},set())
validation={'reviewer_task':'/root/backfill_tech_t_11','report':target.relative_to(ROOT).as_posix(),
            'report_sha256':sha(target),'source_sha256':report['source_sha256'],'format_errors':errors,
            'scope':'_validate checks schema/hashes/identity, not substantive truth; substantive verification is separately archived.'}
write(HERE/'T.11-format-validation.json',validation)
print(json.dumps(validation,ensure_ascii=False,indent=2))
assert not errors
