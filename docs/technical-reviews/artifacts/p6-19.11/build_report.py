import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path.cwd()
OUT = Path(__file__).parent
PREFIX = OUT.relative_to(ROOT).as_posix()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path):return Path(path).relative_to(ROOT).as_posix() if Path(path).is_absolute() else str(path)
environment=json.loads((OUT/'cpu-check-result.json').read_text())['environment']
revision=json.loads((OUT/'saved-raw-check-result.json').read_text())['repository_revision']
artifacts=[]
def artifact(identifier,name,kind,description,command=None,result=None,env=None):
    path=OUT/name
    item={'id':identifier,'kind':kind,'path':rel(path),'sha256':sha(path),'description':description}
    if kind=='execution':item.update(command=command,result=result,environment=env or environment)
    artifacts.append(item)
    return identifier

artifact('cpu','cpu-check-result.json','execution','本人的有界CPU原程式與變化檢查；零training update、零heldout題。',
    '/workspace/tiny-perceptron-vlm/.venv/bin/python '+PREFIX+'/cpu_check.py > '+PREFIX+'/cpu-check-stdout.txt 2> '+PREFIX+'/cpu-check-stderr.txt',
    'exit0；完整151個神經tensor；既知文字示例33tokens/EOS；1token變化token_limit；所有預期拒絕、RNG/sampler/optimizer邊界一致。')
artifact('cpu-code','cpu_check.py','code','實際執行的有界CPU檢查原碼；synthetic receipt僅測local_source分支，不冒充真實训练。')
artifact('cpu-stdout','cpu-check-stdout.txt','source_snapshot','此次CPU檢查的原始stdout bytes。')
artifact('cpu-stderr','cpu-check-stderr.txt','source_snapshot','此次CPU檢查的原始stderr bytes。')
artifact('cli','current-cli-stdout.json','execution','本人用固定公開revision、原chat.py與當前教材訊息檔實際生成的完整JSON。',
    json.dumps(json.loads((OUT/'current-cli-argv.json').read_text())['argv'],ensure_ascii=False),
    'exit0；answer與正文精確一致，1次generation，33generated IDs含EOS，無modality，public authentication disabled。')
artifact('cli-argv','current-cli-argv.json','source_snapshot','真正的cwd與原始argv；依派遣復用既有Python，model/asset-dir使用新tmp目錄。')
artifact('short-cli','short-cli-stdout.json','execution','同一既知示例僅將max-new-tokens由128改為1，並重用完整目的地。',
    json.dumps(json.loads((OUT/'short-cli-argv.json').read_text())['argv'],ensure_ascii=False),
    'exit0；answer為1，generated_ids長度1，generation_status為token_limit。')
artifact('short-cli-argv','short-cli-argv.json','source_snapshot','1token變化的真實原始argv。')
artifact('metadata','public-metadata-result.json','execution','匿名讀取真正HF commit metadata及12個小型metadata檔，不讀model card、舊review或index結論。',
    '/workspace/tiny-perceptron-vlm/.venv/bin/python '+PREFIX+'/inspect_public_metadata.py',
    'exit0；revision精確符合，private=false/gated=false；selftrained/v2恰16files；4x4名稱與三payload指紋一致。')
artifact('metadata-code','inspect_public_metadata.py','code','真正HF metadata檢查原碼，限必要pointers與4份config/tokenizer/manifest。')
artifact('saved-raw','saved-raw-check-result.json','execution','親自核對原始training receipts及public-cpu argv/stdout；不重訓、不重測heldout。',
    '/workspace/tiny-perceptron-vlm/.venv/bin/python '+PREFIX+'/verify_saved_raw.py > '+PREFIX+'/saved-raw-check-stdout.txt 2> '+PREFIX+'/saved-raw-check-stderr.txt',
    'exit0；四份public export逐一匹配原receipt best.pt與每個export SHA；7組/8chat/1append；当次文字IDs與原始stdout精確相同。')
artifact('saved-raw-code','verify_saved_raw.py','code','原始receipts/argv/stdout實際核驗程式，明確限定JSON pointers。')
artifact('training-pointers','training-inspected-pointers.json','source_snapshot','真正檢查的training原資料pointers、值與全檔SHA；避開notes/review/*scope_correction與job額外說明。')
artifact('public-text-pointers','public-text-inspected-pointers.json','source_snapshot','原始text stdout的answer/generations/model/public_source，沒有沿用作者答案比對旗標。')
artifact('scenario-pointers','scenario-argv-check.json','source_snapshot','9份原argv的SHA與實際CPU/task分類；7種task、8chat、1history append。')
artifact('tensor-header','safetensors-header.json','source_snapshot','已核對原始public model.safetensors SHA後保存8-byte length之後的原始header bytes；完整151神經tensor和metadata，未提交大權重。')
artifact('cache-manifest','cache-inference-manifest.json','source_snapshot','本機safeweights原始manifest永久副本，原件/副本SHA實際一致。')
artifact('cache-config','cache-model-config.json','source_snapshot','本機safeweights原始結構JSON永久副本，原件/副本SHA實際一致。')
artifact('cache-tokenizer','cache-tokenizer.json','source_snapshot','本機safeweights原始字元表永久副本，原件/副本SHA實際一致。')
artifact('source-copy-manifest','inputs/source-copy-manifest.json','source_snapshot','30份初始原件/永久副本的byte數與SHA逐一相等；整章SHA只標為frozen input。')
artifact('uv-dry-run','uv-sync-dry-run-stderr.txt','execution','核對正文frozen CPU+selftrained依賴命令與現有lock；dry-run不下載安裝。',
    'uv sync --frozen --extra cpu --extra selftrained --dry-run > '+PREFIX+'/uv-sync-dry-run-stdout.txt 2> '+PREFIX+'/uv-sync-dry-run-stderr.txt',
    'exit0；54 locked packages，其中torch2.14.1+cpu/safetensors0.8.0/huggingface-hub1.33.0；依本次任務復用外部已存在.venv實跑。',
    {'uv':'0.12.19','python':'3.13.5','device':'cpu'})
artifact('wrapper-help','wrapper-help.txt','execution','真實本機wrapper入口可解析help且不依賴Modal或私人checkpoint。',
    '/workspace/tiny-perceptron-vlm/.venv/bin/python scripts/selftrained/train_local_stage.py --help',
    'exit0；wrapper參數manifest/data-root及--後trainer參數如原文。')
artifact('figure640','figure-640.png','figure_render','本人渲染並實際view的640px圖，公開上路與best/latest下路箭頭標示一致。')
artifact('figure360','figure-360.png','figure_render','本人渲染並實際view的360px圖，無標籤截斷，四文件與兩種checkpoint用途清楚。')
official_names=['pytorch-saving-loading.html','pytorch-saving-loading.txt','pytorch-randomness-v2.11.0.rst','hf-download.html','hf-download.txt','hf-file-download.py','safetensors-readme.md','safetensors-torch.py','hf-immutable-metadata.json','fetch-results.json']
for name in official_names:artifact('official-'+name.replace('.','-'),'official/'+name,'source_snapshot','本人讀取的官方原文快照；URL/version/定位見逐source記錄。')
for stage in ['moe-pretrain','moe-sft','moe-joint','dense-joint']:
    for name in ['model-config.json','tokenizer.json','inference-manifest.json']:
        artifact(stage+'-'+name.replace('.','-'),'public-metadata/'+stage+'/'+name,'source_snapshot','固定HF revision匿名取得的'+stage+'原始'+name+'，內容指紋親自核對。')
raw_artifacts={}
for stage in ['moe-pretrain','moe-sft','moe-native','dense-weighted']:
    raw_artifacts[stage]=[]
    for name in ['receipt.json','raw/train-receipt.json','raw/execution.json']:
        identifier=stage+'-'+name.replace('/','-').replace('.','-')
        artifact(identifier,'inputs/docs/selftrained/results/training-raw/'+stage+'/'+name,'source_snapshot','指定原始訓練receipts永久byte-identical副本；只讀training-pointers記錄的必要欄位。')
        raw_artifacts[stage].append(identifier)
for name in ['text-1-argv.json','text-1-stdout.txt','text-1-result.json','text-1-stderr.txt']:
    artifact('raw-'+name.replace('.','-'),'inputs/docs/selftrained/results/public-cpu-raw/'+name,'source_snapshot','指定原始文字CPU記錄的完整SHA；results僅定位topkeys，正文實測取自原stdout。')
code_paths={
    'inference':'tiny_perceptron/selftrained/inference.py',
    'train':'scripts/selftrained/train.py','wrapper':'scripts/selftrained/train_local_stage.py',
    'tokenizer':'tiny_perceptron/selftrained/tokenizer.py','config':'tiny_perceptron/selftrained/model.py',
    'chat':'scripts/selftrained/chat.py','dataset':'tiny_perceptron/selftrained/dataset.py',
    'packaging':'pyproject.toml','lock':'uv.lock'}
sources=[]
def original(identifier,kind,title,url,version,note,snapshot,authority):
    sources.append({'id':identifier,'kind':kind,'title':title,'url':url,'version':version,
        'verified':True,'checked_original':True,'accessed_on':'2026-10-06',
        'authority_reason':authority,'inspection_note':note,
        'snapshot_path':PREFIX+'/official/'+snapshot,'snapshot_sha256':sha(OUT/'official'/snapshot)})
original('torch-checkpoint','official_docs','PyTorch Saving and Loading Models',
    'https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html','官方tutorial於2026-10-06取得，凍結HTML SHA256 '+sha(OUT/'official/pytorch-saving-loading.html'),
    '讀Saving & Loading a General Checkpoint for Inference and/or Resuming Training；text lines937-1083。原文要求model之外optimizer buffers與epoch/progress。沒有把範例2~3x大小當本文主張。',
    'pytorch-saving-loading.html','PyTorch官方教學維護者，直接說明model與optimizer checkpoint分工。')
original('torch-rng','official_source','PyTorch Reproducibility notes',
    'https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/docs/source/notes/randomness.rst','PyTorch v2.11.0原始文件；本次執行安裝版本2.14.1+cpu',
    '讀lines1-73：PyTorch/Python/NumPy多個RNG來源；同環境同序列與跨release/platform不保證完全重現。stable HTML是redirect；版本2.14頁403，改讀可取得的固定v2.11.0官方原檔，只支持一般RNG與範圍，當前API另由真實CPU執行查證。',
    'pytorch-randomness-v2.11.0.rst','PyTorch官方repo原始notes，明確區分隨機控制與跨平台復現。')
original('hf-download','official_docs','Hugging Face Hub Download files',
    'https://huggingface.co/docs/huggingface_hub/v1.33.0/guides/download','huggingface_hub v1.33.0，與安裝版本相同',
    'From specific version，text lines274-320：revision接受branch/tag/commit，commit必須full-length。再讀官方file_download.py857-975的每commit snapshot cache/API契約。',
    'hf-download.html','Hugging Face維護的官方Hub下載文件與原始client實作。')
original('safe-format','official_source','Safetensors v0.8.0 format and PyTorch bindings',
    'https://raw.githubusercontent.com/huggingface/safetensors/v0.8.0/README.md','safetensors v0.8.0，與安裝版本相同',
    'README lines21-26,73-85：named tensor dtype/shape/offset與string-only metadata；torch.py288-360：save_file dict of tensors與load_file Dict[str, Tensor]，metadata不影響tensor loading。安全格式本身不能補出optimizer/RNG，也不保證tensor是正確模型；本repo另外檢查完整性。',
    'safetensors-readme.md','Safetensors官方repository與bindings原碼，直接定義格式與讀寫契約。')
original('hf-actual','official_source','Hugging Face immutable model-revision API response',
    'https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/revision/979cdfacc588ad0536f1c64fff96f264571cf054?blobs=true',
    'revision979cdfacc588ad0536f1c64fff96f264571cf054',
    '原始metadata先列topkeys；實讀/sha,/private,/gated,/siblings中的selftrained/v2檔名size/LFS.sha256。沒有讀config/modelcard/作者額外解說。再匿名讀12個小型config/tokenizer/manifest並自行比對hash。',
    'hf-immutable-metadata.json','HF官方託管服務回應，直接證明該immutable revision的可讀性與實際檔案，並非作者release摘要。')
notes={
    'inference':'AST先定位；實讀74-136 verify_export/fetch_public_export及215-328 InferenceAssistant；檢查hash、origin/preprocess、config/vocab、完整keys、shape/dtype/finite/tied、strict load、eval/no_grad與generations。',
    'train':'AST先定位；實讀106-232 sampler/RNG、235-281 export、631-917 parser/main checkpoint/select；載入trusted .pt weights_only=False，fresh新optimizer/RNG/sampler，resume恢復optimizer/state/count；未執行training main或讀作者額外結論字串。',
    'wrapper':'AST定位且讀23-251原契約：local_source的best/latest與sidecars驗證、source_identity、manifest/assets/source檢查、新output dir及local subprocess/receipts；CPU只跑local_source fixture和help，不執行訓練。',
    'tokenizer':'讀CharacterTokenizer12-77，sorted unique characters+11specials決定ID，to/from_dict/load；非通用HuggingFace BPE tokenizer。',
    'config':'讀SelftrainedConfig25-53的結構/version/architecture/dimension驗證；LimitedAssistant實際state_dict由CPU初始化確認全部151keys；不讀作者結果解說。',
    'chat':'讀parser37-64/main67-110：repo+revision成對、torch threads、anonymous fetch、public messages、CPU reply及輸出model/generations/history。',
    'dataset':'讀PREPROCESS_VERSION及RecordEncoder108-241必要encoding/asset路徑契約；當次文字無image/audio，沒有讀取asset內容。',
    'packaging':'讀CPU/selftrained extras及torchCPU專用uv index；frozen dry-run核對54包，沒有寫dependency或lock。',
    'lock':'UV frozen dry-run使用本版本lock；CPU torch2.14.1+cpu、safetensors0.8.0/hub1.33.0與實跑Python環境一致。'}
for identifier,path in code_paths.items():
    sources.append({'id':identifier,'kind':'repository_code','title':path,'path':path,'sha256':sha(path),
                    'version':'Git '+revision+'；本人凍結完整原始bytes並核對副本SHA',
                    'inspection_note':notes[identifier],'verified':True})
    artifact('code-'+identifier,'inputs/'+path,'code','本人實查原實作/設定的完整frozen byte副本，定位見source '+identifier+'。')

claims=[]
def claim(identifier,kind,statement,location,evidence,ids,scope,expected=None,observed=None,details=None,denominators=None):
    item={'id':identifier,'kind':kind,'statement':statement,'location':location,'status':'verified',
          'evidence':[{'source_id':s,'locator':l,'supports':support} for s,l,support in evidence],
          'artifact_ids':ids,'scope':scope}
    if kind in ['numeric','software','empirical']:
        item['verification']={'method':'executed','expected':expected,'observed':observed,'details':details}
        if kind=='numeric':item['verification']['tolerance']='精確相等，檔案數/字元ID/內容SHA沒有浮點容忍差。'
        if denominators:item['verification']['denominators']=denominators
    claims.append(item)

claim('c1','concept','推論需要權重、模型結構、字表與前處理；精確接回訓練另需optimizer、RNG、sampler與進度，只有權重不能還原全部狀態。','開頭段落',
    [('torch-checkpoint','Saving & Loading a General Checkpoint…; text937-1083','optimizer buffers與training progress超出model state_dict。'),('torch-rng','lines1-73','不同RNG來源和同環境復現範圍。'),('safe-format','README73-85; torch.py288-360','Safetensors命名tensor/string metadata不自行包含本repo完整訓練狀態。')],
    ['cpu','code-train','tensor-header'],'精確恢復指保存的執行/資料/config與受支援同階段契約；不保證跨PyTorch版本、CPU/GPU或不同平台bitwise trajectory。')
claim('c2','concept','完整40hex commit revision固定同一發布內容，避免下載隨可移動branch/tag而改變。','固定HF連結段落',
    [('hf-download','From specific version; text274-320; file_download.py857-918','full-length commit和per-commit snapshot契約。'),('hf-actual','/sha','匿名真實API回應精確為正文revision。')],
    ['metadata','official-hf-download-html','official-hf-file-download-py'],'固定內容身分；不聲稱永久託管可用性或每次都需要重新下載。')
claim('c3','numeric','固定revision的4個selftrained/v2公開目錄，每個恰有4檔，共16檔。','四版本表及每份4檔段落',
    [('hf-actual','/siblings restricted to selftrained/v2/','真實目錄/檔名和大小。')],
    ['metadata','metadata-code','official-hf-immutable-metadata-json'],'只核對本節指定的immutable revision和selftrained/v2前綴。',
    '4份各model.safetensors/model-config.json/tokenizer.json/inference-manifest.json，共16files',
    '4x4=16；moe-pretrain/moe-sft/moe-joint/dense-joint名稱精確符合',
    '匿名API metadata逐項集合比對；12個小metadata來源原始bytes全部保存，tensor SHA與LFS SHA相同。')
claim('c4','software','4檔分別保存完整神經權重、結構、字元ID表和來源/前處理/內容指紋；公開包不含optimizer/RNG/sampler工作checkpoint。','四檔角色段落',
    [('train','export_inference235-281','只寫checkpoint.model tensors與三JSON metadata。'),('tokenizer','CharacterTokenizer12-77','固定special IDs加sorted字元ID映射。'),('inference','74-98;215-257','manifest payload/hash/config/vocab/complete state checks。'),('safe-format','README73-85; torch.py288-360','safe weights格式和string metadata。')],
    ['cpu','tensor-header','cache-manifest','cache-config','cache-tokenizer','metadata','code-train'],
    '公開manifest含描述性來源/選版/目標metadata，不包含實際optimizer moments、RNG bit state或sampler cursor。早期完整tensor存在不表示它們全受過該stage訓練。',
    'safe weights鍵等於完整model.state_dict；manifest exactly三payload，tokenizer vocab等於config',
    '151tensor鍵精確匹配；4份JSON角色與hash均符合；沒有optimizer/rng/sampler tensor或公開.pt',
    '讀取真實cache tensor並比較原始SHA到remote LFS；保存原header和byte-identical三metadata副本。')
claim('c5','software','下載器先驗證manifest及三payload指紋，載入器再驗證結構、完整權重keys/shape/dtype/finite/tied並strict load。','下载程式核對指紋、載入器再核對結構與所有權重一句',
    [('inference','verify_export74-98;fetch_public_export101-136;InferenceAssistant215-257','雙層內容驗證與完整model核對的原分支。')],
    ['cpu','cpu-code','code-inference'],'外部manifest pin或immutable revision提供版本身分；單純內部重新綁定hash不能代替模型結構完整性檢查。',
    '錯manifest/payload SHA拒絕；內容hash合法但少tensor仍拒絕',
    'wrong pin與config bytes變化各ValueError；刪audio_encoder.head.bias重新綁SHA後verify_export可過而loader拒絕complete model keys',
    '真實公開state的小變化僅在tmp；閱讀shape/dtype/finite/tied所有分支且實跑完整模型成功與缺key反例。')
claim('c6','software','四版本代表pretrain、SFT、選定MoE joint及從自己隨機起點承接的Dense joint；前兩份不等於joint完整訓練歷程/能力。','四版本用途表及前兩份比較權重一句',
    [('train','main726-775,804-834;export235-281','origin和stage history承接、Dense/MoE架構拒絕交叉、best export。'),('hf-actual','四份manifest /stage,/origin,/selected_step;config /architecture','公開檔真實stage與architecture。')],
    ['saved-raw','training-pointers','metadata']+sum(raw_artifacts.values(),[]),'核對保存的完成/選版/lineage與實作來源；不由CPU示例評判早期或joint未知任務品質。',
    'public manifests對應原training receipt best SHA與4個export，Dense origin own random且自身stage history',
    'pretrain完成1000/selected250；SFT8000/8000；MoE native joint4000/1000；Dense weighted joint10000/1000，全部SHA匹配；兩architecture各自history且origin all-neural-weights-random',
    '原receipts先topkeys、再限定/steps,/stage_history,/origin,/files及job必要pointers；未使用index或作者比較勝負旗標。')
claim('c7','software','正文CPU命令使用同一immutable MoE joint、manifest pin與公開訊息，匿名取得四檔後直接生成，不需登入或私人.pt；文字不需圖音assets。','操作fence与chat.py匿名/文字資產段落',
    [('chat','parser37-64;main67-105','CPU/device/threads/public fetch/messages/reply。'),('inference','fetch101-136;InferenceAssistant215-328','token=False、safetensors载入、public文字encoding。'),('packaging','CPU+selftrained extras及CPU index','frozen CPU依賴選擇。'),('lock','frozen dry-run解析的locked distributions','torchCPU/hub/safeweights安裝版本。')],
    ['cli','cli-argv','cpu','uv-dry-run','raw-text-1-argv-json'],'依派遣復用/workspace/tiny-perceptron-vlm/.venv/bin/python，未執行完整uv sync安裝；dry-run只證明鎖檔命令解析。模型與asset目的地改新tmp以保留既有資料；其餘教材參數相同。',
    'public_source authentication disabled；固定revision/pin；不開.pt或assets；CPU原碼生成',
    'exit0、anonymous/source revision/prefix符合；不存在的asset目錄仍生成；manifest pinned；model stage joint/selected1000；uv frozen dry-run exit0',
    '實際argv與原stdout保存；執行當前examples/v2/text.messages.json，原歷史public_messages與該檔值完全相同。')
claim('c8','empirical','本次已知文字示例answer為正文兩點App建議，JSON generations保存實際生成而model標示所選權重。','兩點引用及generations/model一句',
    [('inference','reply268-328','prompt/generated IDs和model receipt的實際輸出來源。')],
    ['cli','cli-argv','raw-text-1-stdout-txt','saved-raw','public-text-pointers'],'只支持此固定已知validation示例的CPU介面；沒有估計未知問題成功率，也沒有重測heldout。',
    '1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。',
    '正文answer精確一致；1次generation、33generated IDs、EOS，與原始歷史stdout的IDs和answer完全一致',
    '直接比較本人實跑原stdout與保存raw text stdout，不使用result.json中作者answer_matches等判定欄位。',
    {'known_public_examples':'1 text conversation','current_generation_calls':'1 full-output + 1 one-token limit variation on same conversation','max_new_tokens':'128/full and 1/variation','full_output_generated_ids_including_eos':'33','device':'CPU','heldout_test_runs':'0'})
claim('c9','software','相同下載可重用；目的地任一檔內容不同就拒絕覆寫，應選新model-dir。','同下載可重用與拒絕覆寫段落',
    [('inference','fetch_public_export121-136','existing destination SHA比較在所有copy前完成，等同檔不重寫。')],
    ['cpu','short-cli','code-inference'],'核對此下載器目前契約；不把拒絕錯目的地當一般平台檔案安全保證。',
    '同內容可重用；改tokenizer內容必須FileExistsError且不覆寫',
    '第二次原CLI與fetch均成功重用；更改目的地tokenizer後Refusing to overwrite a different local inference file',
    '本人tmp下載目的地上的實際小變化；原公共資料與本機cache保持byte-identical。')
claim('c10','software','自己的best.pt作--init-checkpoint開新階段/optimizer/RNG/sampler；自己的同階段latest.pt作--resume恢復optimizer/RNG/sampler與累積steps/tokens，steps是總目標。','圖後best/latest解釋、最後兩段及練習',
    [('train','parser631-650;main724-819;checkpoint851-882;while913','fresh新AdamW/sampler/counters與resume load_state/restore_rng/counters，while比較總steps。'),('wrapper','local_source52-85;main164-175','fresh只best，resume只latest；兩flag互斥。'),('torch-checkpoint','general checkpoint section','optimizer state與progress保存是resume所需。')],
    ['cpu','cpu-code','training-pointers','code-train','code-wrapper'],'本次不執行任何trainer步驟；實跑邊界函数與RNG/sampler、手工optimizer state load，原main分支親自查核。exact指同契約恢復，非跨平台bitwise保證。',
    'best=fresh_stage_init/latest=exact_resume；交換名字拒絕；RNG和抽樣接續一致，fresh從0且無moments',
    '模式精確符合；兩交換ValueError；三種RNG再抽值完全相同；sampler draw7→16九個ID全部一致、fresh draw0；載入手工moments step7/LR.001，fresh state空，零optimizer.step',
    '原程式真正的BalancedSampler、rng_state/restore_rng與local_source函式執行；原trainer main的resume counters與fresh init邏輯以AST定位後精讀。')
claim('c11','software','本機wrapper核對自己的可信工作檔、資料/config/source與執行record，每次attempt新目錄；沒有依賴作者私人training storage。','本機stage wrapper說明',
    [('wrapper','verified_file36-49;local_source52-85;source_identity88-115;main118-251','本機manifest/assets/Git source/sidecar SHA以及local subprocess和exist_ok=False。'),('train','main729-775','torch.load weights_only=False只對可信own training checkpoint，config/data/architecture驗證。')],
    ['cpu','wrapper-help','code-wrapper','code-train'],'synthetic fixtures只檢查輸入契約，不宣稱它們是真訓練完成receipt；完整production或新local training本次未重跑。',
    'genuine matching own receipt才能fresh；incomplete fresh拒絕但latest resume允許；新output必須不存在；入口help可用',
    'local_source完整fixture接受正確模式，failed/incomplete best拒絕、latest可resume，已有attempt dir拒絕；wrapper --help exit0',
    '閱讀本機subprocess command與same-directory receipts生成分支；有界fixture只提供必要bytes/SHA，不載入任何synthetic.pt。')
claim('c12','software','公開safetensors只支援推論/評測；目前trainer沒有safe export訓練起點載入器，公開下載路徑不能直接代入init/resume。','最後句',
    [('wrapper','local_source52-85;main166-174','嚴格只收own best.pt/latest.pt與local execution/receipt。'),('train','main729-751','source_path僅torch.load .pt checkpoint schema/tokenizer/data契約，未使用safetensors初始化。'),('inference','InferenceAssistant215-257','safe loader為推論且eval，和training source分開。')],
    ['cpu','code-wrapper','code-train','code-inference'],'這是當前專案接口限制，不聲稱safetensors格式在原理上不能做transfer/fresh training。',
    'public-model directory或model.safetensors不能作own best/latest工作檔',
    '目前local_source檔名/sidecar門檻與trainer .pt schema分支確定排除；真safe package151keys可推論但無實際optimizer/RNG/sampler狀態',
    '執行原local_source正反例與真public loader，逐段確認沒有safe train source loader。')
claim('c13','software','圖中公開四安全檔→回答/評測；自己的完整工作檔另有optimizer/RNG/sampler/progress，best→新階段、latest→同階段恢復，與程式及正文一致。','p6-19-delivery-package-roles.svg全部標籤與箭頭',
    [('train','export235-281;fresh/resume775-819','公開推論檔與full checkpoint用途分工。'),('wrapper','local_source52-85','兩個檔名與兩flag的實際規則。')],
    ['figure640','figure360','cpu','metadata'],'圖是檔案角色與操作方向，不代表模型達成未知能力、所有平台復現或public package可續訓。',
    '兩條路與4files、best/latest標示匹配；640/360可實際查看',
    '本人實際render/view兩尺寸，無截斷；上路及兩個向右箭頭與實作模式一致',
    'inkscape原SVG分別--export-width=640/360生成PNG並view_image；逐一核對素材文字、箭頭方向與完整checkpoint字段。')
claim('c14','numeric','完整CPU操作頁提供7組已知情境供安裝/介面檢查，不可當未知問題成功率。','CPU操作頁連結末段',
    [('chat','parser task/tools/history API37-64','各情境實際CLI task對應。'),('torch-rng','lines1-14','單次固定環境行為支持範圍而非任意平台泛化。')],
    ['saved-raw','scenario-pointers'],'只重新核對原argv的7種情境計數與分工，未重新跑圖音/工具或heldout，也未用選定成功示例估計泛化率。',
    '7組場景，voice續對話多步共8chat+1append',
    '7 distinct tasks，8個chat argv全部device cpu，1個history append',
    '9份指定原argv先確認top-level形狀再讀/argv,/cwd；目前教材連結的操作頁段落作为待核主張。')

all_ids=[c['id'] for c in claims]
report={'schema_version':1,'review_stage':'technical','lesson_id':'19.11',
        'source':'course/chapters/19.md#19.11','reviewer_task':'/root/p6_fact_19_11','reviewer_context':'fresh',
        'source_sha256':sha(OUT/'inputs/19.11.md'),
        'figure_sha256':{'course/figures/p6-19-delivery-package-roles.svg':sha('course/figures/p6-19-delivery-package-roles.svg')},
        'verdict':'pass','claims':claims,'sources':sources,'artifacts':artifacts,'issues':[],
        'checks':{
            'factual_accuracy':{'status':'pass','details':'14項逐主張查核。官方checkpoint/RNG/commit/safe格式與原實作、真實HF metadata及raw receipts一致。','claim_ids':all_ids},
            'numeric_verification':{'status':'pass','details':'4x4=16files；7組/8chat/1append；1既知文字示例33IDs/EOS與正文答案及歷史raw精確相同。','claim_ids':['c3','c8','c14']},
            'figure_consistency':{'status':'pass','details':'本人用Inkscape渲染並view 640與360px；所有標籤、方向與best/latest功能一致。','claim_ids':['c13']},
            'source_verification':{'status':'pass','details':'親讀官方原文並保存HTTPS URL/version/raw SHA；先AST定位原碼，結果檔先topkeys再限定pointers，沒有使用舊審閱判定。','claim_ids':all_ids},
            'limitations':{'status':'pass','details':'固定已知CPU示例仅支持介面；無training update/heldout測試。local fixture不冒充真completed run，safe未支援training init不等於格式本質禁止training，exact不外推跨平台。','claim_ids':all_ids}},
        'actual_read_scope':{
            'course':'僅當前19.11及其SVG；原始UTF8 section bytes與整章frozen副本已保存，未把整章hash當目前節hash。',
            'linked_operating_docs':'docs/selftrained/v2-public-cpu-commands.md lines1-160與docs/selftrained/TRAINING.md lines1-240作待核主張/方法，不把作者當前結果說明當權威。',
            'guides':'本輪factual-reviewer-instructions、technical-review-guide與checker schema，未讀其他reader/technical報告。',
            'raw':'指定training/public-cpu raw原件，實讀pointers見training-inspected-pointers及saved-raw-check；原檔全bytes保留，notes/review/*scope_correction未讀。',
            'prior_report':'opaque-prior-report.json僅二進位副本保存，沒有讀其正文、判定或hash來作證。'},
        'frozen_input':{'path':PREFIX+'/inputs/19.md','sha256':sha(OUT/'inputs/19.md'),'meaning':'最初讀取的整章frozen bytes，並非目前整章版本宣告'},
        'execution_limits':{'training_steps':0,'heldout_runs':0,'paid_compute':False,'uploads':False,'training_dataset_downloads':False,'safeweights_policy':'大權重維持HF immutable artifact與已驗SHA的cache；永久保存必要header/metadata/raw證據，不強制加入普通Git。'}}
# Ensure the live section/figure still match this independently reviewed input.
raw=Path('course/chapters/19.md').read_bytes()
start=raw.index(b'## 19.11 ');end=raw.find(b'\n## ',start+1)
section=raw[start:end+1 if end!=-1 else len(raw)]
assert hashlib.sha256(section).hexdigest()==report['source_sha256']
Path('docs/technical-reviews/19.11.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
shutil.copyfile('docs/technical-reviews/19.11.json',OUT/'initial-review.json')
print('Wrote independent fresh report with',len(claims),'claims,',len(sources),'sources,',len(artifacts),'artifacts; preserved initial-review.json.')
