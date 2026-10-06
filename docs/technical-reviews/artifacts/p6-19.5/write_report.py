from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
prefix=str(BASE.relative_to(ROOT))+'/'
def sha(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
env={'python':'3.13.5','torch':'2.14.1+cpu','device':'cpu','threads':'2','platform':'Linux-6.18.44-x86_64-with-glibc2.41'}
artifacts=[]
def artifact(i,name,kind,desc,command=None,result=None):
    item={'id':i,'path':prefix+name,'sha256':sha(prefix+name),'kind':kind,'description':desc}
    if command is not None: item.update(command=command,result=result,environment=env)
    artifacts.append(item)
artifact('a_verify','verification-output.txt','execution','原例、角色／歷史／字表／context變化、ignore-index梯度、固定資料與既有CPU／SFT原始證據的實際CPU核對輸出。','/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.5/verify.py > docs/technical-reviews/artifacts/p6-19.5/verification-output.txt 2>&1','exit 0；所有有界斷言通過；没有訓練、heldout模型評測或新模型生成。')
artifact('a_membership','public-membership-output.json','execution','公開訊息與原始JSONL prefix的精確比對；只確認資料份，不進行模型生成或評分。','/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.5/public_membership.py > docs/technical-reviews/artifacts/p6-19.5/public-membership-output.json','exit 0；validation第74行唯一吻合，train/test沒有同一prefix。')
for i,name,kind,desc in [
('a_verify_code','verify.py','code','實際執行的獨立有界CPU驗證原碼。'),
('a_membership_code','public_membership.py','code','實際執行的資料prefix比對原碼。'),
('a_section','section.md','source_snapshot','19.5首次讀取的原始UTF-8 bytes。'),
('a_chapter','chapter-frozen-input.md','source_snapshot','首次讀取時整章的frozen input，僅19.5與必要19.3/19.4前文實際閱讀；不表示全章完成審閱。'),
('a_inspection','source-inspection.md','source_snapshot','本人的權威原文、函式定位、讀取範圍與支持範圍紀錄。'),
('a_pointers','inspected-json-pointers.json','source_snapshot','本人實際讀取原始JSON測量／provenance的具名pointers。'),
('a_examples','train-selected-raw-lines.jsonl','source_snapshot','固定資料中必要train示範的完整原始lines，非作者的結果摘要。'),
('a_example_map','train-selected-line-map.json','source_snapshot','原始train完整SHA及保存lines與來源行號映射。'),
('a_val','validation-public-example-raw-line.jsonl','source_snapshot','公開示範所對應的validation原始記錄完整一行。'),
('a_val_map','validation-public-example-provenance.json','source_snapshot','validation原始檔完整SHA及第74行、本人讀取pointers。'),
('a_instruct','instructgpt.pdf','source_snapshot','親自下載閱讀的arXiv:2203.02155v1原PDF。'),
('a_instruct_text','instructgpt.txt','source_snapshot','上述原PDF以pdftotext -layout擷取的可定位原文。'),
('a_gpt2','gpt2.pdf','source_snapshot','親自下載閱讀的OpenAI原始2019 GPT-2報告。'),
('a_gpt2_text','gpt2.txt','source_snapshot','上述GPT-2原PDF的pdftotext原文，核對§2／Eq(1)。'),
('a_torch_ce','torch-functional.py','source_snapshot','安裝版本v2.14.1的PyTorch官方functional原碼。'),
('a_torch_loss','torch-loss.py','source_snapshot','安裝版本v2.14.1的PyTorch官方CrossEntropyLoss原碼。'),
('a_downloads','source-fetch.json','source_snapshot','原始HTTPS下載URL、狀態、bytes及SHA紀錄（前三個來源）。'),
('a_tokenizer','reconstructed-tokenizer.json','source_snapshot','僅用train內容及固定alphabet重建並核對SFT指紋的字表。'),
('a_demo','original-demo.py','code','從19.5原始fence精確擷取並實際執行的原例。'),
]:artifact(i,name,kind,desc)
raws=[('a_cpu_argv','docs/selftrained/results/public-cpu-raw/text-1-argv.json'),('a_cpu_result','docs/selftrained/results/public-cpu-raw/text-1-result.json'),('a_cpu_stdout','docs/selftrained/results/public-cpu-raw/text-1-stdout.txt'),('a_cpu_stderr','docs/selftrained/results/public-cpu-raw/text-1-stderr.txt'),('a_messages','docs/selftrained/examples/v2/text.messages.json'),('a_sft_receipt','docs/selftrained/results/training-raw/moe-sft/raw/train-receipt.json'),('a_sft_execution','docs/selftrained/results/training-raw/moe-sft/raw/execution.json'),('a_sft_outer','docs/selftrained/results/training-raw/moe-sft/receipt.json'),('a_manifest','docs/selftrained/v2-manifest.json')]
for i,p in raws:artifact(i,'raw/'+p,'source_snapshot','已對原件全檔SHA核對的原始證據副本：'+p+'；實際讀取欄位見pointer紀錄。')
code_paths={'s_dataset':'tiny_perceptron/selftrained/dataset.py','s_tokenizer':'tiny_perceptron/selftrained/tokenizer.py','s_model':'tiny_perceptron/selftrained/model.py','s_train':'scripts/selftrained/train.py','s_core':'tiny_perceptron/model.py','s_attention':'tiny_perceptron/attention.py','s_inference':'tiny_perceptron/selftrained/inference.py','s_chat':'scripts/selftrained/chat.py','s_evaluate':'scripts/selftrained/evaluate.py'}
for i,p in code_paths.items():artifact('a_'+i,'frozen-source/'+p,'code','保留本人核對的完整原碼及全檔SHA；真正讀取函式範圍見source-inspection.md。')
sources=[
{'id':'s_instruct','kind':'paper','title':'Training language models to follow instructions with human feedback','url':'https://arxiv.org/pdf/2203.02155v1','version':'arXiv:2203.02155v1 (2022-03-04)','verified':True,'checked_original':True,'accessed_on':'2026-10-06','authority_reason':'方法提出者的原論文；可直接核對示範微調、模型評價與誠實／真實性支持範圍。','inspection_note':'親讀§3.1 Step 1、§3.5 SFT及§3.6；只支持示範微調機制與誠實測量限制，不支持本課程模型的品質。'},
{'id':'s_gpt2','kind':'paper','title':'Language Models are Unsupervised Multitask Learners','url':'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf','version':'OpenAI original 2019 technical report','verified':True,'checked_original':True,'accessed_on':'2026-10-06','authority_reason':'GPT-2方法作者發布的原報告，直接列出自回歸機率分解及序列內條件化目標。','inspection_note':'親讀§2 Eq(1)、p(output|input, task)與supervised objective只計序列子集段落；對應原文字擷取67–107、119–124行。'},
{'id':'s_torch','kind':'official_source','title':'PyTorch functional.cross_entropy and CrossEntropyLoss','url':'https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/functional.py','version':'PyTorch v2.14.1; installed torch 2.14.1+cpu','verified':True,'checked_original':True,'accessed_on':'2026-10-06','authority_reason':'PyTorch官方版本標籤原碼，與本次執行版本匹配。','inspection_note':'親讀functional.py 3478–3510及同標籤loss.py 1221–1244：ignore_index=-100排除class-index目標的loss／gradient與平均分母，不刪除模型輸入。'},
{'id':'s_verify','kind':'execution','title':'19.5 independent bounded CPU verification','verified':True,'artifact_id':'a_verify'},
{'id':'s_membership','kind':'execution','title':'Public prompt exact split membership verification','verified':True,'artifact_id':'a_membership'},
]
inspections={
's_dataset':'AST定位後讀48–95、108–116、174–241：train-only字表、group隔離、role輸入、assistant/EOS目標與labels移位。',
's_tokenizer':'AST定位後讀1–77：控制編號、Unicode單字元字表、顯式EOS及未知字顯示。',
's_model':'AST定位後讀25–53、108–130、153–185、357–419：shifted loss、文字核心與autoregressive generate／EOS停止；未讀description額外說明。',
's_train':'AST定位後讀1–103、284–298、519–572、689–792、902–1004：SFT資料選擇、lm-only參數、編碼與optimizer update分支；未執行訓練。',
's_core':'AST定位後讀1–50及92–100：因果Block及loss_sum按有效target计数。',
's_attention':'AST定位後讀10–28、48–74：causal allowed mask及cache下一字路徑。',
's_inference':'AST定位後讀139–173、267–328：公開末輪是user、沒有當前assistant target；generate後捕捉真實token。',
's_chat':'AST定位後讀67–110：讀公開messages、呼叫InferenceAssistant.reply及保存回傳生成結果。',
's_evaluate':'AST定位後讀225–306、395–457、495–570、567–631、659–704、874–899：finite keyword／exact內容、格式、intent分層及EOS stop_reason；只執行三個helper於train target及人造錯答，未執行heldout runner。'}
for i,p in code_paths.items():sources.append({'id':i,'kind':'repository_code','title':p,'path':p,'sha256':sha(p),'version':'2026-10-06 inspected working tree; full frozen source retained under p6-19.5/frozen-source','verified':True,'inspection_note':inspections[i]})

claims=[]
def ev(s,l,support):return {'source_id':s,'locator':l,'supports':support}
def claim(i,kind,statement,location,evidence,arts,scope,status='verified',verification=None):
    c={'id':i,'kind':kind,'statement':statement,'location':location,'status':status,'evidence':evidence,'artifact_ids':arts,'scope':scope}
    if verification:c['verification']=verification
    claims.append(c)
def verification(expected,observed,details,denominators=None,tolerance=None):
    v={'method':'executed','expected':expected,'observed':observed,'details':details}
    if denominators is not None:v['denominators']=denominators
    if tolerance is not None:v['tolerance']=tolerance
    return v

claim('c1','empirical','公開MoE在CPU以所列四則公開訊息生成所引兩點App回答；這是從validation選出的單一成功生成例。','course/chapters/19.md:252–257',[
ev('s_verify','verification-output.txt public_cpu_existing_run_verification；原始argv/result /actual_generations/0、/model/config/architecture、/public_source/revision','核對真實生成ID、stdout/result、CPU argv、MoE架構和 immutable revision，而非照抄作者摘要。'),
ev('s_membership','public-membership-output.json /membership/validation/0；原始text-tools-validation.jsonl:74','核對公開message prefix屬validation；train/test沒有相同prefix。')],['a_verify','a_membership','a_cpu_argv','a_cpu_result','a_cpu_stdout','a_val','a_val_map'],'只支持revision 979cdfacc588ad0536f1c64fff96f264571cf054、該prompt及既有一次CPU生成；沒有重跑模型，也不支持成功率。',verification=verification('raw prompt、CPU argv、stdout和generated_ids應相互吻合，輸出為所引兩點。','完全吻合；92 prompt tokens，33 generated tokens含1 EOS，1 generation，returncode 0。','按train-only重建字表解碼原生成IDs，核對原stdout/stderr SHA和model SHA；不是新的生成測試。',{'existing_selected_validation_prompt':'1','historical_generations':'1','generated_tokens_including_EOS':'33','new_model_generations':'0'}))
claim('c2','empirical','該單例已能斷言回答「同時用了兩種線索」，且「當前問題決定內容，先前要求決定兩點格式」。','course/chapters/19.md:257 第一句',[
ev('s_verify','verification-output.txt public_cpu_existing_run_verification；原始actual_generations/0 prompt_messages/raw_output','只能核實同一輸入下的內容及格式同時符合要求，沒有隔離當前問題或歷史要求的因果作用。'),
ev('s_inference','InferenceAssistant.reply lines 277–313','歷史確實被送入模型；送入輸入不等於证明这部分决定了输出。')],['a_verify','a_cpu_result','a_messages'],'目前支持範圍是觀察到的內容與格式對應。原文「用了／決定」將對應寫成確定因果；本輪不為修字追加模型能力或因果實驗。',status='unresolved',verification=verification('若按原文因果強度，需有既有同條件對照或其他能隔離線索作用的原始證據。','目前指定原始證據只有1個輸入與1個生成；沒有可隔離歷史要求作用的對照。','區分原始prefix與生成結果可证明沒有預填當前答案，但不能由單一符合要求的輸出判定內部使用來源。',{'selected_prompt':'1','historical_generation':'1','isolating_counterfactuals_in_checked_evidence':'0'}))
claim('c3','software','表中四個問題、條件與回答是固定資料中的train示範。','course/chapters/19.md:259–266',[
ev('s_verify','verification-output.txt table_training_example；train-selected-line-map.json；原始text-tools-train.jsonl:121,201,565,241','逐題檢查精確user/assistant配對、split=train、單句條件與計算器不可用system條件。')],['a_verify','a_examples','a_example_map','a_manifest'],'只是示範資料存在與語義條件對應；不把標準答案當成模型生成能力。',verification=verification('四組問題/答案及額外條件均在固定train資料中。','四組均精確找到；App行121、澄清行201、tool_unavailable行565、範圍外行241。','完整train原件SHA吻合manifest，保存必要raw lines與原行號映射。'))
claim('c4','concept','SFT用prompt下的希望回答作監督以調整參數、优化示範回答的條件機率。','course/chapters/19.md:259,270',[
ev('s_instruct','§3.1 Step 1；§3.5 Supervised fine-tuning','人類示範的期望行為用於監督式微調；不是偏好比較或PPO目標。'),
ev('s_gpt2','§2 Eq(1)及supervised objective subset-of-sequence段落','將task/input/output表示成序列，監督可只計指定輸出子集的條件預測目標。')],['a_instruct','a_instruct_text','a_gpt2','a_gpt2_text'],'這是目標與方法，不保證每筆訓練或每個新問題的機率／正確率均提升；成熟論文效果也不外推到本模型。')
claim('c5','software','本輪SFT只允许更新自己的文字核心。','course/chapters/19.md:270',[
ev('s_train','set_trainable lines 284–296；stage_records 84–103；main 767–777,902–929','sft無感知prefix分支僅lm.*可訓練，使用自己的前段checkpoint後進optimizer步驟。'),
ev('s_verify','verification-output.txt sft_parameter_scope、sft_raw_execution_binding、sft_receipt_crosscheck','小型實例執行trainability函式，核對原始8000步SFT receipt及outer/raw指紋绑定。')],['a_verify','a_sft_execution','a_sft_receipt','a_sft_outer','a_s_train'],'有界檢查只設定requires_grad，未訓練；已完成SFT由既有原始執行／收據支持。',verification=verification('sft可訓練參數均為lm.*；既有執行是completed sft。','小型實例每個可訓練參數均lm.*，感知參數全部凍結；原始execution returncode0，完成8000步。','讀真實函式与原始command，核對manifest與outer所列execution/train-receipt SHA；沒有新增optimizer update。'))
claim('c6','software','SFT作答代價只計每則assistant的文字與EOS，角色標记及system/user/tool文字被忽略。','course/chapters/19.md:270',[
ev('s_dataset','RecordEncoder.encode lines 174–220','assistant內容與EOS入labels；非assistant文字和role是-100；labels最後左移一格。'),
ev('s_verify','verification-output.txt original_encoding、history_role_variation','原例及加入system/tool/历史assistant的變化，均按期望解碼有效目標。')],['a_verify','a_s_dataset'],'包含历史assistant確認句的所有目標，不是只監督最後一則；這裡只說語言作答代價，MoE router auxiliary與另階段感知代價不被誤稱為assistant token。',verification=verification('原例assistant回答+EOS；歷史變化兩則assistant各自内容+EOS。','原例19 targets；變化有效目標為「好。<eos>請說明是地址、App還是卡片的問題。<eos>」，共2 EOS。','檢查labels非-100序列及固定role IDs均不在目標中。'))
claim('c7','software','EOS在此訊息格式表示該則回答結束，並作為assistant監督目標與生成停止條件。','course/chapters/19.md:270,289',[
ev('s_tokenizer','SPECIAL_NAMES lines 7–8；decode lines 43–56','EOS是固定特殊編號且可顯式解碼成<eos>。'),
ev('s_model','LimitedAssistant.generate lines 384–419','生成的eos_id觸發break。'),
ev('s_verify','original_encoding、history_role_variation、public_cpu_existing_run_verification','所有assistant末尾EOS有監督；原始生成最后token就是EOS。')],['a_verify','a_s_tokenizer','a_s_model'],'針對此專案每則message的結束協議；不表示全對話結束，也不保證其他輸入都會完整生成EOS。',verification=verification('原例末尾EOS是有效目標；此公開生成以EOS停止。','原例目標末尾2；公開generated_ids末尾2，eos=true。','固定CharacterTokenizer.eos_id=2，匹配字表、原始ids與generate分支。'))
claim('c8','numeric','本節短程式輸出「請說明是地址、App還是卡片的問題。<eos>」。','course/chapters/19.md:274–289',[
ev('s_verify','verification-output.txt original_fence_stdout、original_encoding','精確抽取原fence並實際執行，與原文輸出逐字相等。')],['a_verify','a_verify_code','a_demo'],'只展示編碼／目標選擇，不下載權重、不生成模型回答、不更新模型。',verification=verification('完整解碼值是18個回答字元加1 EOS與print換行。','stdout精確相等；33 input tokens，19 target tokens。','沒有用手寫預期取代原例執行；保留fence與完整對齊表。',tolerance='字串及整数計數精確相等。'))
claim('c9','software','-100只排除該預測位置的作答損失；使用者問題與歷史仍在input中。','course/chapters/19.md:289',[
ev('s_torch','functional.cross_entropy v2.14.1 lines 3478–3510；CrossEntropyLoss lines 1221–1244','ignore_index不貢獻loss／gradient與有效平均分母。'),
ev('s_core','loss_sum lines 92–100','按非IGNORE目標计数并计算sum CE。'),
ev('s_dataset','encode lines 175–220','input_ids獨立保存全部訊息，labels mask不刪input。'),
ev('s_verify','ignore_index_and_denominator、original_encoding、context_limit','原input有user問題；忽略位置gradient為0；loss與只取有效targets的算值吻合。')],['a_verify','a_torch_ce','a_torch_loss','a_s_core','a_s_dataset'],'class-index labels、此安裝版本；忽略預測位置不等於問題tokens不可作为attention上下文，也不等於共享參數完全無梯度。',verification=verification('mask不刪input；有效targets是19；忽略位置梯度0。','完整input包括user問題；count=19；CE sum 91.82449251362873與直接有效targets 91.82449251362871吻合；ignored gradient最大絕對值0。','在CPU以float64随机logits核對loss_sum与torch API；沒有更新模型。'))
claim('c10','concept','訓練用給定問題與回答前綴預測下一字，推論接續自身已生成字。','course/chapters/19.md:289',[
ev('s_gpt2','§2 Eq(1)，對前面符號條件化與sampling','支持自回歸條件預測的訓練及抽樣機制。'),
ev('s_dataset','encode lines 217–220','labels shift使assistant role預測首字、前一字預測下一字。'),
ev('s_attention','attention_mask 10–18；forward48–74','因果mask不能讀未来答案；cache累積preceding tokens。'),
ev('s_model','generate384–419','generated next_id追加到sequence并作下一步輸入。')],['a_gpt2','a_gpt2_text','a_verify','a_s_attention','a_s_model'],'此字元模型的teacher-forced目標與自回歸生成；不是在訓練時先完整生成答案，也不由此證明模型已學會目標行為。')
claim('c11','software','示範還包含選項記憶、格式改寫、工具請求與tool結果讀回。','course/chapters/19.md:291',[
ev('s_verify','history_or_format_training_example、tool_training_example；原始train行305,441,561,562','原始messages分別有記憶包選項、兩點改一句、calculator JSON call及tool JSON回填後結果。')],['a_verify','a_examples','a_example_map'],'資料存在與協議示範，不證明模型記憶任何新選項或工具題均正確。',verification=verification('這四種訊息模式在train資料中。','四種均有具體原始記錄、group和role訊息。','保存必要完整原始line及來源完整SHA。'))
claim('c12','software','文字／工具訓練、驗證與最後題按指定資料家族隔離。','course/chapters/19.md:291及必要19.3前文',[
ev('s_dataset','read_records48–81','資料讀取器拒絕id重複及group跨split。'),
ev('s_verify','fixed_data；sft_receipt_crosscheck','manifest12原檔指紋相符，text/tools三份group交集0，SFT train/validation數目與receipt吻合。')],['a_verify','a_manifest','a_sft_receipt'],'本節文字／工具的指定group標記隔離，不等於排除所有語義相似，也不把validation成功當test驗收；沒有重跑留出題模型評分。',verification=verification('原件固定且text/tools group不跨train/validation/test。','全部12檔SHA/bytes吻合；文字工具記錄15370/928/1460、group2483/60/90，三對交集0。','read_records實際驗證全原件，分母另按文字/工具task统计；不將讀資料視為推論。'))
claim('c13','software','驗收流程分開記錄有限內容／範圍／格式／澄清行為與完整生成結束。','course/chapters/19.md:291',[
ev('s_evaluate','score_reply240–306、format_pass225–237；summarize495–570；Generator.generate395–457','內容按keyword或exact評，格式獨立，intent/case分層包括澄清及範圍外；tokens與EOS stop_reason獨立記錄。'),
ev('s_verify','scoring_contract_synthetic_checks','只對四個train target及人造錯答執行helper，驗證semantic與format可分離；沒有將其寫成heldout結果。')],['a_verify','a_s_evaluate'],'這裡核對評分及記錄契約；有限keyword/rubric不是通用語義可靠性或任意問題可信度模型。完整結束指tokens/EOS記錄，沒有宣稱EOS成功率。',verification=verification('正確train target在有限rubric下semantic/format正確，人造错答semantic錯且可格式仍對。','四筆train target semantic/format均true；四个「不知道。」semantic false而format true；two_points正反例判準符合。','AST選擇normalize/format_pass/score_reply函式於人造回覆執行，heldout rows scored=0、model outputs scored=0。'))
claim('c14','concept','有限澄清／工具狀態／範圍示範、編碼成功及一題成功，不足以证明任意問題可靠程度或全體新問法成功。','course/chapters/19.md:257,268,289',[
ev('s_instruct','§3.6 honesty/truthfulness evaluation','原論文明說honesty難以直接測量，兩種truthfulness指標只涵蓋小部分意義。'),
ev('s_verify','原例標籤與公開單例檢查的分母','本次軟體檢查沒有模型學習更新；歷史成功例的樣本分母只有1。')],['a_instruct','a_instruct_text','a_verify'],'限制聲明本身正確；既有line257因果措辭仍需收窄，不把本節資料／輸出偷換成普遍誠實能力。')

report={'schema_version':1,'review_stage':'technical','lesson_id':'19.5','source':'course/chapters/19.md#19.5','reviewer_task':'/root/p6_fact_19_5','reviewer_context':'fresh','source_sha256':sha(prefix+'section.md'),'figure_sha256':{},'verdict':'revise','claims':claims,'sources':sources,'artifacts':artifacts,'read_scope':{'current_lesson':'19.5 full original UTF-8; lines250–300','necessary_context':'19.3 and19.4 manuscript, not old review conclusions','frozen_full_chapter_artifact':'a_chapter','reviewed_code_and_original_authorities':'a_inspection','original_json_pointers':'a_pointers','no_prior_report_read':True},'issues':[{'id':'i1','claim_id':'c2','status':'unresolved','location':'course/chapters/19.md:257 第一句','original':'這個回答同時用了兩種線索：當前問題決定內容，先前要求決定兩點格式。','details':'原始CPU紀錄和真實生成流程證明回答符合當前App需求及既有兩點格式要求，但單一prompt/output沒有隔離各線索的因果作用。「用了／決定」比查到的觀察證據強；送入歷史不等於單獨证明它决定格式。','impact':'容易將挑選出的成功展示當成已核實的模型內部因果使用證據。','recommendation':'改為「這個回答的內容對應當前 App 問題，也符合先前要求的兩點格式。」保留下一句的單例／不外推限制即可；不需要追加模型能力或因果實驗。'}], 'checks':{
'factual_accuracy':{'status':'revise','details':'四筆示範、SFT目標、EOS、歷史、字表、原例及生成出处核實；line257因果表述超過單例證據，需按i1收窄。','claim_ids':[c['id'] for c in claims]},
'numeric_verification':{'status':'pass','details':'原fence精確輸出、19targets、有效loss/梯度、固定資料SHA/計數与33個生成tokens核實。c2不是數字錯，而是因果支持範圍未建立。','claim_ids':['c1','c8','c9','c12']},
'figure_consistency':{'status':'not_applicable','details':'19.5本身沒有引用圖解；必要19.3/19.4前文中的圖不當成本節圖解證據。','claim_ids':[]},
'source_verification':{'status':'pass','details':'親讀原InstructGPT v1、OpenAI GPT-2原報告、PyTorch v2.14.1官方source；AST定位原碼并读取指定分支；原始JSON先keys/pointers、完整SHA与副本核對。','claim_ids':[c['id'] for c in claims]},
'limitations':{'status':'revise','details':'原文多数範圍限制正確：手寫／train答案不同於生成，單例不代表全體成功，標籤正確不證明學會；但line257同時寫確定因果需修正。未執行訓練、GPU、付費、新prompt生成、heldout evaluator、上傳或commit。','claim_ids':['c1','c2','c3','c4','c14']}}}
(ROOT/'docs/technical-reviews/19.5.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':'docs/technical-reviews/19.5.json','verdict':report['verdict'],'source_sha256':report['source_sha256'],'claims':len(claims),'unresolved':['c2'],'artifacts':len(artifacts)},ensure_ascii=False,indent=2))
