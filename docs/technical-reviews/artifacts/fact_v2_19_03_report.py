"""Build the independently evidenced report; never infer prior review status."""

import hashlib
import json
from pathlib import Path

from scripts.check_technical_reviews import sections

ROOT = Path.cwd()
BASE = "docs/technical-reviews/artifacts/"
PREFIX = "fact_v2_19_03_"


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    audit = json.loads((ROOT / f"{BASE}{PREFIX}audit.json").read_text())
    read = json.loads((ROOT / f"{BASE}{PREFIX}read_manifest.json").read_text())
    pytest = json.loads((ROOT / f"{BASE}{PREFIX}pytest.json").read_text())
    body = dict(sections(ROOT / "course/chapters/19.md"))["19.3"]
    assert body.encode() == (ROOT / f"{BASE}{PREFIX}section_19_3.md").read_bytes()
    assert hashlib.sha256(body.encode()).hexdigest() == audit["source_sha256"]
    artifacts = []

    def artifact(identifier, name, kind, description, command=None, result=None, environment=None):
        path = name if name.startswith("docs/") else f"{BASE}{PREFIX}{name}"
        item = {"id":identifier,"kind":kind,"path":path,"sha256":sha(path),"description":description}
        if kind == "execution":
            item.update(command=command,result=result,environment=environment)
        artifacts.append(item)

    artifact("a_read", "read_manifest.json", "source_snapshot", "本 task 的真讀順序、checker.sections 精確節文版本、原始下載 URL／HTTP200／SHA；B.7 因初次合併輸出截斷而完整重讀。")
    for target in read["section_reads"]:
        identifier = target["source"].split("#")[1].replace(".", "_")
        artifact("a_section_" + identifier,target["snapshot"],"source_snapshot","本人完整讀取的指定節文，未正規化 UTF-8。")
    artifact("a_prepare","prepare.py","code","擷取當次節文與下載原始官方文檔、公開 pinned data/test 的可重播程式。")
    artifact("a_replay","audit.py","code","直接執行本節原程式與種子43練習，獨立核對所有726筆欄位、真輸入bytes、標籤、分母、跨階段manifest。")
    artifact("a_report_code","report.py","code","由本task自己的真讀／執行證據建立17個逐項主張與五項checks的可重播報告程式；不读舊review。")
    artifact("a_audit","audit.json","execution","CPU完整逐筆重建与核對紀錄；含所有row原欄位、prompt ids、真媒體shape/dtype/bytes SHA、配置、交集與當次結果。",audit["command"],audit["result"],audit["environment"])
    artifact("a_pytest","pytest.json","execution","本 task 真正執行資料完整性test；7 passed、22 deselected，沒有跑模型訓練／生成測試。",pytest["command"],pytest["stdout"].strip(),pytest["environment"])
    for seed in (42,43):
        artifact(f"a_snippet{seed}",f"snippet_seed_{seed}.py","code",f"從實讀19.3提取的原snippet，seed={seed}。")
        artifact(f"a_stdout{seed}",f"snippet_seed_{seed}.txt","source_snapshot",f"a_audit中compile/exec這份snippet的實際標準輸出，seed={seed}。")
    artifact("a_crossval","sklearn_cross_validation_1_7_2.rst","source_snapshot","原始官方scikit-learn1.7.2 user guide，讀取開頭train/validation/test與group_cv、GroupKFold、StratifiedGroupKFold。")
    artifact("a_dummy","sklearn_dummy_1_7_2.py","source_snapshot","原始官方DummyClassifier1.7.2文檔與原碼；feature-independent fixed-label baseline定義。")
    artifact("a_collections","cpython_collections_3_13_5.rst","source_snapshot","CPython3.13.5官方Counter定義與未出現key回傳0。")
    artifact("a_stdtypes","cpython_stdtypes_3_13_5.rst","source_snapshot","CPython3.13.5官方set intersection、distinct elements和dict.items insertion order。")
    artifact("a_shortcut","shortcut_v2.txt","source_snapshot","真正原始論文的PDF文字，頁首arXiv2004.07780v2／19May2020；本人讀abstract、§1/3與§6.1/6.2。")
    artifact("a_pinned_data","pinned_data.json","source_snapshot","正文HTTPS固定commit1df3353的完整726筆data；當次HTTP200下載且與本地資料位元組一致。")
    artifact("a_pinned_tests","pinned_test_capstone.py","source_snapshot","正文HTTPS固定commit1df3353的原始test；當次HTTP200下载且與本地test位元組一致。")
    artifact("a_frozen","docs/course-experiments/capstone-evidence/deployment/data.json","source_snapshot","正式原始完整資料／manifest；全欄位與seed42當次重建精確相等。")
    artifact("a_selection","docs/course-experiments/capstone-selection.json","source_snapshot","真驗證選擇紀錄；joint75/84、dpo71/84，其來源hash亦當次核對。19.3僅用以查固定資料及test用途。")
    for stage in ("pretrain","sft","joint","dpo"):
        artifact(f"a_{stage}_report",f"docs/course-experiments/capstone-evidence/{stage}/train-report.json","source_snapshot","正式訓練原始配置／步數／有效位置分母／data manifest／code SHA；只核對資料沿用，未重訓或重報其CUDA性能。")
        artifact(f"a_{stage}_manifest",f"docs/course-experiments/capstone-evidence/{stage}/data-manifest.json","source_snapshot","該階段正式凍結家族切分，與当次seed42manifest精確一致。")
    artifact("a_deployment","docs/course-experiments/capstone-evidence/deployment/deployment-report.json","source_snapshot","部署各比較仍用同一manifest，當次逐欄核對完整manifest。")
    artifact("a_student","docs/course-experiments/capstone-evidence/student/student-report.json","source_snapshot","CE/KD比較仍用同一manifest，當次逐欄核對完整manifest。")
    sources = []

    def original(identifier,kind,title,url,version,reason,note):
        sources.append({"id":identifier,"kind":kind,"title":title,"url":url,"version":version,"verified":True,"checked_original":True,"accessed_on":"2026-10-04","authority_reason":reason,"inspection_note":note})

    original("s_group","official_docs","Cross-validation: evaluating estimator performance","https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst","scikit-learn1.7.2 fixed release tag","scikit-learn維護者原始user guide；直接定義train/validation/test用途與dependent groups留出。","本人讀lines10–86與631–704。groups是domain-specific；group外推需validation groups完全不出現在train。GroupKFold只保證給定group分離，沒有保證任意定義的group是合理家族。test若用來調參會洩漏；類別分布保持與群組保持是不同限制。原件見a_crossval。")
    original("s_shortcut","paper","Shortcut Learning in Deep Neural Networks","https://arxiv.org/abs/2004.07780v2","arXiv:2004.07780v2, 19May2020","Geirhos等作者原始研究perspective，直接討論benchmark高分與依賴非預期線索。","本人讀原PDF轉文字a_shortcut的abstract、§1（pp1–2）、§3（pp4–6）、§6.1/6.2（pp11–13）。論文定義shortcut為標準benchmark有效但更難條件不轉移的規則；§6.1要求強基線與區分dataset和ability。只支持方法上的限制，不能證明本專題某個已訓練模型用了哪個內部捷徑。")
    original("s_dummy","official_source","DummyClassifier constant/most_frequent strategies","https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/sklearn/dummy.py","scikit-learn1.7.2 fixed release tag","官方原碼與API docstring；明確定義忽略X的常數預測baseline。","本人讀a_dummy lines35–99；constant始終選指定label，most_frequent選fit y的多數label。本教材沒有呼叫DummyClassifier：它按該份清單直接計人工常數答案的成功上限，不能把它誤稱訓練出的純文字模型。")
    original("s_counter","official_docs","collections.Counter","https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/collections.rst","CPythonv3.13.5, matches executed Python3.13.5","Python官方文檔，對Counter與字典記數契約有直接權威。","本人讀a_collections lines243–278；iterable元素成為key，次數為value；缺key返回0；insertion order可影響印出的key順序但不影響基準計數。")
    original("s_sets","official_docs","set types and dictionary views","https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst","CPythonv3.13.5, matches executed Python3.13.5","Python官方文檔直接說明set、&和dict.items。","本人讀a_stdtypes lines4414–4505及4910–4932；set只存不同元素，&是交集；items按dict insertion order列name/rows。這只支持程式語義，不證明資料模型泛化。")

    def repository(identifier,path,note):
        sources.append({"id":identifier,"kind":"repository_code","title":path,"path":path,"sha256":sha(path),"version":"Working-tree source independently inspected on2026-10-04; exact full-file SHA binds source; executed scope is stated separately","inspection_note":note,"verified":True})

    repository("s_generator","tiny_perceptron/capstone.py","本人完整讀lines1–360，重點139–282 build_dataset：55數字、6圖片、20純音、30情境families；先以family選分組再全量派生record一起進份。雖物件先生成，分組沒有逐row shuffle；去重只在已分family內，故不造成洩漏。285–306是真媒體與含system/user/markers的前文；341–352偏好对僅從caller提供rows產生。本 task CPU重建全726筆。")
    repository("s_media","tiny_perceptron/multimodal.py","本人讀44–69 tone/log_mel與177–195 scene：真正sin波→STFT/mel→時間平均、RGB mask scene；capstone明暗/位移variant真轉tensor。沒有把pitch/color規格文字加入問題。")
    repository("s_stage","scripts/course_experiments/capstone.py","本人完整讀stage runner：97–119固定seed manifest、父checkpoint manifest相等、train-only文字或全部train；120偏好pairs只從train；201–213 train batch來源；246–276只validation而test_evaluated=False；312–369部署才從固定test評估。本 task核對正式各stage原始data manifest及原碼SHA，不重做長訓練。")
    repository("s_tests","tests/test_capstone.py","本人完整讀全檔。45–63 actual_input_sha把json前文編號與真正image/audio numpy bytes納入hash；66–98核對audio分布与1,2家族；當次執行的7個selected tests全部pass，其餘22 deselected不作已執行宣稱。")
    repository("s_student","scripts/course_experiments/capstone_student.py","本人讀35–149及run155–225：student沿用匹配teacher manifest，更新只吃splits[train]，validation/test只作完成後evaluation。student-report complete manifest當次與seed42重建相等。")
    repository("s_deploy","scripts/course_experiments/capstone_deployment.py","本人讀run272–380，特別278 build_dataset(ctx.seed)及358–367完整data.json writer。圖／音對照複製test row、保持文字並按新素材更新真值；本節只提出檢查需求，不重報19.6的模型能力分數。")
    sources.append({"id":"s_math","kind":"derivation","title":"分組題數與常數答案基準的手算","verified":True,"details":"數字0<=a<=b<=9共10+9+...+1=55families，44/5/6各6row得264/30/36。6個color/shape families固定4/1/1；每family pitch2×variation3×offset3×task3=54原衍生，其中image_color與image_shape的9種素材各在兩個pitch重複，去18row，餘36：144/36/36。audio20families各3row，16/2/2得48/6/6；各low/high8/1/1families，每family3variants得24/24、3/3、3/3。context30families各4row，24/3/3得96/12/12。總train264+144+48+96=552；validation30+36+6+12=84；test36+36+6+12=90；總726。audio固定high正確3/(3+3)=1/2；image test只有green square，9種variant/offset得color9/9、shape9/9。joint是pitch2×variation3×offset3=18，low/high各9。偏好用train tool_return44、rag24、style24，共92pairs。換入pitch使頻率群及DIRECT音高真值一起改變，應依新真值而非原答案評分；輸出與原答案不同本身不構成正確。"})
    sources.append({"id":"s_audit","kind":"execution","title":"19.3真正CPU重播與726筆全欄審計","verified":True,"artifact_id":"a_audit"})
    sources.append({"id":"s_pytest","kind":"execution","title":"selected dataset integrity tests","verified":True,"artifact_id":"a_pytest"})
    claims = []

    def evidence(identifier,locator,supports):
        return {"source_id":identifier,"locator":locator,"supports":supports}

    def claim(identifier,kind,statement,location,refs,ids,scope,expected=None,observed=None,details=None,denominators=None):
        item={"id":identifier,"kind":kind,"statement":statement,"location":location,"status":"verified","evidence":refs,"artifact_ids":ids,"scope":scope}
        if kind in {"software","numeric","empirical"}:
            item["verification"]={"method":"executed","expected":expected,"observed":observed,"details":details}
            if kind == "numeric":
                item["verification"]["tolerance"]="整數、字串、集合交集、JSON欄位與SHA均精確相等；媒體hash以當次CPU float32 tensor bytes比對，不做浮點近似外推。"
            if denominators:
                item["verification"]["denominators"]=denominators
        claims.append(item)

    group=evidence("s_group","lines10–86、631–704（group_cv/GroupKFold）","支持train更新、validation選設定、test終評，並以dependent group為留出單位。")
    shortcut=evidence("s_shortcut","§6.1 pp11–12及§6.2 pp12–13","支持強常數／非預期線索基線與窄資料成績不足證明一般能力的限制。")
    run=evidence("s_audit","seed42.records/manifest/overlaps_seed42、snippet_executions","支持當次精確資料、實際輸入、各份計數與原snippet輸出。")
    generator=evidence("s_generator","build_dataset lines139–282","支持家族建構、指定留出、份別、變體、manifest與baseline來源。")
    math=evidence("s_math","數字、圖片、audio、context四類逐項推導與常數基準","給出可追蹤手算；由a_audit真CPU核對。")
    claim("c1","concept","同一來源題目的改寫／工具狀態等相依衍生應以family一起留出，train更新、validation選設定、test最後核對。","lines116–118",[group], ["a_crossval"],"family判定依領域與來源規則；分組留出衡量未見家族，不自動排除所有語意相似或捷徑。")
    claim("c2","software","build_dataset(seed42)回傳splits與manifest；row包含所述欄位，首個train calculator問題為3+6且無圖片聲音。","line120與code lines127–142",[generator,run],["a_audit","a_stdout42"],"固定資料version v2、seed42；只是生成資料，不載入模型。","splits train/validation/test、first user 3+6等於多少？、task calculator、image/audio None","全部精確吻合；726筆所有原欄位與frozen JSON相等。","a_audit逐筆assert row fields及frozen equals；原snippet compile/exec真輸出。")
    claim("c3","software","set合併重複family，&找交集；Counter數task/pitch，dict.items給name/rows，manifest逐層索引讀人工baseline。","code lines128–142與lines143–145",[evidence("s_sets","set/dictionary views lines4414–4505、4910–4932","支持distinct set、intersection和items順序。"),evidence("s_counter","Counter lines243–278","支持iterable記數與缺key=0。"),run],["a_audit","a_stdout42","a_stdout43"],"CPython3.13.5本次執行；key印出順序可能隨家族順序變，但計數相同。","三對重疊0並印12種task、audio與image baseline","seed42與43均三對family重疊0，12task計數正確，Counter基準符合正文。","原snippet兩次執行；人工逐層取manifest核對全部task labels與majority baseline。")
    claim("c4","concept","零family交集仍需核對實際文字、system與媒體；完全相同輸入隔開也不能排除模板／常數答案捷徑。","line147與163後半",[group,shortcut,evidence("s_generator","prompt_ids299–306、modality_tensors285–296","family標記不輸入模型；實際前文system/user/markers與真tensor才是模型可見素材。")],["a_crossval","a_shortcut","a_audit"],"只排除當前輸入的位元組重複，不等於語意獨立或真能力；不同family名稱本身不足保證分離。")
    claim("c5","numeric","version capstone-small-world-v2種子42有552/84/90筆，共726筆。","lines149、161",[generator,math,run],["a_audit","a_frozen","a_pinned_data"],"固定合成規則，raw rows先以family分派，再在家族內去掉完全重複衍生；非726份自然素材。","552/84/90總726","精確552/84/90，原本地與pinned完整資料均726。","手算四類family×去重後衍生數並CPU逐筆len與JSON全欄等式核對。")
    claim("c6","software","數字按無序數字對；圖片按color/shape並固定blue circle驗證、green square test；情境任務同編號一起切。","line149",[generator,run],["a_audit"],"每份按素材組合留出，基本red/green/blue與square/circle均已在train；因此不是新概念測試。","四類families全份分離、指定圖片留出且所有基本類別仍在train","train四個圖片組合含全部3colors/2shapes；validation僅blue circle、test僅green square；family每份分配吻合。","逐row核family/schema/label，按category核每family成員與variant/offset，全split交集為空。")
    claim("c7","numeric","numbers:1:2的六筆衍生含正反求和、兩個工具關閉、concept及tool-return，只在test。","line151",[generator,run,evidence("s_tests","test_one_plus_two... lines84–98","直接測所有後裔test-only並查exact arguments。")],["a_audit","a_pytest"],"留出是數字組合，單獨的1或2仍可在train其他family；示範真生成成功屬其他節。","test6、train0、validation0，包含1+2/2+1 tool requests","全部精確符合，tools答案1+2與2+1；selected test通過。","audit.reserved_1_2保存6筆完整問題與規格；逐份搜尋並計Counter與task集合。")
    claim("c8","numeric","純音每類8/1/1個train/validation/test基頻家族，各3變體，因此test low3/high3。","line153前半",[generator,math,run],["a_audit","a_pytest"],"這裡high/low是generator兩群純音分類；不是12章300Hz界線、自然聽覺類別或語音能力。","各low/highfamilies8/1/1、rows24/3/3","seed42精確families8/1/1與rows24/3/3；seed43同分布；pytest seeds0/1/7/42/99均符合。","audit按family pitch與3variation集合全檢；7selected tests含5seed audio assertions。")
    claim("c9","numeric","test純音始終答high的人工固定基準是3/6=0.5。","line153後半與code140",[math,run,evidence("s_dummy","DummyClassifier docstring35–99","支持忽略特徵的常數答案比較概念。")],["a_audit","a_stdout42"],"依test標籤計人工固定答案，不是執行模型、不等於train-fit DummyClassifier或純文字訓練模型。","3 correct /6 total","labels high3、low3，snippet印3/6；manifest audio majority accuracy0.5。","逐row label規則、Counter與全task baseline重新計；手算3/(3+3)。")
    claim("c10","numeric","test圖音joint有18筆且音高low9/high9、問句相同。","line155前半",[math,run,generator],["a_audit"],"一個green square家族的2pitches×3variation×3offset；多筆變體不當作18個獨立自然來源。","18 joint、low9 high9、single user/system","精確18、9/9，所有user都是圖片顏色與聲音高低？。","全18row完整trace可查原user/system、image/audio規格、tensorSHA與答案，不只summary。")
    claim("c11","concept","換聲測試需保持文字／圖片並按换入音高更新真值；分數或輸出改變本身不足證明聽對。","line155後半",[shortcut,math,evidence("s_deploy","run audio_rows lines324–339","原碼只改pitch、新answer與id，保留image、system、user。")],["a_section_12_10","a_section_19_6","a_shortcut"],"此節提的是正確評分方法，沒有新增模型介入結果；19.6結果不可藉本次資料audit宣稱重新生成。")
    claim("c12","numeric","test image_color固定green與image_shape固定square基準各9/9。","line157",[math,run,shortcut],["a_audit","a_stdout42"],"原9幅均green square；100%人工constant baseline足以說明原題高分不能單獨證明看圖，不宣稱模型一定未看图。","green9/9、square9/9","精確兩項9/9，manifest和真pixel素材標籤全核對。","test family只有green square；3offset×3variant=9，所有兩類row真值各固定；重新計baseline。")
    claim("c13","software","數字題兩種問法在各份共用，這份capstone沒有另做陌生措辭考卷。","line159",[generator,evidence("s_stage","train_stage97–123、248–274與run_deployment312–339","訓練／選擇／最後評估只使用build_dataset三份；無新措辭支線。"),run],["a_audit","a_section_B_7"],"限此固定capstone配方與目前source；B.7是另個選卡實驗，不能把它的陌生措辭成績貼成capstone結果。","calculator/unavailable只有a+b等於多少？與請算b加a兩個模板，test新數字家族","全726筆matches generator；每份numeric模板相同，runner無extra wording split。","完整generator字串構造與all row fields CPU trace核對；不拿B.7的能力分數當本節數據。")
    claim("c14","numeric","train/validation/test指紋前綴依序e0a419727952/afedd4dc84ce/aef4b7a876ff。","line161",[run,evidence("s_generator","digest30–31、manifest sha256268","SHA對ensure_ascii=False/sort_keys=True的完整ordered row清單編碼。")],["a_audit","a_frozen","a_pinned_data"],"三份原文用途不同、完整hash彼此不同；這不是媒體tensor指紋，且同一清單重排也會改這個SHA。","正文三個12hex前綴與manifest完整SHA吻合","train e0a419727952eceb700e5bc8f47bbc7d7c961ba4e14acca3f68b73c360e20cf9；validation afedd4dc84ce22a8867f7ebc5cc7804c7403d01d1cf29a68cf04985c2ff208a9；test aef4b7a876ff8966c0a7413d342017fd69038855ac8f55c28b7c66edb8808ecd。","以reviewer獨立hashlib/json重算三份完整hash，原pinned与local data byte equality一起核對。")
    claim("c15","empirical","正式四階段及比較方法沿用同份固定split/manifest，偏好更新不收回test rows。","lines118、161",[evidence("s_stage","train_stage97–123、run_deployment322–332","固定manifest父檔守衛；train rows与DPO train preference來源。"),evidence("s_student","run162–189","CE/KD只用train且teacher manifest必須匹配。"),evidence("s_audit","formal_stages/comparison_manifests/preference_train_pairs/selection","此次執行審計正式各manifest與來源code SHA。")],["a_audit","a_pretrain_report","a_pretrain_manifest","a_sft_report","a_sft_manifest","a_joint_report","a_joint_manifest","a_dpo_report","a_dpo_manifest","a_deployment","a_student","a_selection"],"審計正式保存配置與全manifest，未重訓300/1400/600/100steps或重播模型能力。pretrain/SFT只用train文字subset，joint全部train，DPO92偏好pair＋train重播；同split不表示各stage每次抽同樣row。","四stage/部署/student完整manifest與seed42都一致，DPO pairs train-only","全部manifest精確相等且正式train code SHA與現在源碼一致；92pairs只來自train；test_evaluated=False於四train reports。","真CPU讀全部原始JSON、重建manifest並逐欄比較；selection来源hash與84題分母核對，不將歷史CUDA訓練時間改稱CPU結果。",{"dataset_version":"capstone-small-world-v2","seed":42,"split_rows":{"train":552,"validation":84,"test":90},"formal_main_stages":4,"comparison_reports":2,"dpo_train_pairs":92,"reviewer_optimizer_updates":0,"reviewer_model_generations":0,"timing_scope":"CPU資料/原始紀錄審計；不包含或外推歷史CUDA訓練、模型能力或速度。"})
    claim("c16","empirical","在CPU真正前文編號與image/audio feature bytes指紋核對，三對split的family與actual input交集都是0。","line163",[evidence("s_tests","actual_input_sha45–50、test_family_split53–63","真正參與hash的是system/user序列與媒體tensor bytes，不是row id或family名。"),evidence("s_pytest","7passed output中的family_split test","此task實際執行所述test。"),run],["a_audit","a_pytest"],"僅bit-exact輸入分離；模板、常數答案與語意相似仍可能共享。CPU float32合成媒體摘要就是此模型真正接收的內容，非未計算聲音標籤名。","3pairs family overlap0/actual input overlap0","seed42三對交集全空；actual inputs distinct552/84/90，完整逐筆SHA/shape/dtype保存；pytest原test pass。","對json.dumps(prompt_ids(row)).encode建立SHA，再順序追加真image/audio numpy bytes，batch變換前輸入；無模型權重或訓練依赖。",{"seed":42,"split_rows":{"train":552,"validation":84,"test":90},"split_family_counts":{"train":88,"validation":11,"test":12},"split_pairs":3,"media_dtype":"torch.float32","device":"cpu","optimizer_updates":0,"test_cases_passed":7,"test_cases_deselected":22})
    claim("c17","software","seed42改43仍零family/real-input交集，但部分family分配及split SHA改變；正式配方不能依test分數挑seed。","line165",[group,generator,evidence("s_audit","seed43.changed_family_memberships/overlaps與snippet_executions seed43","本次真的改只有seed並執行教材練習。")],["a_audit","a_stdout43"],"指定modalities blue circle/green square不隨seed變，numbers/audio/context中部分分配變；這不是全種子品質比較。","seed43三對無交叉、分配至少部分改變","train/validation/test symmetric difference36/16/20個families；三對交集均0，三份hash改變；圖片指定留出仍固定。","seed43獨立重建並真生成全部媒体摘要，逐row同樣標籤/分母核對；只做資料exercise，不評模型。")
    report={
        "schema_version":1,"review_stage":"technical","lesson_id":"19.3","source":"course/chapters/19.md#19.3",
        "reviewer_task":"/root/integration_technical_coordinator/fact_v2_19_03","reviewer_context":"fresh",
        "source_sha256":audit["source_sha256"],"figure_sha256":{},"verdict":"pass",
        "claims":claims,"sources":sources,"artifacts":artifacts,"issues":[],
        "checks":{
            "factual_accuracy":{"status":"pass","details":"17個逐項concept/numeric/software/empirical主張，原文完整真讀、官方原文版本及原碼定位皆可追溯；全726row重建與原始欄位一致。","claim_ids":[c["id"] for c in claims]},
            "numeric_verification":{"status":"pass","details":"手算4類family×衍生規則、dedup後數量；CPU真執行原snippet與seed43練習。552/84/90、726、1,2六筆、audio3/6、joint9+9、image9/9與全部SHA精確核對。","claim_ids":["c5","c7","c8","c9","c10","c12","c14"]},
            "figure_consistency":{"status":"not_applicable","details":"本人完整讀19.3確認本節沒有SVG或其他圖引用；13.1前置的圖未在19.3呈現且不構成本節圖證據。","claim_ids":[]},
            "source_verification":{"status":"pass","details":"原始scikit-learn1.7.2 grouped CV/DummyClassifier及CPython3.13.5官方文檔真讀；原論文arXiv2004.07780v2頁首、§1/3/6.1/6.2真讀。正文commit1df3353的data與test當次HTTP200且byte-equal本地；固定SHA綁source/artifacts。候選庫只使用原件，不讀其他review/作者歷史，不猜author_tasks。","claim_ids":[c["id"] for c in claims]},
            "limitations":{"status":"pass","details":"保存bit-exact輸入分離與語意/模板相似的区别；固定多數基準是按test真值人工計算，非模型結果；基本類別已見、組合留出与陌生措辭分開；image test固定green square；資料audit不當能力、CUDA加速、語音辨識或一般中文理解證據。沒有載入weights、更新器或重訓。","claim_ids":["c1","c4","c6","c8","c9","c10","c11","c12","c13","c14","c15","c16","c17"]},
        },
        "inspection_record":{"read_manifest_artifact":"a_read","execution_artifacts":["a_audit","a_pytest"],"no_subagents":True,"old_reviews_read":False,"author_tasks":"Not supplied or guessed; assignment independence remains coordinator responsibility."},
    }
    (ROOT / "docs/technical-reviews/19.3.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"lesson":"19.3","verdict":report["verdict"],"claims":len(claims),"sources":len(sources),"artifacts":len(artifacts),"source_sha256":report["source_sha256"]},indent=2))


if __name__ == "__main__":
    main()
