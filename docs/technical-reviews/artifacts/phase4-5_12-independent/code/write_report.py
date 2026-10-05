import hashlib
import json
from pathlib import Path
ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/technical-reviews/artifacts/phase4-5_12-independent'
REPORT=ROOT/'docs/technical-reviews/5.12.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path):return Path(path).relative_to(ROOT).as_posix()
env=json.loads((OUT/'cpu-environment.json').read_text())
execution_env={'python':env['python'],'executable':env['executable'],'torch':env['torch'],'device':'cpu',
 'cuda_build':env['cuda_build'],'cuda_available':env['cuda_available'],'bound':'45 seconds',
 'network':'offline for CPU checks','model_training':'not executed'}
receipt=json.loads((OUT/'inspection-execution.json').read_text())
original=json.loads((OUT/'original-cpu/execution.json').read_text())
extraction=json.loads((OUT/'extraction/extraction.json').read_text())
results=json.loads((OUT/'inspection-results.json').read_text())
assert receipt['exit_code']==0 and original['exit_code']==0
assert receipt['executed_script_sha256']==sha(OUT/'code/inspect_cpu.py')
artifacts=[]
by_path={}
def artifact(identifier,path,kind,description,**extra):
 p=OUT/path
 a={'id':identifier,'path':rel(p),'sha256':sha(p),'kind':kind,'description':description,**extra}
 artifacts.append(a);by_path[path]=identifier
artifact('original-execution','original-cpu/execution.json','execution','本輪原始 Python fence 的實際 CPU/offline helper 45秒執行紀錄',
 command='.venv/bin/python docs/review-tools/section_facts.py course/chapters/05.md#5.12 --output outputs/phase4-5_12-independent-original-cpu --execute --timeout 45',
 result='exit 0; attempted fence 1; 4 common, 8 union, score 0.5; stderr empty',environment=execution_env)
artifact('cpu-inspection','inspection-results.json','execution','自行有界 CPU 查驗 n=3、空集合、片段限制、原始資料與指定歷史實作；重建原報告六份切分及held-out分母',
 command=receipt['command'],result='exit 0, all assertions pass; all six split hashes equal original JSON; no model construction or training',environment=execution_env)
for identifier,path,kind,description in [
 ('original-code','extraction/fence-1.py','code','未改寫的本節原始 fence bytes'),
 ('n3-code','code/fence-n3.py','code','原始 fence 中兩個 grams 呼叫加入 n=3 的真正已執行練習變化'),
 ('inspection-code','code/inspect_cpu.py','code','本輪自寫 CPU inspection；僅執行原始切分／去重／窗口契約與數值計算'),
 ('inspection-notes','inspection-notes.md','source_snapshot','親讀來源的定位、支持範圍、讀取範圍、工具限制與圖不適用說明'),
 ('input-provenance','input-provenance.json','source_snapshot','輸入原始身份、SHA-256、歷史revision與永久snapshot清單'),
 ('lee-pdf','sources/lee-2022.acl-long.577.pdf','source_snapshot','ACL出版社取得的 Lee et al. 原始論文，親讀§4.2/§5.2/§5.3/§6.1/Appendix A'),
 ('lee-text','sources/lee-2022.acl-long.577.txt','source_snapshot','本輪從原論文 PDF 實際 pdftotext -layout 產物，保留印刷頁碼'),
 ('python-types','sources/python-3.13.5-stdtypes.rst','source_snapshot','CPython官方v3.13.5原始文件：切片、range、set、union與intersection'),
 ('python-expressions','sources/python-3.13.5-expressions.rst','source_snapshot','CPython官方v3.13.5原始文件：set display/comprehension'),
 ('python-functions','sources/python-3.13.5-functions.rst','source_snapshot','CPython官方v3.13.5原始文件：len與sorted'),
 ('stanford-character','sources/stanford-kgram.html','source_snapshot','作者官方書籍網站 k-gram 字元定義，親讀castle例子'),
 ('stanford-overlap','sources/stanford-shingling.html','source_snapshot','作者官方書籍網站§19.6：完整指紋與近重複、shingling及預設門檻'),
 ('sklearn-groups','sources/sklearn-GroupShuffleSplit.html','source_snapshot','scikit-learn 1.9.1 官方 GroupShuffleSplit 文件；方法佐證，未聲稱本repo使用其API'),
 ('historical-text','inputs/historical/scripts/course_experiments/text.py','code','real_text指定原run revision5581462…原始text.py，hash等於報告code_sha256'),
 ('historical-common','inputs/historical/scripts/course_experiments/common.py','code','real_text指定原run revision5581462…原始common.py，hash等於報告code_sha256'),
 ('original-results','inputs/docs/course-experiments/results/real_text.json','source_snapshot','本次親讀與程式核對的原始GPU實跑結果；未重新訓練或生成新的模型成績'),
]:artifact(identifier,path,kind,description)
for path in sorted(OUT.rglob('*')):
 if (not path.is_file() or path.name=='manifest.json' or path.relative_to(OUT).as_posix() in by_path
     or path.name.startswith(('write-report.','checker.','checker-receipt','finalize.'))):continue
 name=path.relative_to(OUT).as_posix()
 # Every retained file is versioned; reports distinguish actual successful execution from preliminary harness failure.
 kind='code' if path.suffix=='.py' else 'source_snapshot'
 identifier='file-'+name.replace('/','--').replace('.','_')
 artifact(identifier,name,kind,'本輪永久證據配套檔：'+name)

def external(identifier,title,kind,url,version,authority,note,artifact_ids):
 return {'id':identifier,'title':title,'kind':kind,'url':url,'version':version,'authority_reason':authority,
 'verified':True,'checked_original':True,'accessed_on':'2026-10-05','inspection_note':note,'artifact_ids':artifact_ids}
sources=[
 external('lee','Deduplicating Training Data Makes Language Models Better','paper',
 'https://aclanthology.org/2022.acl-long.577.pdf','ACL 2022, DOI 10.18653/v1/2022.acl-long.577, pp.8424–8445',
 '原作者論文由ACL正式出版；本輪直接取得出版社PDF和論文條目。',
 '親讀§4.2 p.8427 Jaccard公式/n-gram表示/5-grams與edit filter，Table1 p.8428，§5.2/5.3/6.1 p.8429，AppendixA p.8436；不是用舊review結論。',['lee-pdf','lee-text']),
 external('ir-character','Introduction to Information Retrieval: k-gram indexes for wildcard queries','official_docs',
 'https://nlp.stanford.edu/IR-book/html/htmledition/k-gram-indexes-for-wildcard-queries-1.html','Manning, Raghavan, Schütze 2008, author-hosted online edition, §3.2.2',
 'Stanford作者官方教材網站；直接讀其字元n-gram定義。',
 '親讀「A k-gram is a sequence of k characters」與castle→cas/ast/stl例子；書中詞邊界$不適用本節未補邊界的函數。',['stanford-character']),
 external('ir-overlap','Introduction to Information Retrieval: Near-duplicates and shingling','official_docs',
 'https://nlp.stanford.edu/IR-book/html/htmledition/near-duplicates-and-shingling-1.html','Manning, Raghavan, Schütze 2008, author-hosted online edition, §19.6',
 'Stanford作者官方書籍原文；直接讀完整指紋和近重複差異與shingling。',
 '親讀fingerprint無法捕捉少字變化段落、consecutive term sequences、預設門檻與syntactic clusters；HTML文字會遺失公式圖，Jaccard公式由ACL原PDF另行核實。',['stanford-overlap']),
 external('python-types','Python Standard Types','official_docs',
 'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst','CPython tag v3.13.5; matches installed Python 3.13.5',
 'Python官方CPython repository中隨版本發布的文件原稿。',
 '親讀行1060–1066切片右端排除、1383–1435 range預設起點/步長/stop排除、4409–4508 sets distinct/len/union/intersection。',['python-types']),
 external('python-expressions','Python Language Reference: Set displays and comprehensions','official_docs',
 'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/reference/expressions.rst','CPython tag v3.13.5; matches installed Python 3.13.5',
 'Python官方語言參考版本原稿。','親讀Set displays行296–321，comprehension建構新的mutable set。',['python-expressions']),
 external('python-functions','Python Built-in Functions: len and sorted','official_docs',
 'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/functions.rst','CPython tag v3.13.5; matches installed Python 3.13.5',
 'Python官方builtins文件版本原稿。','親讀len行1127起以及sorted行1848起，sorted回傳new sorted list，不改計分集合。',['python-functions']),
 external('group-doc','scikit-learn GroupShuffleSplit','official_docs',
 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html','scikit-learn 1.9.1 documentation, title and version-switcher marker checked',
 'scikit-learn官方維護的分組切分API文件。',
 '親讀class description、group-count note、完整兩fold example和split(X,y,groups)參數；group標籤與目標y是不同資訊，不在本輪執行該API。',['sklearn-groups']),
 {'id':'repo-text','kind':'repository_code','title':'Original real_text preprocessing and entrypoint',
 'path':rel(OUT/'inputs/historical/scripts/course_experiments/text.py'),
 'sha256':sha(OUT/'inputs/historical/scripts/course_experiments/text.py'),'version':'git 5581462ef01959636425eb7795ae3153142dbfeb; hash matches original JSON code_sha256',
 'verified':True,'inspection_note':'親讀_json_bytes/_digest/_save_splits行39–62、_deduplicate_text行91–100和run_real_text行317–345；只完整正規化內容去重，family替换成完整SHA-256；先切分後fit，無近重複聚類。'},
 {'id':'repo-common','kind':'repository_code','title':'Original split_records and text_examples contracts',
 'path':rel(OUT/'inputs/historical/scripts/course_experiments/common.py'),
 'sha256':sha(OUT/'inputs/historical/scripts/course_experiments/common.py'),'version':'git 5581462ef01959636425eb7795ae3153142dbfeb; hash matches original JSON code_sha256',
 'verified':True,'inspection_note':'親讀split_records行50–70、text_examples行77–97、_nll行106–119、fit_lm行123–201、evaluate_lm行243–304：先按family切分，byte/BOS/EOS後窗口；只train交給fit，before/after評測validation/test。未啟動完整訓練。'},
 {'id':'original-cpu','kind':'execution','title':'Original unmodified 5.12 fence CPU execution','verified':True,'artifact_id':'original-execution'},
 {'id':'independent-cpu','kind':'execution','title':'Independent bounded numerical and original-result verification','verified':True,'artifact_id':'cpu-inspection'},
]
def evidence(source,locator,supports):return {'source_id':source,'locator':locator,'supports':supports}
def claim(identifier,kind,statement,location,scope,refs,artifact_ids,verification=None):
 item={'id':identifier,'kind':kind,'statement':statement,'location':location,'scope':scope,
 'status':'verified','evidence':refs,'artifact_ids':artifact_ids}
 if verification is not None:item['verification']=verification
 return item
claims=[
 claim('C1','concept','只改一字可能讓完整指紋不同，仍是需要另外檢查的相近材料；完整匹配不能涵蓋近重複。',
 'course/chapters/05.md:430','完整內容身份與表面相近是不同檢查；不主張只差一字必然可刪除或答案同義。',[
 evidence('lee','§5.2 p.8429 and Table1 p.8428','少量字段/字詞變化會使exact string matching漏掉近重複；沒有替本例規定門檻。'),
 evidence('ir-overlap','§19.6 fingerprint and near-duplication paragraphs','整篇fingerprint匹配無法處理內容只差few characters的near duplication。'),
 evidence('independent-cpu','inspection-results.json toy.left_right_have_distinct_full_hash','兩個指定句子完整正規化SHA-256確實不同；此運算不推斷一般碰撞率。')],['cpu-inspection']),
 claim('C2','concept','character n-gram是連續n個字元片段；Jaccard相似度是交集基數除以聯集基數，與訓練前文窗口用途不同。',
 'course/chapters/05.md:432,450','本節以Python字串中的中文字元、不加詞邊界、不計重複次數，形成集合；不是詞級BPE、MinHash或訓練語言模型。',[
 evidence('ir-character','§3.2.2 paragraph A k-gram is a sequence of k characters','連續字元片段定義；該書邊界符與索引用途不自動套到本函數。'),
 evidence('lee','§4.2 p.8427, displayed Jaccard equation','n-gram集合交/聯基數比的精確公式。'),
 evidence('repo-common','text_examples lines77–97','真正模型輸入是token化後的窗口；本節文字比較函數沒有模型輸入/更新流程。')],['lee-pdf','stanford-character','historical-common','original-code']),
 claim('N1','numeric','七字句n=2各有六個不同片段，共有四個/聯集八個=0.5；n=3各有五個，共有三個/聯集七個=3/7≈0.429。',
 'course/chapters/05.md:432,448,450,454','僅指定兩個七字句的手工算例及真正執行的n=3練習；不是資料集聚類品質或模型得分。',[
 evidence('original-cpu','original-cpu/stdout.txt, fence1','原碼共有圓形/形在/紅色/色圓四段，聯集八段及0.5。'),
 evidence('independent-cpu','inspection-results.json toy.n2 and toy.n3; code/fence-n3.py','原碼與n=3兩呼叫變化均實跑；n=3紅色圓/色圓形/圓形在，3/7。')],['original-execution','cpu-inspection','n3-code'],
 {'method':'executed','expected':'n2: 6/6 segments, 4 intersection, 8 union, 0.5; n3: 5/5, 3 intersection, 7 union, 3/7',
 'observed':'n2 score 0.5 exactly; n3 score 0.42857142857142855; displayed 0.429 has <0.0005 rounding error',
 'details':'完整片段列表與assertions保存在inspection-results.json与代码；改變n保留其餘運算。','tolerance':'整數與0.5精確相等；3/7按Python浮點比值相等；三位小數四捨五入絕對差<0.0005。'}),
 claim('S1','software','range/slicing/set-comprehension/&/|/len/sorted按本文解釋運作；重复片段只留一份，兩集合皆空則原碼除以0需定規則。',
 'course/chapters/05.md:435–450','教學函數預期正整數n且字串至少n長；正文不是宣稱此函數已是通用輸入驗證或空集合處理器。',[
 evidence('python-types','stdtypes.rst Common Sequence Operations note4 lines1060–1066; range lines1383–1435; Set Types lines4409–4508','切片右端不含，range預設從0、步長1、stop不含；集合唯一元素，&交集、|聯集、len基數。'),
 evidence('python-expressions','expressions.rst Set displays lines296–321','大括號comprehension逐項建構set。'),
 evidence('python-functions','functions.rst len lines1127–1134, sorted lines1848–1878','len回傳項目數；sorted回傳新排序清單，原程式只用於print。'),
 evidence('independent-cpu','inspection-results.json toy.empty_edges and repeated_grams_unique','空字串與兩個一字句n2均真實ZeroDivisionError；人人人只保留一個人人片段。')],['original-execution','cpu-inspection','original-code','inspection-code'],
 {'method':'executed','expected':'原碼0.5；n3變化3/7；人人人的gram集合只有人人；空集合比值ZeroDivisionError。',
 'observed':'所有預期已執行且assertion通過；原helper與獨立harness均CPU-only，exit0。',
 'details':'標準API親讀v3.13.5官方原稿與已安装版本一致。第一版自寫harness少帶_digest，修正後保留failed logs且最終script hash與執行receipt一致；該失敗不是教材錯誤。'}),
 claim('C3','concept','文字相近不保證同義；來源/改寫家族切分可以保留不同答案。片段法可漏改寫或誤合短句，門檻必須隨n與比較方法定義，沒有通用0.5標準。',
 'course/chapters/05.md:452,454','左/右不是同一正解；分組標籤不是目標標籤。此處為方法限制與後續設計建議，不宣稱目前已建成語意聚類或證明所有改寫都能找到。',[
 evidence('lee','§4.2 p.8427; §6.1 p.8429; AppendixA p.8436','原方法使用5-grams、Jaccard與edit filtering，選0.8且討論0.9與門檻對結果的影響，不能把本例0.5當普遍標準。'),
 evidence('group-doc','GroupShuffleSplit class description, Examples; split(X,y,groups) parameters','group為來源/領域分組資訊，與y分離；同一group在一次切分中放同側。'),
 evidence('independent-cpu','inspection-results.json toy and toy_family_preserves_distinct_labels','兩句相近但完整hash和左/右標籤不同；相同family保留left/right於一側；同義改寫詞面例可無共同bigram，短不等串可相同gram集，n2→n3分數改變。')],['lee-pdf','sklearn-groups','cpu-inspection']),
 claim('E1','empirical','T.4指定故事與詩文原始real_text實跑只做完整正規化內容去重，未聚近重複；可核對無完整重複跨組，但不能宣稱相似版本全隔開。',
 'course/chapters/05.md:459; linked 5.11 and T.4','核對既有原始報告與同版本實作/输入身份，不是重新訓練或重新跑模型評測。JSON的family在此是完整正規化SHA-256，非近重複家族。',[
 evidence('repo-text','_deduplicate_text lines91–100; run_real_text lines317–345','只normalization+complete-textSHA256覆寫family；run直接split，warning明說near-duplicate clustering not performed。'),
 evidence('repo-common','split_records lines50–70; text_examples lines77–97; fit_lm and evaluate_lm callers','先按完整內容family切分，只有train進fit，validation/test按窗口評測；無語意或近重複聚類。'),
 evidence('independent-cpu','inspection-results.json runs.*, original-results results.runs.*, input-provenance.json','既有原始512/365 input SHA等於run註冊input，6份完整重建split hash全等original JSON；counts與零跨組完整hash重疊；before/after windows/token分母及NLL/BPB比值重算。')],['cpu-inspection','original-results','input-provenance','historical-text','historical-common'],
 {'method':'executed','expected':'TinyStories512→512且409/51/52；prepared詩文365→365且292/36/37；六份split hash等原報告，完整fingerprint pairwise overlap=0；held-out分母等原JSON。',
 'observed':'全部精確一致；上游詩集366行已於資料準備移除1個exact rendered duplicate後得到365，real_text沒有再刪；512篇故事與完整prefix逐篇一致。',
 'details':'指定revision5581462…text.py與common.py SHA等於原報告code_sha256，親讀並只執行其去重/切分/保存/窗口生成函數。原GPU每來源800 steps僅核對記錄，未重跑。NLL nll_sum/targets与BPB nll_sum/(bytes*ln2)絕對差<1e-12，無新模型分數。',
 'denominators':{'source_records':{'tinystories':512,'chinese-poetry_prepared':365,'chinese-poetry_upstream':366},
 'train_validation_test_records':{'tinystories':[409,51,52],'chinese-poetry':[292,36,37]},
 'heldout':{name:run['denominators'] for name,run in results['runs'].items()},'recorded_training_steps_per_source':800,'training_steps_executed_this_review':0}}),
 claim('C4','concept','這份留出成績只支持小型訓練流程檢查；近重複未審計時，不能排除相似訓練材料提示或據此宣稱廣泛泛化。',
 'course/chapters/05.md:459','限制性結論與下一步家族判定建議，沒有指控已找到具體洩漏，也不把原論文大模型實驗套成此小模型的新成績。',[
 evidence('lee','§5.3 p.8429 Train/Test Set Leakage','近重複train-validation overlap可能提高偏向記憶模型的評価，故exact去重本身不是排除提示的證明。'),
 evidence('independent-cpu','inspection-results.json runs.*.split_warning; original-results results.runs.*.scope','原指定JSON保留not official test/near clustering not performed以及small fixed pilot，實作與數據身份核對一致。')],['lee-pdf','cpu-inspection','original-results']),
]
report={'schema_version':1,'review_stage':'technical','lesson_id':'5.12','source':'course/chapters/05.md#5.12',
 'source_sha256':extraction['source_sha256'],'figure_sha256':{},'reviewer_task':'/root/phase4_factual_coordinator/factual_5_12',
 'reviewer_context':'fresh','author_tasks':[],'reviewed_on':'2026-10-05','verdict':'pass',
 'read_scope':['5.12完整原始UTF-8 bytes','必要前置5.11完整節','明確引用T.4完整節（長recipe只讀契約）',
 '完整factual-reviewer-instructions/schema/section_facts/review-protocol','本文相關當前實作及報告revision5581462…原始實作','官方原始來源精確段落見inspection-notes.md'],
 'summary':'原文概念、原碼與n=3算例、原始完整內容切分及數據分母一致。它明確區分full-fingerprint family與尚未處理的near-duplicate family，保留pilot限制，無未解決實質問題。',
 'checks':{
 'factual_accuracy':{'status':'pass','details':'逐項查原論文/官方文件；文字片段、集合重疊、來源分組與正解標籤不同，no universal0.5的範圍一致。','claim_ids':['C1','C2','S1','C3','E1','C4']},
 'numeric_verification':{'status':'pass','details':'原碼4/8=0.5；真实n3變化3/7=0.42857142857142855；六份split hashes/records与四組heldout windows/targets/bytes及ratio逐項精确或1e-12核对。','claim_ids':['N1','E1']},
 'figure_consistency':{'status':'not_applicable','details':'本節沒有image/SVG/figure引用；字串與交集可直接讀文字和已執行stdout，不需要空間素材。沒有渲染，也不冒稱桌面/手機頁面視覺驗證。','claim_ids':[]},
 'source_verification':{'status':'pass','details':'本輪親讀ACL原PDF、出版社條目、Stanford作者原文、CPythonv3.13.5原稿、sklearn1.9.1官方API；原run實作hash與原報告相等，每claim各列locator/supports。','claim_ids':['C1','C2','N1','S1','C3','E1','C4']},
 'limitations':{'status':'pass','details':'原例是假定有效n/字串長度的比較函數，非訓練；n=3與边界反例不當成熟方法績效。原run只核身份/切分/分母，未用新模型成績補不確定；near duplicates未排除正文已明說。','claim_ids':['S1','C3','E1','C4']}},
 'sources':sources,'claims':claims,'issues':[],'artifacts':artifacts}
REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
manifest={'report':rel(REPORT),'report_sha256':sha(REPORT),'input_scope':'only 5.12 and necessary contracts/original inputs',
 'files':[{'path':rel(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'],
 'commands':['.venv/bin/python docs/review-tools/section_facts.py course/chapters/05.md#5.12 --output outputs/phase4-5_12-independent-extraction',
 '.venv/bin/python docs/review-tools/section_facts.py course/chapters/05.md#5.12 --output outputs/phase4-5_12-independent-original-cpu --execute --timeout 45',
 '.venv/bin/python docs/technical-reviews/artifacts/phase4-5_12-independent/code/prepare.py',
 'timeout 90 .venv/bin/python docs/technical-reviews/artifacts/phase4-5_12-independent/code/fetch_sources.py',
 'pdftotext -layout docs/technical-reviews/artifacts/phase4-5_12-independent/sources/lee-2022.acl-long.577.pdf docs/technical-reviews/artifacts/phase4-5_12-independent/sources/lee-2022.acl-long.577.txt',
 receipt['command'],'.venv/bin/python docs/technical-reviews/artifacts/phase4-5_12-independent/code/write_report.py'],
 'no_textbook_or_figure_edits':True,'no_model_construction_training_or_download':True,'weights_copied':False}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':rel(REPORT),'sha256':sha(REPORT),'artifacts':len(artifacts),'claims':len(claims),'verdict':'pass'}))
