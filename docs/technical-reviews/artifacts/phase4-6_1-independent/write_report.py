import hashlib
import json
from pathlib import Path

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=Path(__file__).parent
REL=OUT.relative_to(ROOT).as_posix()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
probe=json.loads((OUT/'probe.json').read_text())
meta=json.loads((OUT/'original/extraction.json').read_text())
intro=json.loads((OUT/'input-provenance.json').read_text())
env={k:str(v) for k,v in probe['environment'].items()}
original_env=json.loads((OUT/'original/environment.json').read_text())
original_command=json.loads((OUT/'original/execution.json').read_text())['command']
probe_command="CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python "+REL+'/probe.py'
artifacts=[]
artifact_names={}
for path in sorted(OUT.rglob('*')):
 if not path.is_file() or path.name in ['checker.stdout.txt','checker.stderr.txt','checker-receipt.json','write-report.stdout.txt','write-report.stderr.txt'] or '__pycache__' in path.parts:continue
 name=path.relative_to(OUT).as_posix()
 aid='A'+str(len(artifacts)+1)
 artifact_names[name]=aid
 kind='code' if path.suffix=='.py' else 'source_snapshot'
 entry={'id':aid,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'kind':kind,
        'description':'獨立審閱保存的原始輸入、原官方來源或實際命令記錄：'+name}
 if name in ['probe.json','probe.stdout.txt','original/execution.json','original/stdout.txt']:
  entry.update(kind='execution',command=probe_command if name.startswith('probe') else original_command,
   result='exit 0；原碼輸出3/9、3/3、1/4' if name.startswith('original') else 'exit 0；所有assert通過，原split SHA吻合；32新byte末尾e5 a4構成半字；貓在CharTokenizer(cat)還原成<unk>',
   environment=env if name.startswith('probe') else {k:str(original_env[k]) for k in ['python','torch','device_requested','cuda_build','cuda_available']})
 artifacts.append(entry)
def aids(*names):return [artifact_names[n] for n in names]
def primary(sid,kind,title,url,version,authority,note):
 return {'id':sid,'kind':kind,'title':title,'url':url,'version':version,'accessed_on':'2026-10-05',
         'authority_reason':authority,'verified':True,'checked_original':True,'inspection_note':note}
def repo(sid,title,path,version,note):
 p=OUT/path
 return {'id':sid,'kind':'repository_code','title':title,'path':p.relative_to(ROOT).as_posix(),
         'sha256':sha(p),'version':version,'verified':True,'inspection_note':note}
sources=[
 primary('S1','official_docs','Python Unicode HOWTO','https://docs.python.org/3.13/howto/unicode.html','Python 3.13 documentation, retrieved 2026-10-05','Python作者維護的語言文件',
   '親讀Definitions與Encodings（snapshot txt297–345、359–445）、The String Type（479–547）與Comparing Strings（731–744）；碼點/編碼/replace/組合重音長度皆依原頁核對，不靠摘要。'),
 primary('S2','official_docs','Python Built-in Types: str.encode and bytes','https://docs.python.org/3.13/library/stdtypes.html','Python 3.13 documentation; final URL /3.13/builtins/stdtypes.html','Python原官方API文件',
   '親讀str.encode（txt2597–2626）與Bytes Objects（4494–4504）：encode返回bytes，strict是預設，bytes為單byte序列；安裝Python3.13.5與文件同minor。'),
 primary('S3','official_docs','Python codecs Error Handlers','https://docs.python.org/3.13/library/codecs.html','Python 3.13 documentation','Python原官方codec契約',
   '親讀Error Handlers replace（txt752–759）及replace_errors（947–954）；解碼非法資料用U+FFFD，編碼替換規則與解碼不同。'),
 primary('S4','official_docs','Unicode Standard Annex #29: Unicode Text Segmentation','https://www.unicode.org/reports/tr29/tr29-47.html','Unicode 17.0.0; revision 47; 2025-08-17','Unicode Consortium發布文字分段規格',
   '親讀版本表、§3 Grapheme Cluster Boundaries（txt563–611）、GB11（1594–1605）及emoji sequence單grapheme註（1655）；使用者看見的字不必等於一碼點。'),
 primary('S5','official_docs','RFC 3629 UTF-8, a transformation format of ISO 10646','https://www.rfc-editor.org/rfc/rfc3629.txt','RFC 3629, November 2003','IETF Standards Track原標準，RFC Editor保存',
   '親讀§3（txt175–235）與§4（270–300）；UTF-8有效Unicode scalar值需1–4 octet，0x0800..FFFF通常3byte，0x10000..10FFFF為4byte，禁止surrogate。'),
 primary('S6','paper','Attention Is All You Need','https://arxiv.org/pdf/1706.03762v7','arXiv:1706.03762v7, 2 August 2023','作者上傳arXiv原論文版本',
   '親讀首頁作者/版本、§3.2.1 Eq.(1)、§3.4、Table1：QK^T產生query×key表，n為序列長度，dense self-attention O(n²d)。由index找到舊不可變原PDF後，另直接下載官方v7，SHA完全一致（bdfaa68d...），保存兩份並核對abs版本。'),
 primary('S7','paper','Language Models are Unsupervised Multitask Learners','https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf','Radford et al., OpenAI GPT-2 technical report, 2019','原作者OpenAI官方CDN的原研究報告',
   '親讀首頁作者與§2.2 Input Representation（p3，txt169–231）：UTF-8 bytes、byte base vocabulary256、codepoint基表可能大、byte/BPE中間粒度、無lossy preprocessing才可對任意Unicode字串建模。'),
 primary('S8','official_source','PyTorch Embedding source','https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/nn/modules/sparse.py','PyTorch v2.8.0 tag, torch/nn/modules/sparse.py','PyTorch官方repo標籤原始碼',
   '親讀Embedding docstring16–49與建構器140–175：weight=(num_embeddings,embedding_dim)。docs.pytorch.org/2.8的API頁403已保存，不算查證；改查此原碼。環境2.14.1+cpu與來源v2.8.0不同，僅使用兩者一致的Embedding形狀契約，已CPU親跑確認。'),
 primary('S9','official_docs','Hugging Face Transformers Tokenizer summary','https://huggingface.co/docs/transformers/v4.57.1/tokenizer_summary','Transformers documentation v4.57.1','工具維護者官方tokenizer規則文件',
   '親讀Introduction（txt208–255）、BPE未見字（337–343）、Byte-level BPE（347–353）。mug→[<unk>,ug]不能恢复m，證明tokenizer一般無exact inverse保證。本文概念不要求安裝/執行Transformers模型。'),
 repo('R1','當時ByteTokenizer/CharTokenizer程式','historical-code/tiny_perceptron/data.py','Git5581462ef01959636425eb7795ae3153142dbfeb；已吻合原result code_sha256',
      '親讀L15–26 byte±8與errors=replace、L32–44的字元UNK；不是從舊審閱抄判定。'),
 repo('R2','當時固定文字評估入口','historical-code/scripts/course_experiments/common.py','Git5581462ef01959636425eb7795ae3153142dbfeb；已吻合原result code_sha256',
      '親讀split_records L50–70及evaluate_lm L243–305；samples只取前8筆，逐碼點累加至token limit24，generate新token limit32。'),
 repo('R3','當時generate實作','historical-code/tiny_perceptron/model.py','Git5581462ef01959636425eb7795ae3153142dbfeb；已吻合原result code_sha256',
      '親讀L109–133；每步argmax一ID，限max_new_tokens、context、EOS，沒有UTF-8合法性constraint。'),
 repo('R4','當時real_text實驗與去重','historical-code/scripts/course_experiments/text.py','Git5581462ef01959636425eb7795ae3153142dbfeb；已吻合原result code_sha256',
      '親讀L91–99去重、L317–343固定入口與L65–72評估包裝。原始Json報292/36/37；本次只用AST執行2個純資料函式，重建split SHA，未訓練。'),
 repo('R5','當前manual_attention原實作','inputs/tiny_perceptron/attention.py','本次凍結內容SHA',
      '親讀L22–30，q@k.transpose及softmax(-1)；shape B,H,Tq,Tk，key是最後維。CPU跑B=H=1，Tq=Tk=3/9。'),
 {'id':'E1','kind':'execution','title':'原fence執行','verified':True,'artifact_id':artifact_names['original/execution.json']},
 {'id':'E2','kind':'execution','title':'長度變化/矩陣/原詩紀錄/UNK反例CPU核對','verified':True,'artifact_id':artifact_names['probe.json']},
 {'id':'D1','kind':'derivation','title':'byte budget與矩陣格數','verified':True,
  'details':'dense表每head每batch是Tq×Tk，同長3²=9、9²=81，比例9；V×D僅token embedding格數。32byte全ASCII可容32字元；全3byte常用中文floor(32/3)=10完整碼點餘2byte。非ASCII英文、emoji、EOS/control、提前停止會改變實際數量，正文常/約不能讀成保證。'}
]
def evidence(sid,locator,supports):return {'source_id':sid,'locator':locator,'supports':supports}
def verify(expected,observed,details,method='executed',**kw):
 return {'method':method,'expected':expected,'observed':observed,'details':details,**kw}
def claim(cid,kind,statement,location,scope,ev,ar,verification=None,status='verified'):
 r={'id':cid,'kind':kind,'statement':statement,'location':location,'scope':scope,'status':status,
    'evidence':ev,'artifact_ids':ar}
 if verification:r['verification']=verification
 return r
claims=[
 claim('C1','concept','token是模型處理的片段；tokenizer可切一字、多字或byte部分並轉ID；但原句把解碼還原原文寫成未限定的通用契約。','6.1段2，原chapter06 L9',
       '切分單位與ID部分成立；無損逆轉未成立，未知字或會改文字的前處理可失去資訊。此claim保留矛盾，需修句。',
       [evidence('S9','Introduction; Byte-Pair Encoding mug example; Byte-level BPE','多種切法及ID規則；未見m編為<unk>反證所有tokenizer必能還原。'),
        evidence('R1','CharTokenizer L32–44','未見貓會轉0，再decode為<unk>。'),
        evidence('E2','probe.json char_unseen_counterexample','實跑CharTokenizer(cat) encode/decode貓，equal_original=false。')],
       aids('probe.json','probe.py','sources/hf-tokenizer-summary.html','historical-code/tiny_perceptron/data.py'),status='contradicted'),
 claim('C2','concept','Unicode碼點是編號；可見字/emoji與Python碼點長度不是普遍一對一，組合重音和emoji可用多碼點。','6.1段5 Unicode句；開場三元素句',
       '小小貓、cat、🙂在指定字串中3/3/1；不是對全部文字或所有emoji的字元數定義。',
       [evidence('S1','Definitions; Comparing Strings','碼點編號與重音ê可為1或2碼點。'),
        evidence('S4','§3; GB11; emoji sequence note','user-perceived character可多碼點；emoji ZWJ sequence不分開。')],
       aids('sources/python-unicode.html','sources/unicode-uax29.html','probe.json')),
 claim('C3','software','原fence encode/len與練習cat🙂、貓🙂的UTF-8 byte和碼點數一致。','6.1原fence、程式後段與練習',
       '只是字串編碼/長度計算，不訓練、不梯度、不測語言模型品質；一般valid Unicode text，strict UTF-8編碼。',
       [evidence('S2','str.encode; Bytes Objects','encode返回UTF-8 bytes；len(raw)数byte而len(str)数碼點。'),
        evidence('S1','Comparing Strings','len(str)與code point序列一致，不是grapheme個數。'),
        evidence('S5','§3 encoding range table','指定字元UTF-8碼寬。'),
        evidence('E1','original stdout/execution/environment','原fence實際執行輸出3/9,3/3,1/4。'),
        evidence('E2','length_rows','獨立必要小變化cat🙂4/7、貓🙂2/7及多碼點反例。')],
       aids('original/fence-1.py','original/execution.json','original/stdout.txt','original/environment.json','probe.json','probe.py'),
       verify('小小貓3碼點9byte、cat3/3、🙂1/4；cat🙂4/7、貓🙂2/7','全部精確一致；byte tokenizer count等於UTF-8 byte數。','逐一保存碼點數值、raw bytes、roundtrip；原碼在.venv CPU執行，小變化另短CPU核對。')),
 claim('C4','numeric','一字元一token時三格，一byte一token時小小貓九格cat三格；dense注意力表T×T在3→9時9→81格。','6.1原碼解說與成本段（重複句一併核對）',
       '單batch單head、同長query/key的完整dense表；T指模型token，不是字串byte或grapheme；不宣稱全模型RAM/時間必9倍。',
       [evidence('S6','§3.2.1 Eq.(1); Table1','QK^T query×key且n是序列長；dense表T²。'),
        evidence('R5','manual_attention L22–30','key軸最後一維，weights.shape=B,H,T,T。'),
        evidence('E2','attention; length_rows','T3/9表numel9/81，B=H=1，row sums1。'),
        evidence('D1','3² and 9²','精確格數與比例。')],
       aids('probe.json','probe.py','sources/transformer-v7-direct.pdf','inputs/tiny_perceptron/attention.py'),
       verify('3×3=9；9×9=81','weights shape[1,1,3,3]/[1,1,9,9]；numel9/81','親查query/key軸，3与9是普通文字token數，未加BOS/EOS。',tolerance='整數精確相等；row sum float允許通常roundoff，實得1。')),
 claim('C5','concept','V是可用token種類；token embedding V×D隨詞表成本改變，常見粒度取捨是較細V小T長、較粗V大T短，不能僅字表小稱全模型省。','6.1成本段與Unicode段',
       'V×D僅embedding參數；常見趨勢而非所有字表與文本的單調定理，實際T取決於規則/文本，整體成本還有attention、其他參數。',
       [evidence('S8','Embedding attributes L39–40; constructor L166–170','weight shape(num_embeddings,embedding_dim)=V,D。'),
        evidence('S6','§3.4 and Table1','token embedding向量及序列長成本兩方面。'),
        evidence('S7','§2.2 Input Representation','byte base256對完整codepoint大基表，以及word/character間BPE粒度。'),
        evidence('E2','embedding_shape/parameters','親跑Embedding(256,32)的weight256×32=8192。')],
       aids('sources/pytorch-v2_8_0-sparse.py','sources/gpt2-report.pdf','probe.json')),
 claim('C6','concept','完整256-byte起始集合可表示未見字；表示完好文字不保證模型生成合法UTF-8。','6.1Unicode段與補充開頭',
       '指valid Unicode scalar文字先UTF-8編碼，具備全部256 byte並原樣拼回；不含任意surrogate字串、不含normalizer無損契約，也不是任意生成byte序列必合法。',
       [evidence('S7','§2.2 Input Representation','256基byte表能建模任意Unicode字串，无未知byte。'),
        evidence('S5','§3 and §4','UTF-8合法多byte序列有範圍與continuation要求，不是任意byte串。'),
        evidence('R1','ByteTokenizer L21–26','UTF-8 bytes±8；decode errors replace。'),
        evidence('R3','generate L116–129','每步選一ID，无UTF-8 constraint。'),
        evidence('E2','length_rows; poem.strict_decode_error','未見字🦊roundtrip；原錄詩生成byte末尾非法，實證這兩件事不同。')],
       aids('sources/gpt2-report.pdf','sources/rfc3629.txt','probe.json')),
 claim('C7','empirical','原中文詩驗證prompt渡漢江\n李頻\n嶺外保持完整碼點，新增孟沙，天夜天天天天春�為32個byte，最後2byte達到上限而非prompt半字或EOS。','6.1補充實作原始紀錄',
       '核對既有固定小型訓練的單筆展示，不重跑模型或訓練，不概括中文能力。原始驗證36篇，僅展示前8篇，此筆index7；32個新增ID全部普通byte，BOS額外1個。',
       [evidence('R2','evaluate_lm L288–305; split_records L50–70','整碼點prompt limit24、前8样本、生成32上限。'),
        evidence('R3','generate L116–129','EOS/context/新位置限停止；此筆无EOS，總56位置遠小于context128。'),
        evidence('R4','_deduplicate_text L91–99; run_real_text L317–343','歷史資料/seed42与评估入口。'),
        evidence('E2','probe.json poem; original raw result snapshot /results/runs/chinese-poetry/after/validation/samples/7','重建已有local archive/split SHA，確認樣本prompt23byte；raw新32byte解碼与原结果字串相等，末e5 a4在byte30–32非法。'),
        evidence('S3','Error Handlers replace','decode errors replace用U+FFFD。')],
       aids('inputs/docs/course-experiments/results/real_text.json','inputs/poem-selected-original-record.json','probe.json','probe.py','input-provenance.json','historical-code/scripts/course_experiments/common.py','historical-code/tiny_perceptron/data.py','historical-code/tiny_perceptron/model.py','historical-code/scripts/course_experiments/text.py'),
       verify('prompt完整且≤24byte；新生成32byte末2byte造成�','prompt23byte/9碼點，下一音3byte將超24；32新ID无特殊token，最後e5 a4，前30byte為10完整碼點；strict decode byte30–32 unexpected end of data。','原JSON code hash匹配歷史Git四檔，原local archive SHA與member SHA匹配；用歷史兩純函式重建train/validation/test hash全部吻合；未讀權重/未重跑推論或訓練。',
         denominators={'original_source_records':365,'deduplicated_records':365,'train_records':292,'validation_records':36,'test_records':37,'displayed_validation_samples':8,'selected_sample_zero_based_index':7,'prompt_utf8_bytes':23,'prompt_limit_bytes':24,'new_byte_tokens':32,'complete_generated_codepoints':10,'final_partial_bytes':2,'original_training_steps':800,'current_training_steps':0})),
 claim('C8','numeric','同32-byte預算ASCII英文常可32字母/標點，常用3byte中文字約10字；補完末字不能移除已產生的重複字。','6.1補充末句',
       '条件：普通ASCII一byte，通常中文BMP漢字3byte；無EOS提前停或非法控制token時是容量數而非生成品質保證。emoji/扩展漢字可4byte，重複在前30byte已经存在。',
       [evidence('S5','§3 UTF-8 range table','ASCII1byte与BMP漢字常3byte；其他Unicode字符不必3byte。'),
        evidence('D1','32/1; floor32/3','32/10容量和2剩余byte。'),
        evidence('E2','poem.complete_utf8_prefix','前30byte已有天天天天春；尾部補字不改前綴。')],
       aids('probe.json','sources/rfc3629.txt'),
       verify('ASCII容量32；3byte汉字容量10完整碼點餘2','手算32/1=32，floor32/3=10餘2；原sample前30byte10碼點','是UTF-8 byte预算推導；不能拿32 token当32中文或把容量当语言品質。',method='hand_calculation',tolerance='整數精確相等'))
]
issue={'id':'I1','status':'open','claim_ids':['C1'],'severity':'material_scope',
 'original_text':'tokenizer將原文拆片段並轉ID，再能解碼還原。',
 'finding':'解碼得到文字不等於無損還原原文；原敘述未限定字表或前處理。',
 'evidence':'官方Transformers v4.57.1 BPE段的mug→<unk>,ug；本課CharTokenizer(cat)的貓→[0]→<unk>（probe.json）。',
 'impact':'讀者會誤認所有tokenizer都可還原未見文字，混淆本節完整byte表和後續BPE的特定還原契約。',
 'suggestion':'改為「tokenizer將原文拆片段並轉ID；decoder把ID轉回文字，能否完整還原原文取決於字表與前處理規則。」或等價限定。',
 'resolution':'尚未修正；由協調者局部修句，原審閱者再親讀新版與反例，保留此原問題。'}
report={'schema_version':1,'review_stage':'technical','lesson_id':'6.1','source':'course/chapters/06.md#6.1',
 'source_sha256':meta['source_sha256'],'figure_sha256':{},'reviewer_task':'/root/phase4_factual_coordinator/factual_6_1',
 'reviewer_context':'fresh','reviewed_on':'2026-10-05','verdict':'revise',
 'read_scope':intro['read_scope'],'intro_sha256':intro['intro_sha256'],'intro_summary':intro['intro_summary'],
 'intro_source':'course/chapters/06.md bytes before first ##6.1 heading; raw snapshot inputs/chapter-intro.raw.md',
 'artifacts':artifacts,'sources':sources,'claims':claims,'issues':[issue],
 'checks':{
   'factual_accuracy':{'status':'revise','details':'C1的分詞/ID定義成立，但普遍無損還原契約被官方文件與實跑反例否定，保持revise；其餘實質claim逐一核對。','claim_ids':[c['id'] for c in claims]},
   'numeric_verification':{'status':'pass','details':'原fence、兩項練習、dense表9/81、V×D、32byte預算皆查單位/軸；原詩原32 IDs/末2byte/23byteprompt均可回追。','claim_ids':['C3','C4','C7','C8']},
   'figure_consistency':{'status':'not_applicable','details':'親讀6.1全文與helper extraction：本節沒有SVG或其他圖引用，figure_sha256={}，因此無可render/view的本節圖；不以未看圖宣稱通過。','claim_ids':[]},
   'source_verification':{'status':'pass','details':'原官方HTML/PDF/原始碼皆親讀具體段；arXiv官方直接PDF hash和索引原PDF一致；GPT2作者CDN；original result版碼hash與input archive/split SHA全部吻合。不引用舊審閱結論。','claim_ids':[c['id'] for c in claims]},
   'limitations':{'status':'revise','details':'其餘scope已明示：dense表非全機memory，常/約非所有文字普遍定律，UTF-8 scalar不含surrogate；一筆既有生成不代表一般模型品質。但C1原文仍欠還原契約條件。','claim_ids':['C1','C4','C5','C6','C7','C8']}}
}
(ROOT/'docs/technical-reviews/6.1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':'docs/technical-reviews/6.1.json','verdict':report['verdict'],'source_sha256':report['source_sha256'],'intro_sha256':report['intro_sha256'],'claims':len(claims),'artifacts':len(artifacts)},ensure_ascii=False))
